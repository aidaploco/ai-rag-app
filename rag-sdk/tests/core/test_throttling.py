import asyncio

from opentelemetry import metrics as otel_metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from rag_sdk.core.throttling import metered_throttled_function, throttled_function


async def test_throttled_respects_concurrency():
    max_concurrent = 0
    current = 0

    @throttled_function(concurrency=3, tps=100)
    async def work():
        nonlocal max_concurrent, current
        current += 1
        max_concurrent = max(max_concurrent, current)
        await asyncio.sleep(0.01)
        current -= 1

    await asyncio.gather(*[work() for _ in range(10)])
    assert max_concurrent <= 3


async def test_locked_reflects_semaphore():
    @throttled_function(concurrency=1, tps=100)
    async def work():
        await asyncio.sleep(0.05)

    assert not work.locked()
    task = asyncio.create_task(work())
    await asyncio.sleep(0.001)
    assert work.locked()
    await task
    assert not work.locked()


async def test_metered_throttled_respects_concurrency():
    max_concurrent = 0
    current = 0

    @metered_throttled_function(concurrency=3, tps=100)
    async def work():
        nonlocal max_concurrent, current
        current += 1
        max_concurrent = max(max_concurrent, current)
        await asyncio.sleep(0.01)
        current -= 1

    await asyncio.gather(*[work() for _ in range(10)])
    assert max_concurrent <= 3


async def test_metered_throttled_emits_metrics():
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    otel_metrics.set_meter_provider(provider)

    @metered_throttled_function(concurrency=5, tps=100)
    async def work():
        await asyncio.sleep(0.001)

    await asyncio.gather(*[work() for _ in range(3)])

    data = reader.get_metrics_data()
    all_names = {
        m.name for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics
    }

    parts = [p for p in work.__qualname__.split(".") if not p.startswith("<")]
    expected_prefix = ".".join(parts[-2:] if len(parts) >= 2 else parts)
    assert f"{expected_prefix}.waiting" in all_names
    assert f"{expected_prefix}.running" in all_names
    assert f"{expected_prefix}.duration_ms" in all_names
