import asyncio
import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import ParamSpec, Protocol, TypeVar, cast

from aiolimiter import AsyncLimiter
from opentelemetry import metrics

P = ParamSpec("P")
R = TypeVar("R")


def _metric_base(fn: Callable) -> str:
    # OTel names are capped at 63 chars; qualnames with <locals> can be arbitrarily long.
    # Keep only the last two meaningful parts (skipping <locals>, <lambda>, etc.).
    parts = [p for p in fn.__qualname__.split(".") if not p.startswith("<")]
    return ".".join(parts[-2:] if len(parts) >= 2 else parts)


class ThrottledCallable(Protocol[P, R]):  # type: ignore[misc]
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Awaitable[R]: ...
    def locked(self) -> bool: ...


def throttled_function(
    concurrency: int = 20,
    tps: int = 15,
) -> Callable[[Callable[P, Awaitable[R]]], ThrottledCallable[P, R]]:
    def decorator(fn: Callable[P, Awaitable[R]]) -> ThrottledCallable[P, R]:
        semaphore = asyncio.Semaphore(concurrency)
        limiter = AsyncLimiter(tps, 1)

        @wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            async with semaphore:
                async with limiter:
                    return await fn(*args, **kwargs)

        wrapper.locked = semaphore.locked  # type: ignore[attr-defined]
        return cast("ThrottledCallable[P, R]", wrapper)

    return decorator


def metered_throttled_function(
    concurrency: int = 20,
    tps: int = 15,
) -> Callable[[Callable[P, Awaitable[R]]], ThrottledCallable[P, R]]:
    meter = metrics.get_meter(__name__)

    def decorator(fn: Callable[P, Awaitable[R]]) -> ThrottledCallable[P, R]:
        semaphore = asyncio.Semaphore(concurrency)
        limiter = AsyncLimiter(tps, 1)
        condition = asyncio.Condition()

        fn_name = _metric_base(fn)
        waiting_counter = meter.create_up_down_counter(
            f"{fn_name}.waiting",
            description="Calls waiting for a throttle slot",
        )
        running_counter = meter.create_up_down_counter(
            f"{fn_name}.running",
            description="Calls currently executing",
        )
        duration_histogram = meter.create_histogram(
            f"{fn_name}.duration_ms",
            unit="ms",
            description="Execution duration in milliseconds",
        )

        @wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            waiting_counter.add(1)
            async with semaphore:
                async with limiter:
                    waiting_counter.add(-1)
                    running_counter.add(1)
                    async with condition:
                        condition.notify_all()
                    start = time.monotonic()
                    try:
                        return await fn(*args, **kwargs)
                    finally:
                        duration_histogram.record(
                            (time.monotonic() - start) * 1000
                        )
                        running_counter.add(-1)
                        async with condition:
                            condition.notify_all()

        wrapper.locked = semaphore.locked  # type: ignore[attr-defined]
        return cast("ThrottledCallable[P, R]", wrapper)

    return decorator
