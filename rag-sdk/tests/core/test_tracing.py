import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rag_sdk.core.tracing import trace


@pytest.fixture(scope="module")
def span_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture(autouse=True)
def clear_spans(span_exporter: InMemorySpanExporter):
    span_exporter.clear()
    yield


async def test_trace_async_creates_span(span_exporter: InMemorySpanExporter):
    @trace()
    async def my_function():
        return 42

    assert await my_function() == 42

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == f"{my_function.__module__}.{my_function.__qualname__}"


def test_trace_sync_creates_span(span_exporter: InMemorySpanExporter):
    @trace()
    def my_function():
        return 42

    assert my_function() == 42

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == f"{my_function.__module__}.{my_function.__qualname__}"


async def test_trace_custom_span_name(span_exporter: InMemorySpanExporter):
    @trace(span_name="custom.operation")
    async def my_function():
        pass

    await my_function()

    spans = span_exporter.get_finished_spans()
    assert spans[0].name == "custom.operation"


async def test_trace_span_ends_on_exception(span_exporter: InMemorySpanExporter):
    @trace()
    async def raises():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        await raises()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1  # span is still recorded even on exception
