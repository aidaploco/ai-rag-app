import inspect
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from opentelemetry import trace as otel_trace

P = ParamSpec("P")
R = TypeVar("R")


def trace(span_name: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        auto_name = span_name or f"{fn.__module__}.{fn.__qualname__}"
        is_async = inspect.iscoroutinefunction(fn)

        tracer = otel_trace.get_tracer(fn.__module__)

        if is_async:

            @wraps(fn)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                with tracer.start_as_current_span(auto_name):
                    return await fn(*args, **kwargs)  # type: ignore[misc]

            return async_wrapper  # type: ignore[return-value]

        @wraps(fn)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with tracer.start_as_current_span(auto_name):
                return fn(*args, **kwargs)

        return sync_wrapper

    return decorator
