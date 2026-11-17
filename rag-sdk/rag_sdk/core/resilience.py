import logging
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import inflect
import tenacity
from opentelemetry import metrics

from rag_sdk.core.naming import metric_base

P = ParamSpec("P")
R = TypeVar("R")

_inflect = inflect.engine()
logger = logging.getLogger(__name__)


def _get_retry_after(exc: BaseException) -> float | None:
    retry_after = getattr(exc, "retry_after", None)
    if isinstance(retry_after, (int, float)):
        return float(retry_after)
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {})
        val = headers.get("Retry-After") or headers.get("retry-after")
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None


def with_retry(
    *exception_types: type[BaseException],
    max_attempts: int = 5,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
    if not exception_types:
        raise ValueError("with_retry requires at least one exception type")
    meter = metrics.get_meter(__name__)

    def decorator(
        fn: Callable[P, Coroutine[Any, Any, R]],
    ) -> Callable[P, Coroutine[Any, Any, R]]:
        retry_counter = meter.create_counter(
            f"{metric_base(fn)}.retries",
            description="Total retry attempts",
        )

        def _before_sleep(retry_state: tenacity.RetryCallState) -> None:
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            wait_secs = retry_state.next_action.sleep if retry_state.next_action else 0.0
            ordinal = _inflect.ordinal(str(retry_state.attempt_number))
            logger.warning(
                "%s: %s retry in %.1fs — %s",
                fn.__qualname__,
                ordinal,
                wait_secs,
                type(exc).__name__ if exc else "unknown",
            )
            retry_counter.add(1)

        def _wait(retry_state: tenacity.RetryCallState) -> float:
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            if exc is not None:
                retry_after = _get_retry_after(exc)
                if retry_after is not None:
                    return retry_after
            return tenacity.wait_random_exponential(min=min_wait, max=max_wait)(retry_state)

        retrying = tenacity.AsyncRetrying(
            retry=tenacity.retry_if_exception_type(tuple(exception_types)),
            wait=_wait,
            stop=tenacity.stop_after_attempt(max_attempts),
            before_sleep=_before_sleep,
            reraise=True,
        )

        @wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            async for attempt in retrying:
                with attempt:
                    return await fn(*args, **kwargs)
            raise AssertionError("unreachable")

        return wrapper

    return decorator
