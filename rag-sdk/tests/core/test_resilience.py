import time

import pytest

from rag_sdk.core.resilience import with_retry


async def test_succeeds_without_retry():
    call_count = 0

    @with_retry(ValueError, max_attempts=3)
    async def always_ok():
        nonlocal call_count
        call_count += 1
        return "ok"

    assert await always_ok() == "ok"
    assert call_count == 1


async def test_retries_on_matching_exception():
    call_count = 0

    @with_retry(ValueError, min_wait=0.0, max_wait=0.0, max_attempts=3)
    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("not yet")
        return "ok"

    assert await flaky() == "ok"
    assert call_count == 3


async def test_raises_after_max_attempts():
    @with_retry(ValueError, min_wait=0.0, max_wait=0.0, max_attempts=3)
    async def always_fails():
        raise ValueError("always")

    with pytest.raises(ValueError):
        await always_fails()


async def test_does_not_retry_non_matching_exception():
    call_count = 0

    @with_retry(ValueError, min_wait=0.0, max_wait=0.0, max_attempts=3)
    async def raises_type_error():
        nonlocal call_count
        call_count += 1
        raise TypeError("wrong type")

    with pytest.raises(TypeError):
        await raises_type_error()
    assert call_count == 1


async def test_uses_retry_after_from_attribute():
    class RateLimitError(Exception):
        retry_after = 0.001

    call_count = 0

    @with_retry(RateLimitError, min_wait=60.0, max_wait=120.0, max_attempts=3)
    async def rate_limited():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RateLimitError()
        return "ok"

    start = time.monotonic()
    assert await rate_limited() == "ok"
    assert time.monotonic() - start < 1.0  # would be 60s+ with exponential backoff


async def test_uses_retry_after_from_response_headers():
    class HttpRateLimitError(Exception):
        class response:
            headers = {"Retry-After": "0.001"}

    call_count = 0

    @with_retry(HttpRateLimitError, min_wait=60.0, max_wait=120.0, max_attempts=3)
    async def http_rate_limited():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise HttpRateLimitError()
        return "ok"

    start = time.monotonic()
    assert await http_rate_limited() == "ok"
    assert time.monotonic() - start < 1.0
