"""Tiny retry helper for provider API calls.

Gemini's free tier enforces per-minute and per-day request caps. When you
exceed them the server returns HTTP 429 with a ``RESOURCE_EXHAUSTED`` error.
The right response is to wait and try again, not to crash, so we wrap every
provider call in a small exponential-backoff loop.

Why exponential? Because retrying immediately after a 429 just earns you
another 429. Doubling the wait (1s, 2s, 4s, 8s) gives the server's per-minute
window time to roll over.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

from openai import APIStatusError, RateLimitError

T = TypeVar("T")

# 1 + 2 + 4 + 8 + 16 + 32 = 63s of total backoff across 6 retries before
# giving up. The 32s tail matters because Gemini's free-tier RPM window
# resets on a ~minute boundary — a 15s ceiling sometimes wasn't enough.
_DEFAULT_BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def _is_rate_limit(exc: BaseException) -> bool:
    """True if ``exc`` looks like an HTTP 429 from the provider."""
    if isinstance(exc, RateLimitError):
        return True
    # Gemini sometimes surfaces quota errors as a generic APIStatusError(429)
    # rather than the dedicated RateLimitError subclass.
    status = getattr(exc, "status_code", None)
    return isinstance(exc, APIStatusError) and status == 429


def with_retries(
    fn: Callable[[], T],
    *,
    backoff_seconds: tuple[float, ...] = _DEFAULT_BACKOFF_SECONDS,
) -> T:
    """Call ``fn``; on a 429, sleep and retry with exponential backoff.

    The last attempt's exception is re-raised so callers still see a real
    error if the provider keeps refusing.
    """
    attempts = len(backoff_seconds) + 1  # initial try + N retries
    last_exc: BaseException | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — re-raised below if non-429
            if not _is_rate_limit(exc) or i == attempts - 1:
                raise
            last_exc = exc
            time.sleep(backoff_seconds[i])
    # Unreachable: the loop either returns or raises.
    assert last_exc is not None
    raise last_exc
