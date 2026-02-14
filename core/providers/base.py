"""
Shared provider infrastructure — re-exported from data_sources_v2.

This module provides the canonical imports for provider shared utilities.
The implementations live in core.data_sources_v2 to avoid circular imports
and shared-state fragmentation. Provider files import from here.
"""
from __future__ import annotations

# Re-export shared infrastructure from data_sources_v2
# Uses lazy imports to avoid circular dependency at module load time
def _http_get_with_retry(*args, **kwargs):
    from core.data_sources_v2 import _http_get_with_retry as _impl
    return _impl(*args, **kwargs)


def _rate_limit(source: str) -> None:
    from core.data_sources_v2 import _rate_limit as _impl
    _impl(source)


def _get_from_cache(cache_key: str, ttl: int = 14400):
    from core.data_sources_v2 import _get_from_cache as _impl
    return _impl(cache_key, ttl)


def _put_in_cache(cache_key: str, data: dict) -> None:
    from core.data_sources_v2 import _put_in_cache as _impl
    _impl(cache_key, data)


class WindowRateLimiter:
    """Re-export of the canonical WindowRateLimiter from data_sources_v2."""
    def __init__(self, max_calls: int, window_seconds: int):
        from core.data_sources_v2 import WindowRateLimiter as _Impl
        self._impl = _Impl(max_calls, window_seconds)

    def acquire(self) -> None:
        self._impl.acquire()
