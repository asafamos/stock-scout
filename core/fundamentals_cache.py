"""Per-(ticker, source) cache of raw fundamentals fetch results.

Why (2026-11-15): fundamentals change about once a quarter, but every scan (up to 5 a day, ~2,000 tickers,
several providers each) re-fetched all of them. That exhausted the FMP plan limit on 2026-11-13; from then on
every fundamentals fetch timed out, the scan dropped to ~29 rows and the integrity guard blocked it.

Behaviour
- `get(ticker, source, max_age_days)` returns the cached raw provider result (a dict) when it is fresh enough.
- `put(ticker, source, data)` stores a successful provider result.
- Two ages are used by the caller: a short TTL (skip the API call entirely) and a long "stale" age (use the old
  value when the live call failed — better than no fundamentals at all during a provider outage).
- Persisted as one JSON file (atomic write), loaded lazily, thread-safe; saved every `AUTOSAVE_EVERY` writes and
  at interpreter exit. GitHub runners are ephemeral, so the workflow restores/saves the file with actions/cache.

Env: FUND_CACHE=0 disables everything; FUND_CACHE_TTL_DAYS (default 7); FUND_CACHE_STALE_MAX_DAYS (default 45);
FUND_CACHE_PATH (default data/cache/fundamentals_cache.json).
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

AUTOSAVE_EVERY = 300

_lock = threading.RLock()
_cache: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
_dirty_writes = 0
_stats = {"hit": 0, "stale_hit": 0, "miss": 0, "put": 0}


def enabled() -> bool:
    return os.getenv("FUND_CACHE", "1").strip().lower() not in ("0", "false", "no", "off")


def ttl_days() -> float:
    return float(os.getenv("FUND_CACHE_TTL_DAYS", "7"))


def stale_max_days() -> float:
    return float(os.getenv("FUND_CACHE_STALE_MAX_DAYS", "45"))


def _path() -> Path:
    return Path(os.getenv("FUND_CACHE_PATH", "data/cache/fundamentals_cache.json"))


def _load() -> Dict[str, Dict[str, Dict[str, Any]]]:
    global _cache
    with _lock:
        if _cache is None:
            _cache = {}
            p = _path()
            try:
                if p.exists():
                    _cache = json.loads(p.read_text())
                    if not isinstance(_cache, dict):
                        _cache = {}
                    logger.info("fundamentals cache: loaded %d tickers from %s", len(_cache), p)
            except (OSError, ValueError) as e:  # corrupt/unreadable -> start empty, never crash a scan
                logger.warning("fundamentals cache unreadable (%s) — starting empty", e)
                _cache = {}
        return _cache


def reset_for_tests(path: Optional[str] = None) -> None:
    """Drop the in-memory cache (tests / path changes)."""
    global _cache, _dirty_writes
    with _lock:
        _cache = None
        _dirty_writes = 0
        for k in _stats:
            _stats[k] = 0
        if path is not None:
            os.environ["FUND_CACHE_PATH"] = path


def get(ticker: str, source: str, max_age_days: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Cached raw result for (ticker, source) if not older than max_age_days (default: the TTL)."""
    if not enabled():
        return None
    age_limit = ttl_days() if max_age_days is None else max_age_days
    with _lock:
        entry = _load().get(ticker, {}).get(source)
    if not entry:
        _stats["miss"] += 1
        return None
    age_days = (time.time() - float(entry.get("ts", 0))) / 86400.0
    if age_days > age_limit:
        _stats["miss"] += 1
        return None
    _stats["hit" if max_age_days is None else "stale_hit"] += 1
    return entry.get("data")


def put(ticker: str, source: str, data: Dict[str, Any]) -> None:
    """Store a successful provider result."""
    global _dirty_writes
    if not enabled() or not data:
        return
    with _lock:
        _load().setdefault(ticker, {})[source] = {"ts": time.time(), "data": data}
        _stats["put"] += 1
        _dirty_writes += 1
        if _dirty_writes >= AUTOSAVE_EVERY:
            save()


def save() -> None:
    """Atomically write the cache to disk (no-op when nothing was loaded or written)."""
    global _dirty_writes
    if not enabled():
        return
    with _lock:
        if _cache is None or _dirty_writes == 0:
            return
        p = _path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(_cache, f, default=str)
            os.replace(tmp, p)
            _dirty_writes = 0
        except (OSError, TypeError, ValueError) as e:
            logger.warning("fundamentals cache save failed: %s", e)


def stats() -> Dict[str, int]:
    return dict(_stats)


atexit.register(save)
