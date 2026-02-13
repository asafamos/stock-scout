"""
Multi-Source Data Aggregation Module (v2) — Canonical Layer
===========================================================

This is the canonical multi-provider data layer used by the pipeline.
It fetches from multiple providers, cross-checks, and exposes structured
metadata required downstream for reliability and risk-aware scoring.

Guarantees / exposed reliability-related fields (when available):
- Fundamental_Coverage_Pct: float [0,100]
- Fundamental_Sources_Count: int
- sources_used: List[str] for fundamentals
- coverage: Dict[field -> List[sources]]
- disagreement_score: float [0,1] summary across fundamentals
- Price_Mean / price_mean: float
- Price_STD / price_std: float
- Price_Sources_Count / price_sources: int
- prices_by_source: Dict[source -> price]

Do NOT implement scoring here. This layer only fetches/aggregates and surfaces
the raw and derived reliability inputs so scoring and classification can consume
canonical fields consistently.

Implementation highlights:
- Priority chain: FMP → Finnhub → Tiingo → Alpha Vantage
- Multi-source fusion: aggregate data from ALL available sources
- Per-source caching with TTL; retry and timeout protection
- NaN-safe aggregation
"""
from __future__ import annotations
import os
import time
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from functools import lru_cache
import logging
from core.api_monitor import record_api_call

# Facade: re-export main provider functions from core/providers
from core.providers import (
    _http_get_with_retry,
    WindowRateLimiter,
    get_fmp_fundamentals,
    get_finnhub_fundamentals,
    get_tiingo_fundamentals,
    get_alpha_vantage_fundamentals,
    get_polygon_fundamentals,
    get_eodhd_fundamentals,
)
from core.exceptions import (
    DataFetchError, RateLimitError, AuthenticationError, 
    ProviderUnavailableError, DataValidationError, classify_http_error
)
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from core.fundamental import DataMode
from core.fundamental_store import init_fundamentals_store, save_fundamentals_snapshot, load_fundamentals_as_of
from core.config import get_secret
from datetime import date
from core.provider_guard import get_provider_guard

logger = logging.getLogger(__name__)
def normalize_ohlcv(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Normalize an OHLCV DataFrame to a canonical schema:
    - Lowercase columns: open, high, low, close, volume
    - Accept common variants: 'Open','High','Low','Close','Adj Close','Volume','o','h','l','c','v'
    - If only 'Adj Close' exists, map it to 'close' (prefer explicit 'Close' if both exist)
    - Ensure sorted by datetime ascending (use index if DatetimeIndex, else 'date'/'Date' columns)
    - Drop rows where all OHLCV are NaN
    - Ensure numeric types; keep NaN as NaN (do NOT fill with 0)

    Returns the normalized DataFrame containing available columns among
    ['open','high','low','close','volume'] and preserves any 'date' column if present.
    """
    try:
        if df is None:
            return pd.DataFrame()
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

        dfn = df.copy()
        # Normalize column names to lowercase for mapping
        rename_map = {}
        for col in dfn.columns:
            lc = str(col).strip().lower()
            if lc in {"open", "o"}:
                rename_map[col] = "open"
            elif lc in {"high", "h"}:
                rename_map[col] = "high"
            elif lc in {"low", "l"}:
                rename_map[col] = "low"
            elif lc in {"close", "c"}:
                rename_map[col] = "close"
            elif lc in {"adj close", "adj_close", "adjusted close"}:
                rename_map[col] = "adj_close"
            elif lc in {"volume", "v"}:
                rename_map[col] = "volume"
            elif lc in {"date", "datetime"}:
                rename_map[col] = "date"
        if rename_map:
            dfn = dfn.rename(columns=rename_map)

        # If close missing but adj_close exists, use adj_close as close
        if ("close" not in dfn.columns) and ("adj_close" in dfn.columns):
            dfn["close"] = dfn["adj_close"]

        # Ensure datetime sorted ascending
        try:
            if isinstance(dfn.index, pd.DatetimeIndex):
                dfn = dfn.sort_index()
            elif "date" in dfn.columns:
                dfn["date"] = pd.to_datetime(dfn["date"])
                dfn = dfn.sort_values("date")
        except (TypeError, ValueError, KeyError) as exc:
            logger.debug(f"Date sorting failed in normalize_ohlcv: {exc}")

        # Enforce numeric types without coercing NaN to 0
        for col in ["open", "high", "low", "close", "volume"]:
            if col in dfn.columns:
                dfn[col] = pd.to_numeric(dfn[col], errors="coerce")

        # Drop rows where all OHLCV are NaN
        subset_cols = [c for c in ["open","high","low","close","volume"] if c in dfn.columns]
        if subset_cols:
            dfn = dfn.dropna(how="all", subset=subset_cols)

        # Keep only canonical columns + date if present
        keep = [c for c in ["date","open","high","low","close","volume"] if c in dfn.columns]
        return dfn[keep] if keep else dfn
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        # On failure, return empty to allow diagnostics to mark missing
        logger.debug(f"normalize_ohlcv failed: {exc}")
        return pd.DataFrame()

# Initialize fundamentals store at import time so DB/table exist for live snapshots
try:
    init_fundamentals_store()
except (IOError, OSError, RuntimeError) as exc:
    # Non-fatal; live snapshot persistence will simply be skipped if init fails
    logger.warning(f"Failed to initialize fundamentals store (SQLite): {exc}")

# API Keys via unified secret loader
FMP_API_KEY = get_secret("FMP_API_KEY", "")
FINNHUB_API_KEY = get_secret("FINNHUB_API_KEY", "")
TIINGO_API_KEY = get_secret("TIINGO_API_KEY", "")
ALPHA_VANTAGE_API_KEY = get_secret("ALPHA_VANTAGE_API_KEY", "")
POLYGON_API_KEY = get_secret("POLYGON_API_KEY", "")
# Additional providers (registry expansion)
EODHD_API_KEY = get_secret("EODHD_API_KEY", "")
SIMFIN_API_KEY = get_secret("SIMFIN_API_KEY", "")
MARKETSTACK_API_KEY = get_secret("MARKETSTACK_API_KEY", "")
NASDAQ_API_KEY = get_secret("NASDAQ_API_KEY", "")

# Track last-used primary source for index/ETF symbols
_LAST_INDEX_SOURCE: Dict[str, str] = {}

def get_last_index_source(symbol: str) -> Optional[str]:
    return _LAST_INDEX_SOURCE.get(symbol)

# Cache configuration (shared across threads)
_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()
CACHE_TTL_SECONDS = 14400  # 4 hours to reduce rate-limit issues

# Rate limiting (shared across threads)
_LAST_CALL_TIME: Dict[str, float] = {}
_RATE_LOCK = threading.Lock()
MIN_INTERVAL_SECONDS = {
    "fmp": 0.1,        # 10 calls/sec
    "finnhub": 0.2,    # 5 calls/sec
    "tiingo": 0.1,     # 10 calls/sec
    "alpha": 12.0,     # ~5 calls/min
    # Polygon free tier: enforce windowed limiter (5/min) separately
    "polygon": 0.0,
    "eodhd": 0.2,
    "simfin": 0.2,
    "marketstack": 0.2,
    "nasdaq": 0.2,
}

# Session-level provider disable flags (e.g., after rate-limit)
_PROVIDER_DISABLED: Dict[str, bool] = {
    "tiingo": False,
    "fmp": False,
}

# Session-level endpoint-category blacklist (e.g., "fmp:fundamentals")
DISABLED_PROVIDERS: set[str] = set()

_FMP_KEY_METRICS_RESTRICTED_MSG_SHOWN: bool = False
_PRIMARY_FUND_ROTATE: int = 0  # round-robin between finnhub and alpha
def disable_provider_category(provider: str, category: str) -> None:
    """Disable a specific category (e.g. 'price', 'fundamentals') for a provider."""
    try:
        key = f"{provider.lower()}:{category.lower()}"
        DISABLED_PROVIDERS.add(key)
        logger.warning(f"Disabled provider category: {key}")
    except Exception as e:
        logger.error(f"Failed to disable provider category {provider}:{category}: {e}")

def disable_provider(provider: str) -> None:
    """Disable an entire provider for the session."""
    try:
        p = provider.lower()
        _PROVIDER_DISABLED[p] = True
        logger.warning(f"Disabled provider completely: {p}")
    except Exception as e:
        logger.error(f"Failed to disable provider {provider}: {e}")

# Global default provider status (set via preflight)
_DEFAULT_PROVIDER_STATUS: Dict[str, bool] = {}
_DEFAULT_PROVIDER_STATUS_LOCK = __import__('threading').Lock()

# Dynamic provider priority list for fundamentals (populated by preflight)
_PROVIDER_PRIORITY: List[str] = []

# Smart Router cache (singleton per process)
_SMART_ROUTER_STATE_LOCK = threading.Lock()
_SMART_ROUTER_STATE: Optional[Dict[str, Any]] = None

# Windowed rate limiter for Polygon (5 requests per 60 seconds)
class WindowRateLimiter:
    def __init__(self, max_calls: int, window_seconds: int):
        self.max_calls = int(max_calls)
        self.window = int(window_seconds)
        self.lock = threading.Lock()
        self.calls: List[float] = []

    def acquire(self) -> None:
        now = time.time()
        with self.lock:
            # drop calls outside window
            self.calls = [t for t in self.calls if now - t < self.window]
            if len(self.calls) >= self.max_calls:
                # sleep until oldest call exits window
                sleep_for = self.window - (now - self.calls[0]) + 0.01
                sleep_for = max(0.0, sleep_for)
            else:
                sleep_for = 0.0
        if sleep_for > 0:
            time.sleep(sleep_for)
            with self.lock:
                # purge again and record
                now2 = time.time()
                self.calls = [t for t in self.calls if now2 - t < self.window]
                self.calls.append(now2)
        else:
            with self.lock:
                self.calls.append(time.time())

_POLYGON_WINDOW_LIMITER = WindowRateLimiter(max_calls=5, window_seconds=60)

def set_default_provider_status(preflight_status: Dict[str, Dict[str, Any]] | None) -> None:
    """Set global default provider status from preflight results (thread-safe)."""
    global _DEFAULT_PROVIDER_STATUS
    try:
        status = preflight_status or {}
        with _DEFAULT_PROVIDER_STATUS_LOCK:
            _DEFAULT_PROVIDER_STATUS = {
                "fmp": bool(status.get("FMP", {"ok": True}).get("ok", True)),
                "finnhub": bool(status.get("FINNHUB", {"ok": True}).get("ok", True)),
                "tiingo": bool(status.get("TIINGO", {"ok": True}).get("ok", True)),
                "alpha": bool(status.get("ALPHAVANTAGE", {"ok": True}).get("ok", True)),
                "polygon": bool(status.get("POLYGON", {"ok": True}).get("ok", True)),
                "eodhd": bool(status.get("EODHD", {"ok": True}).get("ok", True)),
                "simfin": bool(status.get("SIMFIN", {"ok": True}).get("ok", True)),
            }
        fmp_index_ok = bool(status.get("FMP_INDEX", {"ok": True}).get("ok", True))
        if not fmp_index_ok:
            disable_provider_category("fmp", "index")
            logger.warning("Preflight: disabling FMP index category for this session")

        # Compute dynamic priority from measured latencies for fundamentals providers
        fundamentals_providers = [
            "FMP", "FINNHUB", "TIINGO", "ALPHAVANTAGE", "EODHD", "SIMFIN"
        ]
        def _lat(p: str) -> float:
            try:
                meta = status.get(p, {})
                # Skip providers explicitly lacking capability/auth
                if meta.get("status") in ("auth_error", "no_key") or meta.get("can_fund") is False:
                    return float("inf")
                lat = meta.get("latency")
                if lat is None:
                    return float("inf")
                return float(lat)
            except (TypeError, ValueError, KeyError):
                return float("inf")
        sorted_providers = sorted(fundamentals_providers, key=_lat)
        # Filter out those not active for fundamentals
        _PROVIDER_PRIORITY.clear()
        for p in sorted_providers:
            meta = status.get(p, {})
            if meta.get("status") in ("auth_error", "no_key") or meta.get("can_fund") is False:
                continue
            _PROVIDER_PRIORITY.append(p)
        if not _PROVIDER_PRIORITY:
            # Fallback to static default ordering
            _PROVIDER_PRIORITY.extend(["FMP", "FINNHUB", "TIINGO", "ALPHAVANTAGE", "EODHD", "SIMFIN"])
        logger.info(f"[Preflight] Dynamic fundamentals priority: {_PROVIDER_PRIORITY}")
    except (TypeError, KeyError, AttributeError) as exc:
        logger.debug(f"Preflight provider status parse failed: {exc}")
        _DEFAULT_PROVIDER_STATUS = {}

# NOTE: get_fundamentals_safe is defined below (line ~700), NOT imported from
# core.providers (that module has a stub). The single definition below uses
# FMP Profile + Ratios TTM directly.


def _rate_limit(source: str) -> None:
    """Apply rate limiting for a given source.

    Thread-safe: compute wait time without holding the lock during sleep,
    then update last-call timestamp after waiting.
    """
    # Special: enforce windowed limiter for Polygon (5/min)
    if source == "polygon":
        _POLYGON_WINDOW_LIMITER.acquire()
        return
    min_interval = MIN_INTERVAL_SECONDS.get(source, 0.1)
    now = time.time()
    with _RATE_LOCK:
        last = _LAST_CALL_TIME.get(source, 0.0)
        elapsed = now - last
        wait = max(0.0, min_interval - elapsed)
        if wait == 0:
            _LAST_CALL_TIME[source] = now
    if wait > 0:
        time.sleep(wait)
        with _RATE_LOCK:
            _LAST_CALL_TIME[source] = time.time()


def _http_get_with_retry(
    url: str,
    params: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    timeout: int = 3,
    max_retries: int = 1,
    provider: Optional[str] = None,
    capability: Optional[str] = None,
) -> Optional[Dict]:
    """
    HTTP GET with retry logic and exponential backoff.
    
    Returns:
        Parsed JSON dict or None on failure
    """
    prov_name = str(provider).upper() if provider else None
    # ProviderGuard pre-check: skip immediately if not allowed (cooldown or permanent)
    try:
        if provider and capability:
            guard = get_provider_guard()
            allowed, reason, decision = guard.allow(prov_name, str(capability))
            if not allowed:
                # Return a sentinel to signal an upstream fast-fail without raw request
                return {"__blocked__": True, "reason": reason, "decision": decision}
    except (AttributeError, TypeError) as guard_exc:
        logger.debug(f"Provider guard check failed: {guard_exc}")
    start_time = time.time()
    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout
        )
        if response.status_code == 200:
            # Success: record and reset guard state
            try:
                if prov_name:
                    get_provider_guard().record_success(prov_name)
                record_api_call(
                    provider=(prov_name or "UNKNOWN"),
                    endpoint=(capability or "unknown"),
                    status="ok",
                    latency_sec=time.time() - start_time,
                    extra={"url": url}
                )
            except Exception as monitoring_exc:
                logger.debug(f"Monitoring record failed: {monitoring_exc}")
            return response.json()
        # Handle structured failures
        status_code = response.status_code
        if status_code == 429:
            # Rate limit: cooldown via guard, no sleep/backoff, immediate None
            try:
                get_provider_guard().record_failure(prov_name, status_code=status_code, capability=str(capability or ""), reason="rate_limited")
                record_api_call(
                    provider=(prov_name or "UNKNOWN"),
                    endpoint=(capability or "unknown"),
                    status="rate_limited_429",
                    latency_sec=time.time() - start_time,
                    extra={"url": url}
                )
            except Exception as monitoring_exc:
                logger.debug(f"Rate limit monitoring failed: {monitoring_exc}")
            return None
        if status_code in (401, 403):
            # Auth failure: permanent disable via guard and session set
            try:
                get_provider_guard().record_failure(prov_name, status_code=status_code, capability=str(capability or ""), reason="auth_error")
                if provider and capability:
                    disable_provider_category(str(provider).lower(), str(capability).lower())
                record_api_call(
                    provider=(prov_name or "UNKNOWN"),
                    endpoint=(capability or "unknown"),
                    status="http_forbidden",
                    latency_sec=time.time() - start_time,
                    extra={"url": url}
                )
            except Exception as monitoring_exc:
                logger.debug(f"Auth failure monitoring failed: {monitoring_exc}")
            return None
        # Other errors: record failure with short cooldown for 5xx; no retry loops
        try:
            if status_code and 500 <= status_code < 600:
                get_provider_guard().record_failure(prov_name, status_code=status_code, capability=str(capability or ""), reason="server_error")
            else:
                get_provider_guard().record_failure(prov_name, status_code=status_code, capability=str(capability or ""), reason="http_error")
            record_api_call(
                provider=(prov_name or "UNKNOWN"),
                endpoint=(capability or "unknown"),
                status=f"http_{status_code}",
                latency_sec=time.time() - start_time,
                extra={"url": url}
            )
        except Exception as monitoring_exc:
            logger.debug(f"HTTP error monitoring failed: {monitoring_exc}")
        return None
    except requests.Timeout:
        logger.warning(f"Timeout for {url}")
        try:
            get_provider_guard().record_failure(prov_name, status_code=None, capability=str(capability or ""), reason="timeout")
            record_api_call(
                provider=(prov_name or "UNKNOWN"),
                endpoint=(capability or "unknown"),
                status="timeout",
                latency_sec=time.time() - start_time,
                extra={"url": url}
            )
        except Exception as monitoring_exc:
            logger.debug(f"Timeout monitoring failed: {monitoring_exc}")
        return None
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}", exc_info=True)
        try:
            get_provider_guard().record_failure(prov_name, status_code=None, capability=str(capability or ""), reason="exception")
            record_api_call(
                provider=(prov_name or "UNKNOWN"),
                endpoint=(capability or "unknown"),
                status="exception",
                latency_sec=time.time() - start_time,
                extra={"url": url, "error": str(e)[:200]}
            )
        except Exception as monitoring_exc:
            logger.debug(f"Exception monitoring failed: {monitoring_exc}")
        return None


def _get_from_cache(cache_key: str, ttl: int = CACHE_TTL_SECONDS) -> Optional[Dict]:
    """Retrieve from cache if not expired (thread-safe)."""
    with _CACHE_LOCK:
        entry = _CACHE.get(cache_key)
        if not entry:
            return None
        if time.time() - entry["timestamp"] < ttl:
            return entry["data"]
        # Expired entry: remove
        _CACHE.pop(cache_key, None)
        return None


def _put_in_cache(cache_key: str, data: Dict) -> None:
    """Store in cache with timestamp (thread-safe)."""
    with _CACHE_LOCK:
        _CACHE[cache_key] = {
            "data": data,
            "timestamp": time.time()
        }


# ============================================================================
# SMART ROUTER FOR FUNDAMENTALS
# ============================================================================

def _has_key(provider: str) -> bool:
    if provider == "FMP":
        return bool(FMP_API_KEY or os.getenv("FMP_API_KEY") or os.getenv("FMP_KEY"))
    if provider == "AlphaVantage":
        return bool(ALPHA_VANTAGE_API_KEY or os.getenv("ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHAVANTAGE_API_KEY"))
    if provider == "Tiingo":
        return bool(TIINGO_API_KEY or os.getenv("TIINGO_API_KEY"))
    if provider == "Finnhub":
        return bool(FINNHUB_API_KEY or os.getenv("FINNHUB_API_KEY"))
    return False


def _latency_probe(provider: str, timeout: float = 2.0) -> Tuple[bool, float, str]:
    """Quick probe for provider health and latency. Returns (ok, latency_sec, note)."""
    start = time.perf_counter()
    try:
        if provider == "FMP":
            key = os.getenv("FMP_API_KEY") or os.getenv("FMP_KEY") or FMP_API_KEY
            if not key:
                return (False, float("inf"), "no_key")
            # Use stable Screener endpoint confirmed working
            url = f"https://financialmodelingprep.com/stable/company-screener?symbol=AAPL&apikey={key}"
            resp = requests.get(url, timeout=timeout)
            lat = time.perf_counter() - start
            if resp.status_code == 200:
                try:
                    js = resp.json()
                    ok = isinstance(js, list) and len(js) > 0
                    return (ok, lat, "ok" if ok else "empty")
                except (ValueError, KeyError) as json_exc:
                    logger.debug(f"FMP JSON parse failed: {json_exc}")
                    return (False, lat, "bad_json")
            return (False, lat, f"http_{resp.status_code}")

        if provider == "AlphaVantage":
            key = os.getenv("ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHAVANTAGE_API_KEY") or ALPHA_VANTAGE_API_KEY
            if not key:
                return (False, float("inf"), "no_key")
            url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol=AAPL&apikey={key}"
            resp = requests.get(url, timeout=timeout)
            lat = time.perf_counter() - start
            if resp.status_code == 200:
                try:
                    js = resp.json()
                    ok = isinstance(js, dict) and bool(js.get("Symbol"))
                    return (ok, lat, "ok" if ok else "empty")
                except (ValueError, KeyError) as json_exc:
                    logger.debug(f"AlphaVantage JSON parse failed: {json_exc}")
                    return (False, lat, "bad_json")
            return (False, lat, f"http_{resp.status_code}")

        if provider == "Tiingo":
            key = os.getenv("TIINGO_API_KEY") or TIINGO_API_KEY
            if not key:
                return (False, float("inf"), "no_key")
            url = f"https://api.tiingo.com/tiingo/fundamentals/AAPL/statements?token={key}"
            resp = requests.get(url, timeout=timeout)
            lat = time.perf_counter() - start
            if resp.status_code == 200:
                try:
                    js = resp.json()
                    ok = isinstance(js, list) and len(js) > 0
                    return (ok, lat, "ok" if ok else "empty")
                except (ValueError, KeyError) as json_exc:
                    logger.debug(f"Tiingo JSON parse failed: {json_exc}")
                    return (False, lat, "bad_json")
            return (False, lat, f"http_{resp.status_code}")

        if provider == "Finnhub":
            key = os.getenv("FINNHUB_API_KEY") or FINNHUB_API_KEY
            if not key:
                return (False, float("inf"), "no_key")
            url = f"https://finnhub.io/api/v1/stock/profile2?symbol=AAPL&token={key}"
            resp = requests.get(url, timeout=timeout)
            lat = time.perf_counter() - start
            if resp.status_code == 200:
                try:
                    js = resp.json()
                    ok = isinstance(js, dict) and len(js) > 0
                    return (ok, lat, "ok" if ok else "empty")
                except (ValueError, KeyError) as json_exc:
                    logger.debug(f"Finnhub JSON parse failed: {json_exc}")
                    return (False, lat, "bad_json")
            return (False, lat, f"http_{resp.status_code}")

        return (False, float("inf"), "unknown_provider")
    except requests.Timeout:
        return (False, time.perf_counter() - start, "timeout")
    except Exception as e:
        return (False, time.perf_counter() - start, f"exception:{str(e)[:40]}")


@lru_cache(maxsize=1)
def get_optimal_fundamentals_provider() -> str:
    """Run once: probe fundamentals providers and select fastest healthy one."""
    global _SMART_ROUTER_STATE
    providers = ["FMP", "AlphaVantage", "Tiingo", "Finnhub"]
    results: List[Dict[str, Any]] = []
    for p in providers:
        if not _has_key(p):
            results.append({"provider": p, "ok": False, "latency": float("inf"), "note": "no_key"})
            continue
        ok, lat, note = _latency_probe(p, timeout=2.0)
        results.append({"provider": p, "ok": ok, "latency": lat, "note": note})

    ok_results = [r for r in results if r["ok"]]
    ok_results.sort(key=lambda r: r["latency"])  # fastest first
    winner = ok_results[0]["provider"] if ok_results else "FMP"
    ranking = [r["provider"] for r in ok_results] + [r["provider"] for r in results if not r["ok"]]

    with _SMART_ROUTER_STATE_LOCK:
        _SMART_ROUTER_STATE = {"winner": winner, "ranking": ranking, "results": results}

    # Log summary
    parts = []
    for r in results:
        if r["ok"]:
            parts.append(f"{r['provider']}={r['latency']:.2f}s")
        else:
            parts.append(f"{r['provider']}={r['note']}")
    logger.info(f"Smart Router: Selected {winner}. Details: " + ", ".join(parts))
    return winner


def _get_provider_ranking() -> List[str]:
    with _SMART_ROUTER_STATE_LOCK:
        if _SMART_ROUTER_STATE and _SMART_ROUTER_STATE.get("ranking"):
            return list(_SMART_ROUTER_STATE["ranking"])  # copy
    # Ensure router ran at least once
    get_optimal_fundamentals_provider()
    with _SMART_ROUTER_STATE_LOCK:
        if _SMART_ROUTER_STATE and _SMART_ROUTER_STATE.get("ranking"):
            return list(_SMART_ROUTER_STATE["ranking"])  # copy
    return ["FMP", "Finnhub", "Tiingo", "AlphaVantage"]


# ============================================================================
# SAFE FUNDAMENTALS (FMP PROFILE + RATIOS TTM)
# ============================================================================

def get_fundamentals_safe(ticker: str) -> Optional[Dict]:
    """
    Robust, flat fundamentals fetch using FMP Profile and Ratios TTM.

        Returns None on connection/auth errors. On success, returns flat dict:
        {
            'Market_Cap': float, 'PE_Ratio': float, 'PEG_Ratio': float,
            'PB_Ratio': float, 'ROE': float,
            'Beta': float, 'Sector': str, 'Industry': str,
            'Vol_Avg': float, 'Dividend': float, 'Price': float,
            'Debt_to_Equity': float
        }
    """
    try:
        tkr = str(ticker).upper()
    except (TypeError, AttributeError):
        tkr = ticker

    # Key resolution
    FMP_KEY_RUNTIME = os.getenv("FMP_API_KEY") or os.getenv("FMP_KEY") or get_secret("FMP_API_KEY", "")
    if not (FMP_KEY_RUNTIME or FMP_API_KEY):
        return None

    # Cache
    cache_key = f"fund_safe_{tkr}"
    cached = _get_from_cache(cache_key)
    if cached:
        return cached

    # Profile (best-effort; do not early-return on failure) - using stable API
    _rate_limit("fmp")
    prof_url = f"https://financialmodelingprep.com/stable/profile?symbol={tkr}"
    prof_params = {"apikey": (FMP_KEY_RUNTIME or FMP_API_KEY), "limit": 1}
    prof = None
    try:
        r = requests.get(prof_url, params=prof_params, timeout=4)
        if r.status_code == 200:
            js = r.json()
            if js and isinstance(js, list) and js:
                prof = js[0]
                record_api_call("FMP", "profile", "ok", 0.0, {"ticker": tkr})
            else:
                record_api_call("FMP", "profile", "empty", 0.0, {"ticker": tkr})
        else:
            record_api_call("FMP", "profile", f"http_{r.status_code}", 0.0, {"ticker": tkr})
    except Exception as e:
        record_api_call("FMP", "profile", "exception", 0.0, {"ticker": tkr, "error": str(e)[:200]})

    # Ratios TTM (best-effort) - using stable API
    _rate_limit("fmp")
    ratios_url = f"https://financialmodelingprep.com/stable/ratios-ttm?symbol={tkr}"
    ratios_params = {"apikey": (FMP_KEY_RUNTIME or FMP_API_KEY)}
    ratios = None
    try:
        r2 = requests.get(ratios_url, params=ratios_params, timeout=4)
        if r2.status_code == 200:
            js2 = r2.json()
            if js2 and isinstance(js2, list) and js2:
                ratios = js2[0]
                record_api_call("FMP", "ratios-ttm", "ok", 0.0, {"ticker": tkr})
            else:
                record_api_call("FMP", "ratios-ttm", "empty", 0.0, {"ticker": tkr})
        else:
            record_api_call("FMP", "ratios-ttm", f"http_{r2.status_code}", 0.0, {"ticker": tkr})
    except Exception as e:
        record_api_call("FMP", "ratios-ttm", "exception", 0.0, {"ticker": tkr, "error": str(e)[:200]})

    # Parse helpers
    def _f(v: Any) -> Optional[float]:
        try:
            if v in (None, "None", "-", "N/A", "null"):
                return None
            return float(v)
        except (TypeError, ValueError):
            return None

    # Build output using whichever sources succeeded
    out = {}
    if prof:
        out.update({
            "Market_Cap": _f(prof.get("mktCap")),
            "Beta": _f(prof.get("beta")),
            "Vol_Avg": _f(prof.get("volAvg")),
            "Dividend": _f(prof.get("lastDiv")),
            "Price": _f(prof.get("price")),
            "Sector": prof.get("sector") or None,
            "Industry": prof.get("industry") or None,
        })
    if ratios:
        out.update({
            "PE_Ratio": _f(ratios.get("peRatioTTM")),
            "PEG_Ratio": _f(ratios.get("pegRatioTTM")),
            "PB_Ratio": _f(ratios.get("priceToBookRatioTTM")),
            "Debt_to_Equity": _f(ratios.get("debtEquityRatioTTM")),
            "ROE": _f(ratios.get("returnOnEquityTTM")),
        })

    # Fallback to Alpha Vantage OVERVIEW if both FMP calls failed
    if not out:
        ALPHA_KEY_RUNTIME = get_secret("ALPHA_VANTAGE_API_KEY", os.getenv("ALPHA_VANTAGE_API_KEY", "")) or os.getenv("ALPHAVANTAGE_API_KEY", "")
        if ALPHA_KEY_RUNTIME or ALPHA_VANTAGE_API_KEY:
            try:
                _rate_limit("alpha")
                url = "https://www.alphavantage.co/query"
                params = {"function": "OVERVIEW", "symbol": tkr, "apikey": (ALPHA_KEY_RUNTIME or ALPHA_VANTAGE_API_KEY)}
                data = _http_get_with_retry(url, params=params, timeout=4, provider="ALPHAVANTAGE", capability="fundamentals")
            except (requests.RequestException, ValueError, KeyError):
                data = None
            if data and isinstance(data, dict):
                def _fa(key: str) -> Optional[float]:
                    try:
                        v = data.get(key)
                        if v in (None, "None", "-", "N/A", "null"):
                            return None
                        return float(v)
                    except (TypeError, ValueError, KeyError):
                        return None
                out = {
                    "Market_Cap": _fa("MarketCapitalization"),
                    "PE_Ratio": _fa("PERatio"),
                    "PEG_Ratio": _fa("PEGRatio"),
                    "PB_Ratio": _fa("PriceToBookRatio"),
                    "Debt_to_Equity": _fa("DebtToEquity"),
                    "ROE": _fa("ReturnOnEquityTTM"),
                    "Beta": _fa("Beta"),
                    "Sector": data.get("Sector") or None,
                }
        # If still empty, return None
        if not out:
            return None

    _put_in_cache(cache_key, out)
    return out

    

    


# ============================================================================
# FMP (Financial Modeling Prep) - PRIMARY SOURCE
# ============================================================================

def fetch_fundamentals_fmp(ticker: str, provider_status: Dict | None = None) -> Optional[Dict]:
    """
    Fetch fundamentals from FMP via the stable company screener endpoint.

    Notes:
    - Screener provides descriptive fields (marketCap, beta, sector, industry, price).
    - It may not include valuation ratios (PE). Aggregation should complement from others.
    Returns a dict including both canonical lowercase fields and UI-standard aliases.
    """
    # Reload key at runtime from environment first; fallback to get_secret
    FMP_KEY_RUNTIME = os.getenv("FMP_API_KEY") or os.getenv("FMP_KEY") or get_secret("FMP_API_KEY", "")

    if not (FMP_KEY_RUNTIME or FMP_API_KEY):
        return None
    # Respect session-level blacklist for FMP fundamentals
    if "fmp:fundamentals" in DISABLED_PROVIDERS:
        try:
            record_api_call(
                provider="FMP",
                endpoint="key-metrics",
                status="skipped_blacklist",
                latency_sec=0.0,
                extra={"ticker": ticker, "reason": "session_blacklist"},
            )
        except (TypeError, AttributeError):
            pass
        return None
    
    cache_key = f"fmp_fund_{ticker}"
    cached = _get_from_cache(cache_key)
    if cached:
        return cached
    
    _rate_limit("fmp")
    # Query stable profile endpoint (not company-screener which returns wrong tickers)
    url = f"https://financialmodelingprep.com/stable/profile"
    params = {"symbol": ticker, "apikey": (FMP_KEY_RUNTIME or FMP_API_KEY)}
    start = time.time()
    try:
        data = _http_get_with_retry(url, params=params, timeout=3, provider="FMP", capability="fundamentals")
        # Validate ticker match to avoid returning wrong company data
        if isinstance(data, list) and len(data) > 0:
            returned_ticker = data[0].get("symbol", "").upper()
            if returned_ticker != ticker.upper():
                logger.warning(f"FMP returned wrong ticker: requested {ticker}, got {returned_ticker}")
                data = None
        status = "ok" if isinstance(data, list) and len(data) > 0 else ("empty" if data else "empty")
        record_api_call(
            provider="FMP",
            endpoint="profile",
            status=status,
            latency_sec=time.time() - start,
            extra={"ticker": ticker}
        )
    except Exception as e:
        record_api_call(
            provider="FMP",
            endpoint="profile",
            status="exception",
            latency_sec=time.time() - start,
            extra={"ticker": ticker, "error": str(e)[:200]}
        )
        data = None

    if not data or not isinstance(data, list) or len(data) == 0:
        return None
    item = data[0]

    def _f(v):
        try:
            if v in (None, "None", "-", "N/A", "null"):
                return None
            return float(v)
        except (TypeError, ValueError):
            return None

    # Map screener fields
    market_cap = _f(item.get("marketCap"))
    beta = _f(item.get("beta"))
    sector = item.get("sector")
    industry = item.get("industry")
    price = _f(item.get("price"))

    result = {
        "source": "fmp",
        # Canonical lowercase
        "market_cap": market_cap,
        "beta": beta,
        "sector": sector,
        "industry": industry,
        "price_backup": price,
        "timestamp": time.time(),
        # UI-standard aliases
        "Market_Cap": market_cap,
        "Beta": beta,
        "Sector": sector,
        "Industry": industry,
        "Price": price,
    }

    _put_in_cache(cache_key, result)
    return result


# ============================================================================
# FINNHUB - SECONDARY SOURCE
# ============================================================================

def fetch_fundamentals_finnhub(ticker: str, provider_status: Dict | None = None) -> Optional[Dict]:
    """Fetch fundamentals from Finnhub."""
    # Preflight advisory only; do not skip when key present
    FINNHUB_KEY_RUNTIME = get_secret("FINNHUB_API_KEY", os.getenv("FINNHUB_API_KEY", ""))

    if not (FINNHUB_KEY_RUNTIME or FINNHUB_API_KEY):
        return None
    
    cache_key = f"finnhub_fund_{ticker}"
    cached = _get_from_cache(cache_key)
    if cached:
        return cached
    
    _rate_limit("finnhub")
    
    # Fetch basic financials
    url = "https://finnhub.io/api/v1/stock/metric"
    params = {"symbol": ticker, "metric": "all", "token": (FINNHUB_KEY_RUNTIME or FINNHUB_API_KEY)}
    
    start = time.time()
    try:
        data = _http_get_with_retry(url, params=params, timeout=3, provider="FINNHUB", capability="fundamentals")
        status = "ok" if data and "metric" in data else "empty"
        record_api_call(
            provider="Finnhub",
            endpoint="metric",
            status=status,
            latency_sec=time.time() - start,
            extra={"ticker": ticker}
        )
    except Exception as e:
        record_api_call(
            provider="Finnhub",
            endpoint="metric",
            status="exception",
            latency_sec=time.time() - start,
            extra={"ticker": ticker, "error": str(e)[:200]}
        )
        return None
    if not data or "metric" not in data:
        return None
    metric = data.get("metric", {})
    result = {
        "source": "finnhub",
        "pe": metric.get("peBasicExclExtraTTM"),
        "ps": metric.get("psAnnual"),
        "pb": metric.get("pbAnnual"),
        "roe": metric.get("roeTTM"),
        "margin": metric.get("netProfitMarginTTM"),
        "market_cap": metric.get("marketCapitalization"),
        "beta": metric.get("beta"),
        "debt_equity": metric.get("totalDebt/totalEquityAnnual"),
        "rev_yoy": metric.get("revenueGrowthTTMYoy"),
        "eps_yoy": metric.get("epsGrowthTTMYoy"),
        "timestamp": time.time()
    }
    _put_in_cache(cache_key, result)
    return result


# ============================================================================
# TIINGO - EXISTING SOURCE (PRESERVE)
# ============================================================================

def fetch_fundamentals_tiingo(ticker: str, provider_status: Dict | None = None) -> Optional[Dict]:
    """Fetch fundamentals from Tiingo (existing source - preserved)."""
    # Preflight advisory only; do not skip when key present
    TIINGO_KEY_RUNTIME = get_secret("TIINGO_API_KEY", os.getenv("TIINGO_API_KEY", ""))

    if not (TIINGO_KEY_RUNTIME or TIINGO_API_KEY):
        return None
    
    cache_key = f"tiingo_fund_{ticker}"
    cached = _get_from_cache(cache_key)
    if cached:
        return cached
    
    _rate_limit("tiingo")
    # Explicit pacer to avoid 429s even across processes
    try:
        time.sleep(1.2)
    except (ValueError, OSError):
        pass
    
    # Tiingo fundamentals endpoint
    url = f"https://api.tiingo.com/tiingo/fundamentals/{ticker}/statements"
    headers = {"Content-Type": "application/json", "Authorization": f"Token {TIINGO_KEY_RUNTIME or TIINGO_API_KEY}"}
    
    start = time.time()
    try:
        data = _http_get_with_retry(url, headers=headers, timeout=3, provider="TIINGO", capability="fundamentals")
        status = "ok" if data and isinstance(data, list) and len(data) > 0 else "empty"
        record_api_call(
            provider="Tiingo",
            endpoint="statements",
            status=status,
            latency_sec=time.time() - start,
            extra={"ticker": ticker}
        )
    except Exception as e:
        record_api_call(
            provider="Tiingo",
            endpoint="statements",
            status="exception",
            latency_sec=time.time() - start,
            extra={"ticker": ticker, "error": str(e)[:200]}
        )
        return None
    if not data or not isinstance(data, list) or len(data) == 0:
        return None
    # Parse latest annual statement
    latest = data[0] if data else {}
    result = {
        "source": "tiingo",
        "pe": latest.get("priceToEarnings"),
        "ps": latest.get("priceToSales"),
        "pb": latest.get("priceToBook"),
        "roe": latest.get("returnOnEquity"),
        "margin": latest.get("profitMargin"),
        "market_cap": latest.get("marketCap"),
        "debt_equity": latest.get("debtToEquity"),
        "rev_yoy": latest.get("revenueGrowth"),
        "eps_yoy": latest.get("epsGrowth"),
        "timestamp": time.time()
    }
    _put_in_cache(cache_key, result)
    return result


# ============================================================================
# ALPHA VANTAGE - EXISTING SOURCE (PRESERVE)
# ============================================================================

def fetch_fundamentals_alpha(ticker: str, provider_status: Dict | None = None) -> Optional[Dict]:
    """Fetch fundamentals from Alpha Vantage (existing source - preserved)."""
    # Preflight advisory only; do not skip when key present
    ALPHA_KEY_RUNTIME = get_secret("ALPHA_VANTAGE_API_KEY", os.getenv("ALPHA_VANTAGE_API_KEY", "")) or os.getenv("ALPHAVANTAGE_API_KEY", "")

    if not (ALPHA_KEY_RUNTIME or ALPHA_VANTAGE_API_KEY):
        return None
    
    cache_key = f"alpha_fund_{ticker}"
    cached = _get_from_cache(cache_key)
    if cached:
        return cached
    
    _rate_limit("alpha")
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "OVERVIEW",
        "symbol": ticker,
        "apikey": (ALPHA_KEY_RUNTIME or ALPHA_VANTAGE_API_KEY)
    }
    
    start = time.time()
    try:
        data = _http_get_with_retry(url, params=params, timeout=3, provider="ALPHAVANTAGE", capability="fundamentals")
        status = "ok" if data and "Symbol" in data else "empty"
        record_api_call(
            provider="AlphaVantage",
            endpoint="overview",
            status=status,
            latency_sec=time.time() - start,
            extra={"ticker": ticker}
        )
    except Exception as e:
        record_api_call(
            provider="AlphaVantage",
            endpoint="overview",
            status="exception",
            latency_sec=time.time() - start,
            extra={"ticker": ticker, "error": str(e)[:200]}
        )
        return None
    if not data or "Symbol" not in data:
        return None
    # Helper to parse float-like fields safely
    def _f(key: str) -> Optional[float]:
        v = data.get(key)
        try:
            if v is None or v == "None":
                return None
            return float(v)
        except (TypeError, ValueError):
            return None

    result = {
        "source": "alpha",
        "pe": _f("PERatio"),
        "ps": _f("PriceToSalesRatioTTM"),
        "pb": _f("PriceToBookRatio"),
        "roe": _f("ReturnOnEquityTTM"),
        "margin": _f("ProfitMargin"),
        "market_cap": _f("MarketCapitalization"),
        "beta": _f("Beta"),
        "debt_equity": _f("DebtToEquity"),
        "rev_yoy": _f("QuarterlyRevenueGrowthYOY"),
        "eps_yoy": _f("QuarterlyEarningsGrowthYOY"),
        # Additional fields
        "peg": _f("PEGRatio"),
        "sector": data.get("Sector"),
        "timestamp": time.time(),
        # Standard UI aliases to ensure consistent mapping downstream
        "PE_Ratio": _f("PERatio"),
        "Market_Cap": _f("MarketCapitalization"),
        "PEG_Ratio": _f("PEGRatio"),
        "PB_Ratio": _f("PriceToBookRatio"),
    }
    _put_in_cache(cache_key, result)
    return result


# ============================================================================
# MULTI-SOURCE AGGREGATION
# ============================================================================

def fetch_fundamentals_eodhd(ticker: str, provider_status: Dict | None = None) -> Optional[Dict]:
    """Fetch fundamentals from EODHD (secondary group)."""
    # Preflight advisory only; proceed if key present
    EODHD_KEY_RUNTIME = get_secret("EODHD_API_KEY", os.getenv("EODHD_API_KEY", "")) or os.getenv("EODHD_TOKEN", "")
    if not (EODHD_KEY_RUNTIME or EODHD_API_KEY) or ("eodhd:fundamentals" in DISABLED_PROVIDERS):
        return None
    cache_key = f"eodhd_fund_{ticker}"
    cached = _get_from_cache(cache_key)
    if cached:
        return cached
    _rate_limit("eodhd")
    url = f"https://eodhd.com/api/fundamentals/{ticker}"
    params = {"api_token": (EODHD_KEY_RUNTIME or EODHD_API_KEY), "fmt": "json"}
    start = time.time()
    try:
        resp = requests.get(url, params=params, timeout=4)
        if resp.status_code in (401, 403):
            disable_provider_category("eodhd", "fundamentals")
            record_api_call("EODHD", "fundamentals", "http_forbidden", time.time()-start, {"ticker": ticker})
            return None
        if resp.status_code != 200:
            record_api_call("EODHD", "fundamentals", f"http_{resp.status_code}", time.time()-start, {"ticker": ticker})
            return None
        data = resp.json() if resp.content else None
    except Exception as e:
        record_api_call("EODHD", "fundamentals", "exception", time.time()-start, {"ticker": ticker, "error": str(e)[:200]})
        return None
    if not data or not isinstance(data, dict):
        return None
    # Opportunistic extraction from EODHD schema (keys may vary)
    def _g(*keys):
        cur = data
        for k in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k)
        return cur if cur not in ("None", "null", None) else None
    result = {
        "source": "eodhd",
        "pe": _g("Valuation", "TrailingPE") or _g("Highlights", "PERatioTTM"),
        "ps": _g("Valuation", "PriceToSalesTTM"),
        "pb": _g("Valuation", "PriceBookMRQ") or _g("Valuation", "PriceToBookMRQ"),
        "roe": _g("Highlights", "ReturnOnEquityTTM"),
        "margin": _g("Highlights", "ProfitMarginTTM"),
        "market_cap": _g("Highlights", "MarketCapitalization"),
        "beta": _g("Technicals", "Beta"),
        "debt_equity": _g("Highlights", "TotalDebtToEquityQuarterly"),
        "rev_yoy": _g("Highlights", "RevenueGrowthTTMYoy"),
        "eps_yoy": _g("Highlights", "EPSGrowthTTMYoy"),
        "timestamp": time.time(),
    }
    _put_in_cache(cache_key, result)
    return result


def fetch_fundamentals_simfin(ticker: str, provider_status: Dict | None = None) -> Optional[Dict]:
    """Fetch fundamentals from SimFin (secondary group)."""
    # Preflight advisory only; proceed if key present
    SIMFIN_KEY_RUNTIME = get_secret("SIMFIN_API_KEY", os.getenv("SIMFIN_API_KEY", ""))
    if not (SIMFIN_KEY_RUNTIME or SIMFIN_API_KEY) or ("simfin:fundamentals" in DISABLED_PROVIDERS):
        return None
    cache_key = f"simfin_fund_{ticker}"
    cached = _get_from_cache(cache_key)
    if cached:
        return cached
    _rate_limit("simfin")
    # Note: SimFin API specifics vary; implement a guarded request to a common endpoint.
    url = "https://simfin.com/api/v2/companies/statements"
    params = {
        "api-key": (SIMFIN_KEY_RUNTIME or SIMFIN_API_KEY),
        "ticker": ticker,
        "statement": "pl",
        "period": "ttm",
        "fyear": datetime.utcnow().year,
    }
    start = time.time()
    try:
        data = _http_get_with_retry(url, params=params, timeout=5, provider="SIMFIN", capability="fundamentals")
        status = "ok" if isinstance(data, dict) else ("empty" if data is None else "empty")
        record_api_call("SimFin", "statements", status, time.time()-start, {"ticker": ticker})
        if not data:
            return None
    except Exception as e:
        record_api_call("SimFin", "statements", "exception", time.time()-start, {"ticker": ticker, "error": str(e)[:200]})
        return None
    # Best-effort parse; if structure unknown, return None gracefully
    try:
        # Simplified extraction; real mapping would depend on API response schema
        result = {
            "source": "simfin",
            "pe": None,
            "ps": None,
            "pb": None,
            "roe": None,
            "margin": None,
            "market_cap": None,
            "beta": None,
            "debt_equity": None,
            "rev_yoy": None,
            "eps_yoy": None,
            "timestamp": time.time(),
        }
        _put_in_cache(cache_key, result)
        return result
    except (requests.RequestException, ValueError, KeyError):
        return None

def get_prioritized_fetch_funcs(provider_status: Dict | None = None) -> List[Tuple[str, Any]]:
    """
    Build a prioritized list of fundamentals fetch functions respecting
    preflight capabilities and session-level disables.

    Priority: FMP → Finnhub → Tiingo → Alpha → (EODHD, SimFin optional)
    """
    funcs: List[Tuple[str, Any]] = []

    def _can(upper: str, lower: str) -> bool:
        try:
            if _PROVIDER_DISABLED.get(lower, False):
                return False
            if f"{lower}:fundamentals" in DISABLED_PROVIDERS:
                return False
            s = (provider_status or {}).get(upper)
            if s:
                if s.get("status") in ("auth_error", "no_key"):
                    return False
                if s.get("can_fund", True) is False:
                    return False
            # ProviderGuard runtime check (cooldowns, etc.)
            try:
                guard = get_provider_guard()
                allowed, _rsn, _dec = guard.allow(upper, "fundamentals")
                if not allowed:
                    return False
            except (ImportError, AttributeError, TypeError):
                pass
            return True
        except (KeyError, TypeError, AttributeError):
            return True

    if _can("FMP", "fmp"):
        funcs.append(("fmp", fetch_fundamentals_fmp))
    if _can("FINNHUB", "finnhub"):
        funcs.append(("finnhub", fetch_fundamentals_finnhub))
    if _can("TIINGO", "tiingo"):
        funcs.append(("tiingo", fetch_fundamentals_tiingo))
    if _can("ALPHAVANTAGE", "alpha"):
        funcs.append(("alpha", fetch_fundamentals_alpha))
    # Secondary group (optional enrichers if available)
    if _can("EODHD", "eodhd"):
        funcs.append(("eodhd", fetch_fundamentals_eodhd))
    if _can("SIMFIN", "simfin"):
        funcs.append(("simfin", fetch_fundamentals_simfin))

    return funcs

def aggregate_fundamentals(
    ticker: str,
    prefer_source: str = "fmp",
    provider_status: Dict | None = None,
    mode: DataMode = DataMode.LIVE,
    as_of_date: Optional[date] = None,
    telemetry: Optional[Any] = None,
) -> Dict:
    """
    Fetch and aggregate fundamentals from ALL available sources.
    
    Priority for initial fetch: FMP → Finnhub → Tiingo → Alpha
    But then ENRICH with data from other sources.
    
    Aggregation rules:
    - Use preferred source as anchor if available
    - Fill missing fields from other sources
    - If multiple sources provide same field, use median or flag disagreement
    - Track which sources contributed to each field
    
    Returns:
        Aggregated dict with keys:
        - All fundamental fields (pe, ps, pb, etc.)
        - sources_used: List[str]
        - coverage: Dict[str, List[str]] - which sources provided each field
        - disagreement_score: float (0-1) - how much sources disagree
    """
    # Use global rotation counter for primary fundamentals providers (finnhub/alpha)
    global _PRIMARY_FUND_ROTATE

    # BACKTEST: Prefer local point-in-time store to avoid lookahead bias
    if mode == DataMode.BACKTEST and as_of_date is not None:
        try:
            snapshot = load_fundamentals_as_of(ticker, as_of_date)
        except (IOError, OSError, RuntimeError):
            snapshot = None
        if snapshot:
            # Ensure flags and minimal fields
            snapshot = dict(snapshot)
            snapshot["ticker"] = ticker
            snapshot["Fundamental_Backtest_Unsafe"] = False
            snapshot["Fundamental_From_Store"] = True
            snapshot.setdefault("sources_used", [])
            snapshot.setdefault("Fundamental_Sources_Count", len(snapshot.get("sources_used", [])))
            snapshot.setdefault("Fundamental_Coverage_Pct", 0.0)
            snapshot.setdefault("timestamp", time.time())
            logger.info(f"Using stored fundamentals snapshot for {ticker} as of {as_of_date}")
            return snapshot

    # FAST FUNDAMENTAL mode: If FMP and FINNHUB both unavailable (rate-limited/auth),
    # attempt to load a cached snapshot from the last 24 hours (today/yesterday) from SQLite store.
    try:
        fmp_ok = (provider_status or {}).get("FMP", {"ok": True}).get("ok", True)
        finnhub_ok = (provider_status or {}).get("FINNHUB", {"ok": True}).get("ok", True)
        fast_needed = (not fmp_ok) and (not finnhub_ok)
    except (KeyError, TypeError, AttributeError):
        fast_needed = False

    if fast_needed:
        try:
            today = datetime.utcnow().date()
            # Try today, then yesterday for a <=24h snapshot
            snapshot = load_fundamentals_as_of(ticker, today)
            if not snapshot:
                snapshot = load_fundamentals_as_of(ticker, today - timedelta(days=1))
        except (IOError, OSError, RuntimeError):
            snapshot = None
        if snapshot:
            snap = dict(snapshot)
            snap["ticker"] = ticker
            snap["Fundamental_From_Store"] = True
            snap["Fundamental_Backtest_Unsafe"] = False
            snap.setdefault("sources_used", [])
            snap.setdefault("Fundamental_Sources_Count", len(snap.get("sources_used", [])))
            snap.setdefault("Fundamental_Coverage_Pct", 0.0)
            snap.setdefault("timestamp", time.time())
            logger.info(f"FastFundamental: using cached snapshot for {ticker}")
            return snap

    # Fetch from all sources (sequential, rate-limit friendly)
    sources_data = {}

    # Determine fundamentals priority using dynamic latency-based routing
    fetch_funcs = get_prioritized_fetch_funcs(provider_status)
    first_attempted: Optional[str] = fetch_funcs[0][0] if fetch_funcs else None

    for source_name, fetch_func in fetch_funcs:
        # Respect preflight minimally: skip only when auth/no_key or cannot serve fundamentals
        if provider_status is not None:
            key_map = {
                "fmp": "FMP",
                "finnhub": "FINNHUB",
                "tiingo": "TIINGO",
                "alpha": "ALPHAVANTAGE",
                "eodhd": "EODHD",
                "simfin": "SIMFIN",
            }
            s = provider_status.get(key_map.get(source_name, source_name))
            if s and (s.get("status") in ("auth_error", "no_key") or (s.get("can_fund") is False)):
                try:
                    record_api_call(
                        provider=key_map.get(source_name, source_name),
                        endpoint="fundamentals",
                        status="skipped_preflight",
                        latency_sec=0.0,
                        extra={"ticker": ticker, "reason": s.get("status", "capability_off")},
                    )
                except (TypeError, AttributeError):
                    pass
                continue
        # ProviderGuard runtime block (cooldown/permanent)
        try:
            upper = {
                "fmp": "FMP",
                "finnhub": "FINNHUB",
                "tiingo": "TIINGO",
                "alpha": "ALPHAVANTAGE",
                "eodhd": "EODHD",
                "simfin": "SIMFIN",
            }.get(source_name, source_name.upper())
            allowed, reason, decision = get_provider_guard().allow(upper, "fundamentals")
            if not allowed:
                try:
                    record_api_call(
                        provider=upper,
                        endpoint="fundamentals",
                        status="skipped_guard",
                        latency_sec=0.0,
                        extra={"ticker": ticker, "reason": reason, "decision": decision},
                    )
                except (TypeError, AttributeError):
                    pass
                continue
        except (ImportError, AttributeError, TypeError):
            pass
        # Skip category-level blacklists for the session
        if source_name == "fmp" and ("fmp:fundamentals" in DISABLED_PROVIDERS):
            try:
                record_api_call(
                    provider="FMP",
                    endpoint="fundamentals",
                    status="skipped_blacklist",
                    latency_sec=0.0,
                    extra={"ticker": ticker, "reason": "session_blacklist"},
                )
            except (TypeError, AttributeError):
                pass
            continue
        if source_name == "eodhd" and ("eodhd:fundamentals" in DISABLED_PROVIDERS):
            continue
        if source_name == "simfin" and ("simfin:fundamentals" in DISABLED_PROVIDERS):
            continue
        try:
            result = fetch_func(ticker, provider_status=provider_status)
            if result:
                # Validate that at least one key field is present and finite
                key_fields = ["pe", "ps", "pb", "roe", "margin", "rev_yoy", "eps_yoy", "debt_equity", "market_cap", "beta"]
                has_signal = False
                for k in key_fields:
                    v = result.get(k)
                    if v is not None and np.isfinite(v):
                        has_signal = True
                        break
                if has_signal:
                    sources_data[source_name] = result
                    logger.debug(f"✓ {source_name} data fetched for {ticker}")
                else:
                    logger.debug(f"✗ {source_name} returned no usable fundamentals for {ticker}")
        except Exception as e:
            logger.warning(f"Failed to fetch from {source_name} for {ticker}: {e}")
    
    if not sources_data:
        logger.warning(f"No fundamental data available for {ticker}")
        # Return a neutral, non-crashing structure with explicit metadata
        result = {
            "ticker": ticker,
            "sources_used": [],
            "coverage": {},
            "disagreement_score": 1.0,
            # Explicit fundamental coverage and counts
            "Fundamental_Coverage_Pct": 0.0,
            "Fundamental_Sources_Count": 0,
            # Maintain neutral default for compatibility, plus explicit flag
            "Fundamental_S": 50.0,
            "Fundamental_Missing": True,
            # Per-source flags for downstream logic
            "Fund_from_FMP": False,
            "Fund_from_Finnhub": False,
            "Fund_from_Tiingo": False,
            "Fund_from_Alpha": False,
            "timestamp": time.time(),
        }
        # Backtest awareness
        if mode == DataMode.BACKTEST:
            logger.warning(
                "Using snapshot fundamentals for BACKTEST mode on %s – results may suffer from lookahead bias",
                ticker,
            )
            result["Fundamental_Backtest_Unsafe"] = True
            result["Fundamental_From_Store"] = False
        else:
            result["Fundamental_Backtest_Unsafe"] = False
            result["Fundamental_From_Store"] = False
        return result
    
    # Aggregate numeric fields (extend with PEG and selected profile numerics)
    fields = [
        "pe", "ps", "pb", "roe", "margin", "rev_yoy", "eps_yoy",
        "debt_equity", "market_cap", "beta", "peg", "vol_avg", "last_div", "price_backup"
    ]
    aggregated = {"ticker": ticker}
    coverage = {}
    disagreements = []
    
    for field in fields:
        values = []
        contributing_sources = []
        
        for source_name, data in sources_data.items():
            val = data.get(field)
            # Treat NaN, empty strings, and '-' as invalid; skip to allow fallback
            try:
                if val in (None, "", "-", "N/A"):
                    continue
                v = float(val)
                if np.isfinite(v):
                    values.append(v)
                    contributing_sources.append(source_name)
            except (TypeError, ValueError):
                # Non-numeric values should not block aggregation
                continue
        
        if not values:
            aggregated[field] = None
            coverage[field] = []
        elif len(values) == 1:
            aggregated[field] = values[0]
            coverage[field] = contributing_sources
        else:
            # Multiple sources - use median and measure disagreement
            median_val = float(np.median(values))
            aggregated[field] = median_val
            coverage[field] = contributing_sources
            
            # Disagreement: coefficient of variation
            if median_val != 0:
                cv = np.std(values) / abs(median_val)
                disagreements.append(min(cv, 1.0))  # Cap at 1.0
    
    # Overall disagreement score
    if disagreements:
        aggregated["disagreement_score"] = float(np.mean(disagreements))
    else:
        aggregated["disagreement_score"] = 0.0
    
    aggregated["sources_used"] = list(sources_data.keys())
    aggregated["coverage"] = coverage
    aggregated["timestamp"] = time.time()

    # Record fallback if initial preferred/attempted provider yielded no data but others did
    try:
        if telemetry is not None and sources_data:
            first_success = next(iter(sources_data.keys()))
            # Mark only the first successful provider as the one used
            telemetry.mark_used("fundamentals", first_success)
            if first_attempted and first_attempted not in sources_data:
                telemetry.record_fallback("fundamentals", first_attempted, first_success, "no usable data")
    except (StopIteration, AttributeError, TypeError):
        pass

    # Add explicit per-source flags
    aggregated["Fund_from_FMP"] = "fmp" in sources_data
    aggregated["Fund_from_Finnhub"] = "finnhub" in sources_data
    aggregated["Fund_from_Tiingo"] = "tiingo" in sources_data
    aggregated["Fund_from_Alpha"] = "alpha" in sources_data

    # Compute coverage percentage across fundamental fields
    covered_fields = sum(1 for f in fields if (f in aggregated and pd.notna(aggregated.get(f))) )
    total_fields = len(fields)
    coverage_pct = (covered_fields / total_fields) * 100.0 if total_fields > 0 else 0.0
    aggregated["Fundamental_Coverage_Pct"] = float(coverage_pct)
    aggregated["Fundamental_Sources_Count"] = int(len(aggregated["sources_used"]))

    # Maintain neutral default for compatibility if not set upstream
    if "Fundamental_S" not in aggregated:
        aggregated["Fundamental_S"] = 50.0

    # Phase 13: Emergency MarketCap Default
    # If market_cap is missing or NaN after all sources, force assign 500,000,001
    try:
        mc = aggregated.get("market_cap")
        if mc is None or (isinstance(mc, float) and not np.isfinite(mc)):
            aggregated["market_cap"] = float(500000001)
            aggregated["MarketCap_Defaulted"] = True
        else:
            aggregated["MarketCap_Defaulted"] = False
    except (TypeError, ValueError):
        aggregated["market_cap"] = float(500000001)
        aggregated["MarketCap_Defaulted"] = True

    # Backtest awareness flag
    if mode == DataMode.BACKTEST:
        logger.warning(
            "Using snapshot fundamentals for BACKTEST mode on %s – results may suffer from lookahead bias",
            ticker,
        )
        aggregated["Fundamental_Backtest_Unsafe"] = True
        aggregated["Fundamental_From_Store"] = False
    else:
        aggregated["Fundamental_Backtest_Unsafe"] = False
        aggregated["Fundamental_From_Store"] = False
    
    # Cache the merged result
    cache_key = f"merged_fund_{ticker}"
    _put_in_cache(cache_key, aggregated)

    # Persist a point-in-time fundamentals snapshot in LIVE mode only, when we have data
    if mode == DataMode.LIVE and len(aggregated.get("sources_used", [])) > 0 and not aggregated.get("Fundamental_Missing", False):
        try:
            # Use the threaded signal date if provided; fallback to wall-clock
            as_of = as_of_date or date.today()
            save_fundamentals_snapshot(
                ticker=ticker,
                payload=aggregated,
                as_of_date=as_of,
                provider="multi_source",
            )
            logger.debug(f"Saved fundamentals snapshot for {ticker} as of {as_of}")
        except Exception as e:
            logger.warning(f"Failed to save fundamentals snapshot for {ticker}: {e}")
    
    # Phase 14: Final MarketCap enforcement before return
    try:
        mc_final = aggregated.get("market_cap")
        if mc_final is None or (isinstance(mc_final, float) and (not np.isfinite(mc_final) or mc_final == 0.0)) or mc_final == 0:
            aggregated["market_cap"] = float(500000005)
            aggregated["MarketCap_Defaulted_Final"] = True
    except (TypeError, ValueError):
        aggregated["market_cap"] = float(500000005)
        aggregated["MarketCap_Defaulted_Final"] = True
    logger.info(f"Aggregated fundamentals for {ticker} from {len(sources_data)} sources")
    
    # Map descriptive string fields opportunistically (sector)
    try:
        sector = None
        # prefer FMP then Alpha
        if "fmp" in sources_data:
            sector = sources_data.get("fmp", {}).get("sector")
        if (not sector) and ("alpha" in sources_data):
            sector = sources_data.get("alpha", {}).get("sector")
        if sector:
            aggregated["sector"] = sector
    except (KeyError, TypeError, AttributeError):
        pass

    return aggregated


# ============================================================================
# MULTI-SOURCE PRICE VERIFICATION (EXISTING + ENHANCED)
# ============================================================================

def fetch_price_multi_source(ticker: str, provider_status: Dict | None = None, telemetry: Optional[Any] = None) -> Dict[str, Optional[float]]:
    """
    Fetch current price from multiple sources for verification.

    STRICT PRIORITY: Polygon is PRIMARY for real-time checks.
    Other sources (FMP, Finnhub, Tiingo, Alpha) are secondary.

    Returns:
        Dict with numeric prices per source and optional 'primary_source' key
        when Polygon succeeds.
    """
    prices: Dict[str, Optional[float]] = {}

    # Reload keys at runtime
    POLYGON_KEY_RUNTIME = get_secret("POLYGON_API_KEY", os.getenv("POLYGON_API_KEY", ""))
    EODHD_KEY_RUNTIME = get_secret("EODHD_API_KEY", os.getenv("EODHD_API_KEY", "")) or os.getenv("EODHD_TOKEN", "")
    FMP_KEY_RUNTIME = get_secret("FMP_API_KEY", os.getenv("FMP_API_KEY", "")) or os.getenv("FMP_KEY", "")
    FINNHUB_KEY_RUNTIME = get_secret("FINNHUB_API_KEY", os.getenv("FINNHUB_API_KEY", ""))
    TIINGO_KEY_RUNTIME = get_secret("TIINGO_API_KEY", os.getenv("TIINGO_API_KEY", ""))
    ALPHA_KEY_RUNTIME = get_secret("ALPHA_VANTAGE_API_KEY", os.getenv("ALPHA_VANTAGE_API_KEY", "")) or os.getenv("ALPHAVANTAGE_API_KEY", "")
    MARKETSTACK_KEY_RUNTIME = get_secret("MARKETSTACK_API_KEY", os.getenv("MARKETSTACK_API_KEY", ""))

    # Helper to check if a provider can price per preflight
    def _can_price(name_upper: str) -> bool:
        try:
            if not provider_status:
                return True
            s = provider_status.get(name_upper)
            if not s:
                return True
            if s.get("status") in ("auth_error", "no_key"):
                return False
            return s.get("can_price", True)
        except (KeyError, TypeError, AttributeError):
            return True

    # Track attempted primary provider for fallback recording
    attempted_primary: Optional[str] = None

    # Polygon price (PRIMARY)
    if _can_price("POLYGON") and (POLYGON_KEY_RUNTIME or POLYGON_API_KEY):
        try:
            _rate_limit("polygon")
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev"
            params = {"apiKey": (POLYGON_KEY_RUNTIME or POLYGON_API_KEY)}
            # Fast-fail on 429 via session disable; no sleeps/backoff here
            data = _http_get_with_retry(url, params=params, timeout=3, provider="POLYGON", capability="price")
            if data and "results" in data and len(data["results"]) > 0:
                prices["polygon"] = data["results"][0].get("c")
                prices["primary_source"] = "polygon"
                try:
                    if telemetry is not None:
                        telemetry.mark_used("price", "POLYGON")
                        telemetry.set_value(f"price_provider:{ticker}", "POLYGON")
                except (AttributeError, TypeError):
                    pass
            else:
                attempted_primary = "POLYGON"
        except Exception as e:
            logger.warning(f"Polygon price fetch failed: {e}", exc_info=True)
            prices["polygon"] = None
            attempted_primary = "POLYGON"

    # EODHD price (alternative PRIMARY if Polygon absent)
    if _can_price("EODHD") and ("primary_source" not in prices) and (EODHD_KEY_RUNTIME or EODHD_API_KEY) and ("eodhd:price" not in DISABLED_PROVIDERS):
        try:
            _rate_limit("eodhd")
            url = f"https://eodhd.com/api/real-time/{ticker}"
            params = {"api_token": (EODHD_KEY_RUNTIME or EODHD_API_KEY), "fmt": "json"}
            resp_block = _http_get_with_retry(url, params=params, timeout=3, provider="EODHD", capability="price")
            if isinstance(resp_block, dict) and resp_block.get("__blocked__"):
                pass
            else:
                # Fallback to raw resp parsing if not blocked
                resp = requests.get(url, params=params, timeout=3)
            if resp.status_code in (401, 403):
                disable_provider_category("eodhd", "price")
            elif resp.status_code == 200:
                data = resp.json()
                # EODHD returns dict with fields like 'close', 'previousClose'
                close_val = None
                if isinstance(data, dict):
                    close_val = data.get("close") or data.get("previousClose") or data.get("c")
                prices["eodhd"] = close_val
                if close_val is not None:
                    prices["primary_source"] = "eodhd"
                    try:
                        if telemetry is not None:
                            telemetry.mark_used("price", "EODHD")
                            telemetry.set_value(f"price_provider:{ticker}", "EODHD")
                            if attempted_primary and attempted_primary != "EODHD":
                                telemetry.record_fallback("price", attempted_primary, "EODHD", "primary provider returned no data")
                    except (AttributeError, TypeError):
                        pass
        except Exception as e:
            logger.warning(f"EODHD price fetch failed: {e}")

    # FMP price (secondary) - using stable API
    if _can_price("FMP") and (FMP_KEY_RUNTIME or FMP_API_KEY) and ("fmp:price" not in DISABLED_PROVIDERS):
        try:
            _rate_limit("fmp")
            url = f"https://financialmodelingprep.com/stable/quote?symbol={ticker}"
            params = {"apikey": (FMP_KEY_RUNTIME or FMP_API_KEY)}
            # Use raw request to catch 403 and blacklist
            resp_block = _http_get_with_retry(url, params=params, timeout=3, provider="FMP", capability="price")
            if isinstance(resp_block, dict) and resp_block.get("__blocked__"):
                pass
            else:
                resp = requests.get(url, params=params, timeout=3)
            if resp.status_code == 403:
                disable_provider_category("fmp", "price")
                record_api_call(
                    provider="FMP",
                    endpoint="quote",
                    status="http_403_disabled",
                    latency_sec=0.0,
                    extra={"ticker": ticker}
                )
            elif resp.status_code == 200:
                data = resp.json()
                if data and isinstance(data, list) and len(data) > 0:
                    prices["fmp"] = data[0].get("price")
                    if prices.get("primary_source") is None and prices["fmp"] is not None:
                        prices["primary_source"] = "fmp"
                        try:
                            if telemetry is not None:
                                telemetry.mark_used("price", "FMP")
                                telemetry.set_value(f"price_provider:{ticker}", "FMP")
                                if attempted_primary and attempted_primary != "FMP":
                                    telemetry.record_fallback("price", attempted_primary, "FMP", "primary provider returned no data")
                        except (AttributeError, TypeError):
                            pass
            else:
                record_api_call(
                    provider="FMP",
                    endpoint="quote",
                    status=f"http_{resp.status_code}",
                    latency_sec=0.0,
                    extra={"ticker": ticker}
                )
        except Exception as e:
            logger.warning(f"FMP price fetch failed: {e}", exc_info=True)
            prices["fmp"] = None

    # Finnhub price (secondary)
    if _can_price("FINNHUB") and (FINNHUB_KEY_RUNTIME or FINNHUB_API_KEY):
        try:
            _rate_limit("finnhub")
            url = "https://finnhub.io/api/v1/quote"
            params = {"symbol": ticker, "token": (FINNHUB_KEY_RUNTIME or FINNHUB_API_KEY)}
            data = _http_get_with_retry(url, params=params, timeout=3, provider="FINNHUB", capability="price")
            if data:
                prices["finnhub"] = data.get("c")
                if prices.get("primary_source") is None and prices["finnhub"] is not None:
                    prices["primary_source"] = "finnhub"
                    try:
                        if telemetry is not None:
                            telemetry.mark_used("price", "FINNHUB")
                            telemetry.set_value(f"price_provider:{ticker}", "FINNHUB")
                            if attempted_primary and attempted_primary != "FINNHUB":
                                telemetry.record_fallback("price", attempted_primary, "FINNHUB", "primary provider returned no data")
                    except (AttributeError, TypeError):
                        pass
        except Exception as e:
            logger.warning(f"Finnhub price fetch failed: {e}", exc_info=True)
            prices["finnhub"] = None

    # Tiingo price (secondary)
    if _can_price("TIINGO") and (TIINGO_KEY_RUNTIME or TIINGO_API_KEY) and ("tiingo:price" not in DISABLED_PROVIDERS):
        try:
            _rate_limit("tiingo")
            url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
            headers = {"Content-Type": "application/json", "Authorization": f"Token {TIINGO_KEY_RUNTIME or TIINGO_API_KEY}"}
            data = _http_get_with_retry(url, headers=headers, timeout=3, provider="TIINGO", capability="price")
            if data and isinstance(data, list) and len(data) > 0:
                prices["tiingo"] = data[0].get("close")
                if prices.get("primary_source") is None and prices["tiingo"] is not None:
                    prices["primary_source"] = "tiingo"
                    try:
                        if telemetry is not None:
                            telemetry.mark_used("price", "TIINGO")
                            telemetry.set_value(f"price_provider:{ticker}", "TIINGO")
                            if attempted_primary and attempted_primary != "TIINGO":
                                telemetry.record_fallback("price", attempted_primary, "TIINGO", "primary provider returned no data")
                    except (AttributeError, TypeError):
                        pass
        except Exception as e:
            logger.warning(f"Tiingo price fetch failed: {e}", exc_info=True)
            prices["tiingo"] = None

    # Alpha Vantage price (secondary)
    if ALPHA_KEY_RUNTIME or ALPHA_VANTAGE_API_KEY:
        try:
            _rate_limit("alpha")
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": ticker,
                "apikey": (ALPHA_KEY_RUNTIME or ALPHA_VANTAGE_API_KEY)
            }
            data = _http_get_with_retry(url, params=params, timeout=3, provider="ALPHAVANTAGE", capability="price")
            if data and "Global Quote" in data:
                price_str = data["Global Quote"].get("05. price")
                prices["alpha"] = float(price_str) if price_str else None
                if prices.get("primary_source") is None and prices["alpha"] is not None:
                    prices["primary_source"] = "alpha"
                    try:
                        if telemetry is not None:
                            telemetry.mark_used("price", "ALPHAVANTAGE")
                            telemetry.set_value(f"price_provider:{ticker}", "ALPHAVANTAGE")
                            if attempted_primary and attempted_primary != "ALPHAVANTAGE":
                                telemetry.record_fallback("price", attempted_primary, "ALPHAVANTAGE", "primary provider returned no data")
                    except (AttributeError, TypeError):
                        pass
        except Exception as e:
            logger.warning(f"Alpha price fetch failed: {e}", exc_info=True)
            prices["alpha"] = None

    # MarketStack price (fallback for daily bars)
    if _can_price("MARKETSTACK") and ("primary_source" not in prices) and (MARKETSTACK_KEY_RUNTIME or MARKETSTACK_API_KEY) and ("marketstack:price" not in DISABLED_PROVIDERS):
        try:
            _rate_limit("marketstack")
            url = "http://api.marketstack.com/v1/eod/latest"
            params = {"access_key": (MARKETSTACK_KEY_RUNTIME or MARKETSTACK_API_KEY), "symbols": ticker}
            resp_block = _http_get_with_retry(url, params=params, timeout=4, provider="MARKETSTACK", capability="price")
            if isinstance(resp_block, dict) and resp_block.get("__blocked__"):
                pass
            else:
                resp = requests.get(url, params=params, timeout=4)
            if resp.status_code in (401, 403):
                disable_provider_category("marketstack", "price")
            elif resp.status_code == 200:
                data = resp.json()
                close_val = None
                if isinstance(data, dict) and isinstance(data.get("data"), list) and data["data"]:
                    close_val = data["data"][0].get("close")
                prices["marketstack"] = close_val
                if close_val is not None:
                    prices["primary_source"] = "marketstack"
                    try:
                        if telemetry is not None:
                            telemetry.mark_used("price", "MARKETSTACK")
                            telemetry.set_value(f"price_provider:{ticker}", "MARKETSTACK")
                            if attempted_primary and attempted_primary != "MARKETSTACK":
                                telemetry.record_fallback("price", attempted_primary, "MARKETSTACK", "primary provider returned no data")
                    except (AttributeError, TypeError):
                        pass
        except Exception as e:
            logger.warning(f"MarketStack price fetch failed: {e}")

    return prices


def aggregate_price(prices: Dict[str, Optional[float]]) -> Tuple[float, float, int]:
    """
    Aggregate prices from multiple sources.
    
    Returns:
        (mean_price, std_price, num_sources)
    """
    # Only aggregate numeric price values; ignore metadata like 'primary_source'
    numeric_values = []
    for k, v in prices.items():
        if k == "primary_source":
            continue
        try:
            if v is not None and np.isfinite(float(v)) and float(v) > 0:
                numeric_values.append(float(v))
        except (TypeError, ValueError):
            # Ignore non-numeric values
            continue

    valid_prices = numeric_values
    
    if not valid_prices:
        return np.nan, np.nan, 0
    
    mean_price = float(np.mean(valid_prices))
    std_price = float(np.std(valid_prices)) if len(valid_prices) > 1 else 0.0
    
    return mean_price, std_price, len(valid_prices)

# ============================================================================
# EARNINGS / CALENDAR — FINNHUB PRIMARY
# ============================================================================

def get_next_earnings_date(ticker: str) -> Optional[str]:
    """
    Fetch the next earnings date for a ticker.

    STRICT PRIORITY: Finnhub is PRIMARY. If Finnhub responds (even empty), do
    not fallback. Only fallback to FMP on network error (no response).

    Returns a date string 'YYYY-MM-DD' or None if unavailable.
    """
    # Finnhub primary
    if FINNHUB_API_KEY:
        try:
            _rate_limit("finnhub")
            url = "https://finnhub.io/api/v1/calendar/earnings"
            # Finnhub requires 'symbol' and supports 'from'/'to'; use a 1y window
            params = {"symbol": ticker, "token": FINNHUB_API_KEY}
            data = _http_get_with_retry(url, params=params, timeout=3, provider="FINNHUB", capability="earnings")
            if data is not None:
                items = data.get("earningsCalendar") or data.get("result") or []
                # Find nearest future date
                dates = []
                for it in items:
                    d = it.get("date") or it.get("earningsDate") or it.get("epsDate")
                    if d:
                        dates.append(d)
                if dates:
                    return sorted(dates)[0]
                # Finnhub responded but no items — do not fallback
                return None
        except Exception as e:
            logger.warning(f"Finnhub earnings fetch failed for {ticker}: {e}")
            # Network error — allow fallback

    # FMP fallback ONLY if Finnhub errored
    if FMP_API_KEY:
        try:
            _rate_limit("fmp")
            url = f"https://financialmodelingprep.com/stable/earning-calendar-confirmed"
            params = {"symbol": ticker, "apikey": FMP_API_KEY}
            data = _http_get_with_retry(url, params=params, timeout=3, provider="FMP", capability="earnings")
            if data and isinstance(data, list) and len(data) > 0:
                # FMP returns list of entries; pick the next upcoming
                dates = [it.get("date") for it in data if it.get("date")]
                if dates:
                    return sorted(dates)[0]
        except Exception as e:
            logger.warning(f"FMP earnings fetch failed for {ticker}: {e}")
    return None


# ============================================================================
# PUBLIC API
# ============================================================================

def fetch_multi_source_data(ticker: str, provider_status: Dict | None = None, mode: DataMode = DataMode.LIVE, as_of_date: Optional[date] = None, telemetry: Optional[Any] = None) -> Dict:
    """
    Main entry point: fetch fundamentals and price from all sources.
    
    Returns comprehensive dict with:
    - All fundamental fields
    - sources_used, coverage, disagreement_score
    - price_mean, price_std, price_sources
    - Individual source prices for verification
    """
    # Get fundamentals
    fundamentals = aggregate_fundamentals(ticker, provider_status=provider_status, mode=mode, as_of_date=as_of_date, telemetry=telemetry)
    
    # Get prices
    prices = fetch_price_multi_source(ticker, provider_status=provider_status, telemetry=telemetry)
    price_mean, price_std, price_count = aggregate_price(prices)
    
    # Merge everything
    result = {**fundamentals}
    # Lowercase canonical
    result["price_mean"] = price_mean
    result["price_std"] = price_std
    result["price_sources"] = price_count
    result["prices_by_source"] = prices
    # Uppercase aliases for downstream that expect these names
    result["Price_Mean"] = price_mean
    result["Price_STD"] = price_std
    result["Price_Sources_Count"] = price_count
    
    return result


def clear_cache() -> None:
    """Clear all cached data (thread-safe)."""
    global _CACHE
    with _CACHE_LOCK:
        _CACHE = {}
    logger.info("Cache cleared")


def reset_disabled_providers() -> None:
    """Reset all disabled providers to their default state.

    This is useful for ensuring consistent results between runs by
    clearing any provider blacklists that may have accumulated due
    to rate limits or temporary failures.
    """
    global _PROVIDER_DISABLED, DISABLED_PROVIDERS
    _PROVIDER_DISABLED = {
        "tiingo": False,
        "fmp": False,
    }
    DISABLED_PROVIDERS.clear()
    logger.info("Provider blacklists reset")


def get_cache_stats() -> Dict:
    """Get cache statistics (thread-safe snapshot)."""
    with _CACHE_LOCK:
        keys = list(_CACHE.keys())
    return {
        "total_entries": len(keys),
        "sources": {
            source: sum(1 for k in keys if k.startswith(source))
            for source in ["fmp", "finnhub", "tiingo", "alpha", "merged", "index_series"]
        }
    }


def get_index_series(
    symbol: str,
    start_date: str,
    end_date: str,
    provider_status: Optional[Dict[str, bool]] = None,
    telemetry: Optional[Any] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLCV series for market indices (SPY, QQQ, ^VIX, etc.).
    
    Uses FMP as primary, with fallback to Tiingo/Alpha Vantage.
    Applies caching and rate-limiting like other v2 functions.
    
    Args:
        symbol: Index symbol (e.g., 'SPY', 'QQQ', '^VIX')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        provider_status: Optional dict with provider availability flags
    
    Returns:
        DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'volume']
        or None if all sources fail
    """
    # Normalize VIX symbol for different providers
    fmp_symbol = symbol.replace('^', '')  # FMP uses 'VIX' not '^VIX'
    
    # Check cache (thread-safe)
    cache_key = f"index_series_{symbol}_{start_date}_{end_date}"
    _cached = _get_from_cache(cache_key)
    if _cached is not None:
        logger.debug(f"✓ Cache hit for index series {symbol}")
        return _cached
    
    provider_status = provider_status or _DEFAULT_PROVIDER_STATUS
    df_result = None
    
    # Prefer Polygon first for SPY and VIX to avoid FMP 403 delays
    prefer_polygon = symbol.upper() in {"SPY", "^VIX", "VIX"}

    # If we don't prefer Polygon, try FMP first; otherwise skip to Polygon block below
    if (not prefer_polygon) and provider_status.get("fmp", True) and FMP_API_KEY and not _PROVIDER_DISABLED.get("fmp", False):
        try:
            _rate_limit("fmp")
            # Modern stable endpoint for historical price data
            url = f"https://financialmodelingprep.com/stable/historical-price-eod/light?symbol={fmp_symbol}"
            params = {
                "apikey": FMP_API_KEY,
                "from": start_date,
                "to": end_date,
                "serietype": "line",
            }
            
            record_api_call("fmp", f"index_series_{symbol}", status="attempt")
            # Use raw request to detect legacy error text and status codes
            try:
                resp = requests.get(url, params=params, timeout=3)
                if resp.status_code == 403:
                    disable_provider_category("fmp", "index")
                    logger.warning("FMP index 403; disabling fmp:index for session")
                    data = None
                elif resp.status_code != 200:
                    logger.warning(f"FMP HTTP {resp.status_code} for {symbol}; skipping to fallback")
                    data = None
                else:
                    data = resp.json()
                    if isinstance(data, dict) and "Error Message" in data and "Legacy Endpoint" in str(data.get("Error Message", "")):
                        disable_provider("fmp")
                        logger.error("FMP Legacy Endpoint detected — switching to Polygon for this session")
                        data = None
            except Exception as e:
                logger.warning(f"FMP request failed: {e}")
                data = None

            # historical-chart returns list of bars directly
            if data and isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                df = df.rename(columns={
                    "date": "date",
                    "open": "open",
                    "high": "high", 
                    "low": "low",
                    "close": "close",
                    "volume": "volume"
                })
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                df_result = df[["date", "open", "high", "low", "close", "volume"]]
                logger.info(f"✓ FMP: Fetched {len(df_result)} days for {symbol}")
                _LAST_INDEX_SOURCE[symbol] = "FMP"
                try:
                    if telemetry is not None:
                        telemetry.mark_index(symbol, "FMP")
                except (AttributeError, TypeError):
                    pass
                _put_in_cache(cache_key, df_result)
                return df_result
            elif data is not None:
                # FMP responded but with no data — proceed to paid fallback (Polygon)
                logger.error(f"FMP returned no data for {symbol}; attempting Polygon fallback")
                
        except Exception as e:
            logger.warning(f"FMP index series failed for {symbol}: {e}")
    
    # Polygon primary for SPY/VIX, fallback otherwise
    if df_result is None and provider_status.get("polygon", True) and POLYGON_API_KEY:
        try:
            _rate_limit("polygon")
            poly_symbol = symbol.replace('^', '')
            url = f"https://api.polygon.io/v2/aggs/ticker/{poly_symbol}/range/1/day/{start_date}/{end_date}"
            params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY}
            record_api_call("polygon", f"index_series_{symbol}", status="attempt")
            # Immediate fallback on 429/403 without retrying
            resp = requests.get(url, params=params, timeout=3)
            if resp.status_code in (429, 403):
                logger.warning(f"Polygon {resp.status_code} for {symbol}; switching to next source immediately")
                data = None
            elif resp.status_code == 200:
                data = resp.json()
            else:
                logger.warning(f"Polygon HTTP {resp.status_code} for {symbol}")
                data = None
            if data and "results" in data and len(data["results"]) > 0:
                recs = data["results"]
                df = pd.DataFrame(recs)
                # Polygon fields: t (ms), o,h,l,c,v
                df["date"] = pd.to_datetime(df["t"], unit='ms')
                df_result = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})[
                    ["date", "open", "high", "low", "close", "volume"]
                ].sort_values("date").reset_index(drop=True)
                logger.info(f"✓ Polygon: Fetched {len(df_result)} days for {symbol}")
                _LAST_INDEX_SOURCE[symbol] = "POLYGON"
                try:
                    if telemetry is not None:
                        telemetry.mark_index(symbol, "POLYGON")
                        # Minimal price domain mark when SPY fetched via Polygon
                        if str(symbol).upper() == "SPY":
                            telemetry.mark_used("price", "POLYGON")
                except (AttributeError, TypeError):
                    pass
                _put_in_cache(cache_key, df_result)
                return df_result
        except Exception as e:
            logger.warning(f"Polygon index series failed for {symbol}: {e}")

    # Fallback to Tiingo (if not disabled)
    if df_result is None and provider_status.get("tiingo", True) and TIINGO_API_KEY and not _PROVIDER_DISABLED.get("tiingo", False):
        try:
            _rate_limit("tiingo")
            tiingo_symbol = symbol.replace('^', '$')  # $VIX for Tiingo
            url = f"https://api.tiingo.com/tiingo/daily/{tiingo_symbol}/prices"
            headers = {"Content-Type": "application/json", "Authorization": f"Token {TIINGO_API_KEY}"}
            params = {"startDate": start_date, "endDate": end_date}

            record_api_call("tiingo", f"index_series_{symbol}", status="attempt")
            # Use raw request to detect 429
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=3)
                status_code = resp.status_code
                if status_code == 429:
                    disable_provider("tiingo")
                    logger.warning("Tiingo rate-limited (429). Disabling for session.")
                    data = None
                else:
                    data = resp.json()
                    # Detect hourly allocation error text
                    if isinstance(data, dict) and any("limit" in str(v).lower() or "allocation" in str(v).lower() for v in data.values()):
                        disable_provider("tiingo")
                        logger.warning("Tiingo allocation error. Disabling for session.")
                        data = None
            except Exception as e:
                logger.warning(f"Tiingo request failed: {e}")
                data = None
            
            if data and isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                df = df.rename(columns={
                    "date": "date",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume"
                })
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                df_result = df[["date", "open", "high", "low", "close", "volume"]]
                logger.info(f"✓ Tiingo: Fetched {len(df_result)} days for {symbol}")
                _LAST_INDEX_SOURCE[symbol] = "TIINGO"
                try:
                    if telemetry is not None:
                        telemetry.mark_index(symbol, "TIINGO")
                except (AttributeError, TypeError):
                    pass
                
        except Exception as e:
            logger.warning(f"Tiingo index series failed for {symbol}: {e}")
    
    # Fallback to Alpha Vantage (slower but reliable)
    if df_result is None and provider_status.get("alpha", True) and ALPHA_VANTAGE_API_KEY:
        try:
            _rate_limit("alpha")
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "apikey": ALPHA_VANTAGE_API_KEY,
                "outputsize": "full"
            }
            
            record_api_call("alpha", f"index_series_{symbol}", status="attempt")
            data = _http_get_with_retry(url, params=params, timeout=3)
            
            if data and "Time Series (Daily)" in data:
                ts = data["Time Series (Daily)"]
                records = []
                for date_str, values in ts.items():
                    date_obj = pd.to_datetime(date_str)
                    if pd.to_datetime(start_date) <= date_obj <= pd.to_datetime(end_date):
                        records.append({
                            "date": date_obj,
                            "open": float(values.get("1. open", 0)),
                            "high": float(values.get("2. high", 0)),
                            "low": float(values.get("3. low", 0)),
                            "close": float(values.get("4. close", 0)),
                            "volume": float(values.get("5. volume", 0))
                        })
                
                if records:
                    df_result = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
                    logger.info(f"✓ Alpha: Fetched {len(df_result)} days for {symbol}")
                    _LAST_INDEX_SOURCE[symbol] = "ALPHAVANTAGE"
                    try:
                        if telemetry is not None:
                            telemetry.mark_index(symbol, "ALPHAVANTAGE")
                    except (AttributeError, TypeError):
                        pass
                    
        except Exception as e:
            logger.warning(f"Alpha Vantage index series failed for {symbol}: {e}")
    
    # If all sources failed and symbol is VIX, attempt synthetic proxy from SPY
    if df_result is None and symbol.upper() in {"^VIX", "VIX"}:
        try:
            # Build a VIX-like proxy using SPY realized volatility (30D rolling std * sqrt(252) * 100)
            spy_df = get_index_series("SPY", start_date, end_date, provider_status)
            if spy_df is not None and not spy_df.empty and "close" in spy_df.columns:
                spy_df = spy_df.sort_values("date").reset_index(drop=True)
                # Normalize timezone to naive for safe comparisons
                spy_df["date"] = pd.to_datetime(spy_df["date"]).dt.tz_localize(None)
                ret = spy_df["close"].pct_change()
                vol = (ret.rolling(30).std() * (252 ** 0.5)) * 100.0
                proxy = pd.DataFrame({
                    "date": spy_df["date"],
                    "close": vol.bfill().ffill()
                })
                # Construct OHLCV with close as proxy value
                proxy["open"] = proxy["close"]
                proxy["high"] = proxy["close"]
                proxy["low"] = proxy["close"]
                proxy["volume"] = 0.0
                # Filter to requested window and cache
                proxy["date"] = pd.to_datetime(proxy["date"]).dt.tz_localize(None)
                proxy = proxy[(proxy["date"] >= pd.to_datetime(start_date)) & (proxy["date"] <= pd.to_datetime(end_date))]
                if not proxy.empty:
                    df_result = proxy[["date", "open", "high", "low", "close", "volume"]].copy()
                    _LAST_INDEX_SOURCE[symbol] = "SYNTHETIC_VIX_PROXY"
                    try:
                        if telemetry is not None:
                            telemetry.mark_index(symbol, "SYNTHETIC_VIX_PROXY")
                            telemetry.record_fallback("index", "provider_chain", "SYNTHETIC_VIX_PROXY", "constructed proxy from SPY realized volatility")
                    except Exception:
                        pass
                    logger.warning("Using synthetic VIX proxy derived from SPY realized volatility")
        except Exception as e:
            logger.warning(f"Synthetic VIX proxy construction failed: {e}")

    # Final fallback: yfinance for widely available ETFs/indices (e.g., SPY)
    if df_result is None and symbol.upper() not in {"^VIX", "VIX"}:
        try:
            import yfinance as yf
            yf_df = yf.Ticker(symbol).history(start=start_date, end=end_date)
            if yf_df is not None and not yf_df.empty:
                yf_df = yf_df.reset_index().rename(columns={
                    'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                })
                # Ensure expected columns
                cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                if all(c in yf_df.columns for c in cols):
                    df_result = yf_df[cols].copy()
                    _LAST_INDEX_SOURCE[symbol] = 'YFINANCE'
                    try:
                        if telemetry is not None:
                            telemetry.mark_index(symbol, "YFINANCE")
                    except Exception:
                        pass
                    logger.warning(f"Using yfinance fallback for {symbol}")
        except Exception as e:
            logger.warning(f"yfinance fallback failed for {symbol}: {e}")

    # Cache result
    if df_result is not None:
        _put_in_cache(cache_key, df_result)
    else:
        logger.error(f"❌ All sources failed for index series {symbol}")
    
    return df_result


def fetch_fundamentals_batch(tickers: List[str], provider_status: Dict | None = None, mode: DataMode = DataMode.LIVE, as_of_date: Optional[date] = None, telemetry: Optional[Any] = None) -> pd.DataFrame:
    """
    Batch fetch fundamentals for multiple tickers using aggregate_fundamentals.
    Runs serially to avoid thread deadlocks / DB contention.
    Returns a DataFrame indexed by Ticker.
    """
    if not tickers:
        return pd.DataFrame()

    rows = []
    # Build prioritized provider list once per batch
    try:
        prioritized_fetch = get_prioritized_fetch_funcs(provider_status)
    except (ImportError, AttributeError, TypeError):
        prioritized_fetch = []
    if not prioritized_fetch:
        # No providers available per preflight/disable — return neutral DataFrame indexed by tickers
        return pd.DataFrame(index=pd.Index([t for t in tickers], name="Ticker"))
    # Throttled submissions to reduce burst rate limits (tuned for stability)
    # If FMP and FINNHUB are both unavailable, we likely fall back to Tiingo
    # which rejects high concurrency. In that case, cap workers at 2.
    try:
        fmp_ok = (provider_status or {}).get("FMP", {"ok": True}).get("ok", True)
        finnhub_ok = (provider_status or {}).get("FINNHUB", {"ok": True}).get("ok", True)
        if (not fmp_ok) and (not finnhub_ok):
            max_workers = min(len(tickers), 2)
        else:
            max_workers = min(len(tickers), 12)
    except (KeyError, TypeError, AttributeError):
        max_workers = min(len(tickers), 12)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {}
        for t in tickers:
            # Choose first currently allowed provider per ticker
            chosen = None
            for name, _func in prioritized_fetch:
                upper = {
                    "fmp": "FMP",
                    "finnhub": "FINNHUB",
                    "tiingo": "TIINGO",
                    "alpha": "ALPHAVANTAGE",
                    "eodhd": "EODHD",
                    "simfin": "SIMFIN",
                }.get(name, name.upper())
                try:
                    allowed, _rsn, _dec = get_provider_guard().allow(upper, "fundamentals")
                except (ImportError, AttributeError, TypeError):
                    allowed = True
                if allowed:
                    chosen = name
                    break
            if not chosen:
                try:
                    record_api_call(
                        provider="MULTI",
                        endpoint="fundamentals",
                        status="fundamentals_unavailable",
                        latency_sec=0.0,
                        extra={"ticker": t}
                    )
                except (TypeError, AttributeError):
                    pass
                continue
            future = executor.submit(aggregate_fundamentals, t, chosen, provider_status, mode, as_of_date, telemetry)
            future_to_ticker[future] = t
        for future in as_completed(future_to_ticker):
            tkr = future_to_ticker[future]
            try:
                res = future.result()
                rows.append(res)
            except Exception as e:
                logger.error(f"Batch fetch failed for {tkr}: {e}")
                
    if not rows:
        # Return neutral DataFrame with index tickers when batch yielded no rows
        return pd.DataFrame(index=pd.Index([t for t in tickers], name="Ticker"))

    df = pd.DataFrame(rows)
    if "ticker" in df.columns:
        df = df.rename(columns={"ticker": "Ticker"})
        df = df.set_index("Ticker")

    return df
