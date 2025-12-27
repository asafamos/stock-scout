"""
Multi-Source Data Aggregation Module (v2).

This module implements robust multi-source data fetching with:
- Priority chain: FMP → Finnhub → Tiingo → Alpha Vantage
- Multi-source fusion: aggregate data from ALL available sources
- Per-source caching with TTL
- Retry logic with exponential backoff
- Timeout protection
- NaN-safe aggregation

Design principle: Use MORE data, not less. Cross-check and enrich.
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

logger = logging.getLogger(__name__)

# API Keys from environment
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY", "")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

# Cache configuration
_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = 3600  # 1 hour default

# Rate limiting
_LAST_CALL_TIME: Dict[str, float] = {}
MIN_INTERVAL_SECONDS = {
    "fmp": 0.1,        # 10 calls/sec
    "finnhub": 0.2,    # 5 calls/sec
    "tiingo": 0.1,     # 10 calls/sec
    "alpha": 12.0,     # 5 calls/min
    "polygon": 0.2,    # 5 calls/sec
}


def _rate_limit(source: str) -> None:
    """Apply rate limiting for a given source."""
    if source not in _LAST_CALL_TIME:
        _LAST_CALL_TIME[source] = 0
    
    elapsed = time.time() - _LAST_CALL_TIME[source]
    min_interval = MIN_INTERVAL_SECONDS.get(source, 0.1)
    
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    
    _LAST_CALL_TIME[source] = time.time()


def _http_get_with_retry(
    url: str,
    params: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    timeout: int = 10,
    max_retries: int = 3,
) -> Optional[Dict]:
    """
    HTTP GET with retry logic and exponential backoff.
    
    Returns:
        Parsed JSON dict or None on failure
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limited
                wait_time = (2 ** attempt) * 2
                logger.warning(f"Rate limited on {url}, waiting {wait_time}s")
                time.sleep(wait_time)
                continue
            else:
                logger.warning(f"HTTP {response.status_code} for {url}")
                return None
                
        except requests.Timeout:
            logger.warning(f"Timeout on attempt {attempt+1}/{max_retries} for {url}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    return None


def _get_from_cache(cache_key: str, ttl: int = CACHE_TTL_SECONDS) -> Optional[Dict]:
    """Retrieve from cache if not expired."""
    if cache_key in _CACHE:
        entry = _CACHE[cache_key]
        if time.time() - entry["timestamp"] < ttl:
            return entry["data"]
        else:
            del _CACHE[cache_key]
    return None


def _put_in_cache(cache_key: str, data: Dict) -> None:
    """Store in cache with timestamp."""
    _CACHE[cache_key] = {
        "data": data,
        "timestamp": time.time()
    }


# ============================================================================
# FMP (Financial Modeling Prep) - PRIMARY SOURCE
# ============================================================================

def fetch_fundamentals_fmp(ticker: str, provider_status: Dict | None = None) -> Optional[Dict]:
    """
    Fetch fundamentals from FMP (primary source).
    
    Returns standardized dict with keys:
    - pe, ps, pb, roe, margin, rev_yoy, eps_yoy, debt_equity
    - market_cap, beta, etc.
    """
    # Preflight skip
    if provider_status is not None:
        s = provider_status.get("FMP")
        if s and not s.get("ok", True):
            try:
                record_api_call(
                    provider="FMP",
                    endpoint="key-metrics",
                    status="skipped_preflight",
                    latency_sec=0.0,
                    extra={"ticker": ticker, "reason": "disabled_by_preflight"},
                )
            except Exception:
                pass
            return None

    if not FMP_API_KEY:
        return None
    
    cache_key = f"fmp_fund_{ticker}"
    cached = _get_from_cache(cache_key)
    if cached:
        return cached
    
    _rate_limit("fmp")
    
    # Fetch key metrics
    url = f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}"
    params = {"apikey": FMP_API_KEY, "limit": 1}
    
    start = time.time()
    try:
        data = _http_get_with_retry(url, params=params)
        status = "ok" if data and isinstance(data, list) and len(data) > 0 else "empty"
        record_api_call(
            provider="FMP",
            endpoint="key-metrics",
            status=status,
            latency_sec=time.time() - start,
            extra={"ticker": ticker}
        )
    except Exception as e:
        record_api_call(
            provider="FMP",
            endpoint="key-metrics",
            status="exception",
            latency_sec=time.time() - start,
            extra={"ticker": ticker, "error": str(e)[:200]}
        )
        return None
    if not data or not isinstance(data, list) or len(data) == 0:
        return None
    metrics = data[0]
    # Fetch ratios
    url_ratios = f"https://financialmodelingprep.com/api/v3/ratios/{ticker}"
    start_ratios = time.time()
    try:
        ratios_data = _http_get_with_retry(url_ratios, params=params)
        status = "ok" if ratios_data and isinstance(ratios_data, list) and len(ratios_data) > 0 else "empty"
        record_api_call(
            provider="FMP",
            endpoint="ratios",
            status=status,
            latency_sec=time.time() - start_ratios,
            extra={"ticker": ticker}
        )
    except Exception as e:
        record_api_call(
            provider="FMP",
            endpoint="ratios",
            status="exception",
            latency_sec=time.time() - start_ratios,
            extra={"ticker": ticker, "error": str(e)[:200]}
        )
        ratios_data = None
    ratios = ratios_data[0] if ratios_data and isinstance(ratios_data, list) and len(ratios_data) > 0 else {}
    
    # Standardize output
    result = {
        "source": "fmp",
        "pe": metrics.get("peRatio"),
        "ps": metrics.get("priceToSalesRatio"),
        "pb": metrics.get("pbRatio"),
        "roe": ratios.get("returnOnEquity"),
        "margin": ratios.get("netProfitMargin"),
        "market_cap": metrics.get("marketCap"),
        "beta": metrics.get("beta"),
        "debt_equity": ratios.get("debtEquityRatio"),
        "rev_yoy": ratios.get("revenueGrowth"),
        "eps_yoy": metrics.get("earningsYield"),  # Approximate
        "timestamp": time.time()
    }
    
    _put_in_cache(cache_key, result)
    return result


# ============================================================================
# FINNHUB - SECONDARY SOURCE
# ============================================================================

def fetch_fundamentals_finnhub(ticker: str, provider_status: Dict | None = None) -> Optional[Dict]:
    """Fetch fundamentals from Finnhub."""
    # Preflight skip
    if provider_status is not None:
        s = provider_status.get("FINNHUB")
        if s and not s.get("ok", True):
            try:
                record_api_call(
                    provider="Finnhub",
                    endpoint="metric",
                    status="skipped_preflight",
                    latency_sec=0.0,
                    extra={"ticker": ticker, "reason": "disabled_by_preflight"},
                )
            except Exception:
                pass
            return None

    if not FINNHUB_API_KEY:
        return None
    
    cache_key = f"finnhub_fund_{ticker}"
    cached = _get_from_cache(cache_key)
    if cached:
        return cached
    
    _rate_limit("finnhub")
    
    # Fetch basic financials
    url = "https://finnhub.io/api/v1/stock/metric"
    params = {"symbol": ticker, "metric": "all", "token": FINNHUB_API_KEY}
    
    start = time.time()
    try:
        data = _http_get_with_retry(url, params=params)
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
    # Preflight skip
    if provider_status is not None:
        s = provider_status.get("TIINGO")
        if s and not s.get("ok", True):
            try:
                record_api_call(
                    provider="Tiingo",
                    endpoint="statements",
                    status="skipped_preflight",
                    latency_sec=0.0,
                    extra={"ticker": ticker, "reason": "disabled_by_preflight"},
                )
            except Exception:
                pass
            return None

    if not TIINGO_API_KEY:
        return None
    
    cache_key = f"tiingo_fund_{ticker}"
    cached = _get_from_cache(cache_key)
    if cached:
        return cached
    
    _rate_limit("tiingo")
    
    # Tiingo fundamentals endpoint
    url = f"https://api.tiingo.com/tiingo/fundamentals/{ticker}/statements"
    headers = {"Content-Type": "application/json", "Authorization": f"Token {TIINGO_API_KEY}"}
    
    start = time.time()
    try:
        data = _http_get_with_retry(url, headers=headers)
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
    # Preflight skip
    if provider_status is not None:
        s = provider_status.get("ALPHAVANTAGE")
        if s and not s.get("ok", True):
            try:
                record_api_call(
                    provider="AlphaVantage",
                    endpoint="overview",
                    status="skipped_preflight",
                    latency_sec=0.0,
                    extra={"ticker": ticker, "reason": "disabled_by_preflight"},
                )
            except Exception:
                pass
            return None

    if not ALPHA_VANTAGE_API_KEY:
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
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    start = time.time()
    try:
        data = _http_get_with_retry(url, params=params)
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
    result = {
        "source": "alpha",
        "pe": float(data["PERatio"]) if data.get("PERatio") and data["PERatio"] != "None" else None,
        "ps": float(data["PriceToSalesRatioTTM"]) if data.get("PriceToSalesRatioTTM") and data["PriceToSalesRatioTTM"] != "None" else None,
        "pb": float(data["PriceToBookRatio"]) if data.get("PriceToBookRatio") and data["PriceToBookRatio"] != "None" else None,
        "roe": float(data["ReturnOnEquityTTM"]) if data.get("ReturnOnEquityTTM") and data["ReturnOnEquityTTM"] != "None" else None,
        "margin": float(data["ProfitMargin"]) if data.get("ProfitMargin") and data["ProfitMargin"] != "None" else None,
        "market_cap": float(data["MarketCapitalization"]) if data.get("MarketCapitalization") else None,
        "beta": float(data["Beta"]) if data.get("Beta") and data["Beta"] != "None" else None,
        "debt_equity": float(data["DebtToEquity"]) if data.get("DebtToEquity") and data["DebtToEquity"] != "None" else None,
        "rev_yoy": float(data["QuarterlyRevenueGrowthYOY"]) if data.get("QuarterlyRevenueGrowthYOY") and data["QuarterlyRevenueGrowthYOY"] != "None" else None,
        "eps_yoy": float(data["QuarterlyEarningsGrowthYOY"]) if data.get("QuarterlyEarningsGrowthYOY") and data["QuarterlyEarningsGrowthYOY"] != "None" else None,
        "timestamp": time.time()
    }
    _put_in_cache(cache_key, result)
    return result


# ============================================================================
# MULTI-SOURCE AGGREGATION
# ============================================================================

def aggregate_fundamentals(ticker: str, prefer_source: str = "fmp", provider_status: Dict | None = None) -> Dict:
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
    # Fetch from all sources (in parallel conceptually, but sequentially for rate limits)
    sources_data = {}
    
    # Priority order
    fetch_funcs = [
        ("fmp", fetch_fundamentals_fmp),
        ("finnhub", fetch_fundamentals_finnhub),
        ("tiingo", fetch_fundamentals_tiingo),
        ("alpha", fetch_fundamentals_alpha),
    ]
    
    for source_name, fetch_func in fetch_funcs:
        # Respect preflight: skip disabled providers
        if provider_status is not None:
            key_map = {
                "fmp": "FMP",
                "finnhub": "FINNHUB",
                "tiingo": "TIINGO",
                "alpha": "ALPHAVANTAGE",
            }
            s = provider_status.get(key_map.get(source_name, source_name))
            if s and not s.get("ok", True):
                try:
                    record_api_call(
                        provider=key_map.get(source_name, source_name),
                        endpoint="fundamentals",
                        status="skipped_preflight",
                        latency_sec=0.0,
                        extra={"ticker": ticker, "reason": "disabled_by_preflight"},
                    )
                except Exception:
                    pass
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
        return {
            "ticker": ticker,
            "sources_used": [],
            "coverage": {},
            "disagreement_score": 1.0,
            # Explicit fundamental coverage and counts
            "Fundamental_Coverage_Pct": 0.0,
            "Fundamental_Sources_Count": 0,
            # Neutral default score used by downstream when fundamentals missing
            "Fundamental_S": 50.0,
            # Per-source flags for downstream logic
            "Fund_from_FMP": False,
            "Fund_from_Finnhub": False,
            "Fund_from_Tiingo": False,
            "Fund_from_Alpha": False,
            "timestamp": time.time(),
        }
    
    # Aggregate fields
    fields = ["pe", "ps", "pb", "roe", "margin", "rev_yoy", "eps_yoy", "debt_equity", "market_cap", "beta"]
    aggregated = {"ticker": ticker}
    coverage = {}
    disagreements = []
    
    for field in fields:
        values = []
        contributing_sources = []
        
        for source_name, data in sources_data.items():
            val = data.get(field)
            if val is not None and np.isfinite(val):
                values.append(float(val))
                contributing_sources.append(source_name)
        
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

    # Provide neutral Fundamental_S if not computed elsewhere
    if "Fundamental_S" not in aggregated:
        aggregated["Fundamental_S"] = 50.0
    
    # Cache the merged result
    cache_key = f"merged_fund_{ticker}"
    _put_in_cache(cache_key, aggregated)
    
    logger.info(f"Aggregated fundamentals for {ticker} from {len(sources_data)} sources")
    return aggregated


# ============================================================================
# MULTI-SOURCE PRICE VERIFICATION (EXISTING + ENHANCED)
# ============================================================================

def fetch_price_multi_source(ticker: str, provider_status: Dict | None = None) -> Dict[str, Optional[float]]:
    """
    Fetch current price from multiple sources for verification.
    
    Preserves existing sources: Alpha, Finnhub, Polygon, Tiingo
    Adds FMP if available.
    
    Returns:
        Dict with keys: fmp, finnhub, tiingo, alpha, polygon (each Optional[float])
    """
    prices = {}
    
    # FMP price
    if provider_status is not None:
        s = provider_status.get("FMP")
        if s and not s.get("ok", True):
            try:
                record_api_call(
                    provider="FMP",
                    endpoint="quote",
                    status="skipped_preflight",
                    latency_sec=0.0,
                    extra={"ticker": ticker, "reason": "disabled_by_preflight"},
                )
            except Exception:
                pass
        else:
            pass
    if FMP_API_KEY and (provider_status is None or provider_status.get("FMP", {"ok": True}).get("ok", True)):
        try:
            _rate_limit("fmp")
            url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}"
            params = {"apikey": FMP_API_KEY}
            data = _http_get_with_retry(url, params=params)
            if data and isinstance(data, list) and len(data) > 0:
                prices["fmp"] = data[0].get("price")
        except Exception as e:
            logger.debug(f"FMP price fetch failed: {e}")
            prices["fmp"] = None
    
    # Finnhub price
    if provider_status is not None:
        s = provider_status.get("FINNHUB")
        if s and not s.get("ok", True):
            try:
                record_api_call(
                    provider="Finnhub",
                    endpoint="quote",
                    status="skipped_preflight",
                    latency_sec=0.0,
                    extra={"ticker": ticker, "reason": "disabled_by_preflight"},
                )
            except Exception:
                pass
        else:
            pass
    if FINNHUB_API_KEY and (provider_status is None or provider_status.get("FINNHUB", {"ok": True}).get("ok", True)):
        try:
            _rate_limit("finnhub")
            url = "https://finnhub.io/api/v1/quote"
            params = {"symbol": ticker, "token": FINNHUB_API_KEY}
            data = _http_get_with_retry(url, params=params)
            if data:
                prices["finnhub"] = data.get("c")  # Current price
        except Exception as e:
            logger.debug(f"Finnhub price fetch failed: {e}")
            prices["finnhub"] = None
    
    # Tiingo price (existing)
    if provider_status is not None:
        s = provider_status.get("TIINGO")
        if s and not s.get("ok", True):
            try:
                record_api_call(
                    provider="Tiingo",
                    endpoint="daily/prices",
                    status="skipped_preflight",
                    latency_sec=0.0,
                    extra={"ticker": ticker, "reason": "disabled_by_preflight"},
                )
            except Exception:
                pass
        else:
            pass
    if TIINGO_API_KEY and (provider_status is None or provider_status.get("TIINGO", {"ok": True}).get("ok", True)):
        try:
            _rate_limit("tiingo")
            url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
            headers = {"Content-Type": "application/json", "Authorization": f"Token {TIINGO_API_KEY}"}
            data = _http_get_with_retry(url, headers=headers)
            if data and isinstance(data, list) and len(data) > 0:
                prices["tiingo"] = data[0].get("close")
        except Exception as e:
            logger.debug(f"Tiingo price fetch failed: {e}")
            prices["tiingo"] = None
    
    # Alpha Vantage price (existing)
    if provider_status is not None:
        s = provider_status.get("ALPHAVANTAGE")
        if s and not s.get("ok", True):
            try:
                record_api_call(
                    provider="AlphaVantage",
                    endpoint="global_quote",
                    status="skipped_preflight",
                    latency_sec=0.0,
                    extra={"ticker": ticker, "reason": "disabled_by_preflight"},
                )
            except Exception:
                pass
        else:
            pass
    if ALPHA_VANTAGE_API_KEY and (provider_status is None or provider_status.get("ALPHAVANTAGE", {"ok": True}).get("ok", True)):
        try:
            _rate_limit("alpha")
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": ticker,
                "apikey": ALPHA_VANTAGE_API_KEY
            }
            data = _http_get_with_retry(url, params=params)
            if data and "Global Quote" in data:
                price_str = data["Global Quote"].get("05. price")
                prices["alpha"] = float(price_str) if price_str else None
        except Exception as e:
            logger.debug(f"Alpha price fetch failed: {e}")
            prices["alpha"] = None
    
    # Polygon price (existing)
    if provider_status is not None:
        s = provider_status.get("POLYGON")
        if s and not s.get("ok", True):
            try:
                record_api_call(
                    provider="Polygon",
                    endpoint="prev",
                    status="skipped_preflight",
                    latency_sec=0.0,
                    extra={"ticker": ticker, "reason": "disabled_by_preflight"},
                )
            except Exception:
                pass
        else:
            pass
    if POLYGON_API_KEY and (provider_status is None or provider_status.get("POLYGON", {"ok": True}).get("ok", True)):
        try:
            _rate_limit("polygon")
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev"
            params = {"apiKey": POLYGON_API_KEY}
            data = _http_get_with_retry(url, params=params)
            if data and "results" in data and len(data["results"]) > 0:
                prices["polygon"] = data["results"][0].get("c")
        except Exception as e:
            logger.debug(f"Polygon price fetch failed: {e}")
            prices["polygon"] = None
    
    return prices


def aggregate_price(prices: Dict[str, Optional[float]]) -> Tuple[float, float, int]:
    """
    Aggregate prices from multiple sources.
    
    Returns:
        (mean_price, std_price, num_sources)
    """
    valid_prices = [p for p in prices.values() if p is not None and np.isfinite(p) and p > 0]
    
    if not valid_prices:
        return np.nan, np.nan, 0
    
    mean_price = float(np.mean(valid_prices))
    std_price = float(np.std(valid_prices)) if len(valid_prices) > 1 else 0.0
    
    return mean_price, std_price, len(valid_prices)


# ============================================================================
# PUBLIC API
# ============================================================================

def fetch_multi_source_data(ticker: str, provider_status: Dict | None = None) -> Dict:
    """
    Main entry point: fetch fundamentals and price from all sources.
    
    Returns comprehensive dict with:
    - All fundamental fields
    - sources_used, coverage, disagreement_score
    - price_mean, price_std, price_sources
    - Individual source prices for verification
    """
    # Get fundamentals
    fundamentals = aggregate_fundamentals(ticker, provider_status=provider_status)
    
    # Get prices
    prices = fetch_price_multi_source(ticker, provider_status=provider_status)
    price_mean, price_std, price_count = aggregate_price(prices)
    
    # Merge everything
    result = {**fundamentals}
    result["price_mean"] = price_mean
    result["price_std"] = price_std
    result["price_sources"] = price_count
    result["prices_by_source"] = prices
    
    return result


def clear_cache() -> None:
    """Clear all cached data."""
    global _CACHE
    _CACHE = {}
    logger.info("Cache cleared")


def get_cache_stats() -> Dict:
    """Get cache statistics."""
    return {
        "total_entries": len(_CACHE),
        "sources": {
            source: sum(1 for k in _CACHE if k.startswith(source))
            for source in ["fmp", "finnhub", "tiingo", "alpha", "merged"]
        }
    }


def get_index_series(
    symbol: str,
    start_date: str,
    end_date: str,
    provider_status: Optional[Dict[str, bool]] = None
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
    
    # Check cache
    cache_key = f"index_series_{symbol}_{start_date}_{end_date}"
    if cache_key in _CACHE:
        cached = _CACHE[cache_key]
        if time.time() - cached["timestamp"] < CACHE_TTL_SECONDS:
            logger.debug(f"✓ Cache hit for index series {symbol}")
            return cached["data"]
    
    provider_status = provider_status or {}
    df_result = None
    
    # Try FMP first (most reliable for indices)
    if provider_status.get("fmp", True) and FMP_API_KEY:
        try:
            _rate_limit("fmp")
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{fmp_symbol}"
            params = {
                "apikey": FMP_API_KEY,
                "from": start_date,
                "to": end_date
            }
            
            record_api_call("fmp", f"index_series_{symbol}")
            data = _http_get_with_retry(url, params=params, timeout=15)
            
            if data and "historical" in data and data["historical"]:
                df = pd.DataFrame(data["historical"])
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
                
        except Exception as e:
            logger.warning(f"FMP index series failed for {symbol}: {e}")
    
    # Fallback to Tiingo
    if df_result is None and provider_status.get("tiingo", True) and TIINGO_API_KEY:
        try:
            _rate_limit("tiingo")
            # Tiingo uses different URL structure for indices
            tiingo_symbol = symbol.replace('^', '$')  # $VIX for Tiingo
            url = f"https://api.tiingo.com/tiingo/daily/{tiingo_symbol}/prices"
            headers = {"Content-Type": "application/json"}
            params = {
                "token": TIINGO_API_KEY,
                "startDate": start_date,
                "endDate": end_date
            }
            
            record_api_call("tiingo", f"index_series_{symbol}")
            data = _http_get_with_retry(url, params=params, headers=headers, timeout=15)
            
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
            
            record_api_call("alpha", f"index_series_{symbol}")
            data = _http_get_with_retry(url, params=params, timeout=20)
            
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
                    
        except Exception as e:
            logger.warning(f"Alpha Vantage index series failed for {symbol}: {e}")
    
    # Cache result
    if df_result is not None:
        _CACHE[cache_key] = {
            "data": df_result,
            "timestamp": time.time()
        }
    else:
        logger.error(f"❌ All sources failed for index series {symbol}")
    
    return df_result


def fetch_fundamentals_batch(tickers: List[str], provider_status: Dict | None = None) -> pd.DataFrame:
    """
    Batch fetch fundamentals for multiple tickers using aggregate_fundamentals.
    Returns a DataFrame indexed by Ticker.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if not tickers:
        return pd.DataFrame()
        
    rows = []
    # Limit workers to avoid hitting rate limits too hard
    with ThreadPoolExecutor(max_workers=min(len(tickers), 10)) as executor:
        future_to_ticker = {
            executor.submit(aggregate_fundamentals, t, "fmp", provider_status): t
            for t in tickers
        }
        for future in as_completed(future_to_ticker):
            try:
                res = future.result()
                rows.append(res)
            except Exception as e:
                logger.error(f"Batch fetch failed for {future_to_ticker[future]}: {e}")
                
    if not rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(rows)
    if "ticker" in df.columns:
        df = df.rename(columns={"ticker": "Ticker"})
        df = df.set_index("Ticker")
    
    return df
