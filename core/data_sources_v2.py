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

def fetch_fundamentals_fmp(ticker: str) -> Optional[Dict]:
    """
    Fetch fundamentals from FMP (primary source).
    
    Returns standardized dict with keys:
    - pe, ps, pb, roe, margin, rev_yoy, eps_yoy, debt_equity
    - market_cap, beta, etc.
    """
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
    
    data = _http_get_with_retry(url, params=params)
    if not data or not isinstance(data, list) or len(data) == 0:
        return None
    
    metrics = data[0]
    
    # Fetch ratios
    url_ratios = f"https://financialmodelingprep.com/api/v3/ratios/{ticker}"
    ratios_data = _http_get_with_retry(url_ratios, params=params)
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

def fetch_fundamentals_finnhub(ticker: str) -> Optional[Dict]:
    """Fetch fundamentals from Finnhub."""
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
    
    data = _http_get_with_retry(url, params=params)
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

def fetch_fundamentals_tiingo(ticker: str) -> Optional[Dict]:
    """Fetch fundamentals from Tiingo (existing source - preserved)."""
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
    
    data = _http_get_with_retry(url, headers=headers)
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

def fetch_fundamentals_alpha(ticker: str) -> Optional[Dict]:
    """Fetch fundamentals from Alpha Vantage (existing source - preserved)."""
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
    
    data = _http_get_with_retry(url, params=params)
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

def aggregate_fundamentals(ticker: str, prefer_source: str = "fmp") -> Dict:
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
        try:
            result = fetch_func(ticker)
            if result:
                sources_data[source_name] = result
                logger.debug(f"✓ {source_name} data fetched for {ticker}")
        except Exception as e:
            logger.warning(f"Failed to fetch from {source_name} for {ticker}: {e}")
    
    if not sources_data:
        logger.warning(f"No fundamental data available for {ticker}")
        return {
            "ticker": ticker,
            "sources_used": [],
            "coverage": {},
            "disagreement_score": 1.0,
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
    
    # Cache the merged result
    cache_key = f"merged_fund_{ticker}"
    _put_in_cache(cache_key, aggregated)
    
    logger.info(f"Aggregated fundamentals for {ticker} from {len(sources_data)} sources")
    return aggregated


# ============================================================================
# MULTI-SOURCE PRICE VERIFICATION (EXISTING + ENHANCED)
# ============================================================================

def fetch_price_multi_source(ticker: str) -> Dict[str, Optional[float]]:
    """
    Fetch current price from multiple sources for verification.
    
    Preserves existing sources: Alpha, Finnhub, Polygon, Tiingo
    Adds FMP if available.
    
    Returns:
        Dict with keys: fmp, finnhub, tiingo, alpha, polygon (each Optional[float])
    """
    prices = {}
    
    # FMP price
    if FMP_API_KEY:
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
    if FINNHUB_API_KEY:
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
    if TIINGO_API_KEY:
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
    if ALPHA_VANTAGE_API_KEY:
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
    if POLYGON_API_KEY:
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

def fetch_multi_source_data(ticker: str) -> Dict:
    """
    Main entry point: fetch fundamentals and price from all sources.
    
    Returns comprehensive dict with:
    - All fundamental fields
    - sources_used, coverage, disagreement_score
    - price_mean, price_std, price_sources
    - Individual source prices for verification
    """
    # Get fundamentals
    fundamentals = aggregate_fundamentals(ticker)
    
    # Get prices
    prices = fetch_price_multi_source(ticker)
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
