"""
Unified Price Verification Module

This module provides a consistent interface for price verification
across multiple providers. It wraps core.data_sources_v2 functions
for use in the main application.

Usage:
    from core.price_verify import fetch_prices_for_ticker, PRICE_PROVIDERS
    
    prices = fetch_prices_for_ticker("AAPL", provider_status)
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable

from core.data_sources_v2 import fetch_price_multi_source
from core.config import get_secret

# Provider emoji badges
PRICE_PROVIDER_BADGES = {
    "yahoo": "🟡Yahoo",
    "alpha": "🟣Alpha",
    "finnhub": "🔵Finnhub",
    "polygon": "🟢Polygon",
    "tiingo": "🟠Tiingo",
    "marketstack": "🧩Marketstack",
    "nasdaq": "🏛Nasdaq",
    "eodhd": "📘EODHD",
    "fmp": "💰FMP",
}

# Mapping from v2 provider keys to display names
PROVIDER_KEY_TO_DISPLAY = {
    "fmp": "FMP",
    "finnhub": "Finnhub",
    "tiingo": "Tiingo",
    "alpha": "Alpha",
    "polygon": "Polygon",
    "eodhd": "EODHD",
    "marketstack": "Marketstack",
    "nasdaq": "NasdaqDL",
}


def _mark_provider_usage(provider: str, category: str, session_state_getter: Optional[Callable] = None) -> None:
    """
    Mark provider usage in session state (for Streamlit UI tracking).
    
    Args:
        provider: Provider name
        category: Usage category ('price', 'fundamentals', etc.)
        session_state_getter: Optional callable that returns session_state dict
    """
    try:
        # Try Streamlit session state
        try:
            import streamlit as st
            usage = st.session_state.setdefault("provider_usage", {})
            cats = usage.setdefault(provider, set())
            cats.add(category)
            usage[provider] = cats
        except Exception:
            # Not in Streamlit context - use provided getter or skip
            if session_state_getter:
                state = session_state_getter()
                usage = state.setdefault("provider_usage", {})
                cats = usage.setdefault(provider, set())
                cats.add(category)
    except Exception:
        pass  # Best effort - don't fail on tracking


def fetch_prices_for_ticker(
    ticker: str,
    yahoo_price: Optional[float] = None,
    provider_status: Optional[Dict[str, bool]] = None,
    telemetry: Optional[Any] = None,
    track_usage: bool = True,
) -> Tuple[Dict[str, Optional[float]], List[str]]:
    """
    Fetch prices from multiple providers for verification.
    
    Args:
        ticker: Stock ticker symbol
        yahoo_price: Price from Yahoo (if already fetched)
        provider_status: Dict of provider availability
        telemetry: Optional telemetry instance
        track_usage: Whether to track provider usage in session state
        
    Returns:
        Tuple of (prices_dict, source_badges_list)
    """
    vals: Dict[str, Optional[float]] = {}
    srcs: List[str] = []
    
    # Include Yahoo price if provided
    if yahoo_price is not None and np.isfinite(yahoo_price):
        vals["Yahoo"] = float(yahoo_price)
        srcs.append(PRICE_PROVIDER_BADGES["yahoo"])
        if track_usage:
            _mark_provider_usage("Yahoo", "price")
    
    # Fetch from all available providers using v2 infrastructure
    prices = fetch_price_multi_source(
        ticker, 
        provider_status=provider_status, 
        telemetry=telemetry
    )
    
    for src_key, display_name in PROVIDER_KEY_TO_DISPLAY.items():
        price = prices.get(src_key)
        if price is not None and np.isfinite(price):
            vals[display_name] = float(price)
            badge = PRICE_PROVIDER_BADGES.get(src_key, f"📍{display_name}")
            srcs.append(badge)
            if track_usage:
                _mark_provider_usage(display_name, "price")
    
    return vals, srcs


def compute_price_stats(prices: Dict[str, Optional[float]]) -> Tuple[float, float, int]:
    """
    Compute statistics from multiple price sources.
    
    Args:
        prices: Dict of provider -> price
        
    Returns:
        Tuple of (mean, std, count)
    """
    valid_prices = [p for p in prices.values() if p is not None and np.isfinite(p)]
    
    if not valid_prices:
        return np.nan, np.nan, 0
    
    mean_price = float(np.mean(valid_prices))
    std_price = float(np.std(valid_prices)) if len(valid_prices) > 1 else np.nan
    
    return mean_price, std_price, len(valid_prices)


def format_source_badges(sources: List[str], separator: str = " - ") -> str:
    """Format source badges into a display string."""
    return separator.join(sources) if sources else "🟡Yahoo"


def fetch_external_prices_for_verification(
    ticker: str,
    yahoo_price: float,
    provider_status: Optional[Dict[str, Any]] = None,
    telemetry: Optional[Any] = None,
) -> Tuple[str, Dict[str, Optional[float]], List[str]]:
    """
    Drop-in replacement for stock_scout._fetch_external_for.
    
    Args:
        ticker: Stock ticker symbol
        yahoo_price: Price from Yahoo Finance
        provider_status: Dict of provider availability (from preflight)
        telemetry: Optional telemetry instance
        
    Returns:
        Tuple of (ticker, prices_dict, source_badges_list)
        Compatible with existing stock_scout.py usage.
    """
    vals, srcs = fetch_prices_for_ticker(
        ticker=ticker,
        yahoo_price=yahoo_price,
        provider_status=provider_status,
        telemetry=telemetry,
        track_usage=True,
    )
    return ticker, vals, srcs


# Column names for price DataFrame (matching stock_scout.py expectations)
PRICE_COLUMNS = [
    "Price_Alpha",
    "Price_Finnhub", 
    "Price_Polygon",
    "Price_Tiingo",
    "Price_Marketstack",
    "Price_NasdaqDL",
    "Price_EODHD",
    "Price_FMP",
    "Price_Mean",
    "Price_STD",
    "Source_List",
]


def prices_dict_to_row_values(
    vals: Dict[str, Optional[float]],
    srcs: List[str],
) -> List[Any]:
    """
    Convert prices dict to row values for DataFrame assignment.
    
    Args:
        vals: Dict of provider -> price
        srcs: List of source badges
        
    Returns:
        List matching PRICE_COLUMNS order
    """
    mean_price, std_price, _ = compute_price_stats(vals)
    source_list = format_source_badges(srcs)
    
    return [
        vals.get("Alpha", np.nan),
        vals.get("Finnhub", np.nan),
        vals.get("Polygon", np.nan),
        vals.get("Tiingo", np.nan),
        vals.get("Marketstack", np.nan),
        vals.get("NasdaqDL", np.nan),
        vals.get("EODHD", np.nan),
        vals.get("FMP", np.nan),
        mean_price,
        std_price,
        source_list,
    ]
