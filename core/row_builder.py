"""
Canonical row builder: maps multi-source data into unified DataFrame schema.

This module provides the single source of truth for normalizing multi-source
fundamentals, prices, and metadata into a consistent row schema used across
all pipelines (live app, unified_backtest, unified_time_test).

Design principle: One row builder, one schema, zero divergence.
"""
from __future__ import annotations
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def build_row_from_multi_source(
    ticker: str,
    multi_source_payload: Dict[str, Any],
    market_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a canonical row from multi-source payload.
    
    This is the SINGLE entry point for converting multi-source aggregation outputs
    into the unified row schema used by all pipelines.
    
    Args:
        ticker: Stock symbol (str)
        multi_source_payload: Dict from fetch_multi_source_data() containing:
            - Fundamental fields: pe, ps, pb, roe, roic, gm, rev_yoy, eps_yoy, de, beta, market_cap
            - Metadata: Fundamental_Coverage_Pct, Fundamental_Sources_Count
            - Per-source flags: Fund_from_FMP, Fund_from_Finnhub, Fund_from_Tiingo, Fund_from_Alpha
            - Price aggregation: price_mean, price_std, price_sources (count)
            - Individual prices: prices_by_source (dict)
            - Aggregation metadata: sources_used, coverage, disagreement_score
        
        market_data: Optional dict with current price and other real-time signals:
            - Close or current_price (float): Current market price
            - timestamp: When price was fetched
    
    Returns:
        Dict with canonical row schema including:
        - Ticker info: ticker
        - Price fields: Close, Price_Mean, Price_STD, Price_Sources_Count
        - Fundamental raw: PE_f, PS_f, PBRatio, ROE_f, ROIC_f, GM_f, RevG_f, EPSG_f, DE_f, Beta
        - Fundamental metadata: Fundamental_Coverage_Pct, Fundamental_Sources_Count, per-source flags
        - Fundamental scores: Fundamental_S, Quality_Score_F, Growth_Score_F, Valuation_Score_F, Stability_Score_F
        - Technical placeholders (will be filled by build_technical_indicators): RSI, ATR_Pct, etc.
        - Reliability metadata: (used by calculate_reliability_v2)
    
    Raises:
        No exceptions. All missing/invalid fields receive sensible defaults (None or neutral value).
    
    Examples:
        >>> from core.data_sources_v2 import fetch_multi_source_data
        >>> payload = fetch_multi_source_data("AAPL")
        >>> row = build_row_from_multi_source("AAPL", payload, {"Close": 150.5})
        >>> row["ticker"]
        'AAPL'
        >>> row["Fundamental_Coverage_Pct"]
        85.0
        >>> 0 <= row["Quality_Score_F"] <= 100
        True
    """
    from core.scoring.fundamental import compute_fundamental_score_with_breakdown
    
    row = {}
    
    # === TICKER & TIMESTAMP ===
    row["ticker"] = str(ticker).upper()
    row["timestamp"] = multi_source_payload.get("timestamp", pd.Timestamp.now())
    
    # === PRICE FIELDS ===
    price_mean = multi_source_payload.get("price_mean")
    price_std = multi_source_payload.get("price_std", 0.0)
    price_sources = multi_source_payload.get("price_sources", 0)
    
    # Prefer explicit Close from market_data, fallback to price_mean
    if market_data and market_data.get("Close"):
        row["Close"] = float(market_data["Close"])
    elif price_mean and np.isfinite(price_mean) and price_mean > 0:
        row["Close"] = float(price_mean)
    else:
        row["Close"] = np.nan
    
    # Multi-source price aggregation
    row["Price_Mean"] = float(price_mean) if (price_mean and np.isfinite(price_mean)) else np.nan
    row["Price_STD"] = float(price_std) if (price_std and np.isfinite(price_std)) else np.nan
    row["Price_Sources_Count"] = int(price_sources) if price_sources else 0
    
    # Store individual provider prices for debugging/verification
    prices_by_source = multi_source_payload.get("prices_by_source", {})
    for source, price in prices_by_source.items():
        col_name = f"Price_{source.replace('_', '').title()}"
        row[col_name] = float(price) if (price and np.isfinite(price)) else np.nan
    
    # === FUNDAMENTAL RAW FIELDS ===
    # Extract and normalize each fundamental metric
    roe = multi_source_payload.get("roe")
    row["ROE_f"] = float(roe) if (roe and np.isfinite(roe)) else np.nan
    
    roic = multi_source_payload.get("roic")
    row["ROIC_f"] = float(roic) if (roic and np.isfinite(roic)) else np.nan
    
    gm = multi_source_payload.get("gm") or multi_source_payload.get("margin")
    row["GM_f"] = float(gm) if (gm and np.isfinite(gm)) else np.nan
    
    rev_yoy = multi_source_payload.get("rev_yoy") or multi_source_payload.get("rev_g_yoy")
    row["RevG_f"] = float(rev_yoy) if (rev_yoy and np.isfinite(rev_yoy)) else np.nan
    
    eps_yoy = multi_source_payload.get("eps_yoy") or multi_source_payload.get("eps_g_yoy")
    row["EPSG_f"] = float(eps_yoy) if (eps_yoy and np.isfinite(eps_yoy)) else np.nan
    
    pe = multi_source_payload.get("pe")
    row["PE_f"] = float(pe) if (pe and np.isfinite(pe)) else np.nan
    
    ps = multi_source_payload.get("ps")
    row["PS_f"] = float(ps) if (ps and np.isfinite(ps)) else np.nan
    
    pb = multi_source_payload.get("pb")
    row["PBRatio"] = float(pb) if (pb and np.isfinite(pb)) else np.nan
    
    de = multi_source_payload.get("de") or multi_source_payload.get("debt_equity")
    row["DE_f"] = float(de) if (de and np.isfinite(de)) else np.nan
    
    beta = multi_source_payload.get("beta")
    row["Beta"] = float(beta) if (beta and np.isfinite(beta)) else np.nan
    
    market_cap = multi_source_payload.get("market_cap")
    row["MarketCap"] = float(market_cap) if (market_cap and np.isfinite(market_cap)) else np.nan
    
    # === FUNDAMENTAL METADATA ===
    coverage_pct = multi_source_payload.get("Fundamental_Coverage_Pct")
    row["Fundamental_Coverage_Pct"] = float(coverage_pct) if (coverage_pct and np.isfinite(coverage_pct)) else 0.0
    
    sources_count = multi_source_payload.get("Fundamental_Sources_Count")
    row["Fundamental_Sources_Count"] = int(sources_count) if sources_count else 0
    
    # Per-source flags
    row["Fund_from_FMP"] = bool(multi_source_payload.get("Fund_from_FMP", False))
    row["Fund_from_Finnhub"] = bool(multi_source_payload.get("Fund_from_Finnhub", False))
    row["Fund_from_Tiingo"] = bool(multi_source_payload.get("Fund_from_Tiingo", False))
    row["Fund_from_Alpha"] = bool(multi_source_payload.get("Fund_from_Alpha", False))
    
    # Aggregation metadata
    row["Fund_Disagreement_Score"] = float(multi_source_payload.get("disagreement_score", 0.0))
    
    # === COMPUTE FUNDAMENTAL BREAKDOWN SCORES ===
    # Use the robust fundamental scoring with multi-source metadata
    fund_input = {
        "roe": row.get("ROE_f"),
        "roic": row.get("ROIC_f"),
        "gm": row.get("GM_f"),
        "rev_yoy": row.get("RevG_f"),
        "eps_yoy": row.get("EPSG_f"),
        "pe": row.get("PE_f"),
        "ps": row.get("PS_f"),
        "de": row.get("DE_f"),
        "beta": row.get("Beta"),
        "market_cap": row.get("MarketCap"),
        "Fundamental_Coverage_Pct": row.get("Fundamental_Coverage_Pct"),
        "Fundamental_Sources_Count": row.get("Fundamental_Sources_Count"),
    }
    
    try:
        fundamental_score = compute_fundamental_score_with_breakdown(
            fund_input,
            coverage_pct=row.get("Fundamental_Coverage_Pct", 0.0) / 100.0 if row.get("Fundamental_Coverage_Pct") else 0.0
        )
        
        row["Fundamental_S"] = float(fundamental_score.total)
        row["Quality_Score_F"] = float(fundamental_score.breakdown.quality_score)
        row["Growth_Score_F"] = float(fundamental_score.breakdown.growth_score)
        row["Valuation_Score_F"] = float(fundamental_score.breakdown.valuation_score)
        row["Leverage_Score_F"] = float(fundamental_score.breakdown.leverage_score)
        row["Stability_Score_F"] = float(fundamental_score.breakdown.stability_score)
        
        # Store labels for reference
        row["Quality_Label"] = fundamental_score.breakdown.quality_label
        row["Growth_Label"] = fundamental_score.breakdown.growth_label
        row["Valuation_Label"] = fundamental_score.breakdown.valuation_label
        row["Leverage_Label"] = fundamental_score.breakdown.leverage_label
        row["Stability_Label"] = fundamental_score.breakdown.stability_label
    except Exception as e:
        logger.warning(f"Failed to compute fundamental scores for {ticker}: {e}")
        row["Fundamental_S"] = 50.0
        row["Quality_Score_F"] = 50.0
        row["Growth_Score_F"] = 50.0
        row["Valuation_Score_F"] = 50.0
        row["Leverage_Score_F"] = 50.0
        row["Stability_Score_F"] = 50.0
        row["Quality_Label"] = "Unknown"
        row["Growth_Label"] = "Unknown"
        row["Valuation_Label"] = "Unknown"
        row["Leverage_Label"] = "Unknown"
        row["Stability_Label"] = "Unknown"
    
    # === TECHNICAL PLACEHOLDERS ===
    # These will be populated by build_technical_indicators() or set to NaN as placeholders
    technical_fields = [
        "RSI", "ATR", "ATR_Pct", "MACD", "MACD_Signal", "MACD_Hist",
        "ADX", "SMA20", "SMA50", "SMA200", "EMA12", "EMA26",
        "Upper_Band", "Lower_Band", "BB_Position",
        "Mom1M", "Mom3M", "Mom6M", "Price_52W_High", "Price_52W_Low",
        "Near_52W_High", "Volatility", "Volume_MA", "Volume_Surge"
    ]
    for field in technical_fields:
        if field not in row:
            row[field] = np.nan
    
    # === RELIABILITY & SCORING PREREQUISITES ===
    # Ensure fields needed by calculate_reliability_v2 are present
    row["Reliability_v2"] = np.nan  # Will be populated by calculate_reliability_v2
    row["Technical_S"] = np.nan  # Will be populated by compute_technical_score
    row["RR_Score"] = np.nan  # Will be populated by RR evaluation
    row["ML_Probability"] = np.nan  # Optional, for ML integration
    row["Overall_Score"] = np.nan  # Will be populated by compute_overall_score
    
    # === PORTFOLIO ALLOCATION FIELDS ===
    # Placeholders for downstream allocation logic
    row["Risk_Level"] = "speculative"  # Default, may be updated by filters
    row["Quality_Level"] = "medium"
    row["סכום קנייה ($)"] = np.nan  # Hebrew column name as used in live app
    row["Unit_Price"] = row.get("Close")  # Alias for price
    
    # Ensure all float fields are properly typed
    for key, val in row.items():
        if isinstance(val, float):
            if np.isfinite(val):
                row[key] = float(val)  # Don't clip market cap or other large values
            else:
                row[key] = np.nan
        elif isinstance(val, bool):
            row[key] = bool(val)
    
    return row


def build_rows_from_universe(
    universe_tickers: list[str],
    fetch_multi_source_func,
    market_data_dict: Optional[Dict[str, Dict]] = None,
) -> list[Dict[str, Any]]:
    """
    Build normalized rows for a universe of tickers.
    
    This helper orchestrates multi-source fetching and row building for batch operations.
    
    Args:
        universe_tickers: List of ticker symbols
        fetch_multi_source_func: Callable that fetches multi-source data for a ticker
        market_data_dict: Optional dict mapping ticker → market data (current price, etc.)
    
    Returns:
        List of normalized row dicts, one per ticker
    """
    rows = []
    market_data_dict = market_data_dict or {}
    
    for ticker in universe_tickers:
        try:
            # Fetch multi-source data for this ticker
            multi_source_payload = fetch_multi_source_func(ticker)
            
            # Get optional market data
            market_data = market_data_dict.get(ticker)
            
            # Build normalized row
            row = build_row_from_multi_source(ticker, multi_source_payload, market_data)
            rows.append(row)
        except Exception as e:
            logger.error(f"Failed to build row for {ticker}: {e}")
            # Optionally: add a minimal row with just ticker to maintain alignment
            # or skip entirely depending on pipeline requirements
    
    return rows


def rows_to_dataframe(rows: list[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert list of normalized rows to DataFrame.
    
    Handles None/missing values and ensures consistent column ordering.
    
    Args:
        rows: List of row dicts from build_row_from_multi_source
    
    Returns:
        DataFrame with all rows, consistent schema
    """
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Ensure critical columns exist even if empty
    critical_columns = [
        "ticker", "Close", "Fundamental_S", "Quality_Score_F", "Growth_Score_F",
        "Valuation_Score_F", "Leverage_Score_F", "Stability_Score_F",
        "Price_Mean", "Price_STD", "Price_Sources_Count",
        "Fundamental_Coverage_Pct", "Fundamental_Sources_Count"
    ]
    
    for col in critical_columns:
        if col not in df.columns:
            df[col] = np.nan
    
    return df
