"""
Feature Registry V4 - Enhanced ML Features (50+ Features)

This module defines ALL features used in ML models V4 and beyond.
Includes previously unused computed features, sentiment, and institutional signals.

Key improvements over V3:
- Added earnings proximity features (computed but unused in V3)
- Added news sentiment features (from Finnhub)
- Added institutional flow signals
- Added squeeze/breakout indicators
- Added cross-asset context (VIX correlation)

Usage:
    from core.feature_registry_v4 import get_feature_names_v4, get_feature_defaults_v4
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np


@dataclass
class FeatureSpec:
    """Specification for a single feature."""
    name: str
    description: str
    default_value: float
    valid_range: Tuple[float, float]
    category: str
    importance: str = "medium"  # high, medium, low


# =============================================================================
# FEATURE DEFINITIONS V4 (52 features)
# =============================================================================
FEATURE_SPECS_V4: List[FeatureSpec] = [
    # --- Technical Base (5) ---
    FeatureSpec("RSI", "Relative Strength Index (0-100)", 50.0, (0, 100), "technical", "high"),
    FeatureSpec("ATR_Pct", "Average True Range as % of price", 0.02, (0.001, 0.5), "technical", "high"),
    FeatureSpec("Return_20d", "20-day price return", 0.0, (-1.0, 2.0), "technical", "high"),
    FeatureSpec("Return_10d", "10-day price return", 0.0, (-0.5, 1.0), "technical", "medium"),
    FeatureSpec("Return_5d", "5-day price return", 0.0, (-0.3, 0.5), "technical", "medium"),

    # --- Volatility Patterns (6) ---
    FeatureSpec("VCP_Ratio", "Volatility Contraction: ATR(10)/ATR(30)", 1.0, (0.1, 5.0), "volatility", "high"),
    FeatureSpec("Tightness_Ratio", "Range contraction: range_5d/range_20d", 1.0, (0.05, 2.0), "volatility", "high"),
    FeatureSpec("Dist_From_52w_High", "(Close/52w_High)-1", -0.1, (-0.8, 0.1), "volatility", "high"),
    FeatureSpec("MA_Alignment", "1 if Close>MA20>MA50>MA200", 0.0, (0, 1), "volatility", "high"),
    FeatureSpec("Vol_Contraction_Ratio", "20d ATR / 50d ATR", 1.0, (0.3, 3.0), "volatility", "medium"),
    FeatureSpec("Squeeze_On_Flag", "BB inside KC (pre-breakout)", 0.0, (0, 1), "volatility", "high"),

    # --- Volume Patterns (8) ---
    FeatureSpec("Volume_Surge", "vol_5d_avg / vol_20d_avg", 1.0, (0.1, 10.0), "volume", "high"),
    FeatureSpec("Up_Down_Volume_Ratio", "up-day vol / down-day vol", 1.0, (0.1, 10.0), "volume", "medium"),
    FeatureSpec("Momentum_Consistency", "% positive days (last 20)", 0.5, (0, 1), "momentum", "medium"),
    FeatureSpec("Volume_Ratio_20d", "current vol / 20d avg", 1.0, (0.1, 20.0), "volume", "medium"),
    FeatureSpec("Volume_Trend", "volume slope (accumulation)", 0.0, (-1.0, 1.0), "volume", "medium"),
    FeatureSpec("Up_Volume_Ratio", "up-day vol / total vol", 0.5, (0, 1), "volume", "medium"),
    FeatureSpec("Volume_Price_Confirm", "price+vol up confirmation", 0.5, (0, 1), "volume", "medium"),
    FeatureSpec("Relative_Volume_Rank", "vol percentile vs 60d", 0.5, (0, 1), "volume", "medium"),

    # --- Relative Strength (3) ---
    FeatureSpec("RS_vs_SPY_20d", "stock_ret_20d - spy_ret_20d", 0.0, (-1.0, 1.0), "momentum", "high"),
    FeatureSpec("RS_vs_SPY_60d", "stock_ret_60d - spy_ret_60d", 0.0, (-1.0, 1.0), "momentum", "medium"),
    FeatureSpec("RS_Momentum", "RS_20d - RS_60d (RS acceleration)", 0.0, (-0.5, 0.5), "momentum", "medium"),

    # --- Market Regime (5) ---
    FeatureSpec("Market_Regime", "Bull(1), Sideways(0), Bear(-1)", 0.0, (-1, 1), "market", "high"),
    FeatureSpec("Market_Volatility", "SPY 20d volatility (annualized)", 0.15, (0.05, 0.8), "market", "high"),
    FeatureSpec("Market_Trend", "SPY 50d return", 0.0, (-0.5, 0.5), "market", "medium"),
    FeatureSpec("High_Volatility", "1 if vol > 75th percentile", 0.0, (0, 1), "market", "medium"),
    FeatureSpec("VIX_Level", "VIX current level normalized", 0.15, (0.1, 0.8), "market", "high"),

    # --- Sector Relative (3) ---
    FeatureSpec("Sector_RS", "stock_ret_20d - sector_ret_20d", 0.0, (-1.0, 1.0), "sector", "high"),
    FeatureSpec("Sector_Momentum", "sector_etf_ret_20d", 0.0, (-0.5, 0.5), "sector", "medium"),
    FeatureSpec("Sector_Rank", "1 if stock beats sector in 5d", 0.5, (0, 1), "sector", "medium"),

    # --- Price Action (9) ---
    FeatureSpec("Distance_From_52w_Low", "(close-52w_low)/52w_low", 0.5, (-0.5, 5.0), "price_action", "medium"),
    FeatureSpec("Consolidation_Tightness", "(20d_high-low)/avg", 0.1, (0.01, 0.5), "price_action", "high"),
    FeatureSpec("Days_Since_52w_High", "normalized 0-1", 0.5, (0, 1), "price_action", "medium"),
    FeatureSpec("Price_vs_SMA50", "(close-sma50)/sma50", 0.0, (-0.5, 0.5), "price_action", "medium"),
    FeatureSpec("Price_vs_SMA200", "(close-sma200)/sma200", 0.0, (-0.5, 0.5), "price_action", "medium"),
    FeatureSpec("SMA50_vs_SMA200", "(sma50-sma200)/sma200", 0.0, (-0.3, 0.3), "price_action", "medium"),
    FeatureSpec("MA_Slope_20d", "slope of 20d MA", 0.0, (-0.3, 0.3), "price_action", "medium"),
    FeatureSpec("Distance_To_Resistance", "(20d_high-close)/close", 0.05, (0, 0.5), "price_action", "medium"),
    FeatureSpec("Support_Strength", "fraction days near support", 0.2, (0, 1), "price_action", "medium"),

    # --- Streaks & Patterns (4) - NEW ---
    FeatureSpec("UpStreak_Days", "Consecutive up days (0-10)", 0.0, (0, 10), "pattern", "medium"),
    FeatureSpec("DownStreak_Days", "Consecutive down days (0-10)", 0.0, (0, 10), "pattern", "medium"),
    FeatureSpec("OvernightGap_Avg", "Average overnight gap % (5d)", 0.0, (-0.05, 0.05), "pattern", "low"),
    FeatureSpec("Range_Pct_10d", "Average intraday range % (10d)", 0.02, (0.005, 0.1), "pattern", "low"),

    # --- Earnings Context (3) - NEW (was computed but unused) ---
    FeatureSpec("Days_To_Earnings", "Days until next earnings (0-90)", 45.0, (0, 90), "earnings", "high"),
    FeatureSpec("In_Earnings_Window", "1 if within 5 days of earnings", 0.0, (0, 1), "earnings", "high"),
    FeatureSpec("Post_Earnings_Drift", "Return since last earnings", 0.0, (-0.3, 0.5), "earnings", "medium"),

    # --- Sentiment Features (4) - NEW (from Finnhub/AI) ---
    FeatureSpec("News_Sentiment_7d", "Avg news sentiment (-1 to 1)", 0.0, (-1, 1), "sentiment", "high"),
    FeatureSpec("News_Volume_7d", "News article count (normalized)", 0.5, (0, 1), "sentiment", "medium"),
    FeatureSpec("Sentiment_Momentum", "Sentiment change (7d vs 21d)", 0.0, (-0.5, 0.5), "sentiment", "medium"),
    FeatureSpec("Social_Buzz_Score", "Social media mention intensity", 0.5, (0, 1), "sentiment", "low"),

    # --- Institutional Signals (4) - NEW (from Finnhub) ---
    FeatureSpec("Institutional_Change_QoQ", "Quarterly ownership change %", 0.0, (-0.3, 0.3), "institutional", "high"),
    FeatureSpec("Insider_Net_30d", "Net insider buy/sell (normalized)", 0.0, (-1, 1), "institutional", "high"),
    FeatureSpec("Analyst_Rating_Change", "Rating upgrades - downgrades", 0.0, (-3, 3), "institutional", "medium"),
    FeatureSpec("Price_Target_Upside", "Consensus PT / price - 1", 0.1, (-0.3, 1.0), "institutional", "medium"),
]

# Expected feature count for V4
FEATURE_COUNT_V4 = 54


# =============================================================================
# PUBLIC API
# =============================================================================

def get_feature_names_v4() -> List[str]:
    """Get ordered list of V4 feature names (52 features)."""
    return [f.name for f in FEATURE_SPECS_V4]


def get_feature_specs_v4() -> List[FeatureSpec]:
    """Get full V4 feature specifications."""
    return FEATURE_SPECS_V4.copy()


def get_feature_defaults_v4() -> Dict[str, float]:
    """Get dict of V4 feature name -> default value."""
    return {f.name: f.default_value for f in FEATURE_SPECS_V4}


def get_feature_ranges_v4() -> Dict[str, Tuple[float, float]]:
    """Get dict of V4 feature name -> (min, max) valid range."""
    return {f.name: f.valid_range for f in FEATURE_SPECS_V4}


def get_high_importance_features() -> List[str]:
    """Get features marked as high importance for debugging/analysis."""
    return [f.name for f in FEATURE_SPECS_V4 if f.importance == "high"]


def get_features_by_category(category: str) -> List[str]:
    """Get feature names for a specific category."""
    return [f.name for f in FEATURE_SPECS_V4 if f.category == category]


def validate_features_v4(df, strict: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame has all required V4 features.
    
    Args:
        df: DataFrame to validate
        strict: If True, fail on any missing feature
        
    Returns:
        Tuple of (is_valid, missing_features)
    """
    expected = set(get_feature_names_v4())
    present = set(df.columns) if hasattr(df, 'columns') else set()
    missing = list(expected - present)
    
    if strict:
        return len(missing) == 0, missing
    
    # Allow up to 10 missing features in non-strict mode
    return len(missing) <= 10, missing


# Backwards compatibility with V3
def get_feature_names(version: str = "v4") -> List[str]:
    """Get feature names for specified version."""
    if version == "v4":
        return get_feature_names_v4()
    elif version == "v3":
        from core.feature_registry import get_feature_names as get_v3
        return get_v3("v3")
    raise ValueError(f"Unknown version: {version}")
