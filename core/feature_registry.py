"""
Feature Registry - Single Source of Truth for ML Features.

This module defines ALL features used in ML models. Both training and inference
MUST import from here to ensure consistency.

DO NOT hardcode feature lists anywhere else in the codebase!

Usage:
    from core.feature_registry import get_feature_names, get_feature_defaults
    
    # Get ordered list of feature names
    features = get_feature_names("v3")
    
    # Get default values for all features
    defaults = get_feature_defaults("v3")
    
    # Validate a DataFrame has all required features
    is_valid, missing = validate_features(df, "v3")
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass
class FeatureSpec:
    """Specification for a single feature."""
    name: str
    description: str
    default_value: float
    valid_range: Tuple[float, float]  # (min, max)
    category: str  # e.g., "technical", "volume", "market_regime"


# =============================================================================
# FEATURE DEFINITIONS V3 (34 features) — legacy, kept for backward compat
# =============================================================================
FEATURE_SPECS_V3: List[FeatureSpec] = [
    # --- Technical Base (5) ---
    FeatureSpec("RSI", "Relative Strength Index (0-100)", 50.0, (0, 100), "technical"),
    FeatureSpec("ATR_Pct", "Average True Range as % of price", 0.02, (0.001, 0.5), "technical"),
    FeatureSpec("Return_20d", "20-day price return", 0.0, (-1.0, 2.0), "technical"),
    FeatureSpec("Return_10d", "10-day price return", 0.0, (-0.5, 1.0), "technical"),
    FeatureSpec("Return_5d", "5-day price return", 0.0, (-0.3, 0.5), "technical"),

    # --- Volatility Patterns (4) ---
    FeatureSpec("VCP_Ratio", "Volatility Contraction: ATR(10)/ATR(30)", 1.0, (0.1, 5.0), "volatility"),
    FeatureSpec("Tightness_Ratio", "Range contraction: range_5d/range_20d", 1.0, (0.05, 2.0), "volatility"),
    FeatureSpec("Dist_From_52w_High", "(Close/52w_High)-1", -0.1, (-0.8, 0.1), "volatility"),
    FeatureSpec("MA_Alignment", "1 if Close>MA20>MA50>MA200", 0.0, (0, 1), "volatility"),

    # --- Volume Basic (3) ---
    FeatureSpec("Volume_Surge", "vol_5d_avg / vol_20d_avg", 1.0, (0.1, 10.0), "volume"),
    FeatureSpec("Up_Down_Volume_Ratio", "up-day vol / down-day vol", 1.0, (0.1, 10.0), "volume"),
    FeatureSpec("Momentum_Consistency", "% positive days (last 20)", 0.5, (0, 1), "momentum"),

    # --- Relative Strength (1) ---
    FeatureSpec("RS_vs_SPY_20d", "stock_ret_20d - spy_ret_20d", 0.0, (-1.0, 1.0), "momentum"),

    # --- Market Regime (4) ---
    FeatureSpec("Market_Regime", "Bull(1), Sideways(0), Bear(-1)", 0.0, (-1, 1), "market"),
    FeatureSpec("Market_Volatility", "SPY 20d volatility (annualized)", 0.15, (0.05, 0.8), "market"),
    FeatureSpec("Market_Trend", "SPY 50d return", 0.0, (-0.5, 0.5), "market"),
    FeatureSpec("High_Volatility", "1 if vol > 75th percentile", 0.0, (0, 1), "market"),

    # --- Sector Relative (3) ---
    FeatureSpec("Sector_RS", "stock_ret_20d - sector_ret_20d", 0.0, (-1.0, 1.0), "sector"),
    FeatureSpec("Sector_Momentum", "sector_etf_ret_20d", 0.0, (-0.5, 0.5), "sector"),
    FeatureSpec("Sector_Rank", "1 if stock beats sector in 5d", 0.5, (0, 1), "sector"),

    # --- Volume Advanced (5) ---
    FeatureSpec("Volume_Ratio_20d", "current vol / 20d avg", 1.0, (0.1, 20.0), "volume"),
    FeatureSpec("Volume_Trend", "volume slope (accumulation)", 0.0, (-1.0, 1.0), "volume"),
    FeatureSpec("Up_Volume_Ratio", "up-day vol / total vol", 0.5, (0, 1), "volume"),
    FeatureSpec("Volume_Price_Confirm", "price+vol up confirmation", 0.5, (0, 1), "volume"),
    FeatureSpec("Relative_Volume_Rank", "vol percentile vs 60d", 0.5, (0, 1), "volume"),

    # --- Price Action (9) ---
    FeatureSpec("Distance_From_52w_Low", "(close-52w_low)/52w_low", 0.5, (-0.5, 5.0), "price_action"),
    FeatureSpec("Consolidation_Tightness", "(20d_high-low)/avg", 0.1, (0.01, 0.5), "price_action"),
    FeatureSpec("Days_Since_52w_High", "normalized 0-1", 0.5, (0, 1), "price_action"),
    FeatureSpec("Price_vs_SMA50", "(close-sma50)/sma50", 0.0, (-0.5, 0.5), "price_action"),
    FeatureSpec("Price_vs_SMA200", "(close-sma200)/sma200", 0.0, (-0.5, 0.5), "price_action"),
    FeatureSpec("SMA50_vs_SMA200", "(sma50-sma200)/sma200", 0.0, (-0.3, 0.3), "price_action"),
    FeatureSpec("MA_Slope_20d", "slope of 20d MA", 0.0, (-0.3, 0.3), "price_action"),
    FeatureSpec("Distance_To_Resistance", "(20d_high-close)/close", 0.05, (0, 0.5), "price_action"),
    FeatureSpec("Support_Strength", "fraction days near support", 0.2, (0, 1), "price_action"),
]


# =============================================================================
# FEATURE DEFINITIONS V3.1 (39 features) — stock-picking focused
#
# Changes from v3:
#   REMOVED (market-timing, was 42% of model importance):
#     - Market_Trend (SPY 50d return — pure market timer)
#     - Market_Volatility (SPY vol — pure market timer)
#     - High_Volatility (derived from SPY vol)
#
#   KEPT:
#     - Market_Regime (bull/bear context, only 1.9% importance — useful signal)
#
#   ADDED (8 stock-specific, all computable from OHLCV):
#     - Vol_Contraction_Ratio (ATR_20/ATR_50 — tighter = breakout setup)
#     - Squeeze_On_Flag (BB inside KC — pre-breakout detection)
#     - RS_vs_SPY_60d (longer-term relative strength)
#     - RS_Momentum (RS acceleration — 20d RS vs 60d RS)
#     - UpStreak_Days, DownStreak_Days (price persistence)
#     - Range_Pct_10d (avg intraday range — energy measure)
#     - OvernightGap_Avg (institutional interest signal — gap patterns)
# =============================================================================
FEATURE_SPECS_V3_1: List[FeatureSpec] = [
    # --- Technical Base (5) ---
    FeatureSpec("RSI", "Relative Strength Index (0-100)", 50.0, (0, 100), "technical"),
    FeatureSpec("ATR_Pct", "Average True Range as % of price", 0.02, (0.001, 0.5), "technical"),
    FeatureSpec("Return_20d", "20-day price return", 0.0, (-1.0, 2.0), "technical"),
    FeatureSpec("Return_10d", "10-day price return", 0.0, (-0.5, 1.0), "technical"),
    FeatureSpec("Return_5d", "5-day price return", 0.0, (-0.3, 0.5), "technical"),

    # --- Volatility Patterns (6) ---
    FeatureSpec("VCP_Ratio", "Volatility Contraction: ATR(10)/ATR(30)", 1.0, (0.1, 5.0), "volatility"),
    FeatureSpec("Tightness_Ratio", "Range contraction: range_5d/range_20d", 1.0, (0.05, 2.0), "volatility"),
    FeatureSpec("Dist_From_52w_High", "(Close/52w_High)-1", -0.1, (-0.8, 0.1), "volatility"),
    FeatureSpec("MA_Alignment", "1 if Close>MA20>MA50>MA200", 0.0, (0, 1), "volatility"),
    FeatureSpec("Vol_Contraction_Ratio", "ATR(20)/ATR(50) — tighter=breakout", 1.0, (0.3, 3.0), "volatility"),
    FeatureSpec("Squeeze_On_Flag", "BB inside KC (pre-breakout)", 0.0, (0, 1), "volatility"),

    # --- Volume Basic (3) ---
    FeatureSpec("Volume_Surge", "vol_5d_avg / vol_20d_avg", 1.0, (0.1, 10.0), "volume"),
    FeatureSpec("Up_Down_Volume_Ratio", "avg up-day vol / avg down-day vol", 1.0, (0.1, 10.0), "volume"),
    FeatureSpec("Momentum_Consistency", "% positive days (last 20)", 0.5, (0, 1), "momentum"),

    # --- Relative Strength (3) ---
    FeatureSpec("RS_vs_SPY_20d", "stock_ret_20d - spy_ret_20d", 0.0, (-1.0, 1.0), "momentum"),
    FeatureSpec("RS_vs_SPY_60d", "stock_ret_60d - spy_ret_60d", 0.0, (-1.0, 1.0), "momentum"),
    FeatureSpec("RS_Momentum", "RS_20d - RS_60d (acceleration)", 0.0, (-0.5, 0.5), "momentum"),

    # --- Market Regime (1) --- STRIPPED from 4 → 1 (remove market-timing)
    FeatureSpec("Market_Regime", "Bull(1), Sideways(0), Bear(-1)", 0.0, (-1, 1), "market"),

    # --- Sector Relative (3) ---
    FeatureSpec("Sector_RS", "stock_ret_20d - sector_ret_20d", 0.0, (-1.0, 1.0), "sector"),
    FeatureSpec("Sector_Momentum", "sector_etf_ret_20d", 0.0, (-0.5, 0.5), "sector"),
    FeatureSpec("Sector_Rank", "1 if stock beats sector in 5d", 0.5, (0, 1), "sector"),

    # --- Volume Advanced (5) ---
    FeatureSpec("Volume_Ratio_20d", "current vol / 20d avg", 1.0, (0.1, 20.0), "volume"),
    FeatureSpec("Volume_Trend", "volume slope (accumulation)", 0.0, (-1.0, 1.0), "volume"),
    FeatureSpec("Up_Volume_Ratio", "up-day vol / total vol", 0.5, (0, 1), "volume"),
    FeatureSpec("Volume_Price_Confirm", "price+vol up confirmation", 0.5, (0, 1), "volume"),
    FeatureSpec("Relative_Volume_Rank", "vol percentile vs 60d", 0.5, (0, 1), "volume"),

    # --- Price Action (9) ---
    FeatureSpec("Distance_From_52w_Low", "(close-52w_low)/52w_low", 0.5, (-0.5, 5.0), "price_action"),
    FeatureSpec("Consolidation_Tightness", "(20d_high-low)/avg", 0.1, (0.01, 0.5), "price_action"),
    FeatureSpec("Days_Since_52w_High", "normalized 0-1", 0.5, (0, 1), "price_action"),
    FeatureSpec("Price_vs_SMA50", "(close-sma50)/sma50", 0.0, (-0.5, 0.5), "price_action"),
    FeatureSpec("Price_vs_SMA200", "(close-sma200)/sma200", 0.0, (-0.5, 0.5), "price_action"),
    FeatureSpec("SMA50_vs_SMA200", "(sma50-sma200)/sma200", 0.0, (-0.3, 0.3), "price_action"),
    FeatureSpec("MA_Slope_20d", "slope of 20d MA", 0.0, (-0.3, 0.3), "price_action"),
    FeatureSpec("Distance_To_Resistance", "(20d_high-close)/close", 0.05, (0, 0.5), "price_action"),
    FeatureSpec("Support_Strength", "fraction days near support", 0.2, (0, 1), "price_action"),

    # --- Streaks & Patterns (4) --- NEW stock-specific
    FeatureSpec("UpStreak_Days", "Consecutive up days (0-10)", 0.0, (0, 10), "pattern"),
    FeatureSpec("DownStreak_Days", "Consecutive down days (0-10)", 0.0, (0, 10), "pattern"),
    FeatureSpec("Range_Pct_10d", "Avg intraday range pct (10d)", 0.02, (0.005, 0.1), "pattern"),
    FeatureSpec("OvernightGap_Avg", "Avg overnight gap pct (5d)", 0.0, (-0.05, 0.05), "pattern"),
]


# =============================================================================
# PUBLIC API
# =============================================================================

def get_feature_names(version: str = "v3") -> List[str]:
    """
    Get ordered list of feature names.

    Args:
        version: Feature version ("v3" for 34-feature set, "v3.1" for 39-feature,
                 "v3.2" for 20-feature pruned set, "v4" for 72-feature)

    Returns:
        List of feature names in canonical order
    """
    if version == "v3":
        return [f.name for f in FEATURE_SPECS_V3]
    if version == "v3.1":
        return [f.name for f in FEATURE_SPECS_V3_1]
    if version == "v3.2":
        return [f.name for f in FEATURE_SPECS_V3_2]
    if version == "v4":
        return [f.name for f in FEATURE_SPECS_V4]
    raise ValueError(f"Unknown feature version: {version}")


def get_feature_specs(version: str = "v3") -> List[FeatureSpec]:
    """
    Get full feature specifications.

    Args:
        version: Feature version ("v3", "v3.1", "v3.2", or "v4")

    Returns:
        List of FeatureSpec objects with full metadata
    """
    if version == "v3":
        return FEATURE_SPECS_V3.copy()
    if version == "v3.1":
        return FEATURE_SPECS_V3_1.copy()
    if version == "v3.2":
        return FEATURE_SPECS_V3_2.copy()
    if version == "v4":
        return FEATURE_SPECS_V4.copy()
    raise ValueError(f"Unknown feature version: {version}")


def get_feature_defaults(version: str = "v3") -> Dict[str, float]:
    """
    Get dict of feature name -> default value.
    
    Args:
        version: Feature version
        
    Returns:
        Dict mapping feature names to their default values
    """
    return {f.name: f.default_value for f in get_feature_specs(version)}


def get_feature_ranges(version: str = "v3") -> Dict[str, Tuple[float, float]]:
    """
    Get dict of feature name -> (min, max) valid range.
    
    Args:
        version: Feature version
        
    Returns:
        Dict mapping feature names to (min, max) tuples
    """
    return {f.name: f.valid_range for f in get_feature_specs(version)}


def get_features_by_category(version: str = "v3") -> Dict[str, List[str]]:
    """
    Get features grouped by category.
    
    Args:
        version: Feature version
        
    Returns:
        Dict mapping category names to lists of feature names
    """
    result: Dict[str, List[str]] = {}
    for spec in get_feature_specs(version):
        if spec.category not in result:
            result[spec.category] = []
        result[spec.category].append(spec.name)
    return result


def validate_features(df, version: str = "v3") -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame has all required features.
    
    Args:
        df: DataFrame to validate
        version: Feature version to check against
        
    Returns:
        Tuple of (is_valid, missing_features)
    """
    required = set(get_feature_names(version))
    present = set(df.columns)
    missing = required - present
    return len(missing) == 0, sorted(list(missing))


def clip_features_to_range(df, version: str = "v3", inplace: bool = True):
    """
    Clip feature values to valid ranges.
    
    Args:
        df: DataFrame with features
        version: Feature version for ranges
        inplace: If True, modify df in place; otherwise return copy
        
    Returns:
        DataFrame with clipped values
    """
    if not inplace:
        df = df.copy()
    
    ranges = get_feature_ranges(version)
    for feat, (lo, hi) in ranges.items():
        if feat in df.columns:
            df[feat] = np.clip(df[feat], lo, hi)
    return df


def fill_missing_with_defaults(features_dict: Dict[str, float], version: str = "v3") -> Dict[str, float]:
    """
    Fill missing features with default values.
    
    Args:
        features_dict: Dict of feature values (may be incomplete)
        version: Feature version
        
    Returns:
        Complete dict with all features filled in
    """
    defaults = get_feature_defaults(version)
    result = defaults.copy()
    result.update(features_dict)
    return result


# =============================================================================
# FEATURE DEFINITIONS V4 (72 features) — fundamental + cross-sectional + delta
#
# Changes from v3.1:
#   ADDED 15 fundamental features (ML has ZERO fundamental input in v3.1)
#   ADDED 8 cross-sectional rank features (relative positioning vs peers)
#   ADDED 6 temporal delta features (acceleration/deceleration)
#   ADDED 4 interaction features (non-linear combinations)
# =============================================================================
FEATURE_SPECS_V4: List[FeatureSpec] = [
    # ── All 39 from v3.1 ─────────────────────────────────────────────
    *FEATURE_SPECS_V3_1,

    # ── 15 Fundamental Features (NEW — biggest gap in v3.1) ──────────
    FeatureSpec("Fund_Quality_Score", "ROE/ROIC/margin composite (0-100)", 50.0, (0, 100), "fundamental"),
    FeatureSpec("Fund_Growth_Score", "Revenue/EPS YoY composite (0-100)", 50.0, (0, 100), "fundamental"),
    FeatureSpec("Fund_Valuation_Score", "PE/PS composite (0-100, high=cheap)", 50.0, (0, 100), "fundamental"),
    FeatureSpec("PE_Percentile", "PE rank within sector (0-1)", 0.5, (0, 1), "fundamental"),
    FeatureSpec("PS_Percentile", "PS rank within sector (0-1)", 0.5, (0, 1), "fundamental"),
    FeatureSpec("ROE_Percentile", "ROE rank within universe (0-1)", 0.5, (0, 1), "fundamental"),
    FeatureSpec("Revenue_Growth_Bucket", "Discretized: 0=declining,1=flat,2=moderate,3=strong", 1.0, (0, 3), "fundamental"),
    FeatureSpec("EPS_Growth_Bucket", "Discretized: 0=declining,1=flat,2=moderate,3=strong", 1.0, (0, 3), "fundamental"),
    FeatureSpec("Debt_Risk", "D/E ratio normalised (0-1, higher=riskier)", 0.3, (0, 1), "fundamental"),
    FeatureSpec("MarketCap_Log", "log10(market_cap)", 10.0, (6, 13), "fundamental"),
    FeatureSpec("Fund_Coverage", "Fundamental_Coverage_Pct / 100 (0-1)", 0.5, (0, 1), "fundamental"),
    FeatureSpec("Fund_Disagreement", "Data disagreement score (0-1)", 0.2, (0, 1), "fundamental"),
    FeatureSpec("Quality_Value_Combo", "Quality × (100-Valuation)/100", 25.0, (0, 100), "fundamental"),
    FeatureSpec("Growth_Momentum_Combo", "Growth × Momentum_Consistency", 25.0, (0, 100), "fundamental"),
    FeatureSpec("Earnings_Proximity", "1/(1+days_to_earnings) or 0", 0.0, (0, 1), "fundamental"),

    # ── 8 Cross-Sectional Rank Features ──────────────────────────────
    FeatureSpec("RSI_Rank", "RSI percentile within universe (0-1)", 0.5, (0, 1), "rank"),
    FeatureSpec("ATR_Rank", "ATR_Pct percentile within universe (0-1)", 0.5, (0, 1), "rank"),
    FeatureSpec("Momentum_Rank", "Return_20d percentile within universe (0-1)", 0.5, (0, 1), "rank"),
    FeatureSpec("Volume_Rank", "Volume_Surge percentile within universe (0-1)", 0.5, (0, 1), "rank"),
    FeatureSpec("TechScore_Rank", "TechScore_20d percentile within universe (0-1)", 0.5, (0, 1), "rank"),
    FeatureSpec("FundScore_Rank", "Fundamental_Score percentile within universe (0-1)", 0.5, (0, 1), "rank"),
    FeatureSpec("RS_Rank", "RS_vs_SPY_20d percentile within universe (0-1)", 0.5, (0, 1), "rank"),
    FeatureSpec("RR_Rank", "RR_Ratio percentile within universe (0-1)", 0.5, (0, 1), "rank"),

    # ── 6 Temporal Delta Features ────────────────────────────────────
    FeatureSpec("RSI_Delta_5d", "RSI_now - RSI_5d_ago", 0.0, (-50, 50), "delta"),
    FeatureSpec("ATR_Delta_5d", "ATR_Pct_now - ATR_Pct_5d_ago", 0.0, (-0.1, 0.1), "delta"),
    FeatureSpec("Volume_Delta_5d", "Volume_Surge_now - Volume_Surge_5d_ago", 0.0, (-5, 5), "delta"),
    FeatureSpec("RS_Acceleration", "RS_20d - RS_60d", 0.0, (-0.5, 0.5), "delta"),
    FeatureSpec("Momentum_Acceleration", "Return_5d - (Return_20d-Return_5d)/3", 0.0, (-0.3, 0.3), "delta"),
    FeatureSpec("Breadth_Delta_5d", "Market breadth today - 5d ago", 0.0, (-0.5, 0.5), "delta"),

    # ── 4 Interaction Features ───────────────────────────────────────
    FeatureSpec("VCP_x_RS", "VCP_Ratio × RS_vs_SPY_20d", 0.0, (-5, 5), "interaction"),
    FeatureSpec("Momentum_x_Volume", "Momentum_Consistency × Volume_Surge", 0.5, (0, 10), "interaction"),
    FeatureSpec("Quality_x_Momentum", "Fund_Quality_Score/100 × Momentum_Consistency", 0.25, (0, 1), "interaction"),
    FeatureSpec("Squeeze_x_Volume", "Squeeze_On_Flag × Volume_Surge", 0.0, (0, 10), "interaction"),
]


# =============================================================================
# FEATURE DEFINITIONS V3.2 (20 features) — pruned from v3.1 via permutation
#   importance analysis (AUC=0.554, see models/feature_importance_report.txt).
#
#   17 features with NEGATIVE permutation importance removed from training.
#   2 features with zero importance also removed.
#   Result: 20 features that actually help the model predict.
#
#   This is NOT a band-aid: the model is retrained on these 20 features only,
#   so the model's capacity is fully focused on useful signal.
# =============================================================================
FEATURE_SPECS_V3_2: List[FeatureSpec] = [
    # --- Price Action (6) --- top importance features
    FeatureSpec("Support_Strength", "fraction days near support", 0.2, (0, 1), "price_action"),
    FeatureSpec("Distance_From_52w_Low", "(close-52w_low)/52w_low", 0.5, (-0.5, 5.0), "price_action"),
    FeatureSpec("Consolidation_Tightness", "(20d_high-low)/avg", 0.1, (0.01, 0.5), "price_action"),
    FeatureSpec("Distance_To_Resistance", "(20d_high-close)/close", 0.05, (0, 0.5), "price_action"),
    FeatureSpec("Days_Since_52w_High", "normalized 0-1", 0.5, (0, 1), "price_action"),
    FeatureSpec("Price_vs_SMA50", "(close-sma50)/sma50", 0.0, (-0.5, 0.5), "price_action"),

    # --- Sector Relative (3) ---
    FeatureSpec("Sector_Momentum", "sector_etf_ret_20d", 0.0, (-0.5, 0.5), "sector"),
    FeatureSpec("Sector_RS", "stock_ret_20d - sector_ret_20d", 0.0, (-1.0, 1.0), "sector"),
    FeatureSpec("Sector_Rank", "1 if stock beats sector in 5d", 0.5, (0, 1), "sector"),

    # --- Momentum (3) ---
    FeatureSpec("RS_vs_SPY_20d", "stock_ret_20d - spy_ret_20d", 0.0, (-1.0, 1.0), "momentum"),
    FeatureSpec("Return_20d", "20-day price return", 0.0, (-1.0, 2.0), "technical"),
    FeatureSpec("Momentum_Consistency", "% positive days (last 20)", 0.5, (0, 1), "momentum"),

    # --- Volatility Patterns (3) ---
    FeatureSpec("Tightness_Ratio", "Range contraction: range_5d/range_20d", 1.0, (0.05, 2.0), "volatility"),
    FeatureSpec("ATR_Pct", "Average True Range as % of price", 0.02, (0.001, 0.5), "technical"),
    FeatureSpec("VCP_Ratio", "Volatility Contraction: ATR(10)/ATR(30)", 1.0, (0.1, 5.0), "volatility"),

    # --- Volume (5) ---
    FeatureSpec("Up_Volume_Ratio", "up-day vol / total vol", 0.5, (0, 1), "volume"),
    FeatureSpec("Up_Down_Volume_Ratio", "avg up-day vol / avg down-day vol", 1.0, (0.1, 10.0), "volume"),
    FeatureSpec("Volume_Ratio_20d", "current vol / 20d avg", 1.0, (0.1, 20.0), "volume"),
    FeatureSpec("Volume_Surge", "vol_5d_avg / vol_20d_avg", 1.0, (0.1, 10.0), "volume"),
    FeatureSpec("Volume_Price_Confirm", "price+vol up confirmation", 0.5, (0, 1), "volume"),
]


# =============================================================================
# CONSTANTS
# =============================================================================

# Feature count for assertions
FEATURE_COUNT_V3 = len(FEATURE_SPECS_V3)
assert FEATURE_COUNT_V3 == 34, f"Expected 34 features, got {FEATURE_COUNT_V3}"

FEATURE_COUNT_V3_1 = len(FEATURE_SPECS_V3_1)
assert FEATURE_COUNT_V3_1 == 39, f"Expected 39 features, got {FEATURE_COUNT_V3_1}"

FEATURE_COUNT_V3_2 = len(FEATURE_SPECS_V3_2)
assert FEATURE_COUNT_V3_2 == 20, f"Expected 20 features, got {FEATURE_COUNT_V3_2}"

FEATURE_COUNT_V4 = len(FEATURE_SPECS_V4)
assert FEATURE_COUNT_V4 == 72, f"Expected 72 features, got {FEATURE_COUNT_V4}"

# List of all supported versions
SUPPORTED_VERSIONS = ["v3", "v3.1", "v3.2", "v4"]

# Current default version (v3.2 = production model with pruned features)
DEFAULT_VERSION = "v3.2"
