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


# Current default version (v3.5 = production model, 20 features)
DEFAULT_VERSION = "v3.5"

# =============================================================================
# PUBLIC API
# =============================================================================

def get_feature_names(version: str = DEFAULT_VERSION) -> List[str]:
    """
    Get ordered list of feature names.

    Args:
        version: Feature version ("v3", "v3.1", "v3.2", "v3.3", "v3.4", "v3.5")

    Returns:
        List of feature names in canonical order
    """
    specs_map = {
        "v3": FEATURE_SPECS_V3,
        "v3.1": FEATURE_SPECS_V3_1,
        "v3.2": FEATURE_SPECS_V3_2,
        "v3.3": FEATURE_SPECS_V3_3,
        "v3.4": FEATURE_SPECS_V3_4,
        "v3.5": FEATURE_SPECS_V3_5,
    }
    if version not in specs_map:
        raise ValueError(f"Unknown feature version: {version}. Supported: {SUPPORTED_VERSIONS}")
    return [f.name for f in specs_map[version]]


def get_feature_specs(version: str = DEFAULT_VERSION) -> List[FeatureSpec]:
    """
    Get full feature specifications.

    Args:
        version: Feature version ("v3", "v3.1", "v3.2", "v3.3", "v3.4", "v3.5")

    Returns:
        List of FeatureSpec objects with full metadata
    """
    specs_map = {
        "v3": FEATURE_SPECS_V3,
        "v3.1": FEATURE_SPECS_V3_1,
        "v3.2": FEATURE_SPECS_V3_2,
        "v3.3": FEATURE_SPECS_V3_3,
        "v3.4": FEATURE_SPECS_V3_4,
        "v3.5": FEATURE_SPECS_V3_5,
    }
    if version not in specs_map:
        raise ValueError(f"Unknown feature version: {version}. Supported: {SUPPORTED_VERSIONS}")
    return specs_map[version].copy()


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
# FEATURE DEFINITIONS V3.3 (16 features) — pruned further from v3.2
#
#   v3.2 had 20 features. Four were confirmed NEGATIVE permutation importance:
#     - Days_Since_52w_High  (-1.549% importance)
#     - Price_vs_SMA50       (-1.009% importance)
#     - Volume_Price_Confirm (-0.166% importance)
#     - Up_Down_Volume_Ratio (-0.019% importance)
#
#   Removing them frees model capacity for the 16 features that actually help.
# =============================================================================
FEATURE_SPECS_V3_3: List[FeatureSpec] = [
    # --- Price Action (4) --- top importance features
    FeatureSpec("Support_Strength", "fraction days near support", 0.2, (0, 1), "price_action"),
    FeatureSpec("Distance_From_52w_Low", "(close-52w_low)/52w_low", 0.5, (-0.5, 5.0), "price_action"),
    FeatureSpec("Consolidation_Tightness", "(20d_high-low)/avg", 0.1, (0.01, 0.5), "price_action"),
    FeatureSpec("Distance_To_Resistance", "(20d_high-close)/close", 0.05, (0, 0.5), "price_action"),

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

    # --- Volume (3) --- only features with positive importance
    FeatureSpec("Up_Volume_Ratio", "up-day vol / total vol", 0.5, (0, 1), "volume"),
    FeatureSpec("Volume_Ratio_20d", "current vol / 20d avg", 1.0, (0.1, 20.0), "volume"),
    FeatureSpec("Volume_Surge", "vol_5d_avg / vol_20d_avg", 1.0, (0.1, 10.0), "volume"),
]


# =============================================================================
# FEATURE DEFINITIONS V3.4 (13 features) — pruned further from v3.3
#
#   v3.3 training (AUC=0.613) identified 3 more features with NEGATIVE
#   permutation importance:
#     - Volume_Ratio_20d   (-0.0000 ± 0.0002)
#     - Up_Volume_Ratio    (-0.0002 ± 0.0009)
#     - Sector_RS          (not in top 15, negative importance)
#
#   Removing them focuses the model on 13 features that all contribute
#   positively to prediction quality.
# =============================================================================
FEATURE_SPECS_V3_4: List[FeatureSpec] = [
    # --- Price Action (4) --- highest importance features
    FeatureSpec("Support_Strength", "fraction days near support", 0.2, (0, 1), "price_action"),
    FeatureSpec("Distance_From_52w_Low", "(close-52w_low)/52w_low", 0.5, (-0.5, 5.0), "price_action"),
    FeatureSpec("Consolidation_Tightness", "(20d_high-low)/avg", 0.1, (0.01, 0.5), "price_action"),
    FeatureSpec("Distance_To_Resistance", "(20d_high-close)/close", 0.05, (0, 0.5), "price_action"),

    # --- Sector Relative (2) --- removed Sector_RS (negative importance)
    FeatureSpec("Sector_Momentum", "sector_etf_ret_20d", 0.0, (-0.5, 0.5), "sector"),
    FeatureSpec("Sector_Rank", "1 if stock beats sector in 5d", 0.5, (0, 1), "sector"),

    # --- Momentum (3) ---
    FeatureSpec("RS_vs_SPY_20d", "stock_ret_20d - spy_ret_20d", 0.0, (-1.0, 1.0), "momentum"),
    FeatureSpec("Return_20d", "20-day price return", 0.0, (-1.0, 2.0), "technical"),
    FeatureSpec("Momentum_Consistency", "% positive days (last 20)", 0.5, (0, 1), "momentum"),

    # --- Volatility Patterns (3) ---
    FeatureSpec("Tightness_Ratio", "Range contraction: range_5d/range_20d", 1.0, (0.05, 2.0), "volatility"),
    FeatureSpec("ATR_Pct", "Average True Range as % of price", 0.02, (0.001, 0.5), "technical"),
    FeatureSpec("VCP_Ratio", "Volatility Contraction: ATR(10)/ATR(30)", 1.0, (0.1, 5.0), "volatility"),

    # --- Volume (1) --- removed Volume_Ratio_20d & Up_Volume_Ratio (negative importance)
    FeatureSpec("Volume_Surge", "vol_5d_avg / vol_20d_avg", 1.0, (0.1, 10.0), "volume"),
]


# =============================================================================
# FEATURE DEFINITIONS V3.5 (20 features) — V3.4 + cherry-picked delta/interaction
#
#   V3.4 (13 features) + 7 from V4:
#     Delta (4): RSI_Delta_5d, ATR_Delta_5d, Volume_Delta_5d, Momentum_Acceleration
#     Interaction (3): VCP_x_RS, Momentum_x_Volume, Squeeze_x_Volume
#
#   These capture momentum acceleration and non-linear pattern combinations
#   without requiring external APIs (all OHLCV-derived).
# =============================================================================
FEATURE_SPECS_V3_5: List[FeatureSpec] = [
    # ── All 13 from V3.4 ────────────────────────────────────────────
    *FEATURE_SPECS_V3_4,

    # ── 4 Delta Features (momentum/volatility acceleration) ─────────
    FeatureSpec("RSI_Delta_5d", "RSI_now - RSI_5d_ago", 0.0, (-50, 50), "delta"),
    FeatureSpec("ATR_Delta_5d", "ATR_Pct_now - ATR_Pct_5d_ago", 0.0, (-0.1, 0.1), "delta"),
    FeatureSpec("Volume_Delta_5d", "Volume_Surge_now - Volume_Surge_5d_ago", 0.0, (-5, 5), "delta"),
    FeatureSpec("Momentum_Acceleration", "Return_5d - (Return_20d-Return_5d)/3", 0.0, (-0.3, 0.3), "delta"),

    # ── 3 Interaction Features (non-linear combinations) ────────────
    FeatureSpec("VCP_x_RS", "VCP_Ratio * RS_vs_SPY_20d", 0.0, (-5, 5), "interaction"),
    FeatureSpec("Momentum_x_Volume", "Momentum_Consistency * Volume_Surge", 0.5, (0, 10), "interaction"),
    FeatureSpec("Squeeze_x_Volume", "Squeeze_On_Flag * Volume_Surge", 0.0, (0, 10), "interaction"),
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

FEATURE_COUNT_V3_3 = len(FEATURE_SPECS_V3_3)
assert FEATURE_COUNT_V3_3 == 16, f"Expected 16 features, got {FEATURE_COUNT_V3_3}"

FEATURE_COUNT_V3_4 = len(FEATURE_SPECS_V3_4)
assert FEATURE_COUNT_V3_4 == 13, f"Expected 13 features, got {FEATURE_COUNT_V3_4}"

FEATURE_COUNT_V3_5 = len(FEATURE_SPECS_V3_5)
assert FEATURE_COUNT_V3_5 == 20, f"Expected 20 features, got {FEATURE_COUNT_V3_5}"

# List of all supported versions
SUPPORTED_VERSIONS = ["v3", "v3.1", "v3.2", "v3.3", "v3.4", "v3.5"]
