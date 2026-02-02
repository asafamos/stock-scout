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
# FEATURE DEFINITIONS V3 (34 features)
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
# PUBLIC API
# =============================================================================

def get_feature_names(version: str = "v3") -> List[str]:
    """
    Get ordered list of feature names.
    
    Args:
        version: Feature version ("v3" for 34-feature set)
        
    Returns:
        List of feature names in canonical order
    """
    if version == "v3":
        return [f.name for f in FEATURE_SPECS_V3]
    raise ValueError(f"Unknown feature version: {version}")


def get_feature_specs(version: str = "v3") -> List[FeatureSpec]:
    """
    Get full feature specifications.
    
    Args:
        version: Feature version ("v3" for 34-feature set)
        
    Returns:
        List of FeatureSpec objects with full metadata
    """
    if version == "v3":
        return FEATURE_SPECS_V3.copy()
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
# CONSTANTS
# =============================================================================

# Feature count for assertions
FEATURE_COUNT_V3 = len(FEATURE_SPECS_V3)
assert FEATURE_COUNT_V3 == 34, f"Expected 34 features, got {FEATURE_COUNT_V3}"

# List of all supported versions
SUPPORTED_VERSIONS = ["v3"]

# Current default version
DEFAULT_VERSION = "v3"
