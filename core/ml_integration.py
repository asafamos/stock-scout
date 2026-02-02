"""
ML Integration Layer for Stock Scout v2.

Provides bounded ML boost to base conviction scores with graceful fallback.
ML can adjust final conviction by at most ±10% of base score.

Design: ML is a helper, not a replacement. Hard risk rules always apply first.
"""
from __future__ import annotations
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List, Any
import logging
import joblib

# Import from feature registry - Single Source of Truth for features
from core.feature_registry import (
    get_feature_names,
    get_feature_defaults,
    get_feature_ranges,
    FEATURE_COUNT_V3,
)

logger = logging.getLogger(__name__)

# ============================================================================
# EXPECTED_FEATURES - Single Source of Truth from feature_registry
# The registry defines the canonical 34-feature set (v3).
# Model bundles may use a subset; we load from bundle if available.
# ============================================================================
# Default features from registry (used if model bundle doesn't specify)
DEFAULT_FEATURES: List[str] = get_feature_names("v3")

# This will be populated when the model is loaded (from bundle or defaults)
EXPECTED_FEATURES: List[str] = []

# Global model cache
_ML_MODEL = None
_MODEL_LOADED = False
_MODEL_BUNDLE = None  # Store full bundle for metadata access


# Model path priority (first found is used):
# 1. models/model_20d_v3.pkl (latest trained model)
# 2. ml/bundles/latest/model.joblib (production bundle)
# 3. model_xgboost_5d.pkl (legacy fallback)

def _find_model_path() -> Optional[str]:
    """Find the best available model file."""
    project_root = Path(__file__).resolve().parents[1]

    candidates = [
        project_root / "models" / "model_20d_v3.pkl",
        project_root / "ml" / "bundles" / "latest" / "model.joblib",
        project_root / "model_xgboost_5d.pkl",  # Legacy
    ]

    for path in candidates:
        if path.exists():
            logger.info(f"Found ML model at: {path}")
            return str(path)

    logger.warning("No ML model found in any expected location")
    return None


_MODEL_PATH = _find_model_path()


def validate_model_features(model, expected_features: List[str]) -> bool:
    """
    Validate that loaded model matches expected features.
    
    Args:
        model: Loaded model object
        expected_features: List of feature names the model should accept
    
    Returns:
        True if model accepts the expected features, False otherwise
    """
    try:
        # Create dummy input with expected features
        dummy_input = pd.DataFrame([{f: 0.0 for f in expected_features}])

        # Try to predict - will fail if features don't match
        model.predict_proba(dummy_input)
        return True
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return False


def load_ml_model(model_path: Optional[str] = None) -> bool:
    """
    Load the ML model from disk with validation.
    
    Supports multiple model formats:
    - joblib (preferred, newer format)
    - pickle (legacy format)
    - bundled models (dict with 'model' and 'feature_names' keys)
    
    Args:
        model_path: Path to model file (default: auto-discovered)
    
    Returns:
        True if loaded successfully, False otherwise
    """
    global _ML_MODEL, _MODEL_LOADED, _MODEL_PATH, _MODEL_BUNDLE, EXPECTED_FEATURES

    if model_path:
        _MODEL_PATH = model_path
    elif _MODEL_PATH is None:
        _MODEL_PATH = _find_model_path()

    if _MODEL_LOADED:
        return _ML_MODEL is not None

    if _MODEL_PATH is None:
        logger.warning("No ML model file found in any expected location")
        _MODEL_LOADED = True
        return False

    try:
        # Try joblib first (newer format), fall back to pickle
        try:
            loaded = joblib.load(_MODEL_PATH)
            logger.debug(f"Loaded model using joblib from {_MODEL_PATH}")
        except Exception:
            with open(_MODEL_PATH, "rb") as f:
                loaded = pickle.load(f)
            logger.debug(f"Loaded model using pickle from {_MODEL_PATH}")

        # Handle bundled models (dict with 'model' key)
        if isinstance(loaded, dict) and 'model' in loaded:
            logger.info(f"Loaded model bundle with keys: {list(loaded.keys())}")
            _MODEL_BUNDLE = loaded
            _ML_MODEL = loaded['model']
            
            # Extract feature names from bundle
            if 'feature_names' in loaded:
                EXPECTED_FEATURES = list(loaded['feature_names'])
                logger.info(f"Loaded {len(EXPECTED_FEATURES)} features from model bundle")
            elif hasattr(_ML_MODEL, 'feature_names_in_'):
                EXPECTED_FEATURES = list(_ML_MODEL.feature_names_in_)
                logger.info(f"Loaded {len(EXPECTED_FEATURES)} features from model attribute")
            else:
                EXPECTED_FEATURES = DEFAULT_FEATURES.copy()
                logger.warning(f"No feature names in bundle, using defaults: {len(EXPECTED_FEATURES)} features")
        else:
            # Plain model (not bundled)
            _ML_MODEL = loaded
            _MODEL_BUNDLE = None
            if hasattr(_ML_MODEL, 'feature_names_in_'):
                EXPECTED_FEATURES = list(_ML_MODEL.feature_names_in_)
            else:
                EXPECTED_FEATURES = DEFAULT_FEATURES.copy()
                logger.warning("Model has no feature_names_in_, using defaults")

        # Validate the model against expected features
        if not validate_model_features(_ML_MODEL, EXPECTED_FEATURES):
            logger.error("Model features don't match expected features!")
            logger.error(f"Expected {len(EXPECTED_FEATURES)} features: {EXPECTED_FEATURES[:5]}...")
            _ML_MODEL = None
            _MODEL_LOADED = True
            return False

        logger.info(f"✓ ML model loaded from {_MODEL_PATH}")
        logger.info(f"  Model type: {type(_ML_MODEL).__name__}")
        logger.info(f"  Features: {len(EXPECTED_FEATURES)}")
        _MODEL_LOADED = True
        return True

    except Exception as e:
        logger.error(f"Failed to load ML model: {e}")
        _ML_MODEL = None
        _MODEL_LOADED = True
        return False


def get_model_info() -> Dict[str, Any]:
    """
    Get information about the currently loaded model.
    
    Returns:
        Dict with model status, path, feature count, type, and features
    """
    if not _MODEL_LOADED:
        load_ml_model()

    return {
        "loaded": _ML_MODEL is not None,
        "path": _MODEL_PATH,
        "feature_count": len(EXPECTED_FEATURES) if _ML_MODEL else 0,
        "model_type": type(_ML_MODEL).__name__ if _ML_MODEL else None,
        "features": EXPECTED_FEATURES if _ML_MODEL else [],
        "metrics": _MODEL_BUNDLE.get("metrics") if _MODEL_BUNDLE else None,
        "trained_at": _MODEL_BUNDLE.get("trained_at") if _MODEL_BUNDLE else None,
    }


def get_expected_features() -> List[str]:
    """
    Get the list of expected features for ML predictions.
    
    Ensures model is loaded first so feature list is populated from bundle.
    
    Returns:
        List of feature names the model expects
    """
    if not _MODEL_LOADED:
        load_ml_model()
    return EXPECTED_FEATURES.copy() if EXPECTED_FEATURES else DEFAULT_FEATURES.copy()


def get_ml_prediction(features: Dict[str, float]) -> Optional[float]:
    """
    Get ML prediction probability from features.
    
    Args:
        features: Dict of feature names to values (must match training features)
    
    Returns:
        Probability 0-1, or None if model unavailable or prediction fails
    """
    global _ML_MODEL, _MODEL_LOADED
    
    # Ensure model is loaded
    if not _MODEL_LOADED:
        load_ml_model()
    
    if _ML_MODEL is None:
        return None
    
    try:
        # Convert features dict to DataFrame (single row)
        # The model expects specific feature names from training
        df = pd.DataFrame([features])
        
        # Get prediction probability
        # For XGBoost binary classifier: predict_proba returns [[prob_class_0, prob_class_1]]
        proba = _ML_MODEL.predict_proba(df)
        
        if proba is None or len(proba) == 0:
            return None
        
        # Return probability of positive class (win)
        prob_win = float(proba[0][1])
        
        # Sanity check
        if not (0 <= prob_win <= 1):
            logger.warning(f"ML probability out of range: {prob_win}")
            return None
        
        # NOTE: Return raw model probabilities. Proper probability calibration
        # (e.g., Platt scaling, isotonic regression) should be applied during
        # model training, not at inference time. See train_rolling_ml_20d.py.
        
        return prob_win
        
    except Exception as e:
        logger.warning(f"ML prediction failed: {e}")
        return None


def calculate_ml_boost(
    base_conviction: float,
    ml_probability: Optional[float],
    max_boost_pct: float = 10.0
) -> Tuple[float, float, str]:
    """
    Calculate bounded ML boost to base conviction.
    
    Rules:
    - ML can adjust final conviction by at most ±max_boost_pct% of base
    - If ML unavailable, boost = 0 (graceful fallback)
    - ML probability 0.5 = neutral (0 boost)
    - ML probability 1.0 = +max_boost_pct
    - ML probability 0.0 = -max_boost_pct
    
    Args:
        base_conviction: Base conviction score (0-100)
        ml_probability: ML probability (0-1), or None if unavailable
        max_boost_pct: Maximum boost as percentage of base (default 10%)
    
    Returns:
        (final_conviction, ml_boost, status_msg)
        - final_conviction: Adjusted score (0-100)
        - ml_boost: The adjustment applied (-max to +max)
        - status_msg: Description of boost
    """
    # Validate base conviction
    base_conviction = float(np.clip(base_conviction, 0, 100))
    
    # If ML unavailable, return base unchanged
    if ml_probability is None:
        return base_conviction, 0.0, "ML unavailable - using base conviction"
    
    # Validate ML probability
    if not np.isfinite(ml_probability):
        return base_conviction, 0.0, "ML probability invalid"
    
    ml_probability = float(np.clip(ml_probability, 0, 1))
    
    # Calculate boost
    # ML prob 0.5 → 0 boost (neutral)
    # ML prob 1.0 → +max_boost
    # ML prob 0.0 → -max_boost
    boost_fraction = (ml_probability - 0.5) * 2.0  # Range: -1 to +1
    ml_boost = boost_fraction * max_boost_pct
    
    # Apply boost
    final_conviction = base_conviction + ml_boost
    final_conviction = float(np.clip(final_conviction, 0, 100))
    
    # Status message
    if ml_boost > 5:
        status = f"ML boost: +{ml_boost:.1f} (high confidence)"
    elif ml_boost > 0:
        status = f"ML boost: +{ml_boost:.1f} (slight positive)"
    elif ml_boost < -5:
        status = f"ML penalty: {ml_boost:.1f} (low confidence)"
    elif ml_boost < 0:
        status = f"ML penalty: {ml_boost:.1f} (slight negative)"
    else:
        status = "ML neutral (no adjustment)"
    
    return final_conviction, ml_boost, status


def prepare_ml_features(
    ticker_data: Dict,
    technical_indicators: Dict,
    fundamental_scores: Dict
) -> Dict[str, float]:
    """
    Prepare feature dict for ML model from ticker data.
    
    CRITICAL: This function must generate EXACTLY the 34 features used during
    model training in scripts/train_rolling_ml_20d.py. Feature names and order
    must match EXPECTED_FEATURES.
    
    Expected Features (34 total):
    ─────────────────────────────────────────────────────────────────────────
    1.  RSI                  - Relative Strength Index (0-100), default 50
    2.  ATR_Pct              - ATR as % of Close price, default 0.02
    3.  Return_20d           - 20-day return, default 0.0
    4.  Return_10d           - 10-day return, default 0.0  
    5.  Return_5d            - 5-day return, default 0.0
    6.  VCP_Ratio            - ATR(10)/ATR(30), default 1.0
    7.  Tightness_Ratio      - range_5d/range_20d, default 1.0
    8.  Dist_From_52w_High   - (Close/52w_High)-1, default -0.1
    9.  MA_Alignment         - 1 if Close>MA20>MA50>MA200, else 0
    10. Volume_Surge         - vol_5d/vol_20d, default 1.0
    11. Up_Down_Volume_Ratio - up-day vol/down-day vol, default 1.0
    12. Momentum_Consistency - % positive days (last 20), default 0.5
    13. RS_vs_SPY_20d        - stock_ret_20d - spy_ret_20d, default 0.0
    14. Market_Regime        - Bull(1), Sideways(0), Bear(-1), default 0
    15. Market_Volatility    - SPY 20d volatility annualized, default 0.15
    16. Market_Trend         - SPY 50d return, default 0.0
    17. High_Volatility      - 1 if vol > 75th pct, default 0
    18. Sector_RS            - stock_ret_20d - sector_ret_20d, default 0.0
    19. Sector_Momentum      - sector_etf_ret_20d, default 0.0
    20. Sector_Rank          - 1 if stock beats sector in 5d, default 0.5
    21. Volume_Ratio_20d     - current vol / 20d avg (spike), default 1.0
    22. Volume_Trend         - volume slope (accumulation), default 0.0
    23. Up_Volume_Ratio      - up-day vol / total vol, default 0.5
    24. Volume_Price_Confirm - price+vol up confirmation, default 0.5
    25. Relative_Volume_Rank - vol percentile vs 60d, default 0.5
    26. Distance_From_52w_Low - (close-52w_low)/52w_low, default 0.5
    27. Consolidation_Tightness - (20d_high-low)/avg, default 0.1
    28. Days_Since_52w_High  - normalized 0-1, default 0.5
    29. Price_vs_SMA50       - (close-sma50)/sma50, default 0.0
    30. Price_vs_SMA200      - (close-sma200)/sma200, default 0.0
    31. SMA50_vs_SMA200      - (sma50-sma200)/sma200, default 0.0
    32. MA_Slope_20d         - slope of 20d MA, default 0.0
    33. Distance_To_Resistance - (20d_high-close)/close, default 0.05
    34. Support_Strength     - fraction days near support, default 0.2
    ─────────────────────────────────────────────────────────────────────────
    
    Args:
        ticker_data: Raw ticker data dict (may contain OHLCV, price history,
                     market context like spy_return_20d, sector data, etc.)
        technical_indicators: Technical indicators dict (rsi, atr, atr_pct, etc.)
        fundamental_scores: Fundamental scores dict (not used for these features)
    
    Returns:
        Feature dict with exactly 34 keys matching EXPECTED_FEATURES, 
        in the same order as training.
    """
    features = {}
    
    # Helper to safely get numeric value with default
    def safe_get(d: Dict, keys: list, default: float) -> float:
        """Try multiple keys, return first valid numeric value or default."""
        for key in keys:
            val = d.get(key)
            if val is not None:
                try:
                    fval = float(val)
                    if np.isfinite(fval):
                        return fval
                except (TypeError, ValueError):
                    continue
        return default
    
    # Merge all inputs for easier lookup
    all_data = {**ticker_data, **technical_indicators, **fundamental_scores}
    
    # =========================================================================
    # 1. RSI - Relative Strength Index (0-100)
    # =========================================================================
    features['RSI'] = safe_get(all_data, ['RSI', 'rsi', 'Rsi'], 50.0)
    # Clamp to valid RSI range
    features['RSI'] = float(np.clip(features['RSI'], 0, 100))
    
    # =========================================================================
    # 2. ATR_Pct - Average True Range as percentage of Close
    # =========================================================================
    atr_pct = safe_get(all_data, ['ATR_Pct', 'atr_pct', 'ATR_pct'], None)
    if atr_pct is None:
        # Try to compute from ATR and Close
        atr = safe_get(all_data, ['ATR', 'atr'], None)
        close = safe_get(all_data, ['Close', 'close', 'price', 'Price'], None)
        if atr is not None and close is not None and close > 0:
            atr_pct = atr / close
        else:
            atr_pct = 0.02  # Default ~2% volatility
    features['ATR_Pct'] = float(np.clip(atr_pct, 0.001, 0.5))
    
    # =========================================================================
    # 3-5. Return_20d, Return_10d, Return_5d - Price returns
    # =========================================================================
    features['Return_20d'] = safe_get(all_data, ['Return_20d', 'return_20d', 'ret_20d', 'return_20'], 0.0)
    features['Return_10d'] = safe_get(all_data, ['Return_10d', 'return_10d', 'ret_10d', 'return_10'], 0.0)
    features['Return_5d'] = safe_get(all_data, ['Return_5d', 'return_5d', 'ret_5d', 'return_5'], 0.0)
    
    # Clamp returns to reasonable range (-100% to +200%)
    for ret_key in ['Return_20d', 'Return_10d', 'Return_5d']:
        features[ret_key] = float(np.clip(features[ret_key], -1.0, 2.0))
    
    # =========================================================================
    # 6. VCP_Ratio - Volatility Contraction Pattern: ATR(10) / ATR(30)
    # =========================================================================
    vcp = safe_get(all_data, ['VCP_Ratio', 'vcp_ratio', 'VCP', 'vcp'], None)
    if vcp is None:
        # Try to compute from component ATRs
        atr_10 = safe_get(all_data, ['ATR_10', 'atr_10', 'atr10'], None)
        atr_30 = safe_get(all_data, ['ATR_30', 'atr_30', 'atr30'], None)
        if atr_10 is not None and atr_30 is not None and atr_30 > 0:
            vcp = atr_10 / atr_30
        else:
            vcp = 1.0  # Neutral default
    features['VCP_Ratio'] = float(np.clip(vcp, 0.1, 5.0))
    
    # =========================================================================
    # 7. Tightness_Ratio - range_5d / range_20d (price consolidation)
    # =========================================================================
    tightness = safe_get(all_data, ['Tightness_Ratio', 'tightness_ratio', 'tightness'], None)
    if tightness is None:
        range_5d = safe_get(all_data, ['range_5d', 'Range_5d'], None)
        range_20d = safe_get(all_data, ['range_20d', 'Range_20d'], None)
        if range_5d is not None and range_20d is not None and range_20d > 0:
            tightness = range_5d / range_20d
        else:
            tightness = 1.0  # Neutral default
    features['Tightness_Ratio'] = float(np.clip(tightness, 0.05, 2.0))
    
    # =========================================================================
    # 8. Dist_From_52w_High - (Close / 52-week High) - 1
    # =========================================================================
    dist_high = safe_get(all_data, ['Dist_From_52w_High', 'dist_from_52w_high', 
                                     'dist_52w_high', 'near_high_pct'], None)
    if dist_high is None:
        # Try to compute from Close and 52w High
        close = safe_get(all_data, ['Close', 'close', 'price', 'Price'], None)
        high_52w = safe_get(all_data, ['High_52w', 'high_52w', '52w_high', 'year_high'], None)
        if close is not None and high_52w is not None and high_52w > 0:
            dist_high = (close / high_52w) - 1.0
        else:
            # Check if near_high is available (percentage from high, e.g., 95 = 5% below)
            near_high = safe_get(all_data, ['near_high', 'Near_High'], None)
            if near_high is not None:
                dist_high = (near_high / 100.0) - 1.0  # Convert 95% -> -0.05
            else:
                dist_high = -0.1  # Default: 10% below 52w high
    features['Dist_From_52w_High'] = float(np.clip(dist_high, -0.8, 0.1))
    
    # =========================================================================
    # 9. MA_Alignment - 1 if Close > MA20 > MA50 > MA200, else 0
    # =========================================================================
    ma_align = safe_get(all_data, ['MA_Alignment', 'ma_alignment', 'ma_aligned'], None)
    if ma_align is None:
        close = safe_get(all_data, ['Close', 'close', 'price', 'Price'], None)
        ma20 = safe_get(all_data, ['MA20', 'ma20', 'sma_20', 'SMA_20'], None)
        ma50 = safe_get(all_data, ['MA50', 'ma50', 'sma_50', 'SMA_50'], None)
        ma200 = safe_get(all_data, ['MA200', 'ma200', 'sma_200', 'SMA_200'], None)
        
        if all(v is not None for v in [close, ma20, ma50, ma200]):
            ma_align = 1.0 if (close > ma20 > ma50 > ma200) else 0.0
        else:
            ma_align = 0.0  # Default: not aligned
    features['MA_Alignment'] = 1.0 if ma_align > 0.5 else 0.0
    
    # =========================================================================
    # 10. Volume_Surge - vol_5d_avg / vol_20d_avg
    # =========================================================================
    vol_surge = safe_get(all_data, ['Volume_Surge', 'volume_surge', 'VolSurge', 
                                     'vol_surge', 'volume_ratio'], None)
    if vol_surge is None:
        vol_5d = safe_get(all_data, ['vol_5d', 'Vol_5d', 'volume_5d'], None)
        vol_20d = safe_get(all_data, ['vol_20d', 'Vol_20d', 'volume_20d'], None)
        if vol_5d is not None and vol_20d is not None and vol_20d > 0:
            vol_surge = vol_5d / vol_20d
        else:
            vol_surge = 1.0  # Neutral default
    features['Volume_Surge'] = float(np.clip(vol_surge, 0.1, 10.0))
    
    # =========================================================================
    # 11. Up_Down_Volume_Ratio - avg volume on up days / avg volume on down days
    # =========================================================================
    ud_vol = safe_get(all_data, ['Up_Down_Volume_Ratio', 'up_down_volume_ratio',
                                  'ud_vol_ratio', 'updown_volume'], 1.0)
    features['Up_Down_Volume_Ratio'] = float(np.clip(ud_vol, 0.1, 10.0))
    
    # =========================================================================
    # 12. Momentum_Consistency - % of positive days in last 20
    # =========================================================================
    mom_cons = safe_get(all_data, ['Momentum_Consistency', 'momentum_consistency',
                                    'MomCons', 'mom_consistency', 'pct_positive_days'], 0.5)
    features['Momentum_Consistency'] = float(np.clip(mom_cons, 0.0, 1.0))
    
    # =========================================================================
    # 13. RS_vs_SPY_20d - stock_return_20d - spy_return_20d (relative strength)
    # =========================================================================
    rs_spy = safe_get(all_data, ['RS_vs_SPY_20d', 'rs_vs_spy_20d', 'relative_strength_20d',
                                  'Relative_Strength_20d', 'rs_20d'], None)
    if rs_spy is None:
        # Try to compute from stock return and SPY return
        stock_ret = features['Return_20d']  # Already computed above
        spy_ret = safe_get(all_data, ['spy_return_20d', 'SPY_Return_20d', 'spy_ret_20d'], 0.0)
        rs_spy = stock_ret - spy_ret
    features['RS_vs_SPY_20d'] = float(np.clip(rs_spy, -1.0, 1.0))
    
    # =========================================================================
    # 14. Market_Regime - Bull (1), Sideways (0), Bear (-1)
    # =========================================================================
    market_regime = safe_get(all_data, ['Market_Regime', 'market_regime'], 0.0)
    features['Market_Regime'] = float(np.clip(market_regime, -1, 1))
    
    # =========================================================================
    # 15. Market_Volatility - SPY 20d volatility (annualized)
    # =========================================================================
    market_vol = safe_get(all_data, ['Market_Volatility', 'market_volatility', 
                                      'vix', 'spy_volatility'], 0.15)
    features['Market_Volatility'] = float(np.clip(market_vol, 0.05, 0.80))
    
    # =========================================================================
    # 16. Market_Trend - SPY 50d return (momentum)
    # =========================================================================
    market_trend = safe_get(all_data, ['Market_Trend', 'market_trend', 
                                        'spy_trend', 'spy_50d_return'], 0.0)
    features['Market_Trend'] = float(np.clip(market_trend, -0.5, 0.5))
    
    # =========================================================================
    # 17. High_Volatility - 1 if volatility > 75th percentile
    # =========================================================================
    high_vol = safe_get(all_data, ['High_Volatility', 'high_volatility', 'high_vol'], 0.0)
    features['High_Volatility'] = 1.0 if high_vol > 0.5 else 0.0
    
    # =========================================================================
    # 18. Sector_RS - stock_return_20d - sector_etf_return_20d
    # =========================================================================
    sector_rs = safe_get(all_data, ['Sector_RS', 'sector_rs', 'sector_relative_strength'], None)
    if sector_rs is None:
        # Try to compute from stock return and sector return
        stock_ret = features['Return_20d']
        sector_ret = safe_get(all_data, ['sector_return_20d', 'Sector_Return_20d', 
                                          'sector_ret_20d'], 0.0)
        sector_rs = stock_ret - sector_ret
    features['Sector_RS'] = float(np.clip(sector_rs, -1.0, 1.0))
    
    # =========================================================================
    # 19. Sector_Momentum - sector_etf_return_20d (absolute sector strength)
    # =========================================================================
    sector_mom = safe_get(all_data, ['Sector_Momentum', 'sector_momentum', 
                                      'sector_return_20d', 'Sector_Return_20d'], 0.0)
    features['Sector_Momentum'] = float(np.clip(sector_mom, -0.5, 0.5))
    
    # =========================================================================
    # 20. Sector_Rank - 1 if stock beats sector in 5d return, else 0
    # =========================================================================
    sector_rank = safe_get(all_data, ['Sector_Rank', 'sector_rank'], None)
    if sector_rank is None:
        # Try to compute from 5d returns
        stock_ret_5d = features['Return_5d']
        sector_ret_5d = safe_get(all_data, ['sector_return_5d', 'Sector_Return_5d'], 0.0)
        sector_rank = 1.0 if stock_ret_5d > sector_ret_5d else 0.0
    features['Sector_Rank'] = float(np.clip(sector_rank, 0.0, 1.0))
    
    # =========================================================================
    # 21. Volume_Ratio_20d - current volume / 20d avg (spike detection)
    # =========================================================================
    vol_ratio = safe_get(all_data, ['Volume_Ratio_20d', 'volume_ratio_20d', 'vol_ratio'], None)
    if vol_ratio is None:
        current_vol = safe_get(all_data, ['Volume', 'volume', 'vol'], None)
        vol_20d_avg = safe_get(all_data, ['vol_20d', 'Vol_20d', 'volume_20d', 'avg_volume_20d'], None)
        if current_vol is not None and vol_20d_avg is not None and vol_20d_avg > 0:
            vol_ratio = current_vol / vol_20d_avg
        else:
            vol_ratio = 1.0
    features['Volume_Ratio_20d'] = float(np.clip(vol_ratio, 0.1, 20.0))
    
    # =========================================================================
    # 22. Volume_Trend - slope of volume over 20 days (accumulation trend)
    # =========================================================================
    vol_trend = safe_get(all_data, ['Volume_Trend', 'volume_trend', 'vol_trend'], 0.0)
    features['Volume_Trend'] = float(np.clip(vol_trend, -1.0, 1.0))
    
    # =========================================================================
    # 23. Up_Volume_Ratio - up-day volume / total volume (buying pressure)
    # =========================================================================
    up_vol_ratio = safe_get(all_data, ['Up_Volume_Ratio', 'up_volume_ratio', 
                                        'buying_pressure'], 0.5)
    features['Up_Volume_Ratio'] = float(np.clip(up_vol_ratio, 0.0, 1.0))
    
    # =========================================================================
    # 24. Volume_Price_Confirm - price up AND volume up confirmation
    # =========================================================================
    vol_price_confirm = safe_get(all_data, ['Volume_Price_Confirm', 'volume_price_confirm',
                                             'vol_price_confirmation'], 0.5)
    features['Volume_Price_Confirm'] = float(np.clip(vol_price_confirm, 0.0, 1.0))
    
    # =========================================================================
    # 25. Relative_Volume_Rank - percentile rank of volume vs 60 days
    # =========================================================================
    rel_vol_rank = safe_get(all_data, ['Relative_Volume_Rank', 'relative_volume_rank',
                                        'vol_percentile'], 0.5)
    features['Relative_Volume_Rank'] = float(np.clip(rel_vol_rank, 0.0, 1.0))
    
    # =========================================================================
    # 26. Distance_From_52w_Low - (close - 52w_low) / 52w_low
    # =========================================================================
    dist_low = safe_get(all_data, ['Distance_From_52w_Low', 'distance_from_52w_low',
                                    'dist_52w_low'], None)
    if dist_low is None:
        close = safe_get(all_data, ['Close', 'close', 'price', 'Price'], None)
        low_52w = safe_get(all_data, ['Low_52w', 'low_52w', '52w_low', 'year_low'], None)
        if close is not None and low_52w is not None and low_52w > 0:
            dist_low = (close - low_52w) / low_52w
        else:
            dist_low = 0.5  # Default: 50% above 52w low
    features['Distance_From_52w_Low'] = float(np.clip(dist_low, -0.5, 5.0))
    
    # =========================================================================
    # 27. Consolidation_Tightness - (20d_high - 20d_low) / 20d_avg
    # =========================================================================
    consol_tight = safe_get(all_data, ['Consolidation_Tightness', 'consolidation_tightness',
                                        'price_range_pct'], 0.1)
    features['Consolidation_Tightness'] = float(np.clip(consol_tight, 0.01, 0.5))
    
    # =========================================================================
    # 28. Days_Since_52w_High - normalized 0-1
    # =========================================================================
    days_high = safe_get(all_data, ['Days_Since_52w_High', 'days_since_52w_high',
                                     'days_from_high'], 0.5)
    features['Days_Since_52w_High'] = float(np.clip(days_high, 0.0, 1.0))
    
    # =========================================================================
    # 29. Price_vs_SMA50 - (close - sma50) / sma50
    # =========================================================================
    price_sma50 = safe_get(all_data, ['Price_vs_SMA50', 'price_vs_sma50'], None)
    if price_sma50 is None:
        close = safe_get(all_data, ['Close', 'close', 'price', 'Price'], None)
        sma50 = safe_get(all_data, ['SMA50', 'sma50', 'sma_50', 'MA50', 'ma50'], None)
        if close is not None and sma50 is not None and sma50 > 0:
            price_sma50 = (close - sma50) / sma50
        else:
            price_sma50 = 0.0
    features['Price_vs_SMA50'] = float(np.clip(price_sma50, -0.5, 0.5))
    
    # =========================================================================
    # 30. Price_vs_SMA200 - (close - sma200) / sma200
    # =========================================================================
    price_sma200 = safe_get(all_data, ['Price_vs_SMA200', 'price_vs_sma200'], None)
    if price_sma200 is None:
        close = safe_get(all_data, ['Close', 'close', 'price', 'Price'], None)
        sma200 = safe_get(all_data, ['SMA200', 'sma200', 'sma_200', 'MA200', 'ma200'], None)
        if close is not None and sma200 is not None and sma200 > 0:
            price_sma200 = (close - sma200) / sma200
        else:
            price_sma200 = 0.0
    features['Price_vs_SMA200'] = float(np.clip(price_sma200, -0.5, 0.5))
    
    # =========================================================================
    # 31. SMA50_vs_SMA200 - (sma50 - sma200) / sma200 (golden/death cross)
    # =========================================================================
    sma_ratio = safe_get(all_data, ['SMA50_vs_SMA200', 'sma50_vs_sma200', 'ma_cross'], None)
    if sma_ratio is None:
        sma50 = safe_get(all_data, ['SMA50', 'sma50', 'sma_50', 'MA50', 'ma50'], None)
        sma200 = safe_get(all_data, ['SMA200', 'sma200', 'sma_200', 'MA200', 'ma200'], None)
        if sma50 is not None and sma200 is not None and sma200 > 0:
            sma_ratio = (sma50 - sma200) / sma200
        else:
            sma_ratio = 0.0
    features['SMA50_vs_SMA200'] = float(np.clip(sma_ratio, -0.3, 0.3))
    
    # =========================================================================
    # 32. MA_Slope_20d - slope of 20d moving average
    # =========================================================================
    ma_slope = safe_get(all_data, ['MA_Slope_20d', 'ma_slope_20d', 'ma_slope', 'trend_slope'], 0.0)
    features['MA_Slope_20d'] = float(np.clip(ma_slope, -0.3, 0.3))
    
    # =========================================================================
    # 33. Distance_To_Resistance - (20d_high - close) / close
    # =========================================================================
    dist_resist = safe_get(all_data, ['Distance_To_Resistance', 'distance_to_resistance',
                                       'dist_resistance'], None)
    if dist_resist is None:
        close = safe_get(all_data, ['Close', 'close', 'price', 'Price'], None)
        high_20d = safe_get(all_data, ['High_20d', 'high_20d', '20d_high'], None)
        if close is not None and high_20d is not None and close > 0:
            dist_resist = (high_20d - close) / close
        else:
            dist_resist = 0.05  # Default: 5% to resistance
    features['Distance_To_Resistance'] = float(np.clip(dist_resist, 0.0, 0.5))
    
    # =========================================================================
    # 34. Support_Strength - fraction of days near 20d support
    # =========================================================================
    support_str = safe_get(all_data, ['Support_Strength', 'support_strength',
                                       'support_touches'], 0.2)
    features['Support_Strength'] = float(np.clip(support_str, 0.0, 1.0))
    
    # =========================================================================
    # VALIDATION: Ensure we have exactly 34 features in the correct order
    # =========================================================================
    ordered_features = {}
    missing_features = []
    
    for feat_name in EXPECTED_FEATURES:
        if feat_name in features:
            val = features[feat_name]
            # Final sanity check: ensure finite
            if not np.isfinite(val):
                logger.warning(f"Feature '{feat_name}' is not finite ({val}), using 0.0")
                val = 0.0
            ordered_features[feat_name] = float(val)
        else:
            missing_features.append(feat_name)
            ordered_features[feat_name] = 0.0
    
    if missing_features:
        logger.warning(f"Missing features (using defaults): {missing_features}")
    
    # Verify count
    if len(ordered_features) != len(EXPECTED_FEATURES):
        logger.error(f"Feature count mismatch! Expected {len(EXPECTED_FEATURES)}, got {len(ordered_features)}")
    
    return ordered_features


def integrate_ml_with_conviction(
    base_conviction: float,
    ticker_data: Dict,
    technical_indicators: Dict,
    fundamental_scores: Dict,
    enable_ml: bool = True
) -> Tuple[float, Dict[str, any]]:
    """
    High-level integration: Get ML prediction and apply bounded boost.
    
    Args:
        base_conviction: Base conviction from scoring_engine (0-100)
        ticker_data: Raw ticker data
        technical_indicators: Technical indicators dict
        fundamental_scores: Fundamental scores dict
        enable_ml: Whether to use ML (can be toggled off)
    
    Returns:
        (final_conviction, ml_info_dict)
        ml_info_dict contains: ml_probability, ml_boost, ml_status, ml_enabled
    """
    ml_info = {
        "ml_enabled": enable_ml,
        "ml_probability": 0.5,
        "ml_boost": 0.0,
        "ml_status": "Disabled",
        "final_conviction": base_conviction
    }
    if not enable_ml:
        return base_conviction, ml_info
    try:
        # Defensive: If model is a dict, treat as unavailable
        if not _MODEL_LOADED:
            load_ml_model()
        model = _ML_MODEL
        if isinstance(model, dict):
            ml_info["ml_status"] = "ML unavailable (dict instead of model)"
            return base_conviction, ml_info
        if model is None:
            ml_info["ml_status"] = "Model not available"
            return base_conviction, ml_info
        features = prepare_ml_features(ticker_data, technical_indicators, fundamental_scores)
        ml_probability = get_ml_prediction(features)
        if ml_probability is None:
            ml_info["ml_status"] = "Prediction failed"
            return base_conviction, ml_info
        final_conviction, ml_boost, status = calculate_ml_boost(base_conviction, ml_probability)
        ml_info["ml_probability"] = ml_probability
        ml_info["ml_boost"] = ml_boost
        ml_info["ml_status"] = status
        ml_info["final_conviction"] = final_conviction
        return final_conviction, ml_info
    except Exception as e:
        print(f"ML prediction failed: {e}")
        ml_info["ml_status"] = "ML error, using neutral"
        ml_info["ml_probability"] = 0.5
        ml_info["ml_boost"] = 0.0
        ml_info["final_conviction"] = base_conviction
        return base_conviction, ml_info


# Pre-load model on module import (optional, can be disabled)
def auto_load_model() -> None:
    """Attempt to auto-load model on module import."""
    try:
        load_ml_model()
    except:
        pass

# Uncomment to auto-load:
# auto_load_model()
