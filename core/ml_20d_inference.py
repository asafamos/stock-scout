import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)

ML_20D_AVAILABLE = True
BUNDLE_MODEL: Any = None
FEATURE_COLS_20D: list[str] = []
PREFERRED_SCORING_MODE_20D: str = "hybrid"  # Default fallback

# Try Streamlit caching if available; otherwise use functools.lru_cache
try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False


def _load_bundle_impl() -> tuple[bool, Any, list[str], str]:
    """Load model bundle from absolute path for Streamlit Cloud compatibility."""
    try:
        # Use absolute path relative to this file's location
        module_dir = Path(__file__).resolve().parent.parent  # stock-scout-2 root
        # Prefer v3 model; fallback to v2, then v1
        # Prefer newest timestamped v3 bundle if present
        models_dir = module_dir / "models"
        model_path_v3 = models_dir / "model_20d_v3.pkl"
        try:
            candidates = sorted(models_dir.glob("model_20d_v3_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                model_path_v3 = candidates[0]
        except Exception:
            pass
        model_path_v2 = module_dir / "models" / "model_20d_v2.pkl"
        model_path_v1 = module_dir / "models" / "model_20d_v1.pkl"
        
        if model_path_v3.exists():
            model_path = model_path_v3
        elif model_path_v2.exists():
            logger.warning(f"Model v3 not found, falling back to v2 at {model_path_v2}")
            model_path = model_path_v2
        elif model_path_v1.exists():
            logger.warning(f"Model v2 not found, falling back to v1 at {model_path_v1}")
            model_path = model_path_v1
        else:
            logger.warning(f"No ML 20d model found at {model_path_v3}, {model_path_v2}, or {model_path_v1}")
            return False, None, [], "hybrid"
        
        if not model_path.exists():
            logger.warning(f"Model file not found at {model_path}")
            return False, None, [], "hybrid"
        
        # Load bundle using joblib (compatible with sklearn models)
        bundle = joblib.load(model_path)
        
        # Validate bundle structure
        if not isinstance(bundle, dict):
            logger.error(f"Model bundle is not a dict, got {type(bundle)}")
            return False, None, [], "hybrid"
        
        model = bundle.get("model")
        feature_names = bundle.get("feature_names", [])
        preferred_scoring_mode = bundle.get("preferred_scoring_mode_20d", "hybrid")
        
        # Validate model has predict_proba
        if model is None or not hasattr(model, "predict_proba"):
            logger.error("Model missing or lacks predict_proba method")
            return False, None, [], "hybrid"
        
        # Validate feature names
        if not isinstance(feature_names, (list, tuple)):
            logger.error(f"feature_names is not a list, got {type(feature_names)}")
            return False, None, [], "hybrid"
        
        feature_names = list(feature_names)
        if len(feature_names) == 0:
            logger.error("feature_names is empty")
            return False, None, [], "hybrid"
        
        logger.info(f"✓ Loaded ML 20d model with {len(feature_names)} features")
        logger.info(f"✓ Preferred scoring mode: {preferred_scoring_mode}")
        return True, model, feature_names, preferred_scoring_mode
    except Exception as e:
        logger.error(f"Failed to load ML bundle: {e}")
        return False, None, [], "hybrid"


# Conditional caching based on Streamlit availability
if _STREAMLIT_AVAILABLE:
    # Add version key to bust cache when return signature changes
    @st.cache_resource(hash_funcs={type: id})
    def _load_bundle_cached(cache_version: int = 2):
        """Cache version 2: returns (success, model, features, scoring_mode)."""
        return _load_bundle_impl()
    
    _success, BUNDLE_MODEL, FEATURE_COLS_20D, PREFERRED_SCORING_MODE_20D = _load_bundle_cached()
    ML_20D_AVAILABLE = _success
else:
    from functools import lru_cache
    
    @lru_cache(maxsize=1)
    def _load_bundle_cached():
        return _load_bundle_impl()
    
    _success, BUNDLE_MODEL, FEATURE_COLS_20D, PREFERRED_SCORING_MODE_20D = _load_bundle_cached()
    ML_20D_AVAILABLE = _success

def compute_ml_20d_probabilities_raw(row: pd.Series) -> float:
    """
    Compute RAW ML 20d probability from GradientBoosting model.
    This is the base signal before any live adjustments.
    
    Returns:
        - float in [0, 1]: raw positive-class probability if model available
        - np.nan: if model unavailable, missing features, or prediction fails
    """
    if not ML_20D_AVAILABLE or BUNDLE_MODEL is None or not FEATURE_COLS_20D:
        return np.nan
    
    try:
        # Build feature dict with exact columns from training, in exact order
        feature_dict = {}
        missing_features = []
        
        for col in FEATURE_COLS_20D:
            val = row.get(col, np.nan)
            if not isinstance(val, (int, float)) or np.isnan(val):
                missing_features.append(col)
                val = np.nan
            feature_dict[col] = val
        
        # Log missing features for debugging
        if missing_features:
            logger.debug(f"ML 20d: missing features {missing_features} (using fillna=0)")
        
        # Build DataFrame in exact feature order
        X = pd.DataFrame([feature_dict])
        
        # Reorder columns to match training order (just in case)
        X = X[FEATURE_COLS_20D]
        
        # Fill NaN with 0.0
        X = X.fillna(0.0)
        
        # Replace inf/-inf with 0.0 BEFORE clipping
        X = X.replace([np.inf, -np.inf], 0.0)
        
        # Apply exact clipping rules from training (same as in train_ml_20d.py)
        if "ATR_Pct" in X.columns:
            X["ATR_Pct"] = np.clip(X["ATR_Pct"], 0.0, 0.2)
        if "RR" in X.columns:
            X["RR"] = np.clip(X["RR"], 0.0, 10.0)
        if "RSI" in X.columns:
            X["RSI"] = np.clip(X["RSI"], 5.0, 95.0)
        if "TechScore_20d" in X.columns:
            X["TechScore_20d"] = np.clip(X["TechScore_20d"], 0.0, 100.0)
        
        # Reshape to 2D array and predict
        X_input = X.values.reshape(1, -1)
        proba = BUNDLE_MODEL.predict_proba(X_input)
        
        # Get positive class probability (class 1)
        prob = float(proba[0, 1])
        
        # Ensure probability is in valid range [0, 1]
        prob = float(np.clip(prob, 0.0, 1.0))
        return prob
    except Exception as e:
        logger.debug(f"ML 20d prediction failed: {e}")
        return np.nan


def predict_20d_prob_from_row(row: pd.Series) -> float:
    """
    Backward compatibility wrapper - returns raw probability.
    For new code, use compute_ml_20d_probabilities_raw directly.
    """
    return compute_ml_20d_probabilities_raw(row)


def apply_live_v3_adjustments(
    df: pd.DataFrame,
    prob_col: str = "ML_20d_Prob_raw",
) -> pd.Series:
    """
    Apply live_v3 adjustments to raw ML probabilities based on:
      - Volatility bucket (ATR_Pct_percentile)
      - Price bucket (Price_As_Of_Date)
      - Ticker reliability meta (if available)
    
    Args:
        df: DataFrame with columns:
            - prob_col: Raw ML probability (default: ML_20d_Prob_raw)
            - ATR_Pct_percentile: Volatility percentile [0, 1]
            - Price_As_Of_Date: Current price
            - ReliabilityFactor (optional): Per-ticker reliability multiplier
        prob_col: Name of the raw probability column
    
    Returns:
        pd.Series: Adjusted probabilities (ML_20d_Prob_live_v3)
    
    Logic (tuned to match offline research):
      - Volatility buckets:
        * 0.00-0.25 (low vol): -0.01 penalty
        * 0.25-0.50 (mild): neutral
        * 0.50-0.75 (sweet spot): +0.015 boost
        * 0.75-1.00 (high vol): -0.005 penalty
      - Price buckets:
        * 0-20: +0.01 boost (only if raw prob > 0.55)
        * 20-50: +0.01 boost
        * 50-150: neutral
        * 150+: -0.01 penalty
      - Ticker reliability (optional):
        * ReliabilityFactor available: multiply by factor
    """
    if prob_col not in df.columns:
        logger.warning(f"Column {prob_col} not found, returning 0.5 for all rows")
        return pd.Series(0.5, index=df.index)
    
    # Start with raw probabilities
    adjusted = df[prob_col].copy()
    
    # Apply volatility bucket adjustments
    if "ATR_Pct_percentile" in df.columns:
        vol_pct = df["ATR_Pct_percentile"].fillna(0.5)
        
        # Low vol (0.00-0.25): slight penalty
        adjusted = np.where(
            (vol_pct >= 0.0) & (vol_pct < 0.25),
            adjusted - 0.01,
            adjusted
        )
        
        # Mild vol (0.25-0.50): neutral (no change)
        
        # Sweet spot (0.50-0.75): boost
        adjusted = np.where(
            (vol_pct >= 0.50) & (vol_pct < 0.75),
            adjusted + 0.015,
            adjusted
        )
        
        # High vol (0.75-1.00): slight penalty
        adjusted = np.where(
            vol_pct >= 0.75,
            adjusted - 0.005,
            adjusted
        )
    
    # Apply price bucket adjustments
    if "Price_As_Of_Date" in df.columns:
        price = df["Price_As_Of_Date"].fillna(50.0)
        
        # 0-20: boost only if raw prob is already high (avoid garbage)
        adjusted = np.where(
            (price > 0) & (price < 20) & (df[prob_col] > 0.55),
            adjusted + 0.01,
            adjusted
        )
        
        # 20-50: mild boost
        adjusted = np.where(
            (price >= 20) & (price < 50),
            adjusted + 0.01,
            adjusted
        )
        
        # 50-150: neutral (no change)
        
        # 150+: mild penalty
        adjusted = np.where(
            price >= 150,
            adjusted - 0.01,
            adjusted
        )
    
    # Apply ticker reliability multiplier (if available)
    if "ReliabilityFactor" in df.columns:
        reliability = df["ReliabilityFactor"].fillna(1.0)
        adjusted = adjusted * reliability
    
    # Clip final probability to [0.01, 0.99]
    adjusted = np.clip(adjusted, 0.01, 0.99)
    
    return pd.Series(adjusted, index=df.index)


def choose_rank_col_20d(df: pd.DataFrame) -> str:
    """Select the ranking column for 20d scoring based on bundle policy.

    Prefers ML-only when requested, otherwise hybrid/overlay fallbacks. Safe when
    columns are missing by returning the first available technical score.
    """
    cols = pd.Index(df.columns)
    mode = PREFERRED_SCORING_MODE_20D or "hybrid"

    def first_available(candidates: list[str]) -> str:
        for c in candidates:
            if c in cols:
                return c
        return cols[0] if len(cols) else ""

    overlay_candidates = cols[cols.str.contains("OverlayV2_20d|AdjustedScore_20d", regex=True)]

    if mode == "ml_only":
        return first_available([
            "ML_20d_Prob",
            "ML_20d_Prob_live_v3",
            "ML_20d_Prob_raw",
            "FinalScore_20d",
            "FinalScore",
        ])
    if mode == "hybrid_overlay" and len(overlay_candidates):
        return overlay_candidates[0]
    if mode in {"hybrid", "hybrid_overlay"}:
        return first_available([
            "FinalScore_20d",
            "HybridFinalScore_20d",
            "FinalScore",
        ])
    # Fallbacks (legacy/tech-only or missing ML)
    return first_available([
        "FinalScore_20d",
        "FinalScore",
        "HybridFinalScore_20d",
        "TechScore_20d_v2",
        "TechScore_20d",
    ])
