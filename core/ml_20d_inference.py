import joblib
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)

ML_20D_AVAILABLE = True
BUNDLE_HAS_MISSING_METEOR_FEATURES: bool = False
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
    """Load model bundle from absolute path for Streamlit Cloud compatibility.

    Ensures we target the canonical path `models/model_20d_v3.pkl`.
    If not found, logs the absolute path being attempted and lists contents
    of the `models/` directory to aid debugging.
    """
    try:
        # Use absolute path relative to this file's location
        module_dir = Path(__file__).resolve().parent.parent  # stock-scout-2 root
        models_dir = module_dir / "models"
        env_model = os.getenv("ML_MODEL_PATH")
        if env_model:
            model_path = Path(env_model)
        else:
            model_path = models_dir / "model_20d_v3.pkl"
        # Log absolute path being attempted
        try:
            logger.info(f"Attempting to load ML model from: {model_path.resolve()}")
        except Exception:
            logger.info(f"Attempting to load ML model from: {model_path}")
        if not model_path.exists():
            # Print directory contents to help diagnose path issues
            try:
                candidates = list(models_dir.glob("*.pkl"))
                logger.warning(
                    "Model file not found. models/ contains: %s",
                    [str(p.resolve()) for p in candidates]
                )
            except Exception:
                logger.warning("Model file not found and could not list models directory")
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

        # Verify expected Meteor features present in bundle
        expected = {"ADR_Pct", "Dist_52w_High", "Volume_Surge_Ratio"}
        missing_expected = [f for f in expected if f not in feature_names]
        if missing_expected:
            logger.warning(
                "ML bundle missing expected Meteor features: %s", missing_expected
            )
            # Signal upstream to mark ML features fallback in outputs
            global BUNDLE_HAS_MISSING_METEOR_FEATURES
            BUNDLE_HAS_MISSING_METEOR_FEATURES = True
        return True, model, feature_names, preferred_scoring_mode
    except Exception as e:
        logger.error(f"Failed to load ML bundle: {e}", exc_info=True)
        return False, None, [], "hybrid"


# Conditional caching based on Streamlit availability
if _STREAMLIT_AVAILABLE:
    # Add version key to bust cache when return signature changes
    @st.cache_resource(hash_funcs={type: id})
    def _load_bundle_cached(cache_version: int = 2):
        """Cache version 2: returns (success, model, features, scoring_mode)."""
        return _load_bundle_impl()
    
    _success, BUNDLE_MODEL, FEATURE_COLS_20D, PREFERRED_SCORING_MODE_20D = _load_bundle_cached()
    # Force hybrid mode globally regardless of bundle preference
    PREFERRED_SCORING_MODE_20D = "hybrid"
    ML_20D_AVAILABLE = _success
else:
    from functools import lru_cache
    
    @lru_cache(maxsize=1)
    def _load_bundle_cached():
        return _load_bundle_impl()
    
    _success, BUNDLE_MODEL, FEATURE_COLS_20D, PREFERRED_SCORING_MODE_20D = _load_bundle_cached()
    # Force hybrid mode globally regardless of bundle preference
    PREFERRED_SCORING_MODE_20D = "hybrid"
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
        # Map Return_1m → Return_20d when training expects 20d but row provides 1m
        try:
            if ("Return_20d" in FEATURE_COLS_20D) and ("Return_20d" not in row) and ("Return_1m" in row):
                row["Return_20d"] = row.get("Return_1m")
        except Exception:
            pass
        # Build feature dict with exact columns from training, in exact order
        feature_dict = {}
        missing_features = []
        
        for col in FEATURE_COLS_20D:
            # Apply aliases/fallbacks for known features
            if col == "ADR_Pct":
                val = row.get("ADR_Pct", row.get("ATR_Pct", np.nan))
            else:
                val = row.get(col, np.nan)
            if not isinstance(val, (int, float)) or np.isnan(val):
                missing_features.append(col)
                val = np.nan
            feature_dict[col] = val
        
        # Log missing features for debugging
        if missing_features:
            logger.info(f"ML 20d: missing features {missing_features} (fillna=0.0)")
        
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
        logger.warning(f"ML 20d prediction failed: {e}", exc_info=True)
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


def calibrate_ml_20d_prob(
    prob_raw: float,
    *,
    atr_pct_percentile: float | None = None,
    price_as_of: float | None = None,
    reliability_factor: float | None = None,
    market_regime: str | None = None,
    rsi: float | None = None,
) -> float:
    """
    Calibrate a single raw ML 20d probability using the same semantics as live_v3.

    Inputs:
      - prob_raw: Raw model probability in [0,1]
      - atr_pct_percentile: Optional volatility percentile [0,1]
      - price_as_of: Optional last price to bucket by price levels
      - reliability_factor: Optional multiplicative reliability factor

    Returns:
      - float in [0.01, 0.99]: calibrated probability

    Notes:
      - Mirrors apply_live_v3_adjustments logic for scalar usage in code paths
        that don't have a full DataFrame (e.g., single-row computations).
    """
    try:
        if prob_raw is None or not np.isfinite(prob_raw):
            # Explicit missing-value semantics; caller should handle fallback
            return np.nan
        adjusted = float(prob_raw)

        # Volatility bucket adjustments
        if atr_pct_percentile is not None and np.isfinite(atr_pct_percentile):
            v = float(atr_pct_percentile)
            if 0.0 <= v < 0.25:
                adjusted -= 0.01
            elif 0.50 <= v < 0.75:
                # Sweet spot boost varies by regime: stronger in bull markets
                boost = 0.015
                if isinstance(market_regime, str) and market_regime.upper() == 'BULLISH':
                    boost = 0.035
                adjusted += boost
            elif v >= 0.75:
                adjusted -= 0.005

        # Price bucket adjustments
        if price_as_of is not None and np.isfinite(price_as_of):
            p = float(price_as_of)
            if 0 < p < 20 and adjusted > 0.55:
                adjusted += 0.01
            elif 20 <= p < 50:
                adjusted += 0.01
            elif p >= 150:
                adjusted -= 0.01

        # Reliability handling
        if reliability_factor is not None and np.isfinite(reliability_factor):
            # When provided, retain multiplicative semantics
            adjusted *= float(reliability_factor)
        else:
            # Missing reliability: move probability toward neutral (0.5)
            # Blend current adjusted prob with 0.5 equally
            adjusted = 0.5 * adjusted + 0.5 * 0.5

        # Regime Bonus/Penalty adjustments
        try:
            regime = (market_regime or '').upper()
            rsi_val = float(rsi) if (rsi is not None and np.isfinite(rsi)) else 50.0

            # BEARISH or PANIC: suppress overbought names
            if regime in {'BEARISH', 'PANIC'}:
                if rsi_val > 65.0:
                    adjusted -= 0.06

            # CORRECTION: flight to quality via reliability emphasis
            if regime == 'CORRECTION':
                if reliability_factor is not None and np.isfinite(reliability_factor):
                    if float(reliability_factor) > 1.1:
                        adjusted += 0.02
        except Exception:
            # Defensive: ignore regime tweaks if inputs malformed
            pass

        # Clip
        adjusted = float(np.clip(adjusted, 0.01, 0.99))
        return adjusted
    except Exception:
        # Explicit missing-value semantics; caller should handle fallback
        logger.warning("calibrate_ml_20d_prob failed", exc_info=True)
        return np.nan


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
