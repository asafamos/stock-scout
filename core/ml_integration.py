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
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# Global model cache
_ML_MODEL = None
_MODEL_LOADED = False
_MODEL_PATH = str(Path(__file__).resolve().parents[1] / "model_xgboost_5d.pkl")


def load_ml_model(model_path: Optional[str] = None) -> bool:
    """
    Load the XGBoost model from disk.
    
    Args:
        model_path: Path to model file (default: model_xgboost_5d.pkl)
    
    Returns:
        True if loaded successfully, False otherwise
    """
    global _ML_MODEL, _MODEL_LOADED, _MODEL_PATH
    
    if model_path:
        _MODEL_PATH = model_path
    
    if _MODEL_LOADED:
        return _ML_MODEL is not None
    
    try:
        if not os.path.exists(_MODEL_PATH):
            logger.warning(f"ML model not found at {_MODEL_PATH}")
            _MODEL_LOADED = True
            return False
        
        with open(_MODEL_PATH, "rb") as f:
            _ML_MODEL = pickle.load(f)
        
        logger.info(f"✓ ML model loaded from {_MODEL_PATH}")
        _MODEL_LOADED = True
        return True
        
    except Exception as e:
        logger.error(f"Failed to load ML model: {e}")
        _ML_MODEL = None
        _MODEL_LOADED = True
        return False


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
        
        # CALIBRATION FIX: Spread probabilities to use full range
        # Issue: Model returns very uniform 0.79-0.86 range
        # Solution: Apply non-linear transformation to spread values
        # This is a temporary fix until model is retrained with proper feature scaling
        
        # Map [0.7, 0.9] → [0.5, 1.0] to expand variance
        if 0.7 <= prob_win <= 0.9:
            # Linear stretch within this range
            prob_win = 0.5 + ((prob_win - 0.7) / 0.2) * 0.5
        # Map [0.5, 0.7) → [0.3, 0.5] to expand low end
        elif 0.5 <= prob_win < 0.7:
            prob_win = 0.3 + ((prob_win - 0.5) / 0.2) * 0.2
        # Map [0.3, 0.5) → [0.1, 0.3]
        elif 0.3 <= prob_win < 0.5:
            prob_win = 0.1 + ((prob_win - 0.3) / 0.2) * 0.2
        # Keep very low/high as is
        
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
    
    This should match the features used during model training.
    Based on analysis, the model expects:
    - 6 base technical: RSI, ATR_Pct, Overext, RR, MomCons, VolSurge
    - 8 engineered: RR_MomCons, RSI_Neutral, RSI_Squared, Risk_Score, Vol_Mom, 
                     Overext_Mom_Div, RR_Risk_Adj, ATR_Regime
    - 8 context: Market_Trend, Market_Volatility, SPY_RSI, Relative_Strength_20d,
                 Dist_From_52w_High, Vol_Breakout, Price_Breakout, Mom_Acceleration
    
    Args:
        ticker_data: Raw ticker data dict
        technical_indicators: Technical indicators dict
        fundamental_scores: Fundamental scores dict
    
    Returns:
        Feature dict ready for ML model
    """
    features = {}
    
    # Base technical (6)
    features["RSI"] = technical_indicators.get("rsi", 50.0)
    features["ATR_Pct"] = technical_indicators.get("atr_pct", 0.02)
    features["Overext"] = technical_indicators.get("overextension", 0.0)
    features["RR"] = technical_indicators.get("rr_ratio", 2.0)
    features["MomCons"] = technical_indicators.get("momentum_consistency", 0.5)
    features["VolSurge"] = technical_indicators.get("volume_surge", 1.0)
    
    # Engineered features (8)
    rsi = features["RSI"]
    rr = features["RR"]
    mom_cons = features["MomCons"]
    overext = features["Overext"]
    atr_pct = features["ATR_Pct"]
    
    features["RR_MomCons"] = rr * mom_cons if rr and mom_cons else 1.0
    features["RSI_Neutral"] = 1.0 if 40 <= rsi <= 60 else 0.0
    features["RSI_Squared"] = (rsi / 100.0) ** 2
    
    # Risk score: higher = riskier
    risk_score = 0.0
    if overext > 0.1:
        risk_score += 0.3
    if rsi > 70:
        risk_score += 0.3
    if atr_pct > 0.05:
        risk_score += 0.2
    if rr < 1.5:
        risk_score += 0.2
    features["Risk_Score"] = risk_score
    
    features["Vol_Mom"] = features["VolSurge"] * mom_cons if features.get("VolSurge") and mom_cons else 0.5
    features["Overext_Mom_Div"] = overext / max(mom_cons, 0.1) if mom_cons else 1.0
    features["RR_Risk_Adj"] = rr * (1 - risk_score) if rr else 1.0
    
    # ATR regime: 0=low, 1=normal, 2=high
    if atr_pct < 0.015:
        features["ATR_Regime"] = 0
    elif atr_pct < 0.04:
        features["ATR_Regime"] = 1
    else:
        features["ATR_Regime"] = 2
    
    # Context features (8) - use defaults if not available
    features["Market_Trend"] = ticker_data.get("market_trend", 0.0)
    features["Market_Volatility"] = ticker_data.get("market_volatility", 0.02)
    features["SPY_RSI"] = ticker_data.get("spy_rsi", 50.0)
    features["Relative_Strength_20d"] = ticker_data.get("relative_strength_20d", 0.0)
    features["Dist_From_52w_High"] = ticker_data.get("dist_from_52w_high", 0.1)
    features["Vol_Breakout"] = 1.0 if features["VolSurge"] > 1.5 else 0.0
    features["Price_Breakout"] = 1.0 if ticker_data.get("near_high", 90) > 95 else 0.0
    features["Mom_Acceleration"] = ticker_data.get("mom_acceleration", 0.0)
    
    # Ensure all values are floats and finite
    for key, val in features.items():
        if not np.isfinite(val):
            features[key] = 0.0 if "Breakout" not in key and "Neutral" not in key else 0.0
        else:
            features[key] = float(val)
    
    return features


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
        "ml_probability": None,
        "ml_boost": 0.0,
        "ml_status": "Disabled",
        "final_conviction": base_conviction
    }
    
    if not enable_ml:
        return base_conviction, ml_info
    
    # Ensure model is loaded
    if not _MODEL_LOADED:
        load_ml_model()
    
    if _ML_MODEL is None:
        ml_info["ml_status"] = "Model not available"
        return base_conviction, ml_info
    
    # Prepare features
    try:
        features = prepare_ml_features(ticker_data, technical_indicators, fundamental_scores)
        ml_probability = get_ml_prediction(features)
        
        if ml_probability is None:
            ml_info["ml_status"] = "Prediction failed"
            return base_conviction, ml_info
        
        # Calculate boost
        final_conviction, ml_boost, status = calculate_ml_boost(base_conviction, ml_probability)
        
        ml_info["ml_probability"] = ml_probability
        ml_info["ml_boost"] = ml_boost
        ml_info["ml_status"] = status
        ml_info["final_conviction"] = final_conviction
        
        return final_conviction, ml_info
        
    except Exception as e:
        logger.error(f"ML integration error: {e}")
        ml_info["ml_status"] = f"Error: {str(e)[:50]}"
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
