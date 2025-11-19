"""
Unified Scoring Engine for Stock Scout.

This module implements the final conviction score calculation with strict requirements:
- Fundamentals: 35%
- Momentum: 35%
- Risk/Reward: 15%
- Reliability: 15%

All scores are normalized 0-100 and monotonic (never negative).
Deterministic sorting ensures reproducibility.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def normalize_score(value: float, min_val: float = 0.0, max_val: float = 100.0, 
                   default: float = 50.0) -> float:
    """
    Normalize a score to 0-100 range with safety checks.
    
    Args:
        value: Input value to normalize
        min_val: Minimum expected value
        max_val: Maximum expected value
        default: Default value if input is invalid
    
    Returns:
        Normalized score 0-100
    """
    if not np.isfinite(value):
        return default
    
    if max_val == min_val:
        return default
    
    normalized = ((value - min_val) / (max_val - min_val)) * 100.0
    return float(np.clip(normalized, 0.0, 100.0))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with zero and NaN protection."""
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return default
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator


def calculate_fundamental_score(
    pe: Optional[float] = None,
    ps: Optional[float] = None,
    pb: Optional[float] = None,
    roe: Optional[float] = None,
    margin: Optional[float] = None,
    rev_yoy: Optional[float] = None,
    eps_yoy: Optional[float] = None,
    debt_equity: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Calculate fundamental score (0-100) with confidence.
    
    All metrics are normalized and weighted. Missing values reduce confidence.
    
    Returns:
        (score, confidence) tuple where both are 0-100
    """
    metrics = []
    weights = []
    
    # PE ratio: lower is better (inverse scoring)
    if pe is not None and np.isfinite(pe) and pe > 0:
        pe_score = normalize_score(min(pe, 50), 0, 50, 50)
        pe_score = 100 - pe_score  # Invert: lower PE = higher score
        metrics.append(pe_score)
        weights.append(0.15)
    
    # PS ratio: lower is better
    if ps is not None and np.isfinite(ps) and ps > 0:
        ps_score = normalize_score(min(ps, 10), 0, 10, 50)
        ps_score = 100 - ps_score
        metrics.append(ps_score)
        weights.append(0.10)
    
    # PB ratio: lower is better
    if pb is not None and np.isfinite(pb) and pb > 0:
        pb_score = normalize_score(min(pb, 5), 0, 5, 50)
        pb_score = 100 - pb_score
        metrics.append(pb_score)
        weights.append(0.10)
    
    # ROE: higher is better
    if roe is not None and np.isfinite(roe):
        roe_pct = roe * 100 if abs(roe) < 2 else roe  # Handle decimal vs percentage
        roe_score = normalize_score(np.clip(roe_pct, -20, 50), -20, 50, 50)
        metrics.append(roe_score)
        weights.append(0.20)
    
    # Margin: higher is better
    if margin is not None and np.isfinite(margin):
        margin_pct = margin * 100 if abs(margin) < 2 else margin
        margin_score = normalize_score(np.clip(margin_pct, -10, 40), -10, 40, 50)
        metrics.append(margin_score)
        weights.append(0.15)
    
    # Revenue YoY: higher is better
    if rev_yoy is not None and np.isfinite(rev_yoy):
        rev_pct = rev_yoy * 100 if abs(rev_yoy) < 2 else rev_yoy
        rev_score = normalize_score(np.clip(rev_pct, -20, 50), -20, 50, 50)
        metrics.append(rev_score)
        weights.append(0.15)
    
    # EPS YoY: higher is better
    if eps_yoy is not None and np.isfinite(eps_yoy):
        eps_pct = eps_yoy * 100 if abs(eps_yoy) < 2 else eps_yoy
        eps_score = normalize_score(np.clip(eps_pct, -30, 70), -30, 70, 50)
        metrics.append(eps_score)
        weights.append(0.10)
    
    # Debt/Equity: lower is better
    if debt_equity is not None and np.isfinite(debt_equity) and debt_equity >= 0:
        de_score = normalize_score(min(debt_equity, 3), 0, 3, 50)
        de_score = 100 - de_score
        metrics.append(de_score)
        weights.append(0.05)
    
    # Calculate weighted score
    if not metrics:
        return 50.0, 0.0  # No data = neutral score, zero confidence
    
    total_weight = sum(weights)
    if total_weight == 0:
        return 50.0, 0.0
    
    # Normalize weights to sum to 1
    normalized_weights = [w / total_weight for w in weights]
    
    # Weighted average
    final_score = sum(m * w for m, w in zip(metrics, normalized_weights))
    
    # Confidence based on data completeness (0-8 metrics available)
    confidence = (len(metrics) / 8.0) * 100.0
    
    return float(np.clip(final_score, 0, 100)), float(np.clip(confidence, 0, 100))


def calculate_momentum_score(
    rsi: Optional[float] = None,
    atr_pct: Optional[float] = None,
    ma_aligned: Optional[bool] = None,
    mom_1m: Optional[float] = None,
    mom_3m: Optional[float] = None,
    mom_6m: Optional[float] = None,
    near_high: Optional[float] = None,
    overextension: Optional[float] = None,
    volume_surge: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Calculate momentum score (0-100) with confidence.
    
    Returns:
        (score, confidence) tuple where both are 0-100
    """
    metrics = []
    weights = []
    
    # RSI: 30-70 is ideal (bell curve)
    if rsi is not None and np.isfinite(rsi):
        if rsi < 30:
            rsi_score = normalize_score(rsi, 0, 30, 50)
        elif rsi > 70:
            rsi_score = 100 - normalize_score(rsi, 70, 100, 50)
        else:
            # RSI 40-60 gets highest score
            if 40 <= rsi <= 60:
                rsi_score = 100.0
            elif rsi < 40:
                rsi_score = 50 + normalize_score(rsi, 30, 40, 0)
            else:  # rsi > 60
                rsi_score = 100 - normalize_score(rsi, 60, 70, 0)
        metrics.append(rsi_score)
        weights.append(0.20)
    
    # ATR/Price: moderate volatility is good
    if atr_pct is not None and np.isfinite(atr_pct) and atr_pct > 0:
        # 1-4% is ideal
        if 0.01 <= atr_pct <= 0.04:
            atr_score = 100.0
        elif atr_pct < 0.01:
            atr_score = normalize_score(atr_pct, 0, 0.01, 50)
        else:
            atr_score = 100 - normalize_score(min(atr_pct, 0.10), 0.04, 0.10, 50)
        metrics.append(atr_score)
        weights.append(0.15)
    
    # MA Alignment: boolean = 100 or 0
    if ma_aligned is not None:
        metrics.append(100.0 if ma_aligned else 0.0)
        weights.append(0.15)
    
    # 1-month momentum
    if mom_1m is not None and np.isfinite(mom_1m):
        mom_1m_pct = mom_1m * 100 if abs(mom_1m) < 2 else mom_1m
        mom_1m_score = normalize_score(np.clip(mom_1m_pct, -20, 30), -20, 30, 50)
        metrics.append(mom_1m_score)
        weights.append(0.10)
    
    # 3-month momentum
    if mom_3m is not None and np.isfinite(mom_3m):
        mom_3m_pct = mom_3m * 100 if abs(mom_3m) < 2 else mom_3m
        mom_3m_score = normalize_score(np.clip(mom_3m_pct, -30, 50), -30, 50, 50)
        metrics.append(mom_3m_score)
        weights.append(0.15)
    
    # 6-month momentum
    if mom_6m is not None and np.isfinite(mom_6m):
        mom_6m_pct = mom_6m * 100 if abs(mom_6m) < 2 else mom_6m
        mom_6m_score = normalize_score(np.clip(mom_6m_pct, -40, 70), -40, 70, 50)
        metrics.append(mom_6m_score)
        weights.append(0.10)
    
    # Near 52w high: bell curve (80-95% is ideal)
    if near_high is not None and np.isfinite(near_high):
        if 80 <= near_high <= 95:
            near_high_score = 100.0
        elif near_high < 80:
            near_high_score = normalize_score(near_high, 50, 80, 0)
        else:  # > 95
            near_high_score = 100 - normalize_score(near_high, 95, 100, 0)
        metrics.append(near_high_score)
        weights.append(0.10)
    
    # Overextension: lower is better
    if overextension is not None and np.isfinite(overextension):
        overext_score = 100 - normalize_score(np.clip(overextension, 0, 0.15), 0, 0.15, 50)
        metrics.append(overext_score)
        weights.append(0.05)
    
    # Calculate weighted score
    if not metrics:
        return 50.0, 0.0
    
    total_weight = sum(weights)
    if total_weight == 0:
        return 50.0, 0.0
    
    normalized_weights = [w / total_weight for w in weights]
    final_score = sum(m * w for m, w in zip(metrics, normalized_weights))
    
    # Confidence based on data completeness (0-9 metrics)
    confidence = (len(metrics) / 9.0) * 100.0
    
    return float(np.clip(final_score, 0, 100)), float(np.clip(confidence, 0, 100))


def calculate_rr_score(
    rr_ratio: Optional[float] = None,
    atr: Optional[float] = None,
    support: Optional[float] = None,
    resistance: Optional[float] = None,
    current_price: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Calculate Risk/Reward score (0-100) with confidence.
    
    Penalizes RR < 1.5 automatically.
    
    Returns:
        (score, confidence) tuple where both are 0-100
    """
    # If RR ratio is provided directly
    if rr_ratio is not None and np.isfinite(rr_ratio):
        # Penalize RR < 1.5
        if rr_ratio < 1.5:
            rr_score = normalize_score(rr_ratio, 0, 1.5, 0) * 0.5  # Max 50 points
        else:
            # RR 1.5-5.0 is good
            rr_score = normalize_score(np.clip(rr_ratio, 1.5, 5.0), 1.5, 5.0, 50)
        
        confidence = 80.0  # Moderate confidence if only RR provided
        return float(np.clip(rr_score, 0, 100)), confidence
    
    # Calculate RR from support/resistance
    if all(x is not None and np.isfinite(x) and x > 0 
           for x in [current_price, support, resistance, atr]):
        
        # Risk: distance to support in ATR units
        risk_dollars = current_price - support
        risk_atr = safe_divide(risk_dollars, atr, 1.0)
        
        # Reward: distance to resistance in ATR units
        reward_dollars = resistance - current_price
        reward_atr = safe_divide(reward_dollars, atr, 1.0)
        
        # Calculate RR ratio
        calculated_rr = safe_divide(reward_atr, risk_atr, 0.0)
        
        # Apply same logic as above
        if calculated_rr < 1.5:
            rr_score = normalize_score(calculated_rr, 0, 1.5, 0) * 0.5
        else:
            rr_score = normalize_score(np.clip(calculated_rr, 1.5, 5.0), 1.5, 5.0, 50)
        
        # High confidence when calculated from components
        confidence = 100.0
        return float(np.clip(rr_score, 0, 100)), confidence
    
    # No valid data
    return 50.0, 0.0


def calculate_reliability_score(
    price_sources: int = 0,
    fund_sources: int = 0,
    price_std: Optional[float] = None,
    price_mean: Optional[float] = None,
    fundamental_confidence: float = 0.0,
    data_completeness: float = 0.0,
) -> float:
    """
    Calculate unified reliability score (0-100).
    
    Based on:
    - Data completeness (40%)
    - Cross-source price variance (30%)
    - Fundamental coverage (20%)
    - Number of sources (10%)
    
    Returns:
        Reliability score 0-100
    """
    components = []
    weights = []
    
    # Data completeness
    if 0 <= data_completeness <= 100:
        components.append(data_completeness)
        weights.append(0.40)
    
    # Price variance: lower is better
    if price_std is not None and price_mean is not None and np.isfinite(price_std) and np.isfinite(price_mean):
        if price_mean > 0:
            cv = (price_std / price_mean) * 100  # Coefficient of variation
            # CV < 1% is excellent, > 5% is poor
            variance_score = 100 - normalize_score(np.clip(cv, 0, 5), 0, 5, 50)
            components.append(variance_score)
            weights.append(0.30)
    
    # Fundamental coverage
    if 0 <= fundamental_confidence <= 100:
        components.append(fundamental_confidence)
        weights.append(0.20)
    
    # Source count: more sources = better
    total_sources = price_sources + fund_sources
    source_score = normalize_score(min(total_sources, 6), 0, 6, 50)
    components.append(source_score)
    weights.append(0.10)
    
    if not components:
        return 50.0
    
    total_weight = sum(weights)
    if total_weight == 0:
        return 50.0
    
    normalized_weights = [w / total_weight for w in weights]
    final_score = sum(c * w for c, w in zip(components, normalized_weights))
    
    return float(np.clip(final_score, 0, 100))


def calculate_conviction_score(
    fundamental_score: float,
    fundamental_confidence: float,
    momentum_score: float,
    momentum_confidence: float,
    rr_score: float,
    rr_confidence: float,
    reliability_score: float,
    ml_probability: Optional[float] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate final conviction score (0-100) with strict weighting.
    
    Weights:
    - Fundamentals: 35%
    - Momentum: 35%
    - Risk/Reward: 15%
    - Reliability: 15%
    
    ML can only boost/penalize within ±10% if provided.
    
    Returns:
        (final_score, breakdown_dict) where breakdown shows component contributions
    """
    # Validate inputs
    fundamental_score = np.clip(fundamental_score, 0, 100)
    momentum_score = np.clip(momentum_score, 0, 100)
    rr_score = np.clip(rr_score, 0, 100)
    reliability_score = np.clip(reliability_score, 0, 100)
    
    # Apply confidence weighting (low confidence = pull toward 50)
    fundamental_weighted = fundamental_score * (fundamental_confidence / 100.0) + 50 * (1 - fundamental_confidence / 100.0)
    momentum_weighted = momentum_score * (momentum_confidence / 100.0) + 50 * (1 - momentum_confidence / 100.0)
    rr_weighted = rr_score * (rr_confidence / 100.0) + 50 * (1 - rr_confidence / 100.0)
    
    # Calculate base conviction (before ML)
    base_conviction = (
        fundamental_weighted * 0.35 +
        momentum_weighted * 0.35 +
        rr_weighted * 0.15 +
        reliability_score * 0.15
    )
    
    # ML adjustment (capped at ±10%)
    ml_adjustment = 0.0
    if ml_probability is not None and np.isfinite(ml_probability):
        # ML probability should be 0-1
        ml_prob_clamped = np.clip(ml_probability, 0, 1)
        # Convert to ±10% adjustment: 0.5 = no change, 1.0 = +10%, 0.0 = -10%
        ml_adjustment = (ml_prob_clamped - 0.5) * 20.0  # Range: -10 to +10
    
    final_conviction = base_conviction + ml_adjustment
    final_conviction = float(np.clip(final_conviction, 0, 100))
    
    # Breakdown for transparency
    breakdown = {
        "fundamental_component": fundamental_weighted * 0.35,
        "momentum_component": momentum_weighted * 0.35,
        "rr_component": rr_weighted * 0.15,
        "reliability_component": reliability_score * 0.15,
        "ml_adjustment": ml_adjustment,
        "base_conviction": base_conviction,
        "final_conviction": final_conviction,
    }
    
    return final_conviction, breakdown


def calculate_risk_meter(
    rr_ratio: Optional[float],
    beta: Optional[float],
    atr_pct: Optional[float],
    leverage: Optional[float],
) -> Tuple[float, str]:
    """
    Calculate risk meter (0-100) with label.
    
    Higher number = higher risk
    
    Returns:
        (risk_score, risk_label)
    """
    risk_components = []
    
    # RR ratio: lower RR = higher risk
    if rr_ratio is not None and np.isfinite(rr_ratio):
        if rr_ratio < 1.5:
            risk_components.append(80.0)  # High risk
        elif rr_ratio < 2.0:
            risk_components.append(60.0)  # Moderate-high risk
        elif rr_ratio < 3.0:
            risk_components.append(40.0)  # Moderate risk
        else:
            risk_components.append(20.0)  # Low risk
    
    # Beta: > 1.5 is high risk
    if beta is not None and np.isfinite(beta):
        if beta > 1.5:
            risk_components.append(70.0)
        elif beta > 1.2:
            risk_components.append(50.0)
        elif beta < 0.8:
            risk_components.append(30.0)
        else:
            risk_components.append(40.0)
    
    # ATR/Price: high volatility = high risk
    if atr_pct is not None and np.isfinite(atr_pct):
        if atr_pct > 0.05:
            risk_components.append(70.0)
        elif atr_pct > 0.03:
            risk_components.append(50.0)
        else:
            risk_components.append(30.0)
    
    # Debt/Equity: high leverage = high risk
    if leverage is not None and np.isfinite(leverage):
        if leverage > 2.0:
            risk_components.append(70.0)
        elif leverage > 1.0:
            risk_components.append(50.0)
        else:
            risk_components.append(30.0)
    
    if not risk_components:
        return 50.0, "MODERATE"
    
    risk_score = float(np.mean(risk_components))
    
    # Assign label
    if risk_score >= 70:
        label = "HIGH"
    elif risk_score >= 50:
        label = "MODERATE-HIGH"
    elif risk_score >= 35:
        label = "MODERATE"
    else:
        label = "LOW"
    
    return risk_score, label


def generate_warnings(
    rr_ratio: Optional[float],
    fundamental_confidence: float,
    beta: Optional[float],
    atr_pct: Optional[float],
    reliability_score: float,
) -> list[str]:
    """Generate list of warning messages based on risk factors."""
    warnings = []
    
    if rr_ratio is not None and np.isfinite(rr_ratio) and rr_ratio < 1.5:
        warnings.append(f"⚠️ Low Risk/Reward ratio ({rr_ratio:.2f} < 1.5)")
    
    if fundamental_confidence < 40:
        warnings.append(f"⚠️ Low fundamental confidence ({fundamental_confidence:.0f}%)")
    
    if beta is not None and np.isfinite(beta) and beta > 1.5:
        warnings.append(f"⚠️ High beta ({beta:.2f} > 1.5)")
    
    if atr_pct is not None and np.isfinite(atr_pct) and atr_pct > 0.05:
        warnings.append(f"⚠️ High volatility (ATR {atr_pct*100:.1f}% > 5%)")
    
    if reliability_score < 50:
        warnings.append(f"⚠️ Low data reliability ({reliability_score:.0f}/100)")
    
    return warnings
