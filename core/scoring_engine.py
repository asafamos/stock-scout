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


def evaluate_rr_unified(rr_ratio: Optional[float]) -> Tuple[float, float, str]:
    """
    UNIFIED RR evaluation function used across all contexts.
    
    This is the SINGLE source of truth for RR evaluation.
    All warnings, penalties, and displays must use this.
    
    Args:
        rr_ratio: Reward/Risk ratio
        
    Returns:
        (rr_score_0_100, rr_ratio_float, rr_band_label)
        - rr_score_0_100: Normalized score 0-100 with tiered penalties
        - rr_ratio_float: The actual ratio (or 0.0 if invalid)
        - rr_band_label: "Excellent", "Good", "Fair", "Poor", "Very Poor"
    """
    if rr_ratio is None or not np.isfinite(rr_ratio) or rr_ratio < 0:
        return 0.0, 0.0, "N/A"
    
    ratio = float(np.clip(rr_ratio, 0, 10.0))
    
    # Tiered scoring with strong penalties for poor RR
    if ratio < 1.0:
        score = normalize_score(ratio, 0, 1.0, 0) * 0.2  # Max 20 points
        band = "Very Poor"
    elif ratio < 1.5:
        score = normalize_score(ratio, 1.0, 1.5, 0) * 0.2 + 20  # 20-40 points
        band = "Poor"
    elif ratio < 2.0:
        score = normalize_score(ratio, 1.5, 2.0, 0) * 0.3 + 40  # 40-70 points
        band = "Fair"
    elif ratio < 3.0:
        score = normalize_score(ratio, 2.0, 3.0, 0) * 0.2 + 70  # 70-90 points
        band = "Good"
    else:
        score = normalize_score(np.clip(ratio, 3.0, 5.0), 3.0, 5.0, 0) * 0.1 + 90  # 90-100 points
        band = "Excellent"
    
    return float(np.clip(score, 0, 100)), ratio, band


def compute_overall_score(row: pd.Series) -> Tuple[float, Dict[str, float]]:
    """
    Compute mathematically correct overall score with explicit weights and penalties.
    
    Formula: 35% fund + 35% tech + 15% RR + 15% reliability ± ML (max ±10%)
    
    Applies penalties BEFORE final normalization for realistic spread:
    - RR ratio penalties: <1.0 strong, 1.0-1.5 medium, 1.5-2.0 mild
    - RiskMeter >65 penalty
    - Reliability <0.75 penalty
    - Missing data penalty
    
    Target distributions:
    - Core opportunities: 60-80
    - Speculative: 45-60
    - Problematic: 20-45
    
    Args:
        row: DataFrame row with scoring components
    
    Returns:
        (overall_score 0-100, components_dict with all breakdowns)
    """
    # Extract base components (0-100 each)
    fund_score = float(row.get("Fundamental_S", 50.0))
    tech_score = float(row.get("Technical_S", 50.0))
    rr_score = float(row.get("RR_Score", 50.0))
    reliability = float(row.get("Reliability_v2", 50.0))
    
    # Extract ML probability (0-1 range)
    ml_prob = row.get("ML_Probability", None)
    if ml_prob is not None and np.isfinite(ml_prob):
        ml_prob = float(np.clip(ml_prob, 0, 1))
    else:
        ml_prob = None
    
    # Calculate base score (before ML, before penalties)
    base_score = (
        fund_score * 0.35 +
        tech_score * 0.35 +
        rr_score * 0.15 +
        reliability * 0.15
    )
    
    # Calculate ML adjustment (bounded to ±10% of base)
    ml_delta = 0.0
    if ml_prob is not None:
        # ML prob 0.5 = neutral (0 adjustment)
        # ML prob 1.0 = +10%
        # ML prob 0.0 = -10%
        ml_delta = (ml_prob - 0.5) * 2.0 * 10.0  # Range: -10 to +10
        ml_delta = float(np.clip(ml_delta, -10, 10))
    
    # Score before penalties
    score_before_penalties = base_score + ml_delta
    
    # === APPLY PENALTIES FOR REALISTIC SPREAD ===
    penalty_total = 0.0
    penalty_breakdown = {}
    
    # 1. RR ratio penalties
    rr_ratio = float(row.get("RR", 2.0))
    if rr_ratio < 1.0:
        penalty = 15.0  # Strong penalty
        penalty_breakdown["rr_below_1"] = penalty
    elif rr_ratio < 1.5:
        penalty = 8.0  # Medium penalty
        penalty_breakdown["rr_below_1.5"] = penalty
    elif rr_ratio < 2.0:
        penalty = 3.0  # Mild penalty
        penalty_breakdown["rr_below_2"] = penalty
    else:
        penalty = 0.0
    penalty_total += penalty
    
    # 2. RiskMeter penalty (high risk = lower score)
    risk_meter = float(row.get("RiskMeter", 50.0))
    if risk_meter > 65:
        penalty = (risk_meter - 65) * 0.3  # Up to 10.5 penalty at risk=100
        penalty_breakdown["high_risk"] = penalty
        penalty_total += penalty
    
    # 3. Reliability penalty (low reliability = lower score)
    if reliability < 75:
        penalty = (75 - reliability) * 0.2  # Up to 15 penalty at reliability=0
        penalty_breakdown["low_reliability"] = penalty
        penalty_total += penalty
    
    # 4. Missing data penalty (check key fundamentals)
    missing_count = 0
    for field in ["PE_f", "ROE_f", "GM_f", "DE_f", "RevG_f"]:
        val = row.get(field)
        if pd.isna(val) or val == 0 or val == "N/A":
            missing_count += 1
    
    if missing_count > 0:
        penalty = missing_count * 2.0  # 2 points per missing key metric
        penalty_breakdown["missing_data"] = penalty
        penalty_total += penalty
    
    # Apply penalties
    final_score = score_before_penalties - penalty_total
    final_score = float(np.clip(final_score, 0, 100))
    
    # Build component breakdown for transparency
    components = {
        "fund_component": fund_score * 0.35,
        "tech_component": tech_score * 0.35,
        "rr_component": rr_score * 0.15,
        "reliability_component": reliability * 0.15,
        "base_score": base_score,
        "ml_delta": ml_delta,
        "score_before_penalties": score_before_penalties,
        "penalty_total": penalty_total,
        "final_score": final_score,
    }
    
    # Add penalty breakdown
    for key, val in penalty_breakdown.items():
        components[f"penalty_{key}"] = val
    
    return final_score, components


def calculate_quality_score(row: pd.Series) -> Tuple[float, str]:
    """
    Calculate quality score (0-1) based on fundamental health metrics.
    
    Converts to 3-level system:
    - High: ≥0.7 (strong fundamentals)
    - Medium: 0.4-0.69 (acceptable fundamentals)
    - Low: <0.4 (weak fundamentals)
    
    Quality components:
    - Margins (ROE, Gross Margin, Profit Margin): Higher is better
    - Growth (Revenue YoY, EPS YoY): Higher is better
    - Debt (D/E ratio): Lower is better
    
    Args:
        row: DataFrame row with fundamental metrics
    
    Returns:
        (quality_score 0-1, quality_level "High"/"Medium"/"Low")
    """
    scores = []
    weights = []
    
    # 1. Margin Quality (40% weight)
    # ROE
    roe = row.get("ROE_f")
    if pd.notna(roe) and roe != 0:
        roe_pct = roe * 100 if abs(roe) < 2 else roe
        # ROE > 15% = good, < 5% = poor
        roe_score = np.clip((roe_pct - 5) / 10, 0, 1)  # Normalize 5-15% to 0-1
        scores.append(roe_score)
        weights.append(0.20)
    
    # Gross Margin
    gm = row.get("GM_f")
    if pd.notna(gm) and gm != 0:
        gm_pct = gm * 100 if abs(gm) < 2 else gm
        # GM > 30% = good, < 10% = poor
        gm_score = np.clip((gm_pct - 10) / 20, 0, 1)  # Normalize 10-30% to 0-1
        scores.append(gm_score)
        weights.append(0.10)
    
    # Profit Margin
    pm = row.get("ProfitMargin")
    if pd.notna(pm) and pm != 0:
        pm_pct = pm * 100 if abs(pm) < 2 else pm
        # PM > 10% = good, < 0% = poor
        pm_score = np.clip(pm_pct / 10, 0, 1)  # Normalize 0-10% to 0-1
        scores.append(pm_score)
        weights.append(0.10)
    
    # 2. Growth Quality (40% weight)
    # Revenue Growth
    rev_g = row.get("RevG_f") or row.get("RevenueGrowthYoY")
    if pd.notna(rev_g) and rev_g != 0:
        rev_pct = rev_g * 100 if abs(rev_g) < 2 else rev_g
        # Rev growth > 20% = great, 0-20% = acceptable, < 0% = poor
        if rev_pct >= 20:
            rev_score = 1.0
        elif rev_pct >= 0:
            rev_score = 0.5 + (rev_pct / 20) * 0.5  # 0.5-1.0 for 0-20%
        else:
            rev_score = max(0.0, 0.5 + (rev_pct / 20))  # Below 0.5 for negative
        scores.append(rev_score)
        weights.append(0.20)
    
    # EPS Growth
    eps_g = row.get("EPSG_f") or row.get("EPSGrowthYoY")
    if pd.notna(eps_g) and eps_g != 0:
        eps_pct = eps_g * 100 if abs(eps_g) < 2 else eps_g
        # EPS growth > 25% = great, 0-25% = acceptable, < 0% = poor
        if eps_pct >= 25:
            eps_score = 1.0
        elif eps_pct >= 0:
            eps_score = 0.5 + (eps_pct / 25) * 0.5
        else:
            eps_score = max(0.0, 0.5 + (eps_pct / 25))
        scores.append(eps_score)
        weights.append(0.20)
    
    # 3. Debt Quality (20% weight)
    de = row.get("DE_f")
    if pd.notna(de) and de >= 0:
        # D/E < 0.5 = excellent, 0.5-1.5 = acceptable, > 1.5 = concerning
        if de < 0.5:
            de_score = 1.0
        elif de < 1.5:
            de_score = 1.0 - ((de - 0.5) / 1.0) * 0.4  # 1.0 to 0.6
        elif de < 3.0:
            de_score = 0.6 - ((de - 1.5) / 1.5) * 0.4  # 0.6 to 0.2
        else:
            de_score = 0.2
        scores.append(de_score)
        weights.append(0.20)
    
    # Calculate weighted quality score
    if not scores:
        # No data = neutral/medium quality
        return 0.5, "Medium"
    
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.5, "Medium"
    
    normalized_weights = [w / total_weight for w in weights]
    quality_score = sum(s * w for s, w in zip(scores, normalized_weights))
    quality_score = float(np.clip(quality_score, 0, 1))
    
    # Convert to level
    if quality_score >= 0.7:
        level = "High"
    elif quality_score >= 0.4:
        level = "Medium"
    else:
        level = "Low"
    
    return quality_score, level


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
        # Use unified RR evaluation
        rr_score, _, _ = evaluate_rr_unified(rr_ratio)
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
    
    # Apply confidence weighting (REDUCED pull toward 50 to increase spread)
    # Old: full pull to 50 when confidence low
    # New: only 30% pull to 50 (reduced from 100% to allow more variance)
    pull_strength = 0.3  # Reduced from 1.0 to allow scores to vary more
    fundamental_weighted = fundamental_score * (fundamental_confidence / 100.0) + 50 * pull_strength * (1 - fundamental_confidence / 100.0)
    momentum_weighted = momentum_score * (momentum_confidence / 100.0) + 50 * pull_strength * (1 - momentum_confidence / 100.0)
    rr_weighted = rr_score * (rr_confidence / 100.0) + 50 * pull_strength * (1 - rr_confidence / 100.0)
    
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
    Enhanced with stronger penalties and better spread (10-90 range)
    
    Returns:
        (risk_score, risk_label)
    """
    risk_components = []
    weights = []
    
    # RR ratio: lower RR = higher risk (40% weight - most important)
    if rr_ratio is not None and np.isfinite(rr_ratio):
        if rr_ratio < 0.8:
            risk_components.append(95.0)  # Very high risk
        elif rr_ratio < 1.2:
            risk_components.append(75.0)  # High risk
        elif rr_ratio < 1.8:
            risk_components.append(55.0)  # Moderate-high risk
        elif rr_ratio < 2.5:
            risk_components.append(35.0)  # Moderate risk
        elif rr_ratio < 3.5:
            risk_components.append(20.0)  # Low risk
        else:
            risk_components.append(10.0)  # Very low risk
        weights.append(0.40)
    
    # Beta: market sensitivity (25% weight)
    if beta is not None and np.isfinite(beta):
        if beta > 1.8:
            risk_components.append(85.0)  # Very volatile
        elif beta > 1.4:
            risk_components.append(65.0)  # High volatility
        elif beta > 1.1:
            risk_components.append(50.0)  # Moderate-high
        elif beta > 0.9:
            risk_components.append(40.0)  # Market-like
        elif beta > 0.6:
            risk_components.append(25.0)  # Defensive
        else:
            risk_components.append(15.0)  # Very defensive
        weights.append(0.25)
    
    # ATR/Price: intraday volatility (20% weight)
    if atr_pct is not None and np.isfinite(atr_pct):
        if atr_pct > 0.08:
            risk_components.append(90.0)  # Extreme volatility
        elif atr_pct > 0.05:
            risk_components.append(70.0)  # High volatility
        elif atr_pct > 0.03:
            risk_components.append(50.0)  # Moderate
        elif atr_pct > 0.02:
            risk_components.append(30.0)  # Low
        else:
            risk_components.append(15.0)  # Very low
        weights.append(0.20)
    
    # Debt/Equity: leverage risk (15% weight)
    if leverage is not None and np.isfinite(leverage):
        if leverage > 3.0:
            risk_components.append(85.0)  # Very high leverage
        elif leverage > 2.0:
            risk_components.append(65.0)  # High leverage
        elif leverage > 1.5:
            risk_components.append(50.0)  # Moderate-high
        elif leverage > 0.8:
            risk_components.append(35.0)  # Moderate
        elif leverage > 0.3:
            risk_components.append(20.0)  # Low
        else:
            risk_components.append(10.0)  # Minimal debt
        weights.append(0.15)
    
    if not risk_components:
        return 50.0, "MODERATE"
    
    # Weighted average
    risk_score = float(np.average(risk_components, weights=weights))
    
    # Assign label with tighter thresholds
    if risk_score >= 75:
        label = "VERY HIGH"
    elif risk_score >= 60:
        label = "HIGH"
    elif risk_score >= 45:
        label = "MODERATE-HIGH"
    elif risk_score >= 30:
        label = "MODERATE"
    elif risk_score >= 20:
        label = "LOW"
    else:
        label = "VERY LOW"
    
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
