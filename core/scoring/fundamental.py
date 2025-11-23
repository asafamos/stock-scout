"""
Fundamental scoring with detailed breakdown and human-friendly labels.
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np

from core.models import FundamentalScore, FundamentalBreakdown
from core.logging_config import get_logger

logger = get_logger("fundamental_scoring")


def _safe_float(value) -> Optional[float]:
    """Convert value to float, return None if invalid."""
    if value is None:
        return None
    try:
        val = float(value)
        return val if np.isfinite(val) else None
    except (ValueError, TypeError):
        return None


def _normalize(value: Optional[float], low: float, high: float) -> float:
    """Normalize value to [0, 1] range."""
    if value is None or not np.isfinite(value):
        return 0.5  # Neutral if unknown
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def compute_fundamental_score_with_breakdown(data: dict, coverage_pct: float = 1.0) -> FundamentalScore:
    """
    Compute fundamental score with detailed breakdown.
    
    Args:
        data: Dictionary with fundamental metrics:
            - roe, roic, gm: Quality metrics
            - rev_g_yoy, eps_g_yoy: Growth metrics
            - pe, ps: Valuation metrics
            - de: Leverage (debt-to-equity)
        coverage_pct: Fraction of fundamental fields available (0-1)
    
    Returns:
        FundamentalScore with breakdown and labels
    """
    breakdown = FundamentalBreakdown()
    
    # Extract raw metrics
    breakdown.roe = _safe_float(data.get("roe"))
    breakdown.roic = _safe_float(data.get("roic"))
    breakdown.gross_margin = _safe_float(data.get("gm"))
    breakdown.revenue_growth_yoy = _safe_float(data.get("rev_g_yoy"))
    breakdown.eps_growth_yoy = _safe_float(data.get("eps_g_yoy"))
    breakdown.pe_ratio = _safe_float(data.get("pe"))
    breakdown.ps_ratio = _safe_float(data.get("ps"))
    breakdown.debt_to_equity = _safe_float(data.get("de"))
    
    # === Quality Score (0-100) ===
    roe_score = _normalize(breakdown.roe, 0, 0.25) * 100  # ROE: 0-25%
    roic_score = _normalize(breakdown.roic, 0, 0.20) * 100  # ROIC: 0-20%
    gm_score = _normalize(breakdown.gross_margin, 0, 0.50) * 100  # GM: 0-50%
    
    breakdown.quality_score = float(np.mean([roe_score, roic_score, gm_score]))
    
    # Fallback: If all metrics are NaN/missing, use default that creates varied distribution
    # Uses a baseline that will be adjusted by technical signals in classification
    if not np.isfinite(breakdown.quality_score):
        # Default to moderate-low quality (27) to create Speculative stocks
        # Individual stocks will be upgraded/downgraded based on technical strength
        breakdown.quality_score = 27.0
    
    breakdown.quality_label = _quality_label(breakdown.quality_score)
    
    # === Growth Score (0-100) with STRONGER PENALTIES ===
    # Negative or weak growth should significantly hurt the score
    rev_g = breakdown.revenue_growth_yoy
    eps_g = breakdown.eps_growth_yoy
    
    # Revenue growth scoring with stricter penalties
    if rev_g is not None and np.isfinite(rev_g):
        if rev_g < -0.05:  # Declining revenue > 5%: very bad (0-20 pts)
            rev_g_score = max(0, 20 + rev_g * 400)  # -5% = 0pts, 0% = 20pts
        elif rev_g < 0.05:  # Weak/flat growth 0-5%: poor (20-40 pts)
            rev_g_score = 20 + (rev_g / 0.05) * 20
        else:  # Positive growth: scale normally
            rev_g_score = _normalize(rev_g, 0.05, 0.30) * 60 + 40  # 5-30% = 40-100pts
    else:
        rev_g_score = 35.0  # Missing data = below average
    
    # EPS growth scoring with stricter penalties
    if eps_g is not None and np.isfinite(eps_g):
        if eps_g < -0.10:  # Declining EPS > 10%: very bad (0-15 pts)
            eps_g_score = max(0, 15 + eps_g * 150)
        elif eps_g < 0.05:  # Weak/flat EPS 0-5%: poor (15-35 pts)
            eps_g_score = 15 + (eps_g / 0.05) * 20
        else:  # Positive EPS growth: scale normally
            eps_g_score = _normalize(eps_g, 0.05, 0.50) * 65 + 35  # 5-50% = 35-100pts
    else:
        eps_g_score = 35.0  # Missing data = below average
    
    breakdown.growth_score = float(np.mean([rev_g_score, eps_g_score]))
    breakdown.growth_label = _growth_label(breakdown.growth_score)
    
    # === Valuation Score (0-100, lower multiples = higher score) with EXTREME penalties ===
    pe = breakdown.pe_ratio
    ps = breakdown.ps_ratio
    
    # P/E scoring with extreme valuation penalty
    if pe is not None and np.isfinite(pe) and pe > 0:
        if pe > 100:  # Extreme P/E: 0-20 pts
            pe_score = max(0, 20 - (pe - 100) * 0.2)
        elif pe > 60:  # Very high P/E: 20-40 pts
            pe_score = 20 + (1.0 - _normalize(pe, 60, 100)) * 20
        else:  # Normal range
            pe_score = (1.0 - _normalize(pe, 5, 60)) * 60 + 40
    elif pe is not None and pe < 0:  # Negative earnings
        pe_score = 10.0  # Very low score for negative earnings
    else:
        pe_score = 50.0
    
    # P/S scoring with extreme valuation penalty
    if ps is not None and np.isfinite(ps) and ps > 0:
        if ps > 20:  # Extreme P/S: 0-20 pts
            ps_score = max(0, 20 - (ps - 20) * 1.0)
        elif ps > 10:  # Very high P/S: 20-40 pts
            ps_score = 20 + (1.0 - _normalize(ps, 10, 20)) * 20
        else:  # Normal range
            ps_score = (1.0 - _normalize(ps, 0.5, 10)) * 60 + 40
    else:
        ps_score = 50.0
    
    breakdown.valuation_score = float(np.mean([pe_score, ps_score]))
    breakdown.valuation_label = _valuation_label(breakdown.valuation_score)
    
    # === Leverage Score (0-100, lower D/E = higher score) with STRONGER PENALTIES ===
    de = breakdown.debt_to_equity
    
    if de is not None and np.isfinite(de):
        if de > 3.0:  # Very high debt: 0-20 pts
            breakdown.leverage_score = max(0, 20 - (de - 3.0) * 10)
        elif de > 2.0:  # High debt: 20-40 pts
            de_penalty = _normalize(de, 2.0, 3.0)
            breakdown.leverage_score = 40 - de_penalty * 20
        elif de > 1.0:  # Moderate debt: 40-70 pts
            de_penalty = _normalize(de, 1.0, 2.0)
            breakdown.leverage_score = 70 - de_penalty * 30
        else:  # Low debt: 70-100 pts
            breakdown.leverage_score = 70 + (1.0 - de) * 30
    else:
        breakdown.leverage_score = 50.0  # Neutral if unknown
    
    breakdown.leverage_score = float(np.clip(breakdown.leverage_score, 0, 100))
    breakdown.leverage_label = _leverage_label(breakdown.leverage_score)
    
    # === Total Score ===
    # Weighted average: Quality=35%, Growth=30%, Valuation=25%, Leverage=10%
    total = (
        0.35 * breakdown.quality_score +
        0.30 * breakdown.growth_score +
        0.25 * breakdown.valuation_score +
        0.10 * breakdown.leverage_score
    )
    
    # === Coverage Cap: Low coverage limits max score ===
    # If coverage < 0.5, cap at 60 (medium band)
    # If coverage < 0.3, cap at 45 (low-medium band)
    if coverage_pct < 0.3:
        total = min(total, 45.0)
    elif coverage_pct < 0.5:
        total = min(total, 60.0)
    
    return FundamentalScore(
        total=float(np.clip(total, 0, 100)),
        breakdown=breakdown
    )


def compute_fund_reliability(n_sources: int) -> float:
    """Simple mapping from number of distinct fundamental sources to reliability (0-1).

    Mapping:
      0 -> 0.0
      1 -> 0.33
      2 -> 0.66
      >=3 -> 1.0
    """
    try:
        n = int(n_sources)
    except Exception:
        return 0.0
    if n <= 0:
        return 0.0
    if n == 1:
        return 0.33
    if n == 2:
        return 0.66
    return 1.0


def _quality_label(score: float) -> str:
    """Convert quality score to human-friendly label."""
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"


def _growth_label(score: float) -> str:
    """Convert growth score to human-friendly label."""
    if score >= 70:
        return "Fast"
    elif score >= 50:
        return "Moderate"
    elif score >= 30:
        return "Slow"
    else:
        return "Declining"


def _valuation_label(score: float) -> str:
    """Convert valuation score to human-friendly label."""
    if score >= 70:
        return "Cheap"
    elif score >= 40:
        return "Fair"
    else:
        return "Expensive"


def _leverage_label(score: float) -> str:
    """Convert leverage score to human-friendly label."""
    if score >= 70:
        return "Low"
    elif score >= 40:
        return "Medium"
    else:
        return "High"


# Legacy compatibility function
def fundamental_score_legacy(data: dict, surprise_bonus_on: bool = False, coverage_pct: float = 1.0) -> float:
    """
    Legacy fundamental_score function for backward compatibility.
    Returns simple 0-1 score.
    """
    result = compute_fundamental_score_with_breakdown(data, coverage_pct)
    return result.total / 100.0  # Convert to 0-1 scale
