"""
Fundamental scoring with detailed breakdown and human-friendly labels.
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np

from core.models import FundamentalScore, FundamentalBreakdown
from core.logging_config import get_logger

logger = get_logger("fundamental_scoring")


def compute_fundamental_score_with_breakdown(data: dict) -> FundamentalScore:
    """
    Compute fundamental score with detailed breakdown.
    
    Args:
        data: Dictionary with fundamental metrics:
            - roe, roic, gm: Quality metrics
            - rev_g_yoy, eps_g_yoy: Growth metrics
            - pe, ps: Valuation metrics
            - de: Leverage (debt-to-equity)
    
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
    breakdown.quality_label = _quality_label(breakdown.quality_score)
    
    # === Growth Score (0-100) ===
    rev_g_score = _normalize(breakdown.revenue_growth_yoy, -0.10, 0.30) * 100
    eps_g_score = _normalize(breakdown.eps_growth_yoy, -0.20, 0.50) * 100
    
    breakdown.growth_score = float(np.mean([rev_g_score, eps_g_score]))
    breakdown.growth_label = _growth_label(breakdown.growth_score)
    
    # === Valuation Score (0-100, lower multiples = higher score) ===
    pe = breakdown.pe_ratio
    ps = breakdown.ps_ratio
    
    # Invert: lower P/E = higher score
    pe_score = (1.0 - _normalize(pe, 5, 40)) * 100 if np.isfinite(pe) else 50.0
    ps_score = (1.0 - _normalize(ps, 0.5, 10)) * 100 if np.isfinite(ps) else 50.0
    
    breakdown.valuation_score = float(np.mean([pe_score, ps_score]))
    breakdown.valuation_label = _valuation_label(breakdown.valuation_score)
    
    # === Leverage Score (0-100, lower D/E = higher score) ===
    de = breakdown.debt_to_equity
    
    if np.isfinite(de):
        # Penalize high debt: D/E > 2.0 is very bad
        de_penalty = _normalize(de, 0, 2.0)  # 0=good, 1=bad
        breakdown.leverage_score = (1.0 - de_penalty) * 100
    else:
        breakdown.leverage_score = 50.0  # Neutral if unknown
    
    breakdown.leverage_label = _leverage_label(breakdown.leverage_score)
    
    # === Total Score ===
    # Weighted average: Quality=35%, Growth=30%, Valuation=25%, Leverage=10%
    total = (
        0.35 * breakdown.quality_score +
        0.30 * breakdown.growth_score +
        0.25 * breakdown.valuation_score +
        0.10 * breakdown.leverage_score
    )
    
    return FundamentalScore(
        total=float(np.clip(total, 0, 100)),
        breakdown=breakdown
    )


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
def fundamental_score_legacy(data: dict, surprise_bonus_on: bool = False) -> float:
    """
    Legacy fundamental_score function for backward compatibility.
    Returns simple 0-1 score.
    """
    result = compute_fundamental_score_with_breakdown(data)
    return result.total / 100.0  # Convert to 0-1 scale
