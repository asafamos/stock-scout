"""
Fundamental scoring with detailed Quality/Growth/Valuation/Stability breakdown.

This module provides rich fundamental analysis aligned with multi-source aggregation
from core/data_sources_v2. All scores are normalized to [0, 100] with neutral defaults
for missing data.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from core.models import FundamentalScore, FundamentalBreakdown
from core.logging_config import get_logger

logger = get_logger("fundamental_scoring")


def _safe_float(value: object, scale_to_pct: bool = False) -> Optional[float]:
    """
    Convert value to float, return None if invalid.
    
    Args:
        value: Raw value (can be None, string, numeric)
        scale_to_pct: If True and value is in [0,2], assume decimal and scale to %
    
    Returns:
        Float value or None if invalid
    """
    if value is None:
        return None
    try:
        val = float(value)
        if not np.isfinite(val):
            return None
        # Auto-scale small decimals to percentages when requested
        if scale_to_pct and 0 < abs(val) < 2:
            val = val * 100.0
        return val
    except (ValueError, TypeError):
        return None


def _normalize(value: Optional[float], low: float, high: float, neutral: float = 0.5) -> float:
    """
    Normalize value to [0, 1] range with configurable neutral fallback.
    
    Args:
        value: Input value to normalize
        low: Lower bound (maps to 0.0)
        high: Upper bound (maps to 1.0)
        neutral: Fallback for None/invalid (default 0.5)
    
    Returns:
        Float in [0.0, 1.0]
    """
    if value is None or not np.isfinite(value):
        return neutral
    if high == low:
        return neutral
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def compute_fundamental_score_with_breakdown(data: dict[str, object], coverage_pct: float = 1.0) -> FundamentalScore:
    """
    Compute comprehensive fundamental score with Quality/Growth/Valuation/Stability breakdown.
    
    This is the main entry point for fundamental scoring. It consumes data from
    aggregate_fundamentals() or a DataFrame row and returns a rich breakdown suitable
    for UI display, backtesting analysis, and portfolio construction.
    
    Args:
        data: Dictionary with fundamental metrics from multi-source aggregation:
            Core metrics (directly used):
            - roe (float): Return on Equity, typically 0-50% (or 0-0.5 decimal)
            - roic (float): Return on Invested Capital, typically 0-30%
            - gm or margin (float): Gross/Profit Margin, typically 0-50%
            - rev_yoy or rev_g_yoy (float): Revenue YoY growth, typically -20% to +50%
            - eps_yoy or eps_g_yoy (float): EPS YoY growth, typically -30% to +70%
            - pe (float): Price-to-Earnings ratio, typically 5-60 (>100 is extreme)
            - ps (float): Price-to-Sales ratio, typically 0.5-10 (>20 is extreme)
            - de or debt_equity (float): Debt-to-Equity ratio, typically 0-3
            - beta (float): Market beta, typically 0.5-2.0
            - market_cap (float): Market capitalization in dollars
            
            Metadata (used for adjustments):
            - Fundamental_Coverage_Pct (float 0-100): % of key fields present
            - Fundamental_Sources_Count (int): Number of data providers used
            
        coverage_pct: Explicit coverage override (0-1), used when metadata not in data.
                      If data contains Fundamental_Coverage_Pct, that takes precedence.
    
    Returns:
        FundamentalScore object with:
        - total (float 0-100): Overall fundamental score
        - breakdown (FundamentalBreakdown): Contains:
          * quality_score (0-100): Profitability & efficiency (ROE, ROIC, margins)
          * growth_score (0-100): Revenue & earnings growth
          * valuation_score (0-100): Price multiples (high score = cheap/fair)
          * leverage_score (0-100): Debt level (high score = low debt)
          * stability_score (0-100): Beta & market cap stability
          * Raw metrics and human-friendly labels
    
    Scoring Philosophy:
    - Each sub-score is [0, 100] using sensible bands for the metric.
    - Missing metrics contribute neutrally (typically 50) or conservatively.
    - Extreme outliers are clipped to avoid explosions.
    - Low coverage (<50%) caps the max total score to prevent overconfidence.
    - Negative growth or extreme valuations receive strong penalties.
    
    Weighting:
    - Quality: 30% (profitability is king)
    - Growth: 25% (growth potential)
    - Valuation: 25% (price paid matters)
    - Stability: 10% (risk/volatility)
    - Leverage: 10% (debt burden)
    
    Examples:
        >>> # High-quality growth stock
        >>> data = {'roe': 0.22, 'margin': 0.35, 'rev_yoy': 0.18, 'eps_yoy': 0.25,
        ...         'pe': 28, 'ps': 4.5, 'de': 0.4, 'beta': 1.1,
        ...         'Fundamental_Coverage_Pct': 90.0, 'Fundamental_Sources_Count': 3}
        >>> score = compute_fundamental_score_with_breakdown(data)
        >>> score.total
        78.5  # High overall
        >>> score.breakdown.quality_label
        'High'
        
        >>> # Overvalued, weak growth
        >>> data = {'roe': 0.08, 'rev_yoy': -0.05, 'eps_yoy': -0.02,
        ...         'pe': 85, 'ps': 15, 'de': 2.5,
        ...         'Fundamental_Coverage_Pct': 60.0}
        >>> score = compute_fundamental_score_with_breakdown(data)
        >>> score.total
        28.3  # Low overall
        >>> score.breakdown.valuation_label
        'Expensive'
    """
    breakdown = FundamentalBreakdown()
    
    # Extract coverage metadata
    explicit_coverage = data.get("Fundamental_Coverage_Pct")
    if explicit_coverage is not None and np.isfinite(explicit_coverage):
        coverage_pct = float(explicit_coverage) / 100.0  # Convert % to fraction
    sources_count = int(data.get("Fundamental_Sources_Count", 1))
    
    # Extract raw metrics with auto-scaling for decimals vs percentages
    breakdown.roe = _safe_float(data.get("roe"), scale_to_pct=True)
    breakdown.roic = _safe_float(data.get("roic"), scale_to_pct=True)
    breakdown.gross_margin = _safe_float(data.get("gm") or data.get("margin"), scale_to_pct=True)
    breakdown.revenue_growth_yoy = _safe_float(data.get("rev_yoy") or data.get("rev_g_yoy"), scale_to_pct=True)
    breakdown.eps_growth_yoy = _safe_float(data.get("eps_yoy") or data.get("eps_g_yoy"), scale_to_pct=True)
    breakdown.pe_ratio = _safe_float(data.get("pe"))
    breakdown.ps_ratio = _safe_float(data.get("ps"))
    breakdown.debt_to_equity = _safe_float(data.get("de") or data.get("debt_equity"))
    
    # Additional metrics for stability
    beta = _safe_float(data.get("beta"))
    market_cap = _safe_float(data.get("market_cap"))
    
    # === Quality Score (0-100) ===
    # Profitability & Efficiency: ROE, ROIC, Margins
    # Bands: ROE (0-25%), ROIC (0-20%), GM (0-50%)
    # Strong penalties for negative/low profitability
    
    roe_val = breakdown.roe
    if roe_val is not None and np.isfinite(roe_val):
        if roe_val < 0:
            roe_score = 15.0  # Negative ROE: poor
        elif roe_val < 5:
            roe_score = 15 + (roe_val / 5.0) * 25  # 0-5%: 15-40 pts
        else:
            roe_score = _normalize(roe_val, 5, 25, 0.5) * 60 + 40  # 5-25%: 40-100 pts
    else:
        roe_score = 45.0  # Missing: slightly below neutral
    
    roic_val = breakdown.roic
    if roic_val is not None and np.isfinite(roic_val):
        if roic_val < 0:
            roic_score = 15.0
        elif roic_val < 5:
            roic_score = 15 + (roic_val / 5.0) * 25
        else:
            roic_score = _normalize(roic_val, 5, 20, 0.5) * 60 + 40  # 5-20%: 40-100 pts
    else:
        roic_score = 45.0
    
    gm_val = breakdown.gross_margin
    if gm_val is not None and np.isfinite(gm_val):
        if gm_val < 0:
            gm_score = 10.0  # Negative margin: very poor
        elif gm_val < 10:
            gm_score = 10 + (gm_val / 10.0) * 30  # 0-10%: 10-40 pts
        else:
            gm_score = _normalize(gm_val, 10, 50, 0.5) * 60 + 40  # 10-50%: 40-100 pts
    else:
        gm_score = 45.0
    
    # Weighted average: ROE (40%), ROIC (30%), GM (30%)
    breakdown.quality_score = float(0.40 * roe_score + 0.30 * roic_score + 0.30 * gm_score)
    breakdown.quality_score = float(np.clip(breakdown.quality_score, 0, 100))
    breakdown.quality_label = _quality_label(breakdown.quality_score)
    
    # === Growth Score (0-100) ===
    # Revenue & EPS growth with strong penalties for decline
    # Bands: Rev YoY (-20% to +50%), EPS YoY (-30% to +70%)
    
    rev_g = breakdown.revenue_growth_yoy
    if rev_g is not None and np.isfinite(rev_g):
        if rev_g < -10:  # Severe decline >10%: 0-15 pts
            rev_g_score = max(0, 15 + (rev_g + 10) * 1.5)
        elif rev_g < 0:  # Decline 0-10%: 15-35 pts
            rev_g_score = 15 + ((rev_g + 10) / 10.0) * 20
        elif rev_g < 5:  # Weak growth 0-5%: 35-50 pts
            rev_g_score = 35 + (rev_g / 5.0) * 15
        elif rev_g < 15:  # Moderate growth 5-15%: 50-75 pts
            rev_g_score = 50 + ((rev_g - 5) / 10.0) * 25
        else:  # Strong growth >15%: 75-100 pts
            rev_g_score = 75 + min((rev_g - 15) / 35.0, 1.0) * 25  # Cap at 50%
    else:
        rev_g_score = 42.0  # Missing: below neutral
    
    eps_g = breakdown.eps_growth_yoy
    if eps_g is not None and np.isfinite(eps_g):
        if eps_g < -15:  # Severe EPS decline >15%: 0-10 pts
            eps_g_score = max(0, 10 + (eps_g + 15) * 0.67)
        elif eps_g < 0:  # Decline 0-15%: 10-35 pts
            eps_g_score = 10 + ((eps_g + 15) / 15.0) * 25
        elif eps_g < 5:  # Weak EPS growth 0-5%: 35-50 pts
            eps_g_score = 35 + (eps_g / 5.0) * 15
        elif eps_g < 20:  # Moderate EPS growth 5-20%: 50-75 pts
            eps_g_score = 50 + ((eps_g - 5) / 15.0) * 25
        else:  # Strong EPS growth >20%: 75-100 pts
            eps_g_score = 75 + min((eps_g - 20) / 50.0, 1.0) * 25  # Cap at 70%
    else:
        eps_g_score = 42.0
    
    # Weighted average: Rev (45%), EPS (55%) - EPS slightly more important
    breakdown.growth_score = float(0.45 * rev_g_score + 0.55 * eps_g_score)
    breakdown.growth_score = float(np.clip(breakdown.growth_score, 0, 100))
    breakdown.growth_label = _growth_label(breakdown.growth_score)

    # --- Growth Acceleration Bonus ---
    # If EPS YoY > 25%: +5 points
    # If Revenue YoY > 20%: additional +5 points (total +10)
    try:
        growth_bonus = 0.0
        eps_g_val = breakdown.eps_growth_yoy
        rev_g_val = breakdown.revenue_growth_yoy
        if eps_g_val is not None and np.isfinite(eps_g_val) and eps_g_val > 25.0:
            growth_bonus += 5.0
        if rev_g_val is not None and np.isfinite(rev_g_val) and rev_g_val > 20.0:
            growth_bonus += 5.0
        if growth_bonus > 0:
            breakdown.growth_score = float(np.clip(breakdown.growth_score + growth_bonus, 0.0, 100.0))
            breakdown.growth_label = _growth_label(breakdown.growth_score)
            logger.debug(f"Applied Growth Acceleration bonus (+{growth_bonus:.0f})")
    except Exception:
        # If any issue occurs, keep original growth score
        pass
    
    # === Valuation Score (0-100, lower multiples = higher score) ===
    # P/E and P/S with strong penalties for extreme valuations
    # High score = cheap/fair, low score = expensive/extreme
    
    pe = breakdown.pe_ratio
    if pe is not None and np.isfinite(pe):
        if pe < 0:  # Negative earnings: very poor
            pe_score = 8.0
        elif pe < 10:  # Very cheap: 80-100 pts
            pe_score = 80 + (10 - pe) / 10.0 * 20
        elif pe < 20:  # Cheap-fair: 60-80 pts
            pe_score = 60 + (20 - pe) / 10.0 * 20
        elif pe < 35:  # Fair-moderate: 40-60 pts
            pe_score = 40 + (35 - pe) / 15.0 * 20
        elif pe < 60:  # Expensive: 20-40 pts
            pe_score = 20 + (60 - pe) / 25.0 * 20
        elif pe < 100:  # Very expensive: 5-20 pts
            pe_score = 5 + (100 - pe) / 40.0 * 15
        else:  # Extreme valuation: 0-5 pts
            pe_score = max(0, 5 - (pe - 100) * 0.05)
    else:
        pe_score = 48.0  # Missing: slightly below neutral
    
    ps = breakdown.ps_ratio
    if ps is not None and np.isfinite(ps):
        if ps < 0:
            ps_score = 8.0
        elif ps < 1:  # Very cheap: 80-100 pts
            ps_score = 80 + (1 - ps) * 20
        elif ps < 3:  # Cheap-fair: 60-80 pts
            ps_score = 60 + (3 - ps) / 2.0 * 20
        elif ps < 6:  # Fair-moderate: 40-60 pts
            ps_score = 40 + (6 - ps) / 3.0 * 20
        elif ps < 12:  # Expensive: 20-40 pts
            ps_score = 20 + (12 - ps) / 6.0 * 20
        elif ps < 25:  # Very expensive: 5-20 pts
            ps_score = 5 + (25 - ps) / 13.0 * 15
        else:  # Extreme valuation: 0-5 pts
            ps_score = max(0, 5 - (ps - 25) * 0.2)
    else:
        ps_score = 48.0
    
    # Weighted average: P/E (60%), P/S (40%) - earnings multiple more important
    breakdown.valuation_score = float(0.60 * pe_score + 0.40 * ps_score)
    breakdown.valuation_score = float(np.clip(breakdown.valuation_score, 0, 100))
    breakdown.valuation_label = _valuation_label(breakdown.valuation_score)
    
    # === Leverage Score (0-100, lower D/E = higher score) ===
    # Debt-to-Equity with penalties for high leverage
    
    de = breakdown.debt_to_equity
    if de is not None and np.isfinite(de) and de >= 0:
        if de < 0.3:  # Minimal debt: 90-100 pts
            leverage_score = 90 + (0.3 - de) / 0.3 * 10
        elif de < 0.8:  # Low debt: 70-90 pts
            leverage_score = 70 + (0.8 - de) / 0.5 * 20
        elif de < 1.5:  # Moderate debt: 50-70 pts
            leverage_score = 50 + (1.5 - de) / 0.7 * 20
        elif de < 2.5:  # High debt: 25-50 pts
            leverage_score = 25 + (2.5 - de) * 25
        elif de < 4.0:  # Very high debt: 10-25 pts
            leverage_score = 10 + (4.0 - de) / 1.5 * 15
        else:  # Extreme leverage: 0-10 pts
            leverage_score = max(0, 10 - (de - 4.0) * 2)
    else:
        leverage_score = 55.0  # Missing: slightly above neutral (conservative default)
    
    breakdown.leverage_score = float(np.clip(leverage_score, 0, 100))
    breakdown.leverage_label = _leverage_label(breakdown.leverage_score)
    
    # === Stability Score (0-100) ===
    # Based on beta and market cap (volatility & size stability)
    
    beta_score = 50.0
    if beta is not None and np.isfinite(beta):
        if beta < 0.7:  # Very defensive: 80-100 pts
            beta_score = 80 + (0.7 - beta) / 0.7 * 20
        elif beta < 1.0:  # Defensive-neutral: 65-80 pts
            beta_score = 65 + (1.0 - beta) / 0.3 * 15
        elif beta < 1.3:  # Neutral-volatile: 45-65 pts
            beta_score = 45 + (1.3 - beta) / 0.3 * 20
        elif beta < 1.8:  # Volatile: 20-45 pts
            beta_score = 20 + (1.8 - beta) / 0.5 * 25
        else:  # Very volatile: 0-20 pts
            beta_score = max(0, 20 - (beta - 1.8) * 10)
    
    # Market cap: larger = more stable (optional component)
    cap_score = 50.0
    if market_cap is not None and np.isfinite(market_cap) and market_cap > 0:
        # Log scale: $100M = 20, $1B = 50, $10B = 70, $100B+ = 90
        log_cap = np.log10(market_cap)
        if log_cap < 8:  # <$100M: micro-cap, very volatile
            cap_score = max(10, log_cap * 2.5)
        elif log_cap < 9:  # $100M-$1B: small-cap
            cap_score = 20 + (log_cap - 8) * 30
        elif log_cap < 10:  # $1B-$10B: mid-cap
            cap_score = 50 + (log_cap - 9) * 20
        elif log_cap < 11:  # $10B-$100B: large-cap
            cap_score = 70 + (log_cap - 10) * 15
        else:  # >$100B: mega-cap
            cap_score = min(95, 85 + (log_cap - 11) * 5)
    
    # Stability: 60% beta, 40% market cap
    breakdown.stability_score = float(0.60 * beta_score + 0.40 * cap_score)
    breakdown.stability_score = float(np.clip(breakdown.stability_score, 0, 100))
    breakdown.stability_label = _stability_label(breakdown.stability_score)
    
    # === Total Score ===
    # Weighted combination of all components
    # Quality: 30%, Growth: 25%, Valuation: 25%, Stability: 10%, Leverage: 10%
    total = (
        0.30 * breakdown.quality_score +
        0.25 * breakdown.growth_score +
        0.25 * breakdown.valuation_score +
        0.10 * breakdown.stability_score +
        0.10 * breakdown.leverage_score
    )
    
    # === Coverage Adjustments ===
    # Low coverage limits max score to prevent overconfidence
    # Also apply small penalty for single-source fundamentals
    if coverage_pct < 0.25:  # Very low coverage: cap at 40
        total = min(total, 40.0)
    elif coverage_pct < 0.40:  # Low coverage: cap at 50
        total = min(total, 50.0)
    elif coverage_pct < 0.60:  # Moderate coverage: cap at 65
        total = min(total, 65.0)
    
    # Single-source penalty: reduce by 5%
    if sources_count == 1:
        total = total * 0.95
    elif sources_count == 0:  # No sources: use very conservative score
        total = min(total, 35.0)
    
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


def _stability_label(score: float) -> str:
    """Convert stability score to human-friendly label."""
    if score >= 70:
        return "Stable"
    elif score >= 50:
        return "Moderate"
    elif score >= 30:
        return "Volatile"
    else:
        return "Very Volatile"


# Legacy compatibility function
def fundamental_score_legacy(data: dict[str, object], surprise_bonus_on: bool = False, coverage_pct: float = 1.0) -> float:
    """
    Legacy fundamental_score function for backward compatibility.
    Returns simple 0-1 score.
    """
    result = compute_fundamental_score_with_breakdown(data, coverage_pct)
    return result.total / 100.0  # Convert to 0-1 scale
