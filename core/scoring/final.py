"""
Final Score Computation for Stock Scout.

Combines technical, fundamental, ML, big-winner, and pattern components
into a single 0-100 score.  All weights sourced from
:mod:`core.scoring_config` (single source of truth).

Functions:
    compute_final_score              — 3-component (tech + fund + ML)
    compute_final_score_with_patterns — 5-component (+ big-winner + pattern)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from core.scoring_config import (
    FINAL_SCORE_WEIGHTS,
    ML_GATES,
    ML_GATES_STRONG,
    PATTERN_SCORE_WEIGHTS,
    REGIME_MULTIPLIERS,
    RR_GATES,
)


def compute_final_score(
    tech_score: float,
    fundamental_score: Optional[float] = None,
    ml_prob: Optional[float] = None,
    market_regime: Optional[str] = None,
    rr_ratio: Optional[float] = None,
) -> float:
    """Combine tech + fund + ML into a 0-100 score.

    Delegates to :func:`compute_final_score_with_patterns` with neutral
    big-winner/pattern channels so weight logic lives in one place.
    """
    score, _ = compute_final_score_with_patterns(
        tech_score=tech_score,
        fundamental_score=fundamental_score,
        ml_prob=ml_prob,
        big_winner_score=None,
        pattern_score=None,
        bw_weight=0.0,
        pattern_weight=0.0,
        market_regime=market_regime,
        rr_ratio=rr_ratio,
    )
    return score


def compute_final_score_with_patterns(
    tech_score: float,
    fundamental_score: Optional[float] = None,
    ml_prob: Optional[float] = None,
    big_winner_score: Optional[float] = None,
    pattern_score: Optional[float] = None,
    bw_weight: float = 0.10,
    pattern_weight: float = 0.10,
    market_regime: Optional[str] = None,
    rr_ratio: Optional[float] = None,
) -> Tuple[float, Dict]:
    """Enhanced final score with up to 5 components.

    Weights auto-normalized to sum to 1.0.
    ML gate (penalty/bonus) and market-regime multiplier applied last.

    Returns:
        ``(final_score, breakdown_dict)`` where *final_score* is 0-100.
    """
    tech = float(tech_score) if tech_score is not None else 50.0
    fund = float(fundamental_score) if fundamental_score is not None and pd.notna(fundamental_score) else 50.0
    ml_sc = float(ml_prob) if ml_prob is not None and pd.notna(ml_prob) else 0.5
    bw = float(big_winner_score) if big_winner_score is not None and pd.notna(big_winner_score) else 50.0
    patt = float(pattern_score) if pattern_score is not None and pd.notna(pattern_score) else 50.0

    # ML circuit breaker: scale ML weight by model quality
    try:
        from core.ml_20d_inference import get_ml_weight_multiplier
        ml_quality = get_ml_weight_multiplier()
    except Exception:
        ml_quality = 1.0

    # AUC-tiered gate selection:
    #   ml_quality >= 0.5 (AUC > 0.55) → use ML_GATES_STRONG
    #   otherwise → use conservative ML_GATES
    use_strong_gates = ml_quality >= 0.5  # AUC > 0.55
    active_ml_gates = ML_GATES_STRONG if use_strong_gates else ML_GATES

    # Build weights
    if bw_weight == 0.0 and pattern_weight == 0.0:
        base_w: Dict[str, float] = {**FINAL_SCORE_WEIGHTS, "big_winner": 0.0, "pattern": 0.0}
    else:
        base_w = {**PATTERN_SCORE_WEIGHTS, "big_winner": bw_weight, "pattern": pattern_weight}

    # Apply ML circuit breaker: reduce ML weight, redistribute to technical
    if ml_quality < 1.0 and "ml" in base_w:
        original_ml = base_w["ml"]
        base_w["ml"] = original_ml * ml_quality
        # Redistribute the freed weight to technical (most reliable signal)
        base_w["technical"] = base_w.get("technical", 0) + (original_ml - base_w["ml"])

    # Normalize
    total = sum(base_w.values())
    nw = {k: v / total for k, v in base_w.items()}

    # Weighted sum
    final = (
        nw.get("technical", 0) * tech
        + nw.get("fundamental", 0) * fund
        + nw.get("ml", 0) * (ml_sc * 100.0)
        + nw.get("big_winner", 0) * bw
        + nw.get("pattern", 0) * patt
    )

    # ── ML gate (penalty / bonus) ────────────────────────────────
    multiplier = 1.0
    try:
        ml_val = float(ml_prob) if ml_prob is not None else np.nan
    except Exception:
        ml_val = np.nan
    if pd.notna(ml_val):
        if ml_val < float(active_ml_gates.get("penalty_lt", 0.15)):
            multiplier = float(active_ml_gates.get("penalty_mult", 0.60))
        elif ml_val > float(active_ml_gates.get("bonus_gt", 0.62)):
            multiplier = float(active_ml_gates.get("bonus_mult", 1.15))

    # ── Risk/Reward gate (penalty / bonus) ──────────────────────
    rr_mult = 1.0
    try:
        rr_val = float(rr_ratio) if rr_ratio is not None else np.nan
    except Exception:
        rr_val = np.nan
    if pd.notna(rr_val) and rr_val >= 0:
        if rr_val < float(RR_GATES.get("harsh_penalty_lt", 0.5)):
            rr_mult = float(RR_GATES.get("harsh_penalty_mult", 0.80))
        elif rr_val < float(RR_GATES.get("mild_penalty_lt", 1.0)):
            rr_mult = float(RR_GATES.get("mild_penalty_mult", 0.90))
        elif rr_val < float(RR_GATES.get("mediocre_penalty_lt", 1.5)):
            rr_mult = float(RR_GATES.get("mediocre_penalty_mult", 0.95))
        elif rr_val > float(RR_GATES.get("strong_bonus_gt", 4.0)):
            rr_mult = float(RR_GATES.get("strong_bonus_mult", 1.12))
        elif rr_val > float(RR_GATES.get("mild_bonus_gt", 2.5)):
            rr_mult = float(RR_GATES.get("mild_bonus_mult", 1.06))

    # ── Regime adjustment ────────────────────────────────────────
    regime_mult = 1.0
    if isinstance(market_regime, str):
        regime_mult = float(REGIME_MULTIPLIERS.get(market_regime.upper(), 1.0))

    final = float(np.clip(final * multiplier * rr_mult * regime_mult, 0.0, 100.0))

    breakdown = {
        "tech_component": nw.get("technical", 0) * tech,
        "fund_component": nw.get("fundamental", 0) * fund,
        "ml_component": nw.get("ml", 0) * (ml_sc * 100.0),
        "bw_component": nw.get("big_winner", 0) * bw,
        "pattern_component": nw.get("pattern", 0) * patt,
        "rr_multiplier": rr_mult,
        "final_score": final,
        "weights_used": nw,
    }
    return final, breakdown
