"""
Final Score Computation for Stock Scout.

Thin wrappers around :func:`core.scoring_engine.compute_final_score_20d`
(single source of truth for all scoring).  These wrappers accept scalar
arguments for backward compatibility with callers that don't have a full
``pd.Series`` row.

Functions:
    compute_final_score              — scalar-args wrapper (tech + fund + ML)
    compute_final_score_with_patterns — scalar-args wrapper (+ big-winner + pattern)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def compute_final_score(
    tech_score: float,
    fundamental_score: Optional[float] = None,
    ml_prob: Optional[float] = None,
    market_regime: Optional[str] = None,
    rr_ratio: Optional[float] = None,
) -> float:
    """Combine tech + fund + ML into a 0-100 score.

    Builds a minimal ``pd.Series`` and delegates to
    :func:`core.scoring_engine.compute_final_score_20d` so that all
    scoring paths use the same CONVICTION_WEIGHTS and adjustments.
    """
    row = _build_scoring_row(
        tech_score=tech_score,
        fundamental_score=fundamental_score,
        ml_prob=ml_prob,
        market_regime=market_regime,
        rr_ratio=rr_ratio,
    )
    from core.scoring_engine import compute_final_score_20d
    return compute_final_score_20d(row)


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

    Delegates to :func:`core.scoring_engine.compute_final_score_20d`
    with pattern/big-winner fields populated on the row.

    Returns:
        ``(final_score, breakdown_dict)`` where *final_score* is 0-100.
    """
    row = _build_scoring_row(
        tech_score=tech_score,
        fundamental_score=fundamental_score,
        ml_prob=ml_prob,
        market_regime=market_regime,
        rr_ratio=rr_ratio,
        pattern_score=pattern_score,
        big_winner_score=big_winner_score,
    )
    from core.scoring_engine import compute_final_score_20d
    score, breakdown = compute_final_score_20d(row, return_breakdown=True)
    return score, breakdown


def _build_scoring_row(
    tech_score: float,
    fundamental_score: Optional[float] = None,
    ml_prob: Optional[float] = None,
    market_regime: Optional[str] = None,
    rr_ratio: Optional[float] = None,
    pattern_score: Optional[float] = None,
    big_winner_score: Optional[float] = None,
) -> pd.Series:
    """Build a minimal pd.Series with canonical column names for scoring."""
    tech = float(tech_score) if tech_score is not None else 50.0
    fund = float(fundamental_score) if fundamental_score is not None and pd.notna(fundamental_score) else 50.0
    ml_val = float(ml_prob) if ml_prob is not None and pd.notna(ml_prob) else 0.5

    data = {
        "TechScore_20d": tech,
        "Fundamental_S": fund,
        "ML_20d_Prob": ml_val,
        "Reliability_Score": 50.0,  # neutral default when not available
        "RR": rr_ratio if rr_ratio is not None else np.nan,
        "Market_Regime": market_regime or "",
    }
    if pattern_score is not None and pd.notna(pattern_score):
        data["Pattern_Score"] = float(pattern_score)
    if big_winner_score is not None and pd.notna(big_winner_score):
        data["Big_Winner_Signal"] = float(big_winner_score) / 100.0  # normalize 0-100 to 0-1
    return pd.Series(data)
