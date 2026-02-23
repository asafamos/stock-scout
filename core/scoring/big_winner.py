"""
Big Winner Signal Detection for Stock Scout (20-day horizon).

Identifies potential "big winner" setups using purely technical criteria:
ATR volatility, technical score band, RSI, and risk/reward ratio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.scoring_config import BIG_WINNER_THRESHOLDS


def compute_big_winner_signal_20d(row: pd.Series) -> dict:
    """Compute a purely technical big-winner signal for a 20d horizon.

    Required inputs (from :func:`build_technical_indicators`):
        - TechScore_20d, RSI, ATR_Pct, RR (optional), MomCons (optional), VolSurge (optional).

    Returns:
        Dict with ``BigWinnerScore_20d`` (0-100) and ``BigWinnerFlag_20d`` (0 or 1).
    """
    try:
        tech_score = row.get("TechScore_20d", np.nan)
        rsi = row.get("RSI", np.nan)
        atr = row.get("ATR_Pct", np.nan)
        rr = row.get("RR", np.nan)

        if np.isnan(tech_score) or np.isnan(rsi) or np.isnan(atr):
            return {"BigWinnerScore_20d": 0.0, "BigWinnerFlag_20d": 0}

        T = BIG_WINNER_THRESHOLDS
        score = 0.0

        # ATR filter
        if atr >= T["atr_min"]:
            score += 40
            if atr >= T["atr_bonus"]:
                score += 20

        # TechScore band
        if T["score_mid_low"] <= tech_score < T["score_mid_high"]:
            score += 30
        elif T["score_weak_low"] <= tech_score < T["score_weak_high"]:
            score += 15

        # RSI filter
        if rsi < T["rsi_max"]:
            score += 10
            if rsi < T["rsi_bonus"]:
                score += 5

        # R/R bonus
        if not np.isnan(rr) and T["rr_min"] <= rr <= T["rr_max"]:
            score += 5

        score = float(np.clip(score, 0, 100))
        flag = 1 if score >= T["signal_threshold"] else 0
        return {"BigWinnerScore_20d": score, "BigWinnerFlag_20d": flag}
    except (TypeError, ValueError, KeyError):
        return {"BigWinnerScore_20d": 0.0, "BigWinnerFlag_20d": 0}
