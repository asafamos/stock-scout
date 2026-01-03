"""
Portfolio allocation utilities.

This module contains helper functions for portfolio construction
and weight normalization.
"""
from __future__ import annotations
import numpy as np
from typing import Dict
import pandas as pd

def allocate_budget(
    df: pd.DataFrame, total: float, min_pos: float, max_pos_pct: float, *, score_col: str = "Score", dynamic_sizing: bool = True
) -> pd.DataFrame:
    """Allocate budget across tickers proportionally to score with min/max constraints.
    
    NEW (Nov 2025): Dynamic position sizing based on confidence levels:
    - High confidence (RSI<30, RR>3, MomCons>0.7): 2.0x weight
    - Medium confidence: 1.0x weight
    - Low confidence: 0.5x weight

    Pure function extracted from `stock_scout.py` to allow testing without importing UI side-effects.
    Args:
        df: DataFrame containing at least columns ["Ticker", score_col]. Optionally an existing allocation column will be overwritten.
        total: Total budget in dollars.
        min_pos: Minimum dollar allocation per included position (0 for none).
        max_pos_pct: Maximum position as percent of total (e.g. 15 for 15%).
        score_col: Column name representing relative score weights.
        dynamic_sizing: Whether to apply confidence-based position sizing (default True).
    Returns:
        DataFrame copy with new column "סכום קנייה ($)" populated.
    """
    df = df.copy()
    df["סכום קנייה ($)"] = 0.0
    if total <= 0 or df.empty:
        return df
    
    # Calculate risk multipliers if dynamic sizing enabled
    adjusted_scores = df[score_col]
    if dynamic_sizing:
        try:
            from core.v2_risk_engine import compute_position_risk_factor
            # Prefer canonical risk factor when classification/pipeline fields exist
            if ("RiskClass" in df.columns) or ("Risk_Level" in df.columns) or ("SafetyBlocked" in df.columns):
                df["_risk_factor"] = df.apply(compute_position_risk_factor, axis=1)
                adjusted_scores = df[score_col] * df["_risk_factor"].astype(float)
            elif "RSI" in df.columns:
                # Fallback legacy heuristic (kept for compatibility when canonical fields missing)
                df["_confidence_mult"] = 1.0
                rsi_cond = df.get("RSI", pd.Series(50)).fillna(50) < 30
                rr_cond = df.get("RR_Ratio", df.get("RewardRisk", pd.Series(0))).fillna(0) > 3.0
                mom_cond = df.get("Momentum_Consistency", pd.Series(0)).fillna(0) > 0.7
                high_conf = rsi_cond & rr_cond & mom_cond
                df.loc[high_conf, "_confidence_mult"] = 2.0
                rsi_low = df.get("RSI", pd.Series(50)).fillna(50) > 50
                rr_low = df.get("RR_Ratio", df.get("RewardRisk", pd.Series(2))).fillna(2) < 1.5
                mom_low = df.get("Momentum_Consistency", pd.Series(0.5)).fillna(0.5) < 0.4
                low_conf = rsi_low | rr_low | mom_low
                df.loc[low_conf & ~high_conf, "_confidence_mult"] = 0.5
                adjusted_scores = df[score_col] * df["_confidence_mult"]
        except Exception:
            # On any import/apply failure, fall back to raw scores
            adjusted_scores = df[score_col]
    
    df = df.sort_values([score_col, "Ticker"], ascending=[False, True]).reset_index(drop=True)
    remaining = float(total)
    n = len(df)
    max_pos_abs = (max_pos_pct / 100.0) * total if max_pos_pct > 0 else float("inf")
    if min_pos > 0:
        can_min = int(min(n, remaining // min_pos))
        if can_min > 0:
            base = pd.Series(np.full(can_min, min(min_pos, max_pos_abs), dtype=float))
            df.loc[: can_min - 1, "סכום קנייה ($)"] = base
            remaining -= float(base.sum())
    if remaining > 0:
        # Use adjusted scores for proportional allocation
        weights = adjusted_scores.clip(lower=0).to_numpy(dtype=float)
        extras = (
            np.full(n, remaining / n, dtype=float)
            if np.nansum(weights) <= 0
            else remaining * (np.nan_to_num(weights, nan=0.0) / np.nansum(weights))
        )
        current = df["סכום קנייה ($)"].to_numpy(dtype=float)
        proposed = current + extras
        if np.isfinite(max_pos_abs):
            proposed = np.minimum(proposed, max_pos_abs)
        df["סכום קנייה ($)"] = proposed
    s = float(df["סכום קנייה ($)"].sum())
    if s > 0 and abs(s - total) / max(total, 1) > 1e-6:
        df["סכום קנייה ($)"] = df["סכום קנייה ($)"].to_numpy(dtype=float) * (total / s)
    df["סכום קנייה ($)"] = df["סכום קנייה ($)"].round(2)
    
    # Clean up temporary column
    for tmp in ("_confidence_mult", "_risk_factor"):
        if tmp in df.columns:
            df.drop(columns=[tmp], inplace=True)
    
    return df


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize a weights dictionary so that values sum to 1.0.
    
    Args:
        weights: Dictionary mapping keys to numeric weights
        
    Returns:
        Dictionary with same keys but normalized values summing to 1.0
        
    Example:
        >>> _normalize_weights({"a": 2, "b": 3, "c": 5})
        {"a": 0.2, "b": 0.3, "c": 0.5}
    """
    total = sum(weights.values())
    if total <= 0 or not np.isfinite(total):
        # if invalid total, return equal weights
        n = len(weights)
        return {k: 1.0 / n if n > 0 else 0.0 for k in weights}
    return {k: v / total for k, v in weights.items()}
