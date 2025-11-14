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
    df: pd.DataFrame, total: float, min_pos: float, max_pos_pct: float, *, score_col: str = "Score"
) -> pd.DataFrame:
    """Allocate budget across tickers proportionally to score with min/max constraints.

    Pure function extracted from `stock_scout.py` to allow testing without importing UI side-effects.
    Args:
        df: DataFrame containing at least columns ["Ticker", score_col]. Optionally an existing allocation column will be overwritten.
        total: Total budget in dollars.
        min_pos: Minimum dollar allocation per included position (0 for none).
        max_pos_pct: Maximum position as percent of total (e.g. 15 for 15%).
        score_col: Column name representing relative score weights.
    Returns:
        DataFrame copy with new column "סכום קנייה ($)" populated.
    """
    df = df.copy()
    df["סכום קנייה ($)"] = 0.0
    if total <= 0 or df.empty:
        return df
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
        weights = df[score_col].clip(lower=0).to_numpy(dtype=float)
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
