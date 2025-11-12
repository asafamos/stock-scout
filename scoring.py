import numpy as np
import pandas as pd
from typing import Dict


# Default scoring weights (can be overridden by CONFIG in stock_scout.py)
WEIGHTS: Dict[str, float] = {"momentum": 0.4, "trend": 0.3, "fundamental": 0.3}


def normalize_series(s: pd.Series) -> pd.Series:
    """Normalize a pandas Series to 0..1 range.

    Returns a series with same index. If values are all NaN or constant,
    returns a series of zeros (to avoid division by zero).
    """
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return pd.Series(0.0, index=s.index)
    mn = s.min()
    mx = s.max()
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)


def _normalize_weights(d: Dict[str, float]) -> Dict[str, float]:
    keys = [
        "ma",
        "mom",
        "rsi",
        "near_high_bell",
        "vol",
        "overext",
        "pullback",
        "risk_reward",
        "macd",
        "adx",
    ]
    w = {k: float(d.get(k, 0.0)) for k in keys}
    s = sum(max(0.0, v) for v in w.values())
    s = 1.0 if s <= 0 else s
    return {k: max(0.0, v) / s for k, v in w.items()}


def allocate_budget(
    df: pd.DataFrame, total: float, min_pos: float, max_pos_pct: float
) -> pd.DataFrame:
    df = df.copy()
    df["סכום קנייה ($)"] = 0.0
    if total <= 0 or df.empty:
        return df

    # Sort by Score desc, ticker asc for deterministic ordering
    df = df.sort_values(["Score", "Ticker"], ascending=[False, True]).reset_index(
        drop=True
    )

    n = len(df)
    max_pos_abs = (max_pos_pct / 100.0) * total if max_pos_pct > 0 else float("inf")

    # Ensure min_pos is not greater than max_pos_abs
    if np.isfinite(max_pos_abs) and min_pos > max_pos_abs:
        min_pos = float(max_pos_abs)

    # Establish raw weights
    weights = df.get("Score", pd.Series(0.0, index=df.index)).clip(lower=0).to_numpy(dtype=float)
    if np.nansum(weights) <= 0:
        weights = np.ones(n, dtype=float)

    # Initial proportional allocation
    prop = weights / float(np.nansum(weights))
    proposed = prop * float(total)

    # Enforce per-position bounds [min_pos, max_pos_abs]
    if np.isfinite(max_pos_abs):
        proposed = np.minimum(proposed, max_pos_abs)
    if min_pos > 0:
        proposed = np.maximum(proposed, min_pos)

    # Iteratively adjust to match total while respecting bounds
    for _ in range(10):
        s = float(np.nansum(proposed))
        diff = float(total - s)
        if abs(diff) <= 1e-6:
            break

        if diff > 0:
            # distribute extra to positions not at max
            capacity_mask = ~np.isfinite(max_pos_abs) | (proposed < max_pos_abs - 1e-9)
            capacity = np.where(capacity_mask, (max_pos_abs - proposed), 0.0)
            total_capacity = float(np.nansum(np.where(capacity_mask, capacity, 0.0)))
            if total_capacity <= 1e-9:
                break
            # distribute proportionally to available capacity
            add = capacity * (diff / total_capacity)
            proposed = proposed + add
            if min_pos > 0:
                proposed = np.maximum(proposed, min_pos)
        else:
            # remove excess from positions above min_pos
            reducible_mask = proposed > min_pos + 1e-9
            reducible = np.where(reducible_mask, (proposed - min_pos), 0.0)
            total_reducible = float(np.nansum(reducible))
            if total_reducible <= 1e-9:
                break
            remove = reducible * ((-diff) / total_reducible)
            proposed = proposed - remove

    # Final clipping and rounding
    if np.isfinite(max_pos_abs):
        proposed = np.minimum(proposed, max_pos_abs)
    if min_pos > 0:
        proposed = np.maximum(proposed, min_pos)

    # If sum still differs from total due to rounding or bound constraints, scale proportionally among free positions
    s = float(np.nansum(proposed))
    if s > 0 and abs(s - total) / max(total, 1) > 1e-6:
        free_mask = (proposed > min_pos + 1e-9) & (~(np.isfinite(max_pos_abs) & (proposed >= max_pos_abs - 1e-9)))
        if np.any(free_mask):
            scale = (total - np.nansum(np.where(~free_mask, proposed, 0.0))) / float(np.nansum(np.where(free_mask, proposed, 0.0)))
            proposed = np.where(free_mask, proposed * scale, proposed)

    df["סכום קנייה ($)"] = np.round(proposed, 2)
    # Final safety: ensure total does not exceed budget (rounding may push it over by cents)
    total_assigned = float(df["סכום קנייה ($)"].sum())
    if total_assigned > total:
        # scale down proportionally
        df["סכום קנייה ($)"] = np.round(df["סכום קנייה ($)"].to_numpy(dtype=float) * (total / total_assigned), 2)

    return df


def fundamental_score(d: dict, surprise_bonus_on: bool = False) -> float:
    def _to_01(x, low, high):
        if not isinstance(x, (int, float)) or not np.isfinite(x):
            return np.nan
        return np.clip((x - low) / (high - low), 0, 1)

    g_rev = _to_01(d.get("rev_g_yoy", np.nan), 0.00, 0.30)
    g_eps = _to_01(d.get("eps_g_yoy", np.nan), 0.00, 0.30)
    growth = np.nanmean([g_rev, g_eps])

    q_roe = _to_01(d.get("roe", np.nan), 0.05, 0.25)
    q_roic = _to_01(d.get("roic", np.nan), 0.05, 0.20)
    q_gm = _to_01(d.get("gm", np.nan), 0.10, 0.60)
    quality = np.nanmean([q_roe, q_roic, q_gm])

    pe = d.get("pe", np.nan)
    ps = d.get("ps", np.nan)
    val_pe = None if not np.isfinite(pe) else _to_01(40 - np.clip(pe, 0, 40), 0, 40)
    val_ps = None if not np.isfinite(ps) else _to_01(10 - np.clip(ps, 0, 10), 0, 10)
    vals = [v for v in (val_pe, val_ps) if v is not None and np.isfinite(v)]
    valuation = np.nan if not vals else float(np.mean(vals))

    penalty = 0.0
    de = d.get("de", np.nan)
    if isinstance(de, (int, float)) and np.isfinite(de) and de > 2.0:
        penalty += 0.15

    comp = np.nanmean([growth, quality, valuation])
    comp = 0.0 if not np.isfinite(comp) else float(np.clip(comp, 0.0, 1.0))

    if surprise_bonus_on:
        surprise = d.get("surprise", np.nan)
        comp += (0.05 if (isinstance(surprise, (int, float)) and surprise >= 2.0) else 0.0)

    comp -= penalty
    return float(np.clip(comp, 0.0, 1.0))
