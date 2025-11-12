import numpy as np
import pandas as pd
from typing import Dict


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
    df = df.sort_values(["Score", "Ticker"], ascending=[False, True]).reset_index(
        drop=True
    )
    remaining = float(total)
    n = len(df)
    max_pos_abs = (max_pos_pct / 100.0) * total if max_pos_pct > 0 else float("inf")
    if min_pos > 0:
        can_min = int(min(n, remaining // min_pos))
        if can_min > 0:
            base = np.full(can_min, min(min_pos, max_pos_abs), dtype=float)
            df.loc[: can_min - 1, "סכום קנייה ($)"] = base
            remaining -= float(base.sum())
    if remaining > 0:
        weights = df["Score"].clip(lower=0).to_numpy(dtype=float)
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
