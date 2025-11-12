from __future__ import annotations
import math
import pandas as pd
import numpy as np
from typing import Optional, Dict
from fundamentals import compute_bucket_scores, compute_fundamental_score, zscore_by_group

WEIGHTS: Dict[str, float] = {"momentum": 0.4, "trend": 0.3, "fundamental": 0.3}


def normalize_series(s: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0,1]; constant series -> 0."""
    s = s.astype(float)
    if s.isnull().all():
        return s.fillna(0.0)
    mn, mx = s.min(), s.max()
    if mx == mn or not np.isfinite(mn) or not np.isfinite(mx):
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)


def allocate_budget(
    scores: pd.Series,
    prices: pd.Series,
    atr_pct: pd.Series,
    sector: pd.Series,
    dollar_vol: pd.Series,
    budget: float,
    max_pos_pct: float = 0.10,
    max_sector_pct: float = 0.25,
    min_price: float = 3.0,
    min_dollar_vol: float = 1_000_000.0,
) -> pd.DataFrame:
    """
    Risk-aware allocator:
    - filters illiquid/cheap tickers
    - sizes positions proportional to score / ATR%
    - enforces per-position and per-sector caps
    - floors to shares and trims to respect budget
    """
    df = pd.DataFrame(
        {
            "score": scores,
            "price": prices,
            "atr_pct": atr_pct,
            "sector": sector.fillna("Unknown"),
            "dvol": dollar_vol,
        }
    ).dropna(subset=["score", "price"])

    # Filters
    df = df[(df["price"] >= min_price) & (df["dvol"] >= min_dollar_vol)]
    if df.empty:
        return pd.DataFrame(columns=["shares", "cost"])

    # sizing signal
    safe_atr = df["atr_pct"].replace(0, np.nan).abs()
    size_signal = (df["score"].clip(lower=0) / safe_atr).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if size_signal.sum() == 0:
        size_signal = df["score"].clip(lower=0)

    raw = budget * size_signal / size_signal.sum()

    # per-position cap
    max_pos_abs = budget * max_pos_pct
    raw = raw.clip(0.0, max_pos_abs)

    # sector cap enforcement (iterative)
    sector_totals = raw.groupby(df["sector"]).sum()
    over = sector_totals[sector_totals > (budget * max_sector_pct)]
    if not over.empty:
        for s in over.index:
            idxs = df.index[df["sector"] == s]
            sector_sum = raw.loc[idxs].sum()
            if sector_sum <= 0:
                continue
            allowed = budget * max_sector_pct
            factor = allowed / sector_sum
            raw.loc[idxs] = raw.loc[idxs] * factor

    # ensure total <= budget
    total_alloc = raw.sum()
    if total_alloc > budget and total_alloc > 0:
        raw = raw * (budget / total_alloc)

    # convert to shares (floor) & cost
    shares = (raw / df["price"]).fillna(0).apply(np.floor).astype(int)
    cost = shares * df["price"]

    # trim largest positions if rounding pushed over budget
    while cost.sum() > budget and cost.sum() > 0:
        idx = cost.idxmax()
        if shares.loc[idx] <= 0:
            break
        shares.loc[idx] -= 1
        cost.loc[idx] = shares.loc[idx] * df.at[idx, "price"]

    out = pd.DataFrame({"shares": shares, "cost": cost})
    out.index.name = scores.index.name
    return out



def final_score(
    tech_df: pd.DataFrame,
    fund_metrics: pd.DataFrame,
    meta: pd.DataFrame,
    penalties: Optional[pd.DataFrame] = None,
    alpha: float = 0.55,
) -> pd.Series:
    """
    Blend technical and fundamental signals into a final score in [0..100].

    tech_df: normalized technical features (expected columns like 'momentum_z','trend_z','overext_z')
    fund_metrics: raw fundamentals metrics DataFrame (ticker index)
    meta: DataFrame with columns 'sector' and 'market_cap' indexed by ticker
    penalties: DataFrame (same index) with numeric penalty columns to subtract from the score
    alpha: weight on technical side (default 0.55)
    """
    # compute fundamentals (0..100)
    bucket = compute_bucket_scores(fund_metrics, meta)
    fund = compute_fundamental_score(bucket).fillna(0.0)

    # tech aggregate: use z-features (expected to be around ~N(0,1)); map to a 0..100 band
    # prefer columns with _z suffix; fall back to 0-series if missing
    idx = tech_df.index if isinstance(tech_df, pd.DataFrame) else fund.index
    momentum = tech_df.get("momentum_z", pd.Series(0.0, index=idx)).astype(float)
    trend = tech_df.get("trend_z", pd.Series(0.0, index=idx)).astype(float)
    overext = tech_df.get("overext_z", pd.Series(0.0, index=idx)).astype(float)

    tech_raw = 0.35 * momentum + 0.15 * trend - 0.10 * overext
    # map z-like tech_raw to a 0..100-ish band centered ~50 with modest scale
    tech = (50.0 + 15.0 * tech_raw).clip(0.0, 100.0)

    # risk penalties: sum across penalty columns (if provided)
    if penalties is None or penalties.empty:
        risk_pen = pd.Series(0.0, index=tech.index)
    else:
        # align indices
        pen = penalties.reindex(tech.index).fillna(0.0)
        # numeric sum across columns
        risk_pen = pen.select_dtypes(include=[float, int]).sum(axis=1).fillna(0.0)

    # combine (ensure alignment)
    final = (alpha * tech + (1.0 - alpha) * fund.reindex(tech.index).fillna(0.0) - risk_pen).clip(0.0, 100.0)
    return final

def _cap_bucket_arr(mcap: pd.Series) -> pd.Series:
    """Map market_cap series to buckets: small / mid / large / unknown."""
    def _b(x: float):
        if pd.isna(x):
            return "unknown"
        try:
            if x < 2e9:
                return "small"
            if x < 10e9:
                return "mid"
            return "large"
        except Exception:
            return "unknown"
    return mcap.apply(_b)

def normalize_tech(df: pd.DataFrame, sector: pd.Series, market_cap: pd.Series) -> pd.DataFrame:
    """
    Sector- and market-cap-aware z-score normalization for technical metrics.
    For every column in `df` returns a column named '{col}_z' with the group z-score.
    Missing/constant groups return 0.0 (handled by zscore_by_group).
    """
    caps = _cap_bucket_arr(market_cap)
    sec = sector.fillna("Unknown")
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        # safe: if column missing or all-NaN, zscore_by_group will yield zeros
        out[f"{col}_z"] = zscore_by_group(df[col], sec, caps)
    return out
