from __future__ import annotations
import math
import pandas as pd
import numpy as np
from typing import Optional, Dict
from fundamentals import compute_bucket_scores, compute_fundamental_score, zscore_by_group

WEIGHTS: Dict[str, float] = {"momentum": 0.4, "trend": 0.3, "fundamental": 0.3}


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize a weights dictionary so that values sum to 1.0."""
    total = sum(weights.values())
    if total <= 0 or not np.isfinite(total):
        # if invalid total, return equal weights
        n = len(weights)
        return {k: 1.0 / n if n > 0 else 0.0 for k in weights}
    return {k: v / total for k, v in weights.items()}


def fundamental_score(data: dict, surprise_bonus_on: bool = False) -> float:
    """
    Compute a simple fundamental score (0..1) from a bundle dictionary.
    Expected keys: roe, roic, gm, ps, pe, de, rev_g_yoy, eps_g_yoy
    
    This is a legacy/simplified scoring function used by stock_scout.py.
    For more sophisticated scoring, use compute_bucket_scores + compute_fundamental_score.
    """
    def _norm(val, low, high):
        """Normalize value to [0,1] range."""
        if not isinstance(val, (int, float)) or not np.isfinite(val):
            return 0.0
        return np.clip((val - low) / (high - low), 0.0, 1.0)
    
    # Quality metrics (higher is better)
    roe_score = _norm(data.get("roe", 0), 0, 0.25)  # ROE: 0-25%
    roic_score = _norm(data.get("roic", 0), 0, 0.20)  # ROIC: 0-20%
    gm_score = _norm(data.get("gm", 0), 0, 0.50)  # Gross margin: 0-50%
    
    # Growth metrics (higher is better)
    rev_g = data.get("rev_g_yoy", 0)
    eps_g = data.get("eps_g_yoy", 0)
    rev_g_score = _norm(rev_g, -0.10, 0.30)  # Revenue growth: -10% to 30%
    eps_g_score = _norm(eps_g, -0.20, 0.50)  # EPS growth: -20% to 50%
    
    # Valuation metrics (lower is better, so invert)
    pe = data.get("pe", np.nan)
    ps = data.get("ps", np.nan)
    pe_score = 1.0 - _norm(pe, 5, 40) if np.isfinite(pe) else 0.5  # P/E: 5-40
    ps_score = 1.0 - _norm(ps, 0.5, 10) if np.isfinite(ps) else 0.5  # P/S: 0.5-10
    
    # Leverage penalty (lower D/E is better)
    de = data.get("de", np.nan)
    de_penalty = _norm(de, 0, 2.0) if np.isfinite(de) else 0.0  # D/E: 0-2
    
    # Weighted average
    quality = 0.35 * (roe_score + roic_score + gm_score) / 3
    growth = 0.30 * (rev_g_score + eps_g_score) / 2
    valuation = 0.25 * (pe_score + ps_score) / 2
    leverage = 0.10 * (1.0 - de_penalty)
    
    base_score = quality + growth + valuation + leverage
    
    # Optional surprise bonus (not implemented here, placeholder)
    bonus = 0.0
    if surprise_bonus_on:
        bonus = 0.0  # would add earnings surprise logic here
    
    return np.clip(base_score + bonus, 0.0, 1.0)


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
    - sizes positions proportional to score / ATR% (lower ATR -> larger size)
    - enforces per-position and per-sector caps
    - floors to whole shares and trims to respect budget
    Returns a DataFrame indexed by ticker with columns: shares, cost, sector
    """
    import numpy as np

    df = pd.DataFrame(
        {
            "score": scores,
            "price": prices,
            "atr_pct": atr_pct,
            "sector": sector.fillna("Unknown"),
            "dvol": dollar_vol,
        }
    )

    # require score and price at minimum
    df = df.dropna(subset=["score", "price"])
    # liquidity / price filters
    df = df[(df["price"] >= min_price) & (df["dvol"] >= min_dollar_vol)]
    if df.empty:
        return pd.DataFrame(columns=["shares", "cost", "sector"])

    # sizing signal: score / abs(atr_pct) ; treat zero/NaN ATR safely
    safe_atr = df["atr_pct"].replace(0, np.nan).abs()
    size_sig = (df["score"].clip(lower=0) / safe_atr).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # fallback to score-only if ATR-based signal is all zero
    if size_sig.sum() == 0:
        size_sig = df["score"].clip(lower=0).fillna(0.0)
        if size_sig.sum() == 0:
            return pd.DataFrame(columns=["shares", "cost", "sector"])

    # raw dollar allocation proportional to size_sig
    raw = budget * size_sig / size_sig.sum()

    # enforce per-position absolute cap
    max_pos_abs = budget * max_pos_pct
    raw = raw.clip(lower=0.0, upper=max_pos_abs)

    # enforce per-sector cap (proportionally scale positions in sectors over the cap)
    sector_totals = raw.groupby(df["sector"]).sum()
    sector_cap = budget * max_sector_pct
    over_sectors = sector_totals[sector_totals > sector_cap].index.tolist()

    if over_sectors:
        for s in over_sectors:
            idxs = df.index[df["sector"] == s]
            sector_sum = raw.loc[idxs].sum()
            if sector_sum <= 0:
                continue
            allowed = sector_cap
            factor = allowed / sector_sum
            raw.loc[idxs] = raw.loc[idxs] * factor

    # final safety: ensure total allocation does not exceed budget
    total_alloc = raw.sum()
    if total_alloc > budget and total_alloc > 0:
        raw = raw * (budget / total_alloc)

    # convert to whole shares (floor) & compute cost
    shares = (raw / df["price"]).fillna(0).apply(np.floor).astype(int)
    cost = shares * df["price"]

    # if rounding pushed us over budget, trim largest positions until under budget
    # use while loop but safeguard with iteration limit
    iter_limit = max(1000, len(cost) * 10)
    it = 0
    while cost.sum() > budget and cost.sum() > 0 and it < iter_limit:
        # choose the largest cost position to decrement
        idx_max = cost.idxmax()
        if shares.loc[idx_max] <= 0:
            # nothing to trim for this ticker, remove from consideration
            cost.loc[idx_max] = 0.0
            it += 1
            continue
        shares.loc[idx_max] -= 1
        cost.loc[idx_max] = shares.loc[idx_max] * df.at[idx_max, "price"]
        it += 1

    # final result, keep original index name if present
    out = pd.DataFrame(
        {
            "shares": shares.fillna(0).astype(int),
            "cost": cost.fillna(0.0).astype(float),
            "sector": df["sector"],
        }
    )
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

def make_explain_payload(ticker_idx, final, tech_df, bucket_scores, penalties) -> dict:
    """
    Create payload for explanation API.

    ticker_idx: index of the tickers
    final: final scores series
    tech_df: technical features DataFrame
    bucket_scores: fundamental bucket scores DataFrame
    penalties: risk penalties DataFrame
    """
    # ensure all inputs are aligned by ticker index
    idx = ticker_idx
    final = final.reindex(idx).fillna(0.0)
    tech_df = tech_df.reindex(idx).fillna(0.0)
    bucket_scores = bucket_scores.reindex(idx).fillna(0.0)
    penalties = penalties.reindex(idx).fillna(0.0)

    # unpack bucket scores (assuming multi-level columns)
    bucket_df = pd.DataFrame()
    if isinstance(bucket_scores, pd.DataFrame) and not bucket_scores.empty:
        bucket_df = pd.DataFrame(
            bucket_scores.to_list(),
            index=bucket_scores.index,
            columns=[f"bucket_{i+1}" for i in range(bucket_scores.shape[1])],
        )

    # payload construction
    payload = {
        "ticker": idx.tolist(),
        "final_score": final.tolist(),
        "technical_factors": tech_df.reset_index().to_dict(orient="records"),
        "fundamental_buckets": bucket_df.reset_index().to_dict(orient="records"),
        "risk_penalties": penalties.reset_index().to_dict(orient="records"),
    }
    return payload


# Example usage (commented out to avoid NameError when importing module):
# payload = make_explain_payload(ticker_idx=tech_df.index, final=final_series,
#                                tech_df=tech_df, bucket_scores=bucket_df, penalties=pen_df)
# import json
# print(json.dumps(payload[list(payload.keys())[:3]], indent=2))  # show first 3 tickers
