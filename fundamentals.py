# filepath: fundamentals.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional

BUCKET_WEIGHTS: Dict[str, float] = {
    "quality": 0.35, "growth": 0.25, "valuation": 0.25, "leverage": 0.15
}

def _cap_bucket(mcap: float | None) -> str:
    """Bucket market cap into small/mid/large/unknown."""
    if mcap is None or not np.isfinite(mcap):
        return "unknown"
    if mcap < 2e9:
        return "small"
    if mcap < 10e9:
        return "mid"
    return "large"

def zscore_by_group(series: pd.Series,
                    sector: pd.Series,
                    cap: pd.Series,
                    clip_val: float = 3.0) -> pd.Series:
    """
    Z-score within (sector, cap_bucket). Returns 0 for empty/constant groups.
    Preserves original index.
    """
    df = pd.DataFrame({"x": series, "sec": sector, "cap": cap})
    def _z(g: pd.DataFrame) -> pd.Series:
        x = g["x"].astype(float)
        mu = x.mean()
        sd = x.std(ddof=0)
        if sd == 0 or not np.isfinite(sd):
            return pd.Series(0.0, index=g.index)
        z = (x - mu) / sd
        return z.clip(-clip_val, clip_val).fillna(0.0)
    res = df.groupby(["sec", "cap"], dropna=False, observed=False).apply(_z)
    # groupby.apply adds group-level idx — restore original index
    res.index = res.index.droplevel([0,1])
    return res.reindex(series.index).fillna(0.0)

def _safe_weighted_mean(d: Dict[str, float], w: Dict[str, float]) -> float:
    """
    Weighted mean across available metrics. If a metric is missing (NaN), it's skipped
    and the remaining weights are renormalized.
    """
    avail = {k: v for k, v in d.items() if np.isfinite(v)}
    if not avail:
        return 0.0
    ws = {k: w.get(k, 0.0) for k in avail}
    s = sum(ws.values())
    if s <= 0:
        return 0.0
    return sum(avail[k] * (ws[k] / s) for k in avail)

def compute_bucket_scores(metrics: pd.DataFrame,
                          meta: pd.DataFrame) -> pd.DataFrame:
    """
    Compute bucket scores (quality, growth, valuation, leverage) in range [0..100].

    metrics: columns may include roic, fcf_margin, oper_margin, gross_margin_stability,
             rev_yoy, eps_yoy, fcf_yoy, fwd_pe, ev_ebitda, ps, peg,
             debt_to_equity, net_debt_ebitda, interest_cover, share_dilution_yoy
    meta:    columns: sector, market_cap
    """
    sec = meta["sector"].fillna("Unknown")
    cap = meta["market_cap"].apply(_cap_bucket)

    def Z(col: str, invert: bool = False) -> pd.Series:
        if col not in metrics.columns:
            # missing metric — return NaN so safe_weighted_mean handles it
            return pd.Series(np.nan, index=metrics.index)
        z = zscore_by_group(metrics[col], sec, cap)
        return (-z if invert else z)

    QW = {"roic": .4, "fcf_margin": .3, "oper_margin": .2, "gross_margin_stability": .1}
    GW = {"rev_yoy": .5, "eps_yoy": .3, "fcf_yoy": .2}
    VW = {"fwd_pe": .4, "ev_ebitda": .3, "ps": .2, "peg": .1}  # lower is better → invert sign
    LW = {"debt_to_equity": .35, "net_debt_ebitda": .35, "interest_cover": .2, "share_dilution_yoy": .1}

    # prepare dicts of per-metric z-scores (may contain NaN for missing metrics)
    q = {k: Z(k) for k in QW.keys()}
    g = {k: Z(k) for k in GW.keys()}
    v = {k: Z(k, invert=True) for k in VW.keys()}
    l = {**{k: Z(k, invert=True) for k in ("debt_to_equity","net_debt_ebitda","share_dilution_yoy")},
         **{"interest_cover": Z("interest_cover")}}

    out = pd.DataFrame(index=metrics.index)
    out["q_score"] = (pd.DataFrame(q).apply(lambda r: _safe_weighted_mean(r.to_dict(), QW), axis=1) * 100).clip(0, 100)
    out["g_score"] = (pd.DataFrame(g).apply(lambda r: _safe_weighted_mean(r.to_dict(), GW), axis=1) * 100).clip(0, 100)
    out["v_score"] = (pd.DataFrame(v).apply(lambda r: _safe_weighted_mean(r.to_dict(), VW), axis=1) * 100).clip(0, 100)
    out["l_score"] = (pd.DataFrame(l).apply(lambda r: _safe_weighted_mean(r.to_dict(), LW), axis=1) * 100).clip(0, 100)
    return out

def compute_fundamental_score(bucket_scores: pd.DataFrame) -> pd.Series:
    """
    Combine bucket scores into a single fundamental score in 0..100.
    """
    w = BUCKET_WEIGHTS
    f = (w["quality"]   * bucket_scores["q_score"] +
         w["growth"]    * bucket_scores["g_score"] +
         w["valuation"] * bucket_scores["v_score"] +
         w["leverage"]  * bucket_scores["l_score"])
    return f.clip(0, 100)
