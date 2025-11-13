from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional


def zscore_by_group(
    series: pd.Series, group_a: pd.Series, group_b: pd.Series
) -> pd.Series:
    """
    Compute group-wise z-scores for a series split by two grouping dimensions.
    Returns 0.0 for groups with constant or missing values.
    """
    df = pd.DataFrame({"val": series, "ga": group_a, "gb": group_b})
    df = df.dropna(subset=["val"])
    if df.empty:
        return pd.Series(0.0, index=series.index)

    def _zscore_safe(x: pd.Series) -> pd.Series:
        if len(x) < 2:
            return pd.Series(0.0, index=x.index)
        mu = x.mean()
        sigma = x.std(ddof=1)
        if sigma == 0 or not np.isfinite(sigma):
            return pd.Series(0.0, index=x.index)
        return (x - mu) / sigma

    z = df.groupby(["ga", "gb"], group_keys=False)["val"].apply(_zscore_safe)
    return z.reindex(series.index).fillna(0.0)


def _cap_bucket(mcap: float) -> str:
    """Map market_cap to size bucket: small / mid / large / unknown."""
    if not isinstance(mcap, (int, float)) or not np.isfinite(mcap):
        return "unknown"
    if mcap < 2e9:
        return "small"
    if mcap < 10e9:
        return "mid"
    return "large"


def compute_bucket_scores(
    metrics: pd.DataFrame, meta: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute fundamental bucket scores (quality, growth, valuation, leverage).
    
    Expected columns in `metrics`:
    - Quality: roic, fcf_margin, oper_margin, gross_margin_stability
    - Growth: rev_yoy, eps_yoy, fcf_yoy
    - Valuation: fwd_pe, ev_ebitda, ps, peg
    - Leverage: debt_to_equity, net_debt_ebitda, interest_cover, share_dilution_yoy
    
    Expected columns in `meta`:
    - sector (str)
    - market_cap (float)
    
    Returns DataFrame with columns: q_score, g_score, v_score, l_score (each 0..100).
    """
    idx = metrics.index
    out = pd.DataFrame(
        {"q_score": 0.0, "g_score": 0.0, "v_score": 0.0, "l_score": 0.0}, index=idx
    )
    
    # align meta
    meta_aligned = meta.reindex(idx)
    sector = meta_aligned.get("sector", pd.Series("Unknown", index=idx)).fillna("Unknown")
    mcap = meta_aligned.get("market_cap", pd.Series(np.nan, index=idx))
    
    caps = mcap.apply(_cap_bucket)
    
    # === Quality ===
    quality_cols = ["roic", "fcf_margin", "oper_margin", "gross_margin_stability"]
    q_present = [c for c in quality_cols if c in metrics.columns]
    if q_present:
        q_vals = []
        for c in q_present:
            z = zscore_by_group(metrics[c], sector, caps)
            # normalize z-score to 0..100 (clamp extreme values)
            norm = (z.clip(-3, 3) + 3) / 6 * 100
            q_vals.append(norm)
        out["q_score"] = pd.concat(q_vals, axis=1).mean(axis=1).fillna(0.0)
    
    # === Growth ===
    growth_cols = ["rev_yoy", "eps_yoy", "fcf_yoy"]
    g_present = [c for c in growth_cols if c in metrics.columns]
    if g_present:
        g_vals = []
        for c in g_present:
            z = zscore_by_group(metrics[c], sector, caps)
            norm = (z.clip(-3, 3) + 3) / 6 * 100
            g_vals.append(norm)
        out["g_score"] = pd.concat(g_vals, axis=1).mean(axis=1).fillna(0.0)
    
    # === Valuation (lower is better, so invert) ===
    val_cols = ["fwd_pe", "ev_ebitda", "ps", "peg"]
    v_present = [c for c in val_cols if c in metrics.columns]
    if v_present:
        v_vals = []
        for c in v_present:
            z = zscore_by_group(metrics[c], sector, caps)
            # invert: high PE -> low score
            norm = ((-z).clip(-3, 3) + 3) / 6 * 100
            v_vals.append(norm)
        out["v_score"] = pd.concat(v_vals, axis=1).mean(axis=1).fillna(0.0)
    
    # === Leverage (lower is better) ===
    lev_cols = ["debt_to_equity", "net_debt_ebitda", "share_dilution_yoy"]
    l_present = [c for c in lev_cols if c in metrics.columns]
    if l_present:
        l_vals = []
        for c in l_present:
            z = zscore_by_group(metrics[c], sector, caps)
            norm = ((-z).clip(-3, 3) + 3) / 6 * 100
            l_vals.append(norm)
        # interest_cover: higher is better
        if "interest_cover" in metrics.columns:
            z_ic = zscore_by_group(metrics["interest_cover"], sector, caps)
            norm_ic = (z_ic.clip(-3, 3) + 3) / 6 * 100
            l_vals.append(norm_ic)
        out["l_score"] = pd.concat(l_vals, axis=1).mean(axis=1).fillna(0.0)
    
    return out


def compute_fundamental_score(bucket_scores: pd.DataFrame) -> pd.Series:
    """
    Aggregate bucket scores into a single fundamental score (0..100).
    Bucket weights: quality=0.35, growth=0.30, valuation=0.25, leverage=0.10.
    """
    w_q, w_g, w_v, w_l = 0.35, 0.30, 0.25, 0.10
    score = (
        w_q * bucket_scores.get("q_score", 0.0)
        + w_g * bucket_scores.get("g_score", 0.0)
        + w_v * bucket_scores.get("v_score", 0.0)
        + w_l * bucket_scores.get("l_score", 0.0)
    )
    return score.clip(0.0, 100.0).fillna(0.0)
