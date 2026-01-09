import os
import sys
import json
import random
import pandas as pd
import numpy as np

from core.pipeline_runner import fetch_top_us_tickers_by_market_cap, run_scan_pipeline
from core.config import get_config


def quality_label(score: float) -> str:
    try:
        s = float(score)
    except Exception:
        return "Unknown"
    if s >= 70:
        return "High"
    elif s >= 40:
        return "Medium"
    else:
        return "Low"


def to_pct(val):
    try:
        x = float(val)
        if not np.isfinite(x):
            return np.nan
    except Exception:
        return np.nan
    return x * 100.0 if abs(x) <= 2 else x


def main():
    # Universe and sample
    universe = fetch_top_us_tickers_by_market_cap(limit=2000)
    if not universe:
        print(json.dumps({"error": "Universe fetch failed"}))
        return
    sample = random.sample(universe, min(10, len(universe)))

    # Config tuned for a lightweight run
    cfg = get_config().to_dict()
    cfg.update({
        "lookback_days": int(os.getenv("LOOKBACK_DAYS", "45")),
        "fundamental_enabled": True,
        "beta_filter_enabled": False,
        "meteor_mode": False,
        "BUDGET_TOTAL": float(os.getenv("BUDGET_TOTAL", "5000")),
        "MIN_POSITION": float(os.getenv("MIN_POSITION", "500")),
        "MAX_POSITION_PCT": float(os.getenv("MAX_POSITION_PCT", "20")),  # percent
    })

    results, _ = run_scan_pipeline(sample, cfg)
    if results.empty:
        print(json.dumps({"note": "No results from pipeline."}))
        return

    df = results.copy()

    # Coil Bonus flag
    rs_col = None
    for c in ["Blended_RS_Value", "relative_strength_20d"]:
        if c in df.columns:
            rs_col = c
            break
    coil_flag = pd.Series(False, index=df.index)
    if "RangeRatio_5_20" in df.columns and rs_col:
        rr = pd.to_numeric(df["RangeRatio_5_20"], errors="coerce")
        rs = pd.to_numeric(df[rs_col], errors="coerce")
        coil_flag = (rr.notna()) & (rs.notna()) & (rr < 0.7) & (rs > 0.02)

    # Growth Acceleration flag
    eps = None
    rev = None
    for c in ["eps_yoy", "eps_g_yoy"]:
        if c in df.columns:
            eps = df[c].apply(to_pct)
            break
    for c in ["rev_yoy", "rev_g_yoy"]:
        if c in df.columns:
            rev = df[c].apply(to_pct)
            break
    growth_flag = pd.Series(False, index=df.index)
    if eps is not None:
        growth_flag = eps > 25.0
        if rev is not None:
            growth_flag = growth_flag & (rev > 20.0)

    # MarketCap
    mcap = None
    for c in ["market_cap", "MarketCap", "marketCapitalization"]:
        if c in df.columns:
            mcap = pd.to_numeric(df[c], errors="coerce")
            break
    if mcap is None:
        mcap = pd.Series([np.nan] * len(df), index=df.index)

    # Pattern score
    patt = pd.to_numeric(df.get("Pattern_Score", pd.Series([np.nan] * len(df), index=df.index)), errors="coerce")

    report = pd.DataFrame({
        "Ticker": df["Ticker"],
        "MarketCap": mcap,
        "FinalScore_20d": pd.to_numeric(df["FinalScore_20d"], errors="coerce"),
        "Pattern_Score": patt,
        "Quality_Label": df.get("Quality_Score_F", pd.Series([np.nan]*len(df))).apply(quality_label),
        "סכום קנייה ($)": pd.to_numeric(df.get("סכום קנייה ($)", pd.Series([0] * len(df))), errors="coerce"),
        "Coil_Bonus": coil_flag.astype(int),
        "Growth_Acceleration": growth_flag.astype(int),
    }).sort_values("FinalScore_20d", ascending=False)

    print(json.dumps(report.to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()
