"""Real-time Top 500 Scan and Buy List Output.

Runs a full-universe scan (top ~500 liquid US tickers),
filters for Final_Score > 60 and Market_Regime != 'PANIC',
and prints the Top 10 opportunities for today.

Usage:
    python scan_500_live.py
"""
from __future__ import annotations
import sys
import os
from pathlib import Path
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from batch_scan import run_batch_scan


def pick_final_score_col(df: pd.DataFrame) -> str:
    candidates = [
        "FinalScore_20d",
        "FinalScore",
        "Score",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return candidates[-1]


def pick_regime_col(df: pd.DataFrame) -> str:
    for c in ["Market_Regime", "Regime"]:
        if c in df.columns:
            return c
    return "Market_Regime"


def main():
    # Optional environment sanity check for FMP key
    fmp_key = os.getenv("FMP_API_KEY")
    if not fmp_key:
        print("⚠ FMP_API_KEY not set — provider fetches may be limited.")

    print("🔎 Running full universe scan (~500 tickers)...")
    results = run_batch_scan(universe_size=500, lookback_days=180, output_dir=PROJECT_ROOT / "data" / "scans")

    if results is None or len(results) == 0:
        print("❌ No results returned from pipeline.")
        return

    df = pd.DataFrame(results)

    final_col = pick_final_score_col(df)
    regime_col = pick_regime_col(df)

    # Clean and filter
    df[final_col] = pd.to_numeric(df[final_col], errors="coerce")
    df = df.dropna(subset=[final_col])
    df = df[df[final_col] > 60]
    if regime_col in df.columns:
        df = df[df[regime_col].str.upper() != "PANIC"]

    # Sort descending by final score
    df_sorted = df.sort_values(final_col, ascending=False)

    top10 = df_sorted.head(10)
    if len(top10) == 0:
        print("⚠ No opportunities matched the filter today.")
        return

    cols_to_show = [
        "Ticker",
        final_col,
        regime_col if regime_col in top10.columns else None,
        "ML_20d_Prob" if "ML_20d_Prob" in top10.columns else None,
        "ML_Confidence_Status" if "ML_Confidence_Status" in top10.columns else None,
        "TechScore_20d" if "TechScore_20d" in top10.columns else None,
    ]
    cols_to_show = [c for c in cols_to_show if c is not None]

    print("\n📈 Top 10 Opportunities (Final_Score > 60, Regime != PANIC):\n")
    print(top10[cols_to_show].to_string(index=False, justify="left"))


if __name__ == "__main__":
    main()
