"""Run a 2,000-ticker Meteor scan and save results.

Config:
- Universe size: 2000
- Market cap: $300M–$15B
- Meteor mode: enabled
- Earnings blackout: 7 days

Output:
- reports/meteor_results_YYYYMMDD.csv with Meteor_Confidence_Score column
"""
from __future__ import annotations
import os
import sys
from datetime import datetime
import argparse
from pathlib import Path
import pandas as pd

# Ensure project root in path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.pipeline_runner import (
    fetch_top_us_tickers_by_market_cap,
    run_scan_pipeline,
)
from advanced_filters import compute_meteor_signals, fetch_benchmark_data


def compute_meteor_confidence(row: pd.Series, df_map: dict[str, pd.DataFrame], bench_df: pd.DataFrame) -> float:
    """Heuristic confidence: ML prob + VCP tightness + pocket pivot strength.
    Returns 0–100.
    """
    try:
        ml_prob = float(row.get("ML_20d_Prob", 0.5))
    except Exception:
        ml_prob = 0.5
    tkr = str(row.get("Ticker"))
    df = df_map.get(tkr)
    if df is None or len(df) == 0:
        return float(ml_prob * 100.0)
    try:
        sig = compute_meteor_signals(tkr, df.rename(columns=str.title), bench_df.rename(columns=str.title))
        vcp = float(sig.get("VCP_Ratio", 1.0))
        pocket = sig.get("Pocket_Pivot_Ratio")
        pocket = float(pocket) if pocket is not None else 1.0
        # Tightness bonus: lower VCP ratio → higher bonus (max +20)
        vcp_bonus = float(max(0.0, min(20.0, (1.0 - vcp) * 40.0)))
        pocket_bonus = float(max(0.0, min(10.0, (pocket - 1.0) * 10.0)))
        conf = float(max(0.0, min(100.0, ml_prob * 100.0 + vcp_bonus + pocket_bonus)))
        return conf
    except Exception:
        return float(ml_prob * 100.0)


def _compute_rs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Compute blended RS and rank percentile for convenience in output.

    Weighted RS = 0.7*RS_63d + 0.3*RS_21d; RS_Rank = percentile [0,100].
    If inputs are missing, leaves NaN.
    """
    out = df.copy()
    try:
        rs63 = pd.to_numeric(out.get("RS_63d"), errors="coerce")
        rs21 = pd.to_numeric(out.get("RS_21d"), errors="coerce")
        weighted = 0.7 * rs63 + 0.3 * rs21
        out["Weighted_RS"] = weighted
        # Percentile rank (0-100) over available non-NaN values
        valid = weighted.dropna()
        if len(valid) > 0:
            ranks = weighted.rank(pct=True) * 100.0
            out["RS_Rank"] = ranks
        else:
            out["RS_Rank"] = np.nan
    except Exception:
        out["Weighted_RS"] = np.nan
        out["RS_Rank"] = np.nan
    return out


def main():
    parser = argparse.ArgumentParser(description="Run Meteor scan")
    env_limit = int(os.getenv("UNIVERSE_SIZE", os.getenv("AUTO_SCAN_UNIVERSE_LIMIT", "2000")))
    parser.add_argument("--limit", type=int, default=env_limit, help="Universe size")
    args = parser.parse_args()
    # Environment defaults
    os.environ.setdefault("METEOR_MODE", "1")
    os.environ.setdefault("MIN_MCAP", "300000000")
    os.environ.setdefault("MAX_MCAP", "15000000000")
    # Optional blackout threshold (if pipeline honors this)
    os.environ.setdefault("EARNINGS_THRESHOLD", "7")

    # Universe
    universe = fetch_top_us_tickers_by_market_cap(limit=args.limit)

    # Config
    cfg = {
        "meteor_mode": bool(os.getenv("METEOR_MODE", "1") == "1"),
        "fundamental_enabled": True,
        # If earnings blackout supported elsewhere, pass config key
        "earnings_blackout_days": int(os.getenv("EARNINGS_THRESHOLD", "7")),
    }

    # Run pipeline
    results, data_map = run_scan_pipeline(universe, cfg)
    if results is None or results.empty:
        print("No results")
        return

    # Compute Meteor confidence scores
    bench_df = fetch_benchmark_data("SPY", days=200)
    results = results.copy()
    results["Meteor_Confidence_Score"] = results.apply(
        lambda r: compute_meteor_confidence(r, data_map, bench_df), axis=1
    )

    # Add RS ranking convenience columns
    results = _compute_rs_rank(results)

    # Save output
    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d")
    out_path = out_dir / f"meteor_results_{stamp}.csv"
    results.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
