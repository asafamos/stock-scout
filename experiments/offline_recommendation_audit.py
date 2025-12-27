"""
Offline Recommendation Audit Script
- Uses the same technical indicator logic as the live Stock Scout app
- Supports two modes:
    1. Daily recommendation snapshots (for a given date)
    2. Big winners before the jump (analyze what the model would have said before a major price move)
- Exports clean CSVs for manual analysis (only technical + forward returns)
"""

import argparse
from datetime import datetime, timedelta
import os

import numpy as np
import pandas as pd
import yfinance as yf

from core.unified_logic import build_technical_indicators, compute_overall_score_20d, compute_big_winner_signal_20d
from core.ml_20d_inference import (
    predict_20d_prob_from_row,
    PREFERRED_SCORING_MODE_20D,
)
from core.scoring_pipeline_20d import (
    score_universe_20d,
    apply_20d_sorting,
    compute_final_scores_20d,
)


def to_float(x) -> float:
    """Safely convert scalar or Series to float (fixes FutureWarning)."""
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)


def get_sp500_tickers() -> list[str]:
    """
    Return a static demo list of large-cap US tickers for offline audits.
    This is used when --tickers is not provided.
    """
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "TSLA", "BRK-B", "JPM", "V",
        "UNH", "HD", "PG", "MA", "LLY",
        "AVGO", "XOM", "COST", "MRK", "ABBV",
    ]


def fetch_history(ticker: str, start: datetime, end: datetime) -> pd.DataFrame | None:
    """Download adjusted daily OHLCV from yfinance. Return None if empty or on error."""
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty:
            return None
        # Always sort by index ascending
        df = df.sort_index()
        return df
    except Exception:
        return None


# --- Main analysis script ---
def load_universe(args) -> list[str]:
    """
    Decide which tickers to use for the offline run.
    Priority:
    1) --tickers (comma separated)
    2) --universe-csv with a 'Ticker' column
    3) default get_sp500_tickers()
    """
    if getattr(args, "tickers", None):
        return [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    universe_csv = getattr(args, "universe_csv", None)
    if universe_csv and os.path.exists(universe_csv):
        try:
            df_u = pd.read_csv(universe_csv)
            if "Ticker" in df_u.columns:
                tickers = (
                    df_u["Ticker"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .dropna()
                    .unique()
                    .tolist()
                )
                if tickers:
                    return tickers
        except Exception:
            pass
    return get_sp500_tickers()

def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Recommendation Audit")
    parser.add_argument("--mode", choices=["snapshot", "big_winners", "dataset", "audit_ml_20d"], required=True)
    parser.add_argument("--date", type=str, help="Snapshot date (YYYY-MM-DD)")
    parser.add_argument("--tickers", type=str, help="Comma-separated tickers (optional)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument(
        "--horizons",
        type=str,
        default="5,20",
        help="Comma-separated forward return horizons (default: 5,20)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Big winners/dataset: start of scan window (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="Big winners/dataset: end of scan window (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Big winners/dataset: forward return horizon in trading days (default: 20)",
    )
    parser.add_argument(
        "--min-return",
        type=float,
        default=0.4,
        help="Big winners/dataset: minimum forward return to qualify (default: 0.4 = 40%)",
    )
    parser.add_argument(
        "--universe-csv",
        type=str,
        default=None,
        help="Optional CSV with a 'Ticker' column to define the universe",
    )
    parser.add_argument(
        "--drop-neutral",
        action="store_true",
        default=True,
        help="Drop rows with forward returns between +2% and +15% (neutral zone)",
    )
    parser.add_argument(
        "--include-ml",
        action="store_true",
        default=False,
        help="Include ML 20d probability and FinalScore in output (requires trained model)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input CSV path for audit_ml_20d mode",
    )
    args = parser.parse_args()

    if args.mode == "dataset":
        if not args.start_date or not args.end_date:
            raise ValueError("--start-date and --end-date are required for dataset mode")

        start_date = pd.to_datetime(args.start_date)
        end_date = pd.to_datetime(args.end_date)
        horizon = int(args.horizon)
        min_return = float(args.min_return)

        # Prefer explicit universe CSV if provided; else fallback to default ML universe file
        if not args.universe_csv and os.path.exists("data/universe_ml_20d.csv"):
            args.universe_csv = "data/universe_ml_20d.csv"
        tickers = load_universe(args)
        print(
            f"[INFO] Building dataset for {len(tickers)} tickers "
            f"from {start_date.date()} to {end_date.date()} "
            f"(horizon: {horizon}d, label threshold: {min_return:.2%})"
        )

        rows = []
        processed = 0

        for tkr in tickers:
            try:
                # Fetch a longer window (approx 2.5 years) to compute robust indicators
                fetch_start = start_date - timedelta(days=900)
                fetch_end = end_date + timedelta(days=horizon + 5)
                df = fetch_history(tkr, fetch_start, fetch_end)
                if df is None or len(df) < 60:
                    print(f"[WARN] Not enough data for {tkr}")
                    continue
                df = df.sort_index()
                scan_dates = [d for d in df.index if start_date <= d <= end_date]
                if not scan_dates:
                    continue
                for as_of_date in scan_dates:
                    idx = df.index.get_loc(as_of_date)
                    idx_fwd = idx + horizon
                    if idx_fwd >= len(df):
                        continue
                    price_0 = df.iloc[idx]["Close"]
                    price_h = df.iloc[idx_fwd]["Close"]
                    price_0 = to_float(price_0)
                    price_h = to_float(price_h)
                    forward_ret = float(price_h / price_0 - 1.0)
                    df_hist = df.loc[:as_of_date]
                    if len(df_hist) < 60:
                        continue
                    indicators = build_technical_indicators(df_hist)
                    row_ind = indicators.iloc[-1]
                    tech_score_20d = float(compute_overall_score_20d(row_ind))
                    bw = compute_big_winner_signal_20d(row_ind)
                    
                    # Base row dict with existing features
                    row_dict = {
                        "Ticker": tkr,
                        "As_Of_Date": as_of_date,
                        "Price_As_Of_Date": float(price_0),
                        "TechScore_20d": tech_score_20d,
                        "RSI": float(row_ind.get("RSI", np.nan)),
                        "ATR_Pct": float(row_ind.get("ATR_Pct", np.nan)),
                        "RR": float(row_ind.get("RR", np.nan)),
                        "MomCons": float(row_ind.get("MomCons", np.nan)),
                        "VolSurge": float(row_ind.get("VolSurge", np.nan)),
                        "BigWinnerScore_20d": float(bw.get("BigWinnerScore_20d", 0.0)),
                        "BigWinnerFlag_20d": int(bw.get("BigWinnerFlag_20d", 0)),
                        "Forward_Return_20d": forward_ret,
                        # Classification label: 1 if strong winner (>= +15%), 0 if flat/weak (<= +2%), else neutral
                        "Label_20d": (1 if forward_ret >= 0.15 else (0 if forward_ret <= 0.02 else -1)),
                        # Multi-period returns (from build_technical_indicators)
                        "Return_5d": float(row_ind.get("Return_5d", np.nan)),
                        "Return_10d": float(row_ind.get("Return_10d", np.nan)),
                        "Return_20d": float(row_ind.get("Return_20d", np.nan)),
                        "Return_60d": float(row_ind.get("Return_60d", np.nan)),
                        "Return_120d": float(row_ind.get("Return_120d", np.nan)),
                        # Sequential pattern features
                        "UpStreak_Days": float(row_ind.get("UpStreak_Days", 0)),
                        "DownStreak_Days": float(row_ind.get("DownStreak_Days", 0)),
                        "PullbackFromHigh_20d": float(row_ind.get("PullbackFromHigh_20d", np.nan)),
                        "DistanceFromLow_20d": float(row_ind.get("DistanceFromLow_20d", np.nan)),
                        "Range_Pct": float(row_ind.get("Range_Pct", np.nan)),
                        # Placeholders for context features (will be added during post-processing)
                        "RS_SPY_20d": np.nan,
                        "RS_SPY_60d": np.nan,
                        "ATR_Pct_percentile": 0.5,
                        "Vol_Class": 1,
                        "Vol_SweetSpot_Flag": 0,
                        "Vol_Extreme_Flag": 0,
                    }
                    
                    # Add ML 20d prediction if --include-ml is set
                    if args.include_ml:
                        try:
                            ml_prob = predict_20d_prob_from_row(row_ind)
                            row_dict["ML_20d_Prob"] = float(ml_prob) if ml_prob is not None and np.isfinite(ml_prob) else np.nan
                            # Compute FinalScore: 80% tech + 20% ML
                            if np.isfinite(row_dict.get("ML_20d_Prob", np.nan)):
                                final_score = 0.80 * tech_score_20d + 0.20 * (row_dict["ML_20d_Prob"] * 100.0)
                                row_dict["FinalScore"] = float(final_score)
                            else:
                                row_dict["FinalScore"] = tech_score_20d
                        except Exception as ml_err:
                            print(f"[WARN] ML prediction failed for {tkr} on {as_of_date}: {ml_err}")
                            row_dict["ML_20d_Prob"] = np.nan
                            row_dict["FinalScore"] = tech_score_20d
                    
                    rows.append(row_dict)
                processed += 1
            except Exception as e:
                import traceback
                print(f"[ERROR] {tkr}: {e}")
                traceback.print_exc()
        out_df = pd.DataFrame(rows)
        # Optionally drop neutral rows (Label_20d == -1)
        if not out_df.empty and bool(args.drop_neutral):
            before = len(out_df)
            out_df = out_df[out_df["Label_20d"].isin([0, 1])].copy()
            after = len(out_df)
            print(f"[INFO] Dropped neutral rows: {before - after} (kept {after})")
        
        # Compute context-dependent features (relative strength, volatility percentiles)
        if not out_df.empty:
            print("[INFO] Computing relative strength and volatility context features...")
            from core.ml_features_v3 import compute_relative_strength_features, compute_volatility_context_features
            
            # Fetch SPY data for relative strength calculations
            try:
                from core.data_sources_v2 import get_index_series
                spy_df = get_index_series("SPY", start_date, end_date)
                if spy_df is not None and not spy_df.empty:
                    spy_df = spy_df.rename(columns={"Date": "As_Of_Date"})
                    spy_df["SPY_Return_20d"] = spy_df["Close"].pct_change(20, fill_method=None)
                    spy_df["SPY_Return_60d"] = spy_df["Close"].pct_change(60, fill_method=None)
                    
                    # Merge with main dataset
                    out_df = out_df.merge(
                        spy_df[["As_Of_Date", "SPY_Return_20d", "SPY_Return_60d"]],
                        on="As_Of_Date",
                        how="left"
                    )
                    
                    # Compute RS_SPY features
                    out_df["RS_SPY_20d"] = out_df["Return_20d"] - out_df["SPY_Return_20d"]
                    out_df["RS_SPY_60d"] = out_df["Return_60d"] - out_df["SPY_Return_60d"]
                    print(f"[SUCCESS] Added RS_SPY features for {len(out_df)} rows")
                else:
                    print("[WARN] Could not fetch SPY data for RS calculation")
            except Exception as e:
                print(f"[WARN] RS feature computation failed: {e}")
            
            # Compute volatility percentiles per date
            if "ATR_Pct" in out_df.columns and "As_Of_Date" in out_df.columns:
                try:
                    # Compute ATR_Pct percentile within each date
                    out_df["ATR_Pct_percentile"] = out_df.groupby("As_Of_Date")["ATR_Pct"].transform(
                        lambda x: x.rank(pct=True, method='average')
                    )
                    
                    # Classify volatility (0: Low <25th, 1: Medium 25-75th, 2: High 75-90th, 3: Extreme >90th)
                    out_df["Vol_Class"] = pd.cut(
                        out_df["ATR_Pct_percentile"],
                        bins=[0, 0.25, 0.75, 0.90, 1.0],
                        labels=[0, 1, 2, 3],
                        include_lowest=True
                    ).astype(int)
                    
                    # Sweet spot: 50th-75th percentile (elevated but not extreme)
                    out_df["Vol_SweetSpot_Flag"] = (
                        (out_df["ATR_Pct_percentile"] >= 0.50) & 
                        (out_df["ATR_Pct_percentile"] < 0.75)
                    ).astype(int)
                    
                    # Extreme volatility flag (>90th percentile)
                    out_df["Vol_Extreme_Flag"] = (out_df["ATR_Pct_percentile"] >= 0.90).astype(int)
                    
                    print(f"[SUCCESS] Added volatility context features for {len(out_df)} rows")
                except Exception as e:
                    print(f"[WARN] Volatility context computation failed: {e}")
        
        # Build TechScore_20d_v2: hybrid technical formula with per-date normalization
        if not out_df.empty:
            from core.unified_logic import compute_tech_score_20d_v2
            
            # Compute raw TechScore_20d_v2 for each row
            out_df["TechScore_20d_v2_raw"] = out_df.apply(
                lambda row: compute_tech_score_20d_v2(row) * 100.0,  # Scale to 0-100
                axis=1
            )
            
            # Normalize per date using percent-rank (0-100)
            if "As_Of_Date" in out_df.columns:
                out_df["TechScore_20d_v2"] = out_df.groupby("As_Of_Date")["TechScore_20d_v2_raw"].transform(
                    lambda x: x.rank(pct=True, method='average') * 100.0
                )
            else:
                # Fallback: global percent-rank if no date column
                out_df["TechScore_20d_v2"] = out_df["TechScore_20d_v2_raw"].rank(pct=True, method='average') * 100.0
            
            # Compute TechRank_20d_v2 (0-1 scale for FinalScore computation)
            if "As_Of_Date" in out_df.columns:
                out_df["TechRank_20d_v2"] = out_df.groupby("As_Of_Date")["TechScore_20d_v2"].transform(
                    lambda x: x.rank(pct=True, method='average')
                )
            else:
                out_df["TechRank_20d_v2"] = out_df["TechScore_20d_v2"].rank(pct=True, method='average')
            
            # Compute ML_rank_20d and HybridFinalScore_20d if ML is available
            if "ML_20d_Prob" in out_df.columns:
                # Compute ML_rank_20d (0-1 scale, per date)
                if "As_Of_Date" in out_df.columns:
                    out_df["ML_rank_20d"] = out_df.groupby("As_Of_Date")["ML_20d_Prob"].transform(
                        lambda x: x.rank(pct=True, method='average')
                    )
                else:
                    out_df["ML_rank_20d"] = out_df["ML_20d_Prob"].rank(pct=True, method='average')
                
                # HybridFinalScore_20d: 0.20 Tech_rank_v2 + 0.80 ML_rank (ML dominant)
                out_df["HybridFinalScore_20d"] = (
                    0.20 * out_df["TechRank_20d_v2"] + 
                    0.80 * out_df["ML_rank_20d"]
                ) * 100.0  # Scale to 0-100
                
                # Legacy FinalScore (0.5 Tech + 0.5 ML) for backward compatibility
                out_df["FinalScore_Legacy"] = (
                    0.5 * out_df["TechRank_20d_v2"] + 
                    0.5 * out_df["ML_rank_20d"]
                ) * 100.0
        
        out_path = args.output or "experiments/training_dataset_20d_v2.csv"
        if not out_df.empty:
            out_df = out_df.sort_values(
                by=["Label_20d", "As_Of_Date"],
                ascending=[False, True],
            )
        out_df.to_csv(out_path, index=False)
        # Print dataset stats
        n_rows = len(out_df)
        n_tickers = out_df["Ticker"].nunique() if n_rows > 0 else 0
        pos = int((out_df["Label_20d"] == 1).sum()) if n_rows > 0 else 0
        neg = int((out_df["Label_20d"] == 0).sum()) if n_rows > 0 else 0
        
        # Print detailed statistics comparing v1 vs v2 technical scores
        if not out_df.empty:
            print(f"\n[DATASET STATS]")
            corr_v1 = float(out_df["TechScore_20d"].corr(out_df["Forward_Return_20d"]))
            corr_v2 = float(out_df["TechScore_20d_v2"].corr(out_df["Forward_Return_20d"]))
            print(f"  TechScore_20d v1 correlation with Forward_Return_20d: {corr_v1:.6f}")
            print(f"  TechScore_20d_v2 correlation with Forward_Return_20d: {corr_v2:.6f}")
            
            # Top decile performance for v1
            out_df["v1_decile"] = pd.qcut(out_df["TechScore_20d"], q=10, labels=False, duplicates='drop')
            v1_top = out_df[out_df["v1_decile"] == out_df["v1_decile"].max()]
            v1_avg_ret = float(v1_top["Forward_Return_20d"].mean()) if len(v1_top) > 0 else np.nan
            v1_hit = float((v1_top["Forward_Return_20d"] >= 0.15).mean()) if len(v1_top) > 0 else np.nan
            print(f"\n  TechScore_20d v1 top decile:")
            print(f"    Avg forward return: {v1_avg_ret:.6f}")
            print(f"    Hit rate (≥15%): {v1_hit:.1%}")
            
            # Top decile performance for v2
            out_df["v2_decile"] = pd.qcut(out_df["TechScore_20d_v2"], q=10, labels=False, duplicates='drop')
            v2_top = out_df[out_df["v2_decile"] == out_df["v2_decile"].max()]
            v2_avg_ret = float(v2_top["Forward_Return_20d"].mean()) if len(v2_top) > 0 else np.nan
            v2_hit = float((v2_top["Forward_Return_20d"] >= 0.15).mean()) if len(v2_top) > 0 else np.nan
            print(f"\n  TechScore_20d_v2 top decile:")
            print(f"    Avg forward return: {v2_avg_ret:.6f}")
            print(f"    Hit rate (≥15%): {v2_hit:.1%}")
            
            # Baseline
            baseline_avg_ret = float(out_df["Forward_Return_20d"].mean())
            baseline_hit = float((out_df["Forward_Return_20d"] >= 0.15).mean())
            print(f"\n  Baseline (all samples):")
            print(f"    Avg forward return: {baseline_avg_ret:.6f}")
            print(f"    Hit rate (≥15%): {baseline_hit:.1%}")
        
        print(
            f"\n[DONE] Dataset v2 exported to {out_path} | "
            f"Tickers processed: {processed} | Rows: {n_rows} | Unique tickers: {n_tickers} | Class balance: pos={pos}, neg={neg}"
        )
    elif args.mode == "snapshot":
        # --- SNAPSHOT MODE ---
        if not args.date:
            raise ValueError("--date is required for snapshot mode")

        snapshot_date = pd.to_datetime(args.date)
        horizons = [int(h) for h in args.horizons.split(",") if h.strip().isdigit()]
        max_horizon = max(horizons) if horizons else 20

        tickers = (
            [t.strip().upper() for t in args.tickers.split(",")]
            if args.tickers
            else get_sp500_tickers()
        )

        print(
            f"[INFO] Running snapshot for {len(tickers)} tickers "
            f"on {snapshot_date.date()} (horizons: {horizons})"
        )

        # Build price map first
        price_data: dict[str, pd.DataFrame] = {}
        processed = 0
        for tkr in tickers:
            try:
                start = snapshot_date - timedelta(days=365 + max_horizon)
                end = snapshot_date + timedelta(days=max_horizon + 5)
                df = fetch_history(tkr, start, end)
                if df is None or len(df) < 60:
                    print(f"[WARN] Not enough data for {tkr}")
                    continue
                price_data[tkr] = df
                processed += 1
            except Exception as e:
                import traceback
                print(f"[ERROR] {tkr}: {e}")
                traceback.print_exc()

        # Score using shared pipeline (50/50 rank blend) for the first horizon in list
        primary_h = horizons[0] if horizons else 20
        scored = score_universe_20d(
            as_of_date=snapshot_date,
            horizon_days=primary_h,
            universe=tickers,
            price_data=price_data,
            include_ml=args.include_ml,
            logger=None,
        )

        if scored.empty:
            out_df = scored
        else:
            # Compute additional returns for remaining horizons
            for h in horizons:
                col = f"Return_{h}d"
                if col in scored.columns:
                    continue
                vals = []
                for _, r in scored.iterrows():
                    tkr = r["Ticker"]
                    df = price_data.get(tkr)
                    if df is None or df.empty:
                        vals.append(np.nan)
                        continue
                    df = df.sort_index()
                    idx_pos = df.index.searchsorted(snapshot_date, side="right") - 1
                    if idx_pos < 0:
                        vals.append(np.nan)
                        continue
                    idx_fwd = idx_pos + h
                    if idx_fwd < len(df):
                        price_0 = to_float(df["Close"].iloc[idx_pos])
                        price_h = to_float(df["Close"].iloc[idx_fwd])
                        vals.append(float(price_h / price_0 - 1.0))
                    else:
                        vals.append(np.nan)
                scored[col] = vals

            # Sort using shared policy
            scored = apply_20d_sorting(scored, PREFERRED_SCORING_MODE_20D)
            print(f"[SNAPSHOT] Using ranking policy for 20d: {PREFERRED_SCORING_MODE_20D}")
            
            # Add FinalScore alias for UI/export compatibility
            scored["FinalScore"] = scored["FinalScore_20d"]
            
            out_df = scored

        top_n = 20
        df_top = out_df.head(top_n).copy() if not out_df.empty else out_df

        # Ensure required columns exist for export
        base_export_cols = [
            "Ticker",
            "As_Of_Date",
            "Price_As_Of_Date",
            "ML_20d_Prob",
            "FinalScore_20d",
        ]
        export_cols = [c for c in base_export_cols if c in df_top.columns]
        for c in df_top.columns:
            if c not in export_cols:
                export_cols.append(c)

        out_path = args.output or f"experiments/reco_audit_snapshot_{args.date}.csv"
        df_top.to_csv(out_path, index=False, columns=export_cols)
        print(
            f"[DONE] Snapshot exported to {out_path} | "
            f"Tickers processed: {processed} | Rows: {len(out_df)} | Top N: {len(df_top)}"
        )

    elif args.mode == "big_winners":
        # --- BIG WINNERS MODE ---
        if not args.start_date or not args.end_date:
            raise ValueError("--start-date and --end-date are required for big_winners mode")

        start_date = pd.to_datetime(args.start_date)
        end_date = pd.to_datetime(args.end_date)
        horizon = int(args.horizon)
        min_return = float(args.min_return)

        tickers = (
            [t.strip().upper() for t in args.tickers.split(",")]
            if args.tickers
            else get_sp500_tickers()
        )

        print(
            f"[INFO] Scanning {len(tickers)} tickers for big winners "
            f"from {start_date.date()} to {end_date.date()} "
            f"(horizon: {horizon}d, min_return: {min_return:.2%})"
        )

        results: list[dict] = []
        processed = 0
        big_winner_events = 0

        for tkr in tickers:
            try:
                fetch_start = start_date - timedelta(days=365)
                fetch_end = end_date + timedelta(days=horizon + 5)
                df = fetch_history(tkr, fetch_start, fetch_end)
                if df is None or len(df) < 60:
                    print(f"[WARN] Not enough data for {tkr}")
                    continue

                df = df.sort_index()

                # Trading days to scan
                scan_dates = [d for d in df.index if start_date <= d <= end_date]

                for as_of_date in scan_dates:
                    idx = df.index.get_loc(as_of_date)
                    idx_fwd = idx + horizon
                    if idx_fwd >= len(df):
                        continue

                    price_0 = df.iloc[idx]["Close"]
                    price_h = df.iloc[idx_fwd]["Close"]

                    # Use to_float helper to safely extract scalar
                    price_0 = to_float(price_0)
                    price_h = to_float(price_h)

                    forward_return = float(price_h / price_0 - 1.0)

                    if forward_return >= min_return:
                        df_hist = df.loc[:as_of_date]
                        if len(df_hist) < 30:
                            continue

                        indicators = build_technical_indicators(df_hist)
                        row = indicators.iloc[-1]
                        tech_score_20d = compute_overall_score_20d(row)
                        bw = compute_big_winner_signal_20d(row)
                        rec = {
                            "Ticker": tkr,
                            "As_Of_Date": as_of_date,
                            "Price_As_Of_Date": float(price_0),
                            "Event_Horizon_Days": horizon,
                            "Forward_Return": float(forward_return),
                            "TechScore_20d": float(tech_score_20d),
                            "RSI": float(row.get("RSI", float("nan"))),
                            "ATR_Pct": float(row.get("ATR_Pct", float("nan"))),
                            "RR": float(row.get("RR", float("nan"))),
                            "MomCons": float(row.get("MomCons", float("nan"))),
                            "VolSurge": float(row.get("VolSurge", float("nan"))),
                            "BigWinnerScore_20d": float(bw.get("BigWinnerScore_20d", 0.0)),
                            "BigWinnerFlag_20d": int(bw.get("BigWinnerFlag_20d", 0)),
                        }
                        
                        # Add ML 20d prediction if --include-ml is set
                        if args.include_ml:
                            try:
                                ml_prob = predict_20d_prob_from_row(row)
                                rec["ML_20d_Prob"] = float(ml_prob) if ml_prob is not None and np.isfinite(ml_prob) else np.nan
                                # Compute FinalScore: 80% tech + 20% ML
                                if np.isfinite(rec.get("ML_20d_Prob", np.nan)):
                                    final_score = 0.80 * float(tech_score_20d) + 0.20 * (rec["ML_20d_Prob"] * 100.0)
                                    rec["FinalScore"] = float(final_score)
                                else:
                                    rec["FinalScore"] = float(tech_score_20d)
                            except Exception as ml_err:
                                print(f"[WARN] ML prediction failed for {tkr} on {as_of_date}: {ml_err}")
                                rec["ML_20d_Prob"] = np.nan
                                rec["FinalScore"] = float(tech_score_20d)
                        
                        results.append(rec)
                        big_winner_events += 1

                processed += 1

            except Exception as e:
                import traceback

                print(f"[ERROR] {tkr}: {e}")
                traceback.print_exc()

        out_df = pd.DataFrame(results)
        out_path = (
            args.output
            or f"experiments/reco_big_winners_{start_date.date()}_to_{end_date.date()}.csv"
        )
        out_df.to_csv(out_path, index=False)
        print(
            f"[DONE] Big winners exported to {out_path} | "
            f"Tickers processed: {processed} | Rows: {len(out_df)} | "
            f"Big winner events: {big_winner_events}"
        )
    elif args.mode == "audit_ml_20d":
        # --- AUDIT ML 20D MODE ---
        # Evaluate FinalScore deciles against forward returns with improved rank-based FinalScore_20d
        if not args.input:
            raise ValueError("--input is required for audit_ml_20d mode")
        in_path = args.input
        out_path = args.output or "experiments/audit_ml_20d_v2.csv"
        df = pd.read_csv(in_path)
        if df.empty:
            raise ValueError(f"Input dataset {in_path} is empty")
        
        # Build ML prob and FinalScore if missing
        feats = ["TechScore_20d", "RSI", "ATR_Pct", "RR", "MomCons", "VolSurge"]
        def build_row_series(r):
            return pd.Series({c: r.get(c, np.nan) for c in feats})
        if "ML_20d_Prob" not in df.columns:
            ml_probs = []
            for _, r in df.iterrows():
                try:
                    ml = predict_20d_prob_from_row(build_row_series(r))
                except Exception:
                    ml = np.nan
                ml_probs.append(float(ml) if np.isfinite(ml) else np.nan)
            df["ML_20d_Prob"] = ml_probs
        
        # Ensure TechScore_20d_v2 exists (compute if not present)
        if "TechScore_20d_v2" not in df.columns:
            technical_features = ["TechScore_20d", "RSI", "ATR_Pct", "RR", "MomCons", "VolSurge"]
            signs = {
                "TechScore_20d": 1.0,
                "RSI": -1.0,
                "ATR_Pct": 1.0,
                "RR": 1.0,
                "MomCons": -1.0,
                "VolSurge": 1.0,
            }
            for col in technical_features:
                if col in df.columns:
                    df[col + "_rank"] = df[col].fillna(0.0).rank(pct=True, method='average')
            
            def compute_tech_v2(row):
                valid_feats = [col for col in technical_features if col in df.columns]
                if not valid_feats:
                    return 50.0
                numerator = sum(signs.get(col, 1.0) * row.get(col + "_rank", 0.5) for col in valid_feats)
                score = numerator / len(valid_feats)
                score = max(0.0, min(1.0, score))
                return 100.0 * score
            
            df["TechScore_20d_v2"] = df.apply(compute_tech_v2, axis=1)
        
        # Recompute canonical ML/Tech/FinalScore using shared helper
        df = compute_final_scores_20d(df, include_ml=True)
        
        # Print baseline
        baseline_ret = df["Forward_Return_20d"].mean()
        baseline_hit = (df["Forward_Return_20d"] >= 0.15).mean()
        print(f"\n{'='*70}")
        print(f"[AUDIT BASELINE] Avg 20d return={baseline_ret:.6f}, Hit rate (≥15%)={baseline_hit:.3f}")
        print(f"{'='*70}")
        
        # Bin into deciles and compute stats for TechScore_20d v1 (old)
        df = df.copy()
        df["tech_v1_bin"] = pd.qcut(df["TechScore_20d"], q=10, labels=False, duplicates="drop")
        tech_v1_stats = []
        for b in sorted(df["tech_v1_bin"].dropna().unique()):
            sub = df[df["tech_v1_bin"] == b]
            n = len(sub)
            avg_ret = float(sub["Forward_Return_20d"].mean()) if n > 0 else np.nan
            hit_rate = float((sub["Forward_Return_20d"] >= 0.15).mean()) if n > 0 else np.nan
            tech_v1_stats.append({"metric_type": "tech_v1", "bin": int(b), "n": n, "avg_forward_ret": avg_ret, "hit_rate_15pct": hit_rate})
        
        # Bin into deciles and compute stats for TechScore_20d v2 (new)
        df["tech_v2_bin"] = pd.qcut(df["TechScore_20d_v2"], q=10, labels=False, duplicates="drop")
        tech_v2_stats = []
        for b in sorted(df["tech_v2_bin"].dropna().unique()):
            sub = df[df["tech_v2_bin"] == b]
            n = len(sub)
            avg_ret = float(sub["Forward_Return_20d"].mean()) if n > 0 else np.nan
            hit_rate = float((sub["Forward_Return_20d"] >= 0.15).mean()) if n > 0 else np.nan
            tech_v2_stats.append({"metric_type": "tech_v2", "bin": int(b), "n": n, "avg_forward_ret": avg_ret, "hit_rate_15pct": hit_rate})
        
        # Bin into deciles and compute stats for FinalScore_20d (rank-blend)
        df["finalscore_bin"] = pd.qcut(df["FinalScore_20d"], q=10, labels=False, duplicates="drop")
        finalscore_stats = []
        for b in sorted(df["finalscore_bin"].dropna().unique()):
            sub = df[df["finalscore_bin"] == b]
            n = len(sub)
            avg_ret = float(sub["Forward_Return_20d"].mean()) if n > 0 else np.nan
            hit_rate = float((sub["Forward_Return_20d"] >= 0.15).mean()) if n > 0 else np.nan
            finalscore_stats.append({"metric_type": "finalscore_20d", "bin": int(b), "n": n, "avg_forward_ret": avg_ret, "hit_rate_15pct": hit_rate})
        
        # Bin into deciles and compute stats for ML_20d_Prob (pure model)
        df["ml_bin"] = pd.qcut(df["ML_20d_Prob"], q=10, labels=False, duplicates="drop")
        ml_stats = []
        for b in sorted(df["ml_bin"].dropna().unique()):
            sub = df[df["ml_bin"] == b]
            n = len(sub)
            avg_ret = float(sub["Forward_Return_20d"].mean()) if n > 0 else np.nan
            hit_rate = float((sub["Forward_Return_20d"] >= 0.15).mean()) if n > 0 else np.nan
            ml_stats.append({"metric_type": "ml_prob", "bin": int(b), "n": n, "avg_forward_ret": avg_ret, "hit_rate_15pct": hit_rate})
        
        # Combine all stats into one DataFrame
        all_stats = pd.DataFrame(tech_v1_stats + tech_v2_stats + finalscore_stats + ml_stats)
        all_stats = all_stats.sort_values(by=["metric_type", "bin"], ascending=[True, True])
        
        # Save combined CSV
        all_stats.to_csv(out_path, index=False)
        
        # Print detailed results
        print(f"\n[AUDIT] Technical Score v1 (old) deciles:")
        df_v1 = pd.DataFrame(tech_v1_stats).sort_values("bin")
        print(df_v1.to_string(index=False))
        
        print(f"\n[AUDIT] Technical Score v2 (new, data-driven percentile ranks) deciles:")
        df_v2 = pd.DataFrame(tech_v2_stats).sort_values("bin")
        print(df_v2.to_string(index=False))
        
        print(f"\n[AUDIT] FinalScore_20d deciles (0.5×TechScore_v2_rank + 0.5×ML_rank):")
        df_fs = pd.DataFrame(finalscore_stats).sort_values("bin")
        print(df_fs.to_string(index=False))
        
        print(f"\n[AUDIT] ML_20d_Prob deciles (pure model ranking):")
        df_ml = pd.DataFrame(ml_stats).sort_values("bin")
        print(df_ml.to_string(index=False))
        
        print(f"\n[AUDIT] Summary (Top Decile Improvements):")
        v1_top = df_v1.loc[df_v1["bin"].idxmax()] if not df_v1.empty else None
        v2_top = df_v2.loc[df_v2["bin"].idxmax()] if not df_v2.empty else None
        ml_top = df_ml.loc[df_ml["bin"].idxmax()] if not df_ml.empty else None
        
        if v1_top is not None:
            print(f"  TechScore_v1 top: avg_ret={v1_top['avg_forward_ret']:.6f}, hit_rate={v1_top['hit_rate_15pct']:.1%}")
        if v2_top is not None:
            print(f"  TechScore_v2 top: avg_ret={v2_top['avg_forward_ret']:.6f}, hit_rate={v2_top['hit_rate_15pct']:.1%}")
        if ml_top is not None:
            print(f"  ML_Prob top:     avg_ret={ml_top['avg_forward_ret']:.6f}, hit_rate={ml_top['hit_rate_15pct']:.1%}")
        print(f"  Baseline:        avg_ret={baseline_ret:.6f}, hit_rate={baseline_hit:.1%}")

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
