#!/usr/bin/env python3
import os
import sys
import json
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

# Ensure project root is on sys.path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from core.pipeline_runner import (
    fetch_top_us_tickers_by_market_cap,
    run_scan_pipeline,
    LAST_UNIVERSE_PROVIDER,
)
from core.config import get_config
from core.data_sources_v2 import clear_cache, reset_disabled_providers


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


def send_high_confidence_alerts(df: pd.DataFrame, ml_threshold: float = 0.7, max_alerts: int = 10) -> None:
    """Send alerts for high-confidence ML signals.
    
    If ALERT_WEBHOOK env var is set, sends POST request with top signals.
    Otherwise, just logs the signals to stdout.
    
    Args:
        df: Scan results DataFrame
        ml_threshold: Minimum ML_20d_Prob to trigger alert (default 0.7)
        max_alerts: Maximum number of alerts to send (default 10)
    """
    # Find ML probability column
    ml_col = None
    for col in ["ML_20d_Prob", "ml_20d_prob", "ML_Prob"]:
        if col in df.columns:
            ml_col = col
            break
    
    if ml_col is None:
        print(json.dumps({"alert_status": "skipped", "reason": "no_ml_column"}))
        return
    
    # Filter for high-confidence signals
    df_copy = df.copy()
    df_copy["_ml_prob"] = pd.to_numeric(df_copy[ml_col], errors="coerce")
    high_conf = df_copy[df_copy["_ml_prob"] >= ml_threshold].copy()
    
    if high_conf.empty:
        print(json.dumps({"alert_status": "no_signals", "threshold": ml_threshold}))
        return
    
    # Sort by ML probability and take top N
    high_conf = high_conf.sort_values("_ml_prob", ascending=False).head(max_alerts)
    
    # Build alert payload
    timestamp = datetime.utcnow().isoformat() + "Z"
    score_col = "FinalScore_20d" if "FinalScore_20d" in high_conf.columns else "Score"
    
    signals = []
    for _, row in high_conf.iterrows():
        signal = {
            "ticker": str(row.get("Ticker", "UNKNOWN")),
            "score": round(float(row.get(score_col, 0)), 2) if pd.notna(row.get(score_col)) else None,
            "ml_probability": round(float(row["_ml_prob"]), 4),
            "timestamp": timestamp,
        }
        signals.append(signal)
    
    alert_payload = {
        "event": "high_confidence_signals",
        "timestamp": timestamp,
        "signal_count": len(signals),
        "ml_threshold": ml_threshold,
        "signals": signals,
    }
    
    # Check for webhook
    webhook_url = os.environ.get("ALERT_WEBHOOK")
    
    if webhook_url:
        try:
            resp = requests.post(
                webhook_url,
                json=alert_payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            print(json.dumps({
                "alert_status": "sent",
                "webhook_response": resp.status_code,
                "signal_count": len(signals),
            }))
        except Exception as e:
            print(json.dumps({
                "alert_status": "webhook_failed",
                "error": str(e),
                "signal_count": len(signals),
            }))
            # Still log the signals on webhook failure
            print(json.dumps({"alert_payload": alert_payload}))
    else:
        # No webhook - just log the signals
        print(json.dumps({
            "alert_status": "logged",
            "note": "Set ALERT_WEBHOOK env var to enable webhook alerts",
            "signal_count": len(signals),
        }))
        for sig in signals:
            print(json.dumps({"high_confidence_signal": sig}))


def run_pipeline_full(universe: List[str], cfg: dict, batch_size: int = 200) -> Tuple[pd.DataFrame, int]:
    """Run the pipeline over the full universe with batch resilience.

    Returns (results_df, batches_ok)
    """
    results_list: List[pd.DataFrame] = []
    batches_ok = 0
    for i in range(0, len(universe), batch_size):
        batch = universe[i : i + batch_size]
        try:
            df, _ = run_scan_pipeline(batch, cfg)
            if df is not None and not df.empty:
                results_list.append(df)
            batches_ok += 1
        except Exception as e:
            # continue to next batch
            print(json.dumps({"warning": f"batch_failed_{i}_{i+len(batch)}", "error": str(e) }))
            continue
        # short pause to be kind to providers
        time.sleep(0.5)
    if results_list:
        all_df = pd.concat(results_list, ignore_index=True)
    else:
        all_df = pd.DataFrame()
    return all_df, batches_ok


def main():
    t0 = time.perf_counter()

    # Clear cache and reset providers for consistent results between runs
    print(json.dumps({"status": "clearing_cache_and_resetting_providers"}))
    clear_cache()
    reset_disabled_providers()

    # Universe
    universe = fetch_top_us_tickers_by_market_cap(limit=2000)
    if not universe:
        print(json.dumps({"error": "Universe fetch failed"}))
        return

    # Config for full run
    cfg_obj = get_config()
    cfg = cfg_obj.to_dict()
    cfg.update(
        {
            "lookback_days": int(os.getenv("LOOKBACK_DAYS", str(getattr(cfg_obj, "lookback_days", 90)))),
            "fundamental_enabled": True,
            "beta_filter_enabled": False,
            "meteor_mode": False,
            "BUDGET_TOTAL": float(os.getenv("BUDGET_TOTAL", str(getattr(cfg_obj, "budget_total", 5000.0)))),
            "MIN_POSITION": float(os.getenv("MIN_POSITION", str(getattr(cfg_obj, "min_position", 500.0)))),
            "MAX_POSITION_PCT": float(os.getenv("MAX_POSITION_PCT", str(getattr(cfg_obj, "max_position_pct", 20.0)))),
        }
    )

    # Run full pipeline with batch resilience
    try:
        results, batches_ok = run_pipeline_full(universe, cfg, batch_size=200)
    except Exception as e:
        # Last-resort safety
        print(json.dumps({"error": f"Pipeline crashed: {e}"}))
        results = pd.DataFrame()
        batches_ok = 0

    # Prepare output directory
    out_dir = Path(__file__).resolve().parents[1] / "data" / "scans"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "latest_scan.parquet"
    meta_path = out_dir / "latest_scan.meta.json"

    # Save Parquet and metadata even if empty (captures failure context)
    try:
        if results is None:
            results = pd.DataFrame()
        results.to_parquet(parquet_path, index=False)
    except Exception as e:
        print(json.dumps({"warning": f"parquet_save_failed: {e}"}))

    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "universe_count": len(universe),
        "provider": LAST_UNIVERSE_PROVIDER,
        "results_count": 0 if results is None else int(len(results)),
        "batches_ok": int(batches_ok),
        "lookback_days": int(cfg.get("lookback_days", 90)),
        "beta_filter_enabled": bool(cfg.get("beta_filter_enabled", False)),
        "fundamental_enabled": bool(cfg.get("fundamental_enabled", True)),
        "runtime_seconds": round(time.perf_counter() - t0, 2),
    }
    try:
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception as e:
        print(json.dumps({"warning": f"meta_save_failed: {e}"}))

    # Send alerts for high-confidence ML signals
    if results is not None and not results.empty:
        try:
            send_high_confidence_alerts(results, ml_threshold=0.7, max_alerts=10)
        except Exception as e:
            print(json.dumps({"warning": f"alert_failed: {e}"}))

    # Build terminal report (JSON) with flags and FinalScore_20d
    df = results.copy() if results is not None else pd.DataFrame()
    if df.empty:
        print(json.dumps({"note": "No results from pipeline."}))
        return

    # Coil bonus flag (tight + RS)
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

    # Growth Acceleration flag (EPS and optionally Rev YoY)
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

    patt = pd.to_numeric(df.get("Pattern_Score", pd.Series([np.nan] * len(df), index=df.index)), errors="coerce")

    report = pd.DataFrame(
        {
            "Ticker": df["Ticker"],
            "MarketCap": mcap,
            "FinalScore_20d": pd.to_numeric(df.get("FinalScore_20d", df.get("Score", np.nan)), errors="coerce"),
            "Pattern_Score": patt,
            "Quality_Label": df.get("Quality_Score_F", pd.Series([np.nan] * len(df))).apply(quality_label),
            "סכום קנייה ($)": pd.to_numeric(df.get("סכום קנייה ($)", pd.Series([0] * len(df))), errors="coerce"),
            "Coil_Bonus": coil_flag.astype(int),
            "Growth_Acceleration": growth_flag.astype(int),
        }
    ).sort_values("FinalScore_20d", ascending=False)

    print(json.dumps(report.to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()
