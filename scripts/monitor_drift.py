import json
import glob
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp

MODELS_DIR = Path("models")
DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")


def _latest_model_path() -> Path:
    candidates = sorted(MODELS_DIR.glob("model_20d_v3_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    p = MODELS_DIR / "model_20d_v3.pkl"
    if p.exists():
        return p
    raise FileNotFoundError("No model bundle found")


def _latest_live_csv() -> str:
    # Prefer explicit live v3 dataset if present, else fallback to test mini
    candidates = glob.glob(str(DATA_DIR / "training_dataset_20d_v3_with_live_v3.csv"))
    candidates += glob.glob(str(DATA_DIR / "test_20d_v3_*.csv"))
    candidates += glob.glob(str(DATA_DIR / "test_20d_v3_mini.csv"))
    if not candidates:
        raise FileNotFoundError("No live/test CSV found in data/ for drift monitoring")
    candidates.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return candidates[0]


def _psi(base_pct: List[float], cur_pct: List[float]) -> float:
    # Protect against zeros
    eps = 1e-6
    s = 0.0
    for b, c in zip(base_pct, cur_pct):
        b = max(b, eps)
        c = max(c, eps)
        s += (c - b) * np.log(c / b)
    return float(s)


def run_drift_monitor():
    model_path = _latest_model_path()
    bundle = joblib.load(model_path)
    feats = bundle.get("feature_names", [])
    feature_bins: Dict[str, List[float]] = bundle.get("feature_bins", {})
    training_bin_pct: Dict[str, List[float]] = bundle.get("training_bin_pct", {})

    # Load live data
    live_csv = _latest_live_csv()
    df_live = pd.read_csv(live_csv)

    # Optionally load training dataset for KS
    # Try to read training dataset path from latest rolling report
    train_csv = None
    latest_reports = sorted(REPORTS_DIR.glob("rolling_train_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if latest_reports:
        try:
            with open(latest_reports[0], "r", encoding="utf-8") as f:
                rep = json.load(f)
                train_csv = rep.get("csv")
        except Exception:
            train_csv = None
    df_train = None
    if train_csv and Path(train_csv).exists():
        try:
            df_train = pd.read_csv(train_csv)
        except Exception:
            df_train = None

    results = {}
    severe = False
    moderate = False

    for feat in feats:
        if feat not in df_live.columns:
            continue
        bins = feature_bins.get(feat)
        base_pct = training_bin_pct.get(feat)
        psi = None
        ks = None
        try:
            s_live = pd.to_numeric(df_live[feat], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if bins and base_pct:
                cats_live = pd.cut(s_live, bins=bins, include_lowest=True, duplicates='drop')
                cnt_live = cats_live.value_counts(sort=False)
                cur_pct = (cnt_live / max(cnt_live.sum(), 1)).astype(float).tolist()
                # Align lengths if bin collapsing occurred
                if len(cur_pct) != len(base_pct):
                    m = min(len(cur_pct), len(base_pct))
                    cur_pct = cur_pct[:m]
                    base_pct = base_pct[:m]
                psi = _psi(base_pct, cur_pct)
            # KS using raw samples if training CSV available
            if df_train is not None and feat in df_train.columns:
                s_train = pd.to_numeric(df_train[feat], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if len(s_train) > 50 and len(s_live) > 50:
                    ks = float(ks_2samp(s_train.values, s_live.values, alternative='two-sided').statistic)
        except Exception:
            pass

        results[feat] = {"PSI": psi, "KS": ks}
        if psi is not None:
            if psi >= 0.3:
                severe = True
            elif psi >= 0.2:
                moderate = True
        if ks is not None and ks >= 0.3:
            severe = True

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "model": str(model_path),
        "live_csv": live_csv,
        "training_csv": train_csv,
        "results": results,
        "flags": {"moderate": moderate, "severe": severe},
    }
    with open(REPORTS_DIR / "drift_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Write alert for inference fallback if severe
    if severe:
        alert = {"drift_alert": True, "recommend_fallback": True}
        with open(MODELS_DIR / "drift_alert.json", "w", encoding="utf-8") as f:
            json.dump(alert, f)
        print("Drift: severe — fallback recommended")
    elif moderate:
        print("Drift: moderate — monitor closely")
    else:
        print("Drift: normal")


if __name__ == "__main__":
    run_drift_monitor()
