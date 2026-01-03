import os
import glob
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, log_loss, precision_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
import joblib

REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")
DATA_DIR = Path("data")

DEFAULT_RELATIVE_THRESHOLD = float(os.environ.get("RELATIVE_LABEL_TAU", "0.03"))  # 3% vs SPY
ROLLING_WINDOW_MONTHS = int(os.environ.get("ROLLING_MONTHS", "18"))  # use last 18 months
N_SPLITS = int(os.environ.get("TS_SPLITS", "5"))
K_TOP = int(os.environ.get("PRECISION_AT_K", "20"))

NON_FEATURE_COLS = {
    "Ticker","As_Of_Date","date","Sector","Label","y","target","target_20d","Return_20d","SPY_20d_ret","Sector_20d_ret"
}


def _find_latest_training_csv() -> str:
    candidates = glob.glob(str(DATA_DIR / "training_dataset_20d_v3_*.csv"))
    candidates += glob.glob(str(DATA_DIR / "training_dataset_20d_v3_with_*.csv"))
    if not candidates:
        raise FileNotFoundError("No training CSV found in data/ matching training_dataset_20d_v3_*.csv")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _filter_recent_window(df: pd.DataFrame) -> pd.DataFrame:
    # Assume As_Of_Date or date column exists; fallback to index
    date_col = None
    for c in ("As_Of_Date","date","Date"):
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        return df  # cannot filter by date
    try:
        dd = pd.to_datetime(df[date_col])
        cutoff = dd.max() - pd.DateOffset(months=ROLLING_WINDOW_MONTHS)
        return df[dd >= cutoff].copy()
    except Exception:
        return df


def _build_label(df: pd.DataFrame, tau: float) -> pd.Series:
    # Prefer existing label columns
    for c in ("Label","target","y","target_20d"):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    # Compute relative 20d vs SPY
    if "Return_20d" in df.columns:
        ret20 = pd.to_numeric(df["Return_20d"], errors="coerce")
    else:
        ret20 = pd.Series(np.nan, index=df.index)
    if "SPY_20d_ret" in df.columns:
        spy20 = pd.to_numeric(df["SPY_20d_ret"], errors="coerce").fillna(0)
    else:
        spy20 = pd.Series(0.0, index=df.index)
    rel = ret20 - spy20
    y = (rel >= tau).astype(int)
    # Ensure both classes present; fallback to percentile split
    if y.sum() == 0 or y.sum() == len(y):
        if ret20.notna().sum() > 0:
            cutoff = np.nanpercentile(ret20.values, 70)
            y = (ret20 >= cutoff).astype(int)
        else:
            # if no returns, default to zeros
            y = pd.Series(np.zeros(len(df), dtype=int))
    return y


def _select_feature_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if c in NON_FEATURE_COLS:
            continue
        # numeric only
        try:
            s = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            continue
        # skip mostly missing
        if s.notna().sum() < max(50, int(0.2 * len(df))):
            continue
        cols.append(c)
    return cols


def _precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    order = np.argsort(-y_prob)
    topk = order[:k]
    return float(np.mean(y_true[topk]))


def train_and_save_bundle() -> Tuple[str, dict]:
    csv_path = _find_latest_training_csv()
    df = pd.read_csv(csv_path)
    df_recent = _filter_recent_window(df)
    y = _build_label(df_recent, DEFAULT_RELATIVE_THRESHOLD)
    if len(np.unique(y)) < 2:
        # Fallback: use full dataset without window
        df_recent = df
        y = _build_label(df_recent, DEFAULT_RELATIVE_THRESHOLD)
        if len(np.unique(y)) < 2:
            raise RuntimeError("Insufficient class diversity in labels; adjust threshold or dataset.")
    feature_cols = _select_feature_cols(df)
    X = df_recent[feature_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # TimeSeries CV + calibration
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof_prob = np.zeros(len(X))
    models = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        base = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
        base.fit(X_train, y_train)
        prob = base.predict_proba(X_val)[:,1]
        oof_prob[val_idx] = prob
        models.append(base)

    # Metrics
    auc = roc_auc_score(y, oof_prob) if len(np.unique(y)) > 1 else np.nan
    try:
        ll = log_loss(y, oof_prob, labels=[0,1])
    except Exception:
        ll = np.nan
    p_at_k = _precision_at_k(y.values, oof_prob, K_TOP)

    # Refit on full data (stacking last model for simplicity)
    final_base = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    final_base.fit(X, y)
    # Apply calibration only if both classes present and enough positives
    if len(np.unique(y)) > 1 and y.sum() >= 50:
        final_model = CalibratedClassifierCV(final_base, method="isotonic", cv=3)
        final_model.fit(X, y)
    else:
        final_model = final_base

    # Save bundle with feature names and metadata
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    out_path = MODELS_DIR / f"model_20d_v3_{ts}.pkl"
    bundle = {
        "model": final_model,
        "feature_names": feature_cols,
        "preferred_scoring_mode_20d": "hybrid",
        "trained_at": ts,
        "metrics": {"auc": float(auc), "logloss": float(ll), "precision_at_k": float(p_at_k), "k": K_TOP}
    }
    joblib.dump(bundle, out_path)

    # Also update canonical symlink/copy
    latest_path = MODELS_DIR / "model_20d_v3.pkl"
    try:
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(out_path.name)
    except Exception:
        # On platforms without symlinks, copy
        joblib.dump(bundle, latest_path)

    # Write report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rep_path = REPORTS_DIR / f"rolling_train_report_{ts}.json"
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump({"csv": csv_path, "bundle": str(out_path), "metrics": bundle["metrics"]}, f, indent=2)

    return str(out_path), bundle


if __name__ == "__main__":
    p, b = train_and_save_bundle()
    print(f"Saved model bundle to {p}")
    print(json.dumps(b["metrics"], indent=2))
