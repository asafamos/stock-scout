import sys, os
# Ensure project root is on sys.path when running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    """Strict label: positive if relative 20d >= tau. No percentile fallback."""
    # Prefer existing label columns
    for c in ("Label","target","y","target_20d"):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    # Compute relative 20d vs SPY
    # Support both 'Forward_Return_20d' and 'Return_20d'
    if "Forward_Return_20d" in df.columns:
        ret20 = pd.to_numeric(df["Forward_Return_20d"], errors="coerce")
    elif "Return_20d" in df.columns:
        ret20 = pd.to_numeric(df["Return_20d"], errors="coerce")
    else:
        ret20 = pd.Series(np.nan, index=df.index)
    spy20 = pd.to_numeric(df.get("SPY_20d_ret", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0)
    rel = ret20 - spy20
    y = (rel >= float(tau)).astype(int)
    # No percentile fallback; if all zeros, return zeros
    return y.fillna(0).astype(int)


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


def _regime_adjusted_tau(base_tau: float) -> float:
    """Adjust label threshold by detected market regime (bullish/neutral/bearish)."""
    try:
        from core.market_regime import detect_market_regime
        regime = detect_market_regime()
        r = regime.get("regime", "neutral")
        conf = regime.get("confidence", 50)
        tau = base_tau
        if r == "bullish":
            tau = max(0.015, base_tau - 0.01)
            if conf > 70:
                tau = max(0.010, tau - 0.005)
        elif r == "bearish":
            tau = min(0.06, base_tau + 0.02)
            if conf > 70:
                tau = min(0.08, tau + 0.01)
        return float(tau)
    except Exception:
        return float(base_tau)


def train_and_save_bundle() -> Tuple[str, dict]:
    csv_path = _find_latest_training_csv()
    df = pd.read_csv(csv_path)
    # Flatten MultiIndex columns if present (e.g., from yfinance downstream merges)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(lvl) for lvl in col if str(lvl) != ""]) for col in df.columns]
    # Sanity: ensure ticker universe not empty; fallback to alternative dataset if needed
    if "Ticker" not in df.columns or df["Ticker"].nunique() == 0:
        alt = None
        for name in [
            "data/training_dataset_20d_v3_with_live_v3.csv",
            "data/training_dataset_20d_v3_with_overlay_v2_with_meta.csv",
            "data/training_dataset_20d_v3.csv",
        ]:
            try:
                if Path(name).exists():
                    alt = pd.read_csv(name)
                    if "Ticker" in alt.columns and alt["Ticker"].nunique() > 0:
                        df = alt
                        print(f"DEBUG: Fallback to {name} due to empty universe in {csv_path}")
                        break
            except Exception:
                continue
        if "Ticker" not in df.columns or df["Ticker"].nunique() == 0:
            raise RuntimeError("Training dataset has no tickers; verify data ingestion pipeline.")
    df_recent = _filter_recent_window(df)
    base_tau = _regime_adjusted_tau(DEFAULT_RELATIVE_THRESHOLD)
    # Dynamic threshold search: start high (0.20) and reduce until class diversity achieved
    def _dynamic_label(df_in: pd.DataFrame, start_tau: float) -> Tuple[pd.Series, float]:
        # Begin at 0.20 and reduce by 0.02 down to 0.05
        tau_try = max(0.20, float(start_tau))
        while tau_try >= 0.05:
            y_try = _build_label(df_in, tau_try)
            # Require at least 10 positives and both classes present
            if y_try.sum() >= 10 and len(np.unique(y_try)) > 1:
                print(f"Adjusted success threshold to {tau_try:.2f} to ensure class diversity.")
                return y_try, tau_try
            tau_try = round(tau_try - 0.02, 4)
        # Final attempt with original regime-adjusted tau
        y_final = _build_label(df_in, float(start_tau))
        return y_final, float(start_tau)

    y, tau = _dynamic_label(df_recent, base_tau)
    # Data profiling (Probe) prior to diversity checks
    probe_df = df_recent.copy()
    ret_col = "Forward_Return_20d" if "Forward_Return_20d" in probe_df.columns else ("Return_20d" if "Return_20d" in probe_df.columns else None)
    try:
        print(f"DEBUG: Loaded {probe_df['Ticker'].nunique()} unique tickers.")
        print(f"DEBUG: Total rows: {len(probe_df)}")
        if ret_col:
            r = pd.to_numeric(probe_df[ret_col], errors="coerce")
            print(f"DEBUG: Max 20d Return found: {np.nanmax(r.values):.4f}")
            print(f"DEBUG: 95th Percentile Return: {np.nanpercentile(r.values, 95):.4f}")
            print(f"DEBUG: Number of NaN targets: {probe_df[ret_col].isna().sum()}")
        else:
            print("DEBUG: No 20d return column found (Forward_Return_20d/Return_20d)")
    except Exception:
        print("DEBUG: Probe logging failed")
    if len(np.unique(y)) < 2 or y.sum() < 10:
        # Fallback: try full dataset with dynamic threshold
        df_recent = df
        y, tau = _dynamic_label(df_recent, base_tau)
        # Run probe on full dataset too
        try:
            probe_df = df_recent.copy()
            ret_col = "Forward_Return_20d" if "Forward_Return_20d" in probe_df.columns else ("Return_20d" if "Return_20d" in probe_df.columns else None)
            print(f"DEBUG(FULL): Loaded {probe_df['Ticker'].nunique()} unique tickers.")
            print(f"DEBUG(FULL): Total rows: {len(probe_df)}")
            if ret_col:
                r = pd.to_numeric(probe_df[ret_col], errors="coerce")
                print(f"DEBUG(FULL): Max 20d Return found: {np.nanmax(r.values):.4f}")
                print(f"DEBUG(FULL): 95th Percentile Return: {np.nanpercentile(r.values, 95):.4f}")
                print(f"DEBUG(FULL): Number of NaN targets: {probe_df[ret_col].isna().sum()}")
        except Exception:
            pass
        if len(np.unique(y)) < 2 or y.sum() < 10:
            raise RuntimeError("Insufficient class diversity in labels even after dynamic threshold search. Investigate data ingestion and return calculations.")
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
    # Compute training feature bin metadata for drift monitoring
    feature_bins = {}
    training_bin_pct = {}
    quantiles = np.linspace(0.0, 1.0, 11)
    for feat in feature_cols:
        try:
            s = pd.to_numeric(df_recent[feat], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                continue
            edges = np.nanquantile(s.values, quantiles).tolist()
            # Ensure unique and sorted edges; expand slightly if degenerate
            edges = sorted(set(edges))
            if len(edges) < 3:
                mn, mx = float(np.nanmin(s.values)), float(np.nanmax(s.values))
                if mn == mx:
                    mx = mn + 1e-6
                edges = [mn, (mn+mx)/2.0, mx]
            # Build bin counts
            cats = pd.cut(s, bins=edges, include_lowest=True, duplicates='drop')
            cnt = cats.value_counts(sort=False)
            pct = (cnt / cnt.sum()).astype(float).tolist()
            feature_bins[feat] = edges
            training_bin_pct[feat] = pct
        except Exception:
            continue

    bundle = {
        "model": final_model,
        "feature_names": feature_cols,
        "preferred_scoring_mode_20d": "hybrid",
        "trained_at": ts,
        "metrics": {"auc": float(auc), "logloss": float(ll), "precision_at_k": float(p_at_k), "k": K_TOP},
        "feature_bins": feature_bins,
        "training_bin_pct": training_bin_pct,
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
