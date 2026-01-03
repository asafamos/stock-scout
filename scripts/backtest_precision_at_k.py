import os
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import precision_score

MODELS_DIR = Path("models")
DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")

K_TOP = int(os.environ.get("PRECISION_AT_K", "20"))

NON_FEATURE_COLS = {
    "Ticker","As_Of_Date","date","Sector","Label","y","target","target_20d","Return_20d","SPY_20d_ret","Sector_20d_ret"
}


def _latest_model_path() -> Path:
    candidates = sorted(MODELS_DIR.glob("model_20d_v3_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    p = MODELS_DIR / "model_20d_v3.pkl"
    if p.exists():
        return p
    raise FileNotFoundError("No model bundle found")


def _latest_test_csv() -> str:
    candidates = glob.glob(str(DATA_DIR / "test_20d_v3_*.csv"))
    candidates += glob.glob(str(DATA_DIR / "test_20d_v3_mini.csv"))
    if not candidates:
        raise FileNotFoundError("No test CSV found in data/ matching test_20d_v3_*.csv")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _select_feature_cols(df: pd.DataFrame, model_features: list[str]) -> list[str]:
    # Use model's feature list if available, else infer numerics
    if model_features:
        return [c for c in model_features if c in df.columns]
    cols = []
    for c in df.columns:
        if c in NON_FEATURE_COLS:
            continue
        try:
            s = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            continue
        if s.notna().sum() < max(50, int(0.2 * len(df))):
            continue
        cols.append(c)
    return cols


def run_backtest():
    model_path = _latest_model_path()
    bundle = joblib.load(model_path)
    model = bundle.get("model")
    feats = bundle.get("feature_names", [])

    csv_path = _latest_test_csv()
    df = pd.read_csv(csv_path)

    # Build label
    y = None
    for c in ("Label","target","y","target_20d"):
        if c in df.columns:
            y = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).values
            break
    if y is None:
        ret20 = pd.to_numeric(df.get("Return_20d", np.nan), errors="coerce")
        spy20 = pd.to_numeric(df.get("SPY_20d_ret", np.nan), errors="coerce").fillna(0)
        y = ((ret20 - spy20) >= 0.03).astype(int).values

    X_cols = _select_feature_cols(df, feats)
    X = df[X_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    prob = model.predict_proba(X)[:,1]

    # Precision@K
    order = np.argsort(-prob)
    topk = order[:K_TOP]
    p_at_k = float(np.mean(y[topk]))

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "model": str(model_path),
        "csv": csv_path,
        "K": K_TOP,
        "precision_at_k": p_at_k,
    }
    rep = REPORTS_DIR / f"backtest_precision_at_k.json"
    with open(rep, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run_backtest()
