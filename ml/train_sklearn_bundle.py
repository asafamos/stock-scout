from __future__ import annotations

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier

from ml.targets import compute_smart_targets


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    returns = close.diff()
    gain = returns.clip(lower=0)
    loss = -returns.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def fetch_prices(tickers: List[str]) -> pd.DataFrame:
    end = datetime.utcnow().date()
    start = end - timedelta(days=365 + 30)
    frames: List[pd.DataFrame] = []
    for t in tickers:
        data = yf.download(t, start=start, end=end, progress=False)
        if data is None or data.empty:
            continue
        df = data.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            df[c] = df[c].astype(float)
        df["Ticker"] = t
        frames.append(df)
    if not frames:
        raise RuntimeError("No data fetched for training bundle")
    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Compute RSI and ATR_Pct
    out_frames: List[pd.DataFrame] = []
    for t, g in df.groupby("Ticker"):
        g = g.copy()
        g["RSI"] = _compute_rsi(g["Close"]).bfill()
        prev_close = g["Close"].shift(1)
        tr = pd.concat([
            (g["High"] - g["Low"]),
            (g["High"] - prev_close).abs(),
            (g["Low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean().bfill()
        g["ATR_Pct"] = (atr / g["Close"]).fillna(0.0).replace([np.inf, -np.inf], 0.0)
        out_frames.append(g)
    out = pd.concat(out_frames, axis=0, ignore_index=True)
    return out


def train_and_export() -> Path:
    tickers = ["AAPL", "MSFT", "SPY"]
    raw = fetch_prices(tickers)
    feats = build_features(raw)

    # Targets
    frames: List[pd.DataFrame] = []
    for t, g in feats.groupby("Ticker"):
        tg = compute_smart_targets(g, horizon=20)
        tg = tg.dropna(subset=["raw_forward_return", "forward_sharpe"]).copy()
        frames.append(tg)
    data = pd.concat(frames, axis=0, ignore_index=True)

    # Feature matrix and label
    X = data[["RSI", "ATR_Pct"]].to_numpy(dtype=float)
    y = (data["forward_sharpe"] > 1.5).astype(int).to_numpy()

    # Simple classifier
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X, y)

    # Bundle and metadata
    feature_list = ["RSI", "ATR_Pct"]
    metadata = {
        "sklearn_version": __import__("sklearn").__version__,
        "feature_list": feature_list,
        "training_date_utc": datetime.utcnow().isoformat(),
        "label_summary": {
            "type": "binary_sharpe_threshold",
            "horizon_days": 20,
            "sharpe_threshold": 1.5,
        },
    }

    bundle_dir = Path("ml/bundles") / f"ml_bundle_sklearn_1_8_0_{datetime.utcnow().strftime('%Y%m%d')}"
    latest_dir = Path("ml/bundles/latest")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    # Save model.joblib and metadata.json
    joblib.dump(clf, bundle_dir / "model.joblib")
    joblib.dump(clf, latest_dir / "model.joblib")
    with open(bundle_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    with open(latest_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    print(f"Saved bundle to {bundle_dir} and symlink-like latest at {latest_dir}")
    return latest_dir


if __name__ == "__main__":
    train_and_export()
