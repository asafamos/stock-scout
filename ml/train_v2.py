from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, precision_score
from xgboost import XGBClassifier

import yfinance as yf

from ml.feature_pipeline import FeaturePipeline
from ml.targets import compute_smart_targets


DEFAULT_UNIVERSE = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD"]


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    returns = close.diff()
    gain = returns.clip(lower=0)
    loss = -returns.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=window).mean()
    return atr


def fetch_training_data(tickers: List[str]) -> pd.DataFrame:
    """Fetch 2 years of OHLCV data via yfinance and compute simple RSI/ATR.

    Returns a concatenated DataFrame with columns:
    - Ticker, Date, Open, High, Low, Close, Volume, Sector, RSI, ATR, ATR_Pct, PE, MarketCap
    PE defaults to 25 (placeholder) and Sector to 'Technology' for PoC.
    MarketCap set to NaN (placeholder).
    """
    end = datetime.utcnow().date()
    start = end - timedelta(days=365 * 2 + 10)

    frames: List[pd.DataFrame] = []
    for t in tickers:
        data = yf.download(t, start=start, end=end, progress=False)
        if data.empty:
            continue
        dr = data.reset_index()
        # Flatten possible MultiIndex columns from yfinance
        if isinstance(dr.columns, pd.MultiIndex):
            dr.columns = dr.columns.get_level_values(0)
        # Select and ensure types
        df = dr[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["Open"] = df["Open"].astype(float)
        df["High"] = df["High"].astype(float)
        df["Low"] = df["Low"].astype(float)
        df["Close"] = df["Close"].astype(float)
        df["Volume"] = df["Volume"].astype(float)
        df["Ticker"] = t
        df["Sector"] = "Technology"

        # Technicals
        df["RSI"] = _compute_rsi(df["Close"]).bfill()
        atr = _compute_atr(df["High"], df["Low"], df["Close"]).bfill()
        df["ATR"] = atr.astype(float)
        close_vals = df["Close"].astype(float)
        atr_pct_arr = (atr.astype(float).to_numpy() / close_vals.to_numpy())
        atr_pct_arr = np.where(np.isfinite(atr_pct_arr), atr_pct_arr, 0.0)
        # Fill initial NaNs due to windowing
        atr_pct_arr = np.nan_to_num(atr_pct_arr, nan=0.0)
        df["ATR_Pct"] = atr_pct_arr

        # Placeholders for fundamentals and risk
        df["PE"] = 25.0
        df["MarketCap"] = np.nan

        frames.append(df[["Ticker", "Date", "Open", "High", "Low", "Close", "Volume", "Sector", "RSI", "ATR", "ATR_Pct", "PE", "MarketCap"]])

    if not frames:
        raise RuntimeError("No data fetched from yfinance")

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return out


def _time_based_split(df: pd.DataFrame, train_frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.sort(df["Date"].unique())
    split_idx = int(len(dates) * train_frac)
    train_dates = set(dates[:split_idx])
    test_dates = set(dates[split_idx:])
    train_df = df[df["Date"].isin(train_dates)].copy()
    test_df = df[df["Date"].isin(test_dates)].copy()
    return train_df, test_df


def _to_xy(features_list: List, labels: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    # Convert TickerFeatures list into feature matrix in fixed order
    X = np.array([[
        tf.model_features["feat_rsi"],
        tf.model_features["feat_atr_pct"],
        tf.model_features["feat_fund_pe_sector_rel"],
    ] for tf in features_list], dtype=float)
    y = labels.to_numpy()
    return X, y


def _save_artifacts(pipeline: FeaturePipeline, model: XGBClassifier, metrics_text: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(out_dir, "feature_pipeline.joblib"))
    # Save underlying Booster to avoid sklearn wrapper metadata issues
    model.get_booster().save_model(os.path.join(out_dir, "model_xgb.json"))
    with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(metrics_text)


def train_model() -> None:
    # 1. Data
    tickers = DEFAULT_UNIVERSE
    raw = fetch_training_data(tickers)

    # 2. Targets per ticker, drop tail NaNs
    frames = []
    for t, g in raw.groupby("Ticker"):
        tg = compute_smart_targets(g, horizon=20)
        tg = tg.dropna(subset=["target_log_return", "future_volatility", "forward_sharpe", "raw_forward_return"])
        frames.append(tg)
    data = pd.concat(frames, axis=0, ignore_index=True)

    # 3. Time-based split
    train_df, test_df = _time_based_split(data, train_frac=0.8)

    # 4. Pipeline fit on training subset (features only)
    fp = FeaturePipeline()
    X_train = train_df[["Ticker", "Date", "Sector", "PE", "RSI", "ATR_Pct", "MarketCap", "Volume"]].copy()
    fp.fit(X_train)

    # 5. Transform train and test
    as_of_train = pd.to_datetime(train_df["Date"]).max().to_pydatetime()
    as_of_test = pd.to_datetime(test_df["Date"]).max().to_pydatetime()
    tf_train = fp.transform(X_train, as_of_date=as_of_train)
    X_test = test_df[["Ticker", "Date", "Sector", "PE", "RSI", "ATR_Pct", "MarketCap", "Volume"]].copy()
    tf_test = fp.transform(X_test, as_of_date=as_of_test)

    # 6. Prepare X, y
    y_train = train_df["target_class"].astype(int)
    y_test = test_df["target_class"].astype(int)
    X_tr, y_tr = _to_xy(tf_train, y_train)
    X_te, y_te = _to_xy(tf_test, y_test)

    # 7. Model
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        max_depth=4,
        n_estimators=100,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_tr, y_tr)

    # 8. Evaluation
    y_pred = model.predict(X_te)
    report = classification_report(y_te, y_pred, digits=3)
    prec_gold = precision_score(y_te, y_pred, labels=[2], average="macro")
    print(report)
    print(f"Precision (Class 2 - Gold): {prec_gold:.3f}")

    metrics_text = report + "\n" + f"Precision (Class 2 - Gold): {prec_gold:.3f}\n"

    # 9. Serialization
    out_dir = os.path.join("models", "v2")
    _save_artifacts(fp, model, metrics_text, out_dir)


if __name__ == "__main__":
    train_model()
