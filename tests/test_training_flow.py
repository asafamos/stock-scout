import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier

from ml.feature_pipeline import FeaturePipeline
from ml.targets import compute_smart_targets


def make_synth(n_days: int = 100) -> pd.DataFrame:
    start = datetime(2025, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    # Create a simple rising price with some noise
    close = 100.0 * np.exp(np.linspace(0, 0.1, n_days)) * (1.0 + 0.01 * np.random.randn(n_days))
    high = close * (1 + 0.01)
    low = close * (1 - 0.01)
    volume = 1_000_000 + (np.random.randint(0, 1000, size=n_days))

    df = pd.DataFrame({
        "Ticker": ["TEST"] * n_days,
        "Date": dates,
        "Open": close,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
        "Sector": ["Technology"] * n_days,
    })

    # Minimal technicals for pipeline
    # Simple RSI proxy: scaled returns
    returns = pd.Series(close).pct_change().fillna(0.0)
    rsi_proxy = (returns - returns.min()) / (returns.max() - returns.min() + 1e-9) * 100.0
    df["RSI"] = rsi_proxy
    atr = (pd.Series(high) - pd.Series(low)).rolling(14).mean().fillna(0.0)
    df["ATR_Pct"] = (atr / pd.Series(close)).fillna(0.0)

    # Required columns placeholders
    df["PE"] = 20.0
    df["MarketCap"] = np.nan
    return df


def test_mini_training_cycle_creates_artifacts():
    df = make_synth(100)
    # Compute targets and drop tail NaNs
    out = compute_smart_targets(df, horizon=20)
    out = out.dropna(subset=["target_log_return", "future_volatility", "forward_sharpe", "raw_forward_return"])\
             .reset_index(drop=True)

    # Time-based split
    split = int(len(out) * 0.8)
    train_df = out.iloc[:split].copy()
    test_df = out.iloc[split:].copy()

    fp = FeaturePipeline()
    X_train = train_df[["Ticker", "Date", "Sector", "PE", "RSI", "ATR_Pct", "MarketCap", "Volume"]]
    fp.fit(X_train)

    tf_train = fp.transform(X_train, as_of_date=pd.to_datetime(train_df["Date"]).max().to_pydatetime())
    X_test = test_df[["Ticker", "Date", "Sector", "PE", "RSI", "ATR_Pct", "MarketCap", "Volume"]]
    tf_test = fp.transform(X_test, as_of_date=pd.to_datetime(test_df["Date"]).max().to_pydatetime())

    # Prepare features and labels
    def to_xy(lst, labels):
        X = np.array([[o.model_features["feat_rsi"], o.model_features["feat_atr_pct"], o.model_features["feat_fund_pe_sector_rel"]] for o in lst])
        y = labels.to_numpy()
        return X, y

    y_train = train_df["target_class"].astype(int)
    y_test = test_df["target_class"].astype(int)
    # For robustness in this mini test, collapse class 2 into class 1
    y_train_bin = y_train.replace({2: 1})
    y_test_bin = y_test.replace({2: 1})
    X_tr, y_tr = to_xy(tf_train, y_train_bin)
    X_te, y_te = to_xy(tf_test, y_test_bin)

    # Train a small XGBClassifier
    model = XGBClassifier(objective="binary:logistic", n_estimators=10, max_depth=3, random_state=42)
    model.fit(X_tr, y_tr)

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline_path = os.path.join(tmpdir, "feature_pipeline.joblib")
        model_path = os.path.join(tmpdir, "model_xgb.json")

        joblib.dump(fp, pipeline_path)
        model.save_model(model_path)

        assert os.path.exists(pipeline_path)
        assert os.path.exists(model_path)
