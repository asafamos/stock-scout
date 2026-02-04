#!/usr/bin/env python3
"""
Smart ML Training Script - Uses yfinance (free) for fresh data.
Falls back to existing CSV if needed.

Usage:
    python scripts/train_ml_smart.py              # Fresh data (recommended)
    python scripts/train_ml_smart.py --use-csv    # Use existing CSV
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import joblib
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

from core.feature_registry import get_feature_names, get_feature_defaults

# Paths
MODELS_DIR = Path("models")
DATA_DIR = Path("data")

# Feature list (34 features)
FEATURE_NAMES = get_feature_names("v3")
FEATURE_DEFAULTS = get_feature_defaults("v3")


def fetch_ticker_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV data for a single ticker using yfinance."""
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df['Ticker'] = ticker
        return df
    except Exception as e:
        return None


def fetch_spy_returns(start: str, end: str) -> pd.DataFrame:
    """Fetch SPY data for relative strength calculation."""
    try:
        df = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df['SPY_Return_20d'] = df['Close'].pct_change(20)
        df['SPY_Return_60d'] = df['Close'].pct_change(60)
        return df
    except Exception:
        return pd.DataFrame()


def calculate_all_features(df: pd.DataFrame, spy_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Calculate all 34 ML features for a ticker."""
    if df is None or len(df) < 60:
        return None

    result = df.copy()
    close = result['Close']
    high = result['High']
    low = result['Low']
    volume = result['Volume']

    # --- Technical Base (5) ---
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    result['RSI'] = 100 - (100 / (1 + rs))

    # ATR_Pct
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    result['ATR_Pct'] = atr / close

    # Returns
    result['Return_20d'] = close.pct_change(20)
    result['Return_10d'] = close.pct_change(10)
    result['Return_5d'] = close.pct_change(5)

    # --- Volatility Patterns (4) ---
    atr_10 = tr.rolling(10).mean()
    atr_30 = tr.rolling(30).mean()
    result['VCP_Ratio'] = atr_10 / atr_30.replace(0, np.nan)

    std_5 = close.rolling(5).std()
    std_20 = close.rolling(20).std()
    result['Tightness_Ratio'] = std_5 / std_20.replace(0, np.nan)

    high_52w = high.rolling(252, min_periods=60).max()
    result['Dist_From_52w_High'] = (close / high_52w) - 1

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200, min_periods=100).mean()
    result['MA_Alignment'] = ((close > ma20) & (ma20 > ma50) & (ma50 > ma200)).astype(float)

    # --- Volume Basic (3) ---
    vol_5 = volume.rolling(5).mean()
    vol_20 = volume.rolling(20).mean()
    result['Volume_Surge'] = vol_5 / vol_20.replace(0, np.nan)

    price_change = close.diff()
    up_vol = volume.where(price_change > 0, 0).rolling(20).sum()
    down_vol = volume.where(price_change < 0, 0).rolling(20).sum()
    result['Up_Down_Volume_Ratio'] = up_vol / down_vol.replace(0, np.nan)

    result['Momentum_Consistency'] = (price_change > 0).rolling(20).mean()

    # --- Relative Strength (1) ---
    if not spy_df.empty and 'SPY_Return_20d' in spy_df.columns:
        result = result.join(spy_df[['SPY_Return_20d']], how='left')
        result['RS_vs_SPY_20d'] = result['Return_20d'] - result['SPY_Return_20d']
        result.drop(columns=['SPY_Return_20d'], inplace=True, errors='ignore')
    else:
        result['RS_vs_SPY_20d'] = 0.0

    # --- Market Regime (4) - simplified ---
    result['Market_Regime'] = 0.0  # Neutral
    result['Market_Volatility'] = 0.15
    result['Market_Trend'] = 0.0
    result['High_Volatility'] = (result['ATR_Pct'] > 0.04).astype(float)

    # --- Sector Relative (3) - simplified ---
    result['Sector_RS'] = 0.0
    result['Sector_Momentum'] = 0.0
    result['Sector_Rank'] = 0.5

    # --- Volume Advanced (5) ---
    result['Volume_Ratio_20d'] = volume / vol_20.replace(0, np.nan)

    obv = (np.sign(price_change) * volume).cumsum()
    obv_20_ago = obv.shift(20)
    result['Volume_Trend'] = ((obv - obv_20_ago) / obv_20_ago.abs().replace(0, np.nan)).clip(-1, 1)

    total_vol = volume.rolling(20).sum()
    result['Up_Volume_Ratio'] = up_vol / total_vol.replace(0, np.nan)

    price_up_5d = close > close.shift(5)
    vol_up = volume > vol_20
    result['Volume_Price_Confirm'] = (price_up_5d & vol_up).astype(float)

    vol_60_rank = volume.rolling(60).apply(lambda x: (x <= x.iloc[-1]).sum() / len(x), raw=False)
    result['Relative_Volume_Rank'] = vol_60_rank

    # --- Price Action (9) ---
    low_52w = low.rolling(252, min_periods=60).min()
    result['Distance_From_52w_Low'] = (close - low_52w) / low_52w.replace(0, np.nan)

    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()
    avg_20 = close.rolling(20).mean()
    result['Consolidation_Tightness'] = (high_20 - low_20) / avg_20.replace(0, np.nan)

    def days_since_high(series):
        def calc(window):
            if len(window) < 2:
                return np.nan
            max_idx = window.argmax()
            return (len(window) - 1 - max_idx) / 252
        return series.rolling(252, min_periods=60).apply(calc, raw=True)

    result['Days_Since_52w_High'] = days_since_high(high)

    result['Price_vs_SMA50'] = (close - ma50) / ma50.replace(0, np.nan)
    result['Price_vs_SMA200'] = (close - ma200) / ma200.replace(0, np.nan)
    result['SMA50_vs_SMA200'] = (ma50 - ma200) / ma200.replace(0, np.nan)

    ma20_5d_ago = ma20.shift(5)
    result['MA_Slope_20d'] = (ma20 - ma20_5d_ago) / ma20_5d_ago.replace(0, np.nan)

    result['Distance_To_Resistance'] = (high_20 - close) / close.replace(0, np.nan)

    support_threshold = low_20 * 1.05
    result['Support_Strength'] = (close <= support_threshold).rolling(20).mean()

    # Forward return for labeling
    result['Forward_Return_20d'] = close.shift(-20) / close - 1

    # Cleanup
    result['Ticker'] = ticker
    result = result.replace([np.inf, -np.inf], np.nan)

    return result


def load_universe(limit: int = 500) -> list:
    """Load stock universe - S&P 500 + additional tickers."""
    sp500_path = Path("sp500_tickers.txt")
    if sp500_path.exists():
        with open(sp500_path) as f:
            tickers = [line.strip() for line in f if line.strip()]
        return tickers[:limit]

    # Fallback: well-known tickers
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "JNJ",
        "V", "UNH", "HD", "PG", "MA", "DIS", "PYPL", "BAC", "ADBE", "NFLX",
        "CMCSA", "XOM", "VZ", "INTC", "T", "KO", "PFE", "MRK", "PEP", "ABT",
        "TMO", "CVX", "COST", "WMT", "MCD", "CSCO", "ACN", "AVGO", "TXN", "QCOM",
    ][:limit]


def train_fresh_data(n_tickers: int = 200, lookback_days: int = 365):
    """Train model on freshly downloaded data."""
    print("🚀 Training on FRESH DATA (yfinance)")
    print("=" * 50)

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=lookback_days + 60)).strftime('%Y-%m-%d')

    print(f"📅 Date range: {start_date} to {end_date}")

    # Load universe
    tickers = load_universe(n_tickers)
    print(f"📋 Universe: {len(tickers)} tickers")

    # Fetch SPY first
    print("📥 Fetching SPY benchmark...")
    spy_df = fetch_spy_returns(start_date, end_date)
    if spy_df.empty:
        print("   ⚠️ SPY unavailable, RS features will be 0")
    else:
        print(f"   ✅ SPY data: {len(spy_df)} rows")

    # Fetch all tickers in parallel
    print(f"📥 Downloading {len(tickers)} tickers (parallel)...")
    all_data = []
    failed = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_ticker_data, t, start_date, end_date): t for t in tickers}

        for i, future in enumerate(as_completed(futures), 1):
            ticker = futures[future]
            df = future.result()

            if df is not None and len(df) > 60:
                featured = calculate_all_features(df, spy_df, ticker)
                if featured is not None:
                    all_data.append(featured)
            else:
                failed += 1

            if i % 50 == 0:
                print(f"   ... {i}/{len(tickers)} processed")

    print(f"   ✅ Success: {len(all_data)} tickers")
    print(f"   ❌ Failed: {failed} tickers")

    if len(all_data) < 20:
        print("❌ Not enough data! Falling back to existing CSV...")
        return train_from_csv()

    # Combine all data
    full_df = pd.concat(all_data, ignore_index=True)

    # Remove rows without forward return (last 20 days)
    full_df = full_df.dropna(subset=['Forward_Return_20d'])

    print(f"📊 Total training rows: {len(full_df)}")

    # Save fresh training data
    fresh_csv = DATA_DIR / f"training_dataset_fresh_{datetime.now().strftime('%Y%m%d')}.csv"
    full_df.to_csv(fresh_csv, index=True)
    print(f"💾 Saved fresh data: {fresh_csv}")

    return train_model(full_df, source="fresh_yfinance")


def train_from_csv():
    """Train from existing CSV data."""
    print("🚀 Training from EXISTING CSV")
    print("=" * 50)

    csv_candidates = [
        DATA_DIR / "training_dataset_20d_v3_with_overlay_v2_with_meta.csv",
        DATA_DIR / "training_dataset_20d_v3_with_overlay_v2.csv",
        DATA_DIR / "training_dataset_20d_v3.csv",
    ]

    csv_path = None
    for c in csv_candidates:
        if c.exists():
            csv_path = c
            break

    if csv_path is None:
        print("❌ No training CSV found!")
        return False

    print(f"📂 Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Rows: {len(df)}")

    return train_model(df, source=str(csv_path))


def train_model(df: pd.DataFrame, source: str) -> bool:
    """Train model from DataFrame."""

    # Column mapping for compatibility
    column_mapping = {
        'Dist_52w_High': 'Dist_From_52w_High',
        'MomCons': 'Momentum_Consistency',
        'VolSurge': 'Volume_Surge',
        'RS_SPY_20d': 'RS_vs_SPY_20d',
    }
    for old, new in column_mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # Create label if needed
    if 'Label' not in df.columns and 'Forward_Return_20d' in df.columns:
        threshold = df['Forward_Return_20d'].quantile(0.80)
        df['Label'] = (df['Forward_Return_20d'] >= threshold).astype(int)
        print(f"🎯 Created labels (threshold: {threshold*100:.1f}%)")

    if 'Label' not in df.columns:
        print("❌ No label column!")
        return False

    # Fill missing features
    for feat in FEATURE_NAMES:
        if feat not in df.columns:
            df[feat] = FEATURE_DEFAULTS.get(feat, 0.0)

    # Prepare X, y
    X = df[FEATURE_NAMES].copy()
    y = df['Label'].copy()

    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        X[col] = X[col].fillna(FEATURE_DEFAULTS.get(col, 0.0))

    print(f"\n🎯 Label distribution:")
    print(f"   Winners: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"   Losers: {len(y) - y.sum()} ({(1-y.mean())*100:.1f}%)")

    # Cross-validation
    print("\n📊 Time-Series Cross-Validation (5 folds)...")
    tscv = TimeSeriesSplit(n_splits=5)
    aucs = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if y_train.sum() < 10 or y_val.sum() < 5:
            continue

        model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.05,
            random_state=42, class_weight='balanced'
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        aucs.append(auc)
        print(f"   Fold {fold}: AUC = {auc:.4f}")

    mean_auc = np.mean(aucs) if aucs else 0.5
    print(f"\n✅ Mean CV AUC: {mean_auc:.4f}")

    # Train final model
    print("\n🔧 Training final model...")
    final_model = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.05,
        random_state=42, class_weight='balanced'
    )
    final_model.fit(X, y)

    # Save
    MODELS_DIR.mkdir(exist_ok=True)

    bundle = {"model": final_model, "feature_names": FEATURE_NAMES}
    bundle_path = MODELS_DIR / "model_20d_v3.pkl"
    joblib.dump(bundle, bundle_path)
    print(f"💾 Model saved: {bundle_path}")

    import sklearn
    metadata = {
        "sklearn_version": sklearn.__version__,
        "feature_list": FEATURE_NAMES,
        "training_timestamp_utc": datetime.utcnow().isoformat(),
        "model_type": "HistGradientBoostingClassifier",
        "cv_auc": float(mean_auc),
        "training_samples": len(X),
        "source": source,
    }
    meta_path = MODELS_DIR / "model_20d_v3.metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Metadata saved: {meta_path}")

    print("\n" + "=" * 50)
    print("✅ TRAINING COMPLETE!")
    print(f"   Model: {bundle_path}")
    print(f"   AUC: {mean_auc:.4f}")
    print("=" * 50)

    return True


def main():
    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument("--use-csv", action="store_true", help="Use existing CSV instead of fresh data")
    parser.add_argument("--tickers", type=int, default=200, help="Number of tickers for fresh data")
    args = parser.parse_args()

    if args.use_csv:
        train_from_csv()
    else:
        train_fresh_data(n_tickers=args.tickers)


if __name__ == "__main__":
    main()
