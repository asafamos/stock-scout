#!/usr/bin/env python3
"""
V4 ML Training Script for Stock Scout

Enhanced training pipeline using V4 features (52 features vs 34 in V3):
- Comprehensive feature set including sentiment, institutional, earnings
- Walk-forward validation
- Feature importance analysis
- Model calibration
- Monitoring integration

Usage:
    python scripts/train_ml_v4.py --output models/ml_20d_v4.pkl
    python scripts/train_ml_v4.py --skip-sentiment  # Skip slow API calls
"""
from __future__ import annotations
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any
import warnings
import pickle
import json

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score, log_loss, precision_recall_curve, average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

# Project imports
from core.api_keys import get_api_key
from core.feature_registry_v4 import (
    FEATURE_SPECS_V4,
    get_feature_names_v4,
    FEATURE_COUNT_V4,
    get_feature_defaults_v4 as get_registry_defaults
)
from core.ml_feature_builder_v4 import (
    build_features_v4,
    build_features_batch_v4,
    get_feature_defaults_v4
)
from core.sentiment_data import fetch_alternative_data_batch
from core.sector_mapping import get_stock_sector, get_sector_etf, get_all_sector_etfs
from core.monitoring import Monitor, log_api_call

# Configuration
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
DATA_DIR = Path("data")

# Training parameters
TRAIN_CONFIG = {
    "years_history": 5,
    "min_samples_per_ticker": 50,
    "target_threshold": 0.10,  # 10% gain = positive label
    "max_workers": 10,
    "n_splits": 5,  # Time series CV splits
    "test_size_days": 180,  # 6 months test per fold
    "random_state": 42
}


class EnsembleClassifierV4(BaseEstimator, ClassifierMixin):
    """
    V4 Ensemble classifier combining multiple algorithms.
    
    Uses:
    - HistGradientBoosting (40%): Fast, handles missing values
    - RandomForest (35%): Robust, good for feature interactions  
    - Logistic Regression (25%): Calibrated probabilities
    
    Calibrated using isotonic regression for better probability estimates.
    """
    
    def __init__(
        self,
        weights: Optional[List[float]] = None,
        calibrate: bool = True,
        random_state: int = 42
    ):
        self.weights = weights or [0.40, 0.35, 0.25]
        self.calibrate = calibrate
        self.random_state = random_state
        self.models_ = []
        self.scaler_ = StandardScaler()
        self.feature_names_ = []
        self.n_features_in_ = 0
        self.feature_importances_ = None
    
    def fit(self, X, y, feature_names: Optional[List[str]] = None):
        """Fit all base models and calibrate."""
        X_scaled = self.scaler_.fit_transform(X)
        
        self.feature_names_ = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.n_features_in_ = X.shape[1]
        
        # Initialize base models
        hgb = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=8,
            learning_rate=0.05,
            l2_regularization=1.0,
            min_samples_leaf=50,
            random_state=self.random_state
        )
        
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=30,
            n_jobs=-1,
            random_state=self.random_state
        )
        
        lr = LogisticRegression(
            C=0.1,
            max_iter=1000,
            solver='lbfgs',
            random_state=self.random_state
        )
        
        base_models = [hgb, rf, lr]
        
        if self.calibrate:
            # Use calibrated classifiers
            self.models_ = []
            for model in base_models:
                calibrated = CalibratedClassifierCV(
                    model, 
                    method='isotonic',
                    cv=3
                )
                calibrated.fit(X_scaled, y)
                self.models_.append(calibrated)
        else:
            # Fit without calibration
            self.models_ = []
            for model in base_models:
                model.fit(X_scaled, y)
                self.models_.append(model)
        
        # Compute ensemble feature importance from Random Forest
        # (HGB and calibrated models don't expose feature_importances_ directly)
        if self.calibrate:
            # Get feature importance from the RF inside calibrated wrapper
            rf_calibrated = self.models_[1]
            if hasattr(rf_calibrated, 'calibrated_classifiers_'):
                # Get first calibrated classifier's base estimator
                base_rf = rf_calibrated.calibrated_classifiers_[0].estimator
                self.feature_importances_ = base_rf.feature_importances_
        else:
            self.feature_importances_ = self.models_[1].feature_importances_
        
        return self
    
    def predict_proba(self, X):
        """Get weighted average probability predictions."""
        X_scaled = self.scaler_.transform(X)
        
        probas = []
        for model, weight in zip(self.models_, self.weights):
            proba = model.predict_proba(X_scaled)[:, 1]
            probas.append(proba * weight)
        
        combined = np.sum(probas, axis=0)
        
        # Return 2D array [P(neg), P(pos)]
        return np.column_stack([1 - combined, combined])
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


def fetch_polygon_history(ticker: str, start_str: str, end_str: str) -> Optional[pd.DataFrame]:
    """Fetch adjusted daily bars from Polygon.io."""
    api_key = get_api_key("POLYGON_API_KEY", required=False)
    if not api_key:
        return None
    
    import requests
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_str}/{end_str}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key
    }
    
    start_time = time.time()
    try:
        r = requests.get(url, params=params, timeout=10)
        latency_ms = (time.time() - start_time) * 1000
        
        if r.status_code != 200:
            log_api_call("polygon", False, latency_ms, f"HTTP {r.status_code}")
            return None
        
        data = r.json()
        if "results" not in data or not data["results"]:
            return None
        
        log_api_call("polygon", True, latency_ms)
        
        df = pd.DataFrame(data["results"])
        df = df.rename(columns={
            "v": "Volume", "o": "Open", "c": "Close", 
            "h": "High", "l": "Low", "t": "Date"
        })
        df["Date"] = pd.to_datetime(df["Date"], unit="ms")
        df = df.set_index("Date").sort_index()
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        log_api_call("polygon", False, latency_ms, str(e))
        return None


def get_universe_tickers(limit: int = 2000) -> List[str]:
    """Get universe of tickers to train on."""
    try:
        # Try local S&P 500 file
        sp500_file = DATA_DIR / "sp500_tickers.txt"
        if sp500_file.exists():
            with open(sp500_file) as f:
                tickers = [line.strip() for line in f if line.strip()]
            return tickers[:limit]
    except Exception:
        pass
    
    try:
        # Fallback to Wikipedia
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        tickers = tables[0]['Symbol'].str.replace(".", "-").tolist()
        return tickers[:limit]
    except Exception:
        # Minimal fallback
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD", "NFLX", "INTC"]


def calculate_labels(df: pd.DataFrame, threshold: float = 0.10) -> pd.Series:
    """
    Calculate binary labels based on forward returns.
    
    Args:
        df: DataFrame with 'Close' column
        threshold: Return threshold for positive label
    
    Returns:
        Series of 0/1 labels
    """
    forward_return = df['Close'].shift(-20) / df['Close'] - 1.0
    labels = (forward_return >= threshold).astype(int)
    return labels


def build_training_dataset(
    tickers: List[str],
    start_date: datetime,
    end_date: datetime,
    include_sentiment: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Build training dataset with V4 features.
    
    Args:
        tickers: List of tickers to process
        start_date: Start of training period
        end_date: End of training period
        include_sentiment: Whether to fetch sentiment data (slow)
        verbose: Print progress
    
    Returns:
        DataFrame with features and labels
    """
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    if verbose:
        print(f"📥 Building V4 training dataset...")
        print(f"   Period: {start_str} to {end_str}")
        print(f"   Tickers: {len(tickers)}")
        print(f"   Include sentiment: {include_sentiment}")
    
    # Fetch SPY for benchmark
    spy_df = fetch_polygon_history("SPY", start_str, end_str)
    if spy_df is None or len(spy_df) < 50:
        print("   ⚠️ Could not fetch SPY data, relative strength features will be zeros")
        spy_df = None
    
    # Fetch VIX
    vix_df = fetch_polygon_history("VIXY", start_str, end_str)  # VIX ETF proxy
    
    # Fetch sentiment data if requested
    sentiment_data = {}
    if include_sentiment:
        if verbose:
            print("   📰 Fetching sentiment data (this may take a while)...")
        try:
            sentiment_data = fetch_alternative_data_batch(tickers[:100])  # Limit to avoid rate limits
        except Exception as e:
            print(f"   ⚠️ Sentiment fetch failed: {e}")
    
    # Fetch price data for all tickers
    price_data: Dict[str, pd.DataFrame] = {}
    
    def _fetch_one(ticker: str) -> Tuple[str, Optional[pd.DataFrame]]:
        df = fetch_polygon_history(ticker, start_str, end_str)
        return ticker, df
    
    if verbose:
        print("   📥 Fetching price data...")
    
    with ThreadPoolExecutor(max_workers=TRAIN_CONFIG["max_workers"]) as executor:
        futures = [executor.submit(_fetch_one, t) for t in tickers]
        for i, future in enumerate(as_completed(futures)):
            ticker, df = future.result()
            if df is not None and len(df) >= TRAIN_CONFIG["min_samples_per_ticker"]:
                price_data[ticker] = df
            if verbose and (i + 1) % 100 == 0:
                print(f"     Fetched {i + 1}/{len(tickers)} tickers...")
    
    if verbose:
        print(f"   ✅ Loaded price data for {len(price_data)} tickers")
    
    # Build features for each ticker
    all_rows = []
    
    for ticker, price_df in price_data.items():
        try:
            # Calculate labels first
            labels = calculate_labels(price_df, TRAIN_CONFIG["target_threshold"])
            
            # Build features for each date
            for i in range(252, len(price_df) - 20):  # Need lookback and forward
                idx = price_df.index[i]
                
                # Get price data up to this date (no lookahead)
                price_subset = price_df.iloc[:i+1]
                
                # Build V4 features
                ticker_sentiment = sentiment_data.get(ticker, {})
                features = build_features_v4(
                    price_df=price_subset,
                    ticker=ticker,
                    spy_df=spy_df,
                    vix_df=vix_df,
                    sentiment_data=ticker_sentiment
                )
                
                # Add metadata
                features["Ticker"] = ticker
                features["Date"] = idx
                features["Label"] = labels.iloc[i]
                features["Forward_Return"] = (
                    price_df["Close"].iloc[i + 20] / price_df["Close"].iloc[i] - 1
                    if i + 20 < len(price_df) else np.nan
                )
                
                all_rows.append(features)
        
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Error processing {ticker}: {e}")
    
    if not all_rows:
        raise ValueError("No training data generated!")
    
    df = pd.DataFrame(all_rows)
    
    # Clean up
    df = df.dropna(subset=["Label"])
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN in features with defaults
    feature_names = get_feature_names_v4()
    defaults = get_feature_defaults_v4()
    
    for fname in feature_names:
        if fname in df.columns:
            df[fname] = df[fname].fillna(defaults.get(fname, 0.0))
    
    if verbose:
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total samples: {len(df)}")
        print(f"   Unique tickers: {df['Ticker'].nunique()}")
        print(f"   Positive labels: {df['Label'].sum()} ({df['Label'].mean():.1%})")
        print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df


def precision_at_k(y_true, y_pred_proba, k: int = 20) -> float:
    """Calculate precision for top K predictions."""
    if len(y_true) < k:
        k = len(y_true)
    if k == 0:
        return 0.0
    
    top_k_idx = np.argsort(y_pred_proba)[-k:]
    
    if hasattr(y_true, 'iloc'):
        precision = y_true.iloc[top_k_idx].mean()
    else:
        precision = y_true[top_k_idx].mean()
    
    return float(precision)


def walkforward_train_evaluate(
    df: pd.DataFrame,
    n_splits: int = 5,
    verbose: bool = True
) -> Tuple[EnsembleClassifierV4, Dict[str, Any]]:
    """
    Walk-forward training and evaluation.
    
    Splits data temporally, trains on past, tests on future.
    Returns the model trained on all data, plus per-fold metrics.
    """
    feature_names = get_feature_names_v4()
    
    # Ensure all features exist
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        print(f"   ⚠️ Missing features: {missing}")
        defaults = get_feature_defaults_v4()
        for f in missing:
            df[f] = defaults.get(f, 0.0)
    
    X = df[feature_names].values
    y = df["Label"].values
    dates = df["Date"].values
    
    # Sort by date for temporal split
    sort_idx = np.argsort(dates)
    X = X[sort_idx]
    y = y[sort_idx]
    dates = dates[sort_idx]
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_metrics = []
    
    if verbose:
        print(f"\n🔄 Walk-Forward Validation ({n_splits} folds)")
        print("=" * 60)
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        model = EnsembleClassifierV4(random_state=TRAIN_CONFIG["random_state"])
        model.fit(X_train, y_train, feature_names=feature_names)
        
        # Predict
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        p_at_20 = precision_at_k(y_test, y_pred_proba, k=20)
        p_at_50 = precision_at_k(y_test, y_pred_proba, k=50)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        fold_metrics.append({
            "fold": fold + 1,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "auc": auc,
            "precision_at_20": p_at_20,
            "precision_at_50": p_at_50,
            "avg_precision": avg_precision
        })
        
        if verbose:
            print(f"Fold {fold + 1}: AUC={auc:.3f}, P@20={p_at_20:.3f}, P@50={p_at_50:.3f}")
    
    # Calculate aggregate metrics
    avg_metrics = {
        "avg_auc": np.mean([m["auc"] for m in fold_metrics]),
        "avg_precision_at_20": np.mean([m["precision_at_20"] for m in fold_metrics]),
        "avg_precision_at_50": np.mean([m["precision_at_50"] for m in fold_metrics]),
        "std_auc": np.std([m["auc"] for m in fold_metrics]),
        "std_precision_at_20": np.std([m["precision_at_20"] for m in fold_metrics])
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"📊 Average Metrics:")
        print(f"   AUC: {avg_metrics['avg_auc']:.3f} ± {avg_metrics['std_auc']:.3f}")
        print(f"   P@20: {avg_metrics['avg_precision_at_20']:.3f} ± {avg_metrics['std_precision_at_20']:.3f}")
        print(f"   P@50: {avg_metrics['avg_precision_at_50']:.3f}")
    
    # Train final model on all data
    if verbose:
        print("\n🏋️ Training final model on all data...")
    
    final_model = EnsembleClassifierV4(random_state=TRAIN_CONFIG["random_state"])
    final_model.fit(X, y, feature_names=feature_names)
    
    return final_model, {
        "fold_metrics": fold_metrics,
        "aggregate_metrics": avg_metrics
    }


def analyze_feature_importance(
    model: EnsembleClassifierV4,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    verbose: bool = True
) -> pd.DataFrame:
    """Analyze and report feature importance."""
    if verbose:
        print("\n" + "=" * 60)
        print("📊 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
    
    # Built-in importance
    if model.feature_importances_ is not None:
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'builtin_importance': model.feature_importances_
        }).sort_values('builtin_importance', ascending=False)
        
        if verbose:
            print("\n🔹 Built-in Importance (Top 15):")
            for _, row in importance_df.head(15).iterrows():
                bar = "█" * int(row['builtin_importance'] * 50)
                print(f"   {row['feature']:30s} {row['builtin_importance']:.4f} {bar}")
    
    # Permutation importance
    if verbose:
        print("\n   Computing permutation importance...")
    
    perm_importance = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=TRAIN_CONFIG["random_state"],
        n_jobs=-1,
        scoring='roc_auc'
    )
    
    perm_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    if verbose:
        print("\n🔹 Permutation Importance (Top 15):")
        for _, row in perm_df.head(15).iterrows():
            bar = "█" * int(max(0, row['importance_mean']) * 200)
            print(f"   {row['feature']:30s} {row['importance_mean']:+.4f} ± {row['importance_std']:.4f} {bar}")
    
    # Flag harmful features
    harmful = perm_df[perm_df['importance_mean'] < 0]['feature'].tolist()
    if harmful and verbose:
        print(f"\n⚠️  Harmful features (negative importance): {harmful}")
    
    return perm_df


def save_model_bundle(
    model: EnsembleClassifierV4,
    feature_names: List[str],
    metrics: Dict[str, Any],
    output_path: Path,
    verbose: bool = True
) -> None:
    """Save model bundle with metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    bundle = {
        "model": model,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "version": "v4",
        "trained_at": datetime.now().isoformat(),
        "metrics": metrics,
        "config": TRAIN_CONFIG
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)
    
    if verbose:
        print(f"\n✅ Model saved to: {output_path}")
    
    # Also save metrics as JSON
    metrics_path = output_path.with_suffix(".json")
    with open(metrics_path, "w") as f:
        json.dump({
            "version": "v4",
            "trained_at": bundle["trained_at"],
            "feature_count": bundle["feature_count"],
            "metrics": metrics,
            "config": TRAIN_CONFIG
        }, f, indent=2, default=str)
    
    if verbose:
        print(f"   Metrics saved to: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Train V4 ML model for Stock Scout")
    parser.add_argument(
        "--output", "-o",
        default="models/ml_20d_v4.pkl",
        help="Output path for trained model"
    )
    parser.add_argument(
        "--skip-sentiment",
        action="store_true",
        help="Skip sentiment data fetching (faster but fewer features)"
    )
    parser.add_argument(
        "--tickers",
        type=int,
        default=500,
        help="Number of tickers to train on"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Years of historical data to use"
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    TRAIN_CONFIG["years_history"] = args.years
    TRAIN_CONFIG["n_splits"] = args.folds
    
    if verbose:
        print("🚀 Stock Scout V4 ML Training")
        print("=" * 60)
        print(f"   Output: {args.output}")
        print(f"   Tickers: {args.tickers}")
        print(f"   Years: {args.years}")
        print(f"   Folds: {args.folds}")
        print(f"   Include sentiment: {not args.skip_sentiment}")
        print(f"   V4 Feature count: {FEATURE_COUNT_V4}")
        print("=" * 60)
    
    # Get tickers
    tickers = get_universe_tickers(args.tickers)
    if verbose:
        print(f"\n📋 Universe: {len(tickers)} tickers")
    
    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * args.years)
    
    # Build training dataset
    df = build_training_dataset(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        include_sentiment=not args.skip_sentiment,
        verbose=verbose
    )
    
    # Train and evaluate
    model, metrics = walkforward_train_evaluate(
        df=df,
        n_splits=args.folds,
        verbose=verbose
    )
    
    # Feature importance analysis
    feature_names = get_feature_names_v4()
    X = df[feature_names].values
    y = df["Label"].values
    
    # Use last 20% for importance analysis
    split = int(len(X) * 0.8)
    X_test = X[split:]
    y_test = y[split:]
    
    importance_df = analyze_feature_importance(
        model, X_test, y_test, feature_names, verbose=verbose
    )
    
    # Save importance report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(
        REPORTS_DIR / "feature_importance_v4.csv",
        index=False
    )
    
    # Save model
    output_path = Path(args.output)
    save_model_bundle(model, feature_names, metrics, output_path, verbose=verbose)
    
    # Update monitoring baseline
    try:
        monitor = Monitor.get_instance()
        y_pred = model.predict_proba(X)[:, 1]
        monitor.update_baseline(y_pred)
    except Exception as e:
        if verbose:
            print(f"   ⚠️ Could not update monitoring baseline: {e}")
    
    if verbose:
        print("\n" + "=" * 60)
        print("✅ V4 MODEL TRAINING COMPLETE")
        print("=" * 60)
        print(f"   Final AUC: {metrics['aggregate_metrics']['avg_auc']:.3f}")
        print(f"   Final P@20: {metrics['aggregate_metrics']['avg_precision_at_20']:.3f}")
        print(f"   Model path: {output_path}")


if __name__ == "__main__":
    main()
