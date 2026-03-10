#!/usr/bin/env python3
"""
AUC A/B Experiment — test multiple configs with ONE data download.

Downloads yfinance data once, then runs walk-forward CV for each config:
  A: v3.6 features (23) + rank labeling  (local baseline)
  B: v3.6 features (23) + fixed 5% threshold labeling
  C: v3.6 features (23) + rank labeling + tuned hyperparams
  D: v3.7 features (16) + rank labeling  (pruned set)

Run: python3 scripts/experiment_auc.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import gc
from datetime import datetime, timedelta
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from core.feature_registry import get_feature_names

# Reuse data loading from main training script
from scripts.train_ml_20d_v3_local import (
    get_sp500_tickers,
    fetch_yf_history,
    calculate_all_features,
    calculate_market_regime,
    fetch_sector_etf_data,
    precision_at_k,
    PurgedWalkForwardCV,
)


def run_experiment(X, y, dates, config_name, hgb_params=None):
    """Run walk-forward CV and return mean AUC."""
    if hgb_params is None:
        hgb_params = dict(
            max_iter=500, learning_rate=0.05, max_depth=8,
            min_samples_leaf=15, l2_regularization=0.1,
            class_weight='balanced', early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=10,
            random_state=42,
        )

    print(f"\n{'='*60}")
    print(f"  Config {config_name}")
    print(f"  Features: {X.shape[1]}, Samples: {len(X)}")
    print(f"  Class balance: {y.mean():.1%} positive")
    print(f"{'='*60}")

    pwf_cv = PurgedWalkForwardCV(n_splits=5, embargo_td=20, min_train_pct=0.3)
    cv_aucs = []
    cv_p20 = []

    for fold, (train_idx, val_idx) in enumerate(pwf_cv.split(X, dates=dates), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if y_train.sum() < 10 or y_val.sum() < 5:
            print(f"   Fold {fold}: SKIPPED")
            continue

        model = HistGradientBoostingClassifier(**hgb_params)
        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        p20 = precision_at_k(y_val, y_pred, k=20)
        cv_aucs.append(auc)
        cv_p20.append(p20)

        baseline = y_val.mean()
        lift = p20 / baseline if baseline > 0 else 0
        print(f"   Fold {fold}: AUC={auc:.4f}, P@20={p20:.1%} ({lift:.1f}x lift)")

    mean_auc = np.mean(cv_aucs) if cv_aucs else 0
    std_auc = np.std(cv_aucs) if cv_aucs else 0
    mean_p20 = np.mean(cv_p20) if cv_p20 else 0
    print(f"   ➜ AUC={mean_auc:.4f} ± {std_auc:.4f}, P@20={mean_p20:.1%}")
    gc.collect()
    return mean_auc, std_auc, mean_p20


def main():
    print("🔬 AUC A/B Experiment")
    print("=" * 60)

    # ── 1. Download data (ONE TIME) ──
    tickers = get_sp500_tickers(limit=500)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=750)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    print("📥 Fetching SPY benchmark...")
    spy_df = fetch_yf_history("SPY", start_str, end_str)
    spy_returns = spy_df['Close'].pct_change(20) if spy_df is not None else None
    spy_returns_60d = spy_df['Close'].pct_change(60) if spy_df is not None else None
    market_regime_df = calculate_market_regime(spy_df) if spy_df is not None else None
    print(f"   ✅ SPY: {len(spy_df)} days")

    sector_etf_returns = fetch_sector_etf_data(start_str, end_str)

    all_data = []
    print(f"📥 Downloading {len(tickers)} tickers...")
    for i, ticker in enumerate(tickers, 1):
        if i % 25 == 0:
            print(f"   ... {i}/{len(tickers)}")
        df = fetch_yf_history(ticker, start_str, end_str)
        if df is not None and len(df) > 250:
            df = calculate_all_features(df, spy_returns, spy_returns_60d,
                                        market_regime_df, sector_etf_returns, ticker)
            if len(df) > 50:
                df['Ticker'] = ticker
                all_data.append(df)

    full_df = pd.concat(all_data)
    print(f"📊 Total rows: {len(full_df)}, Tickers: {full_df['Ticker'].nunique()}")

    # ── 2. Prepare labels ──
    # Rank-based labels
    from core.ml_targets import make_label_20d_ranked
    full_df['Date'] = full_df.index
    full_df['Label_Rank'] = make_label_20d_ranked(
        full_df['Forward_Return_20d'],
        dates=full_df['Date'],
        top_pct=0.20,
        bottom_pct=0.40,
    )

    # Fixed threshold labels (original v3.6 approach)
    full_df['Label_Fixed'] = (full_df['Forward_Return_20d'] >= 0.05).astype(float)

    full_df = full_df.sort_index()

    # ── 3. Feature sets ──
    v31_features = get_feature_names("v3.1")  # 39 features (full OHLCV-derived)
    v36_features = get_feature_names("v3.6")  # 23 features
    v37_features = get_feature_names("v3.7")  # 16 features

    results = {}

    # ── Config A: v3.6 features + rank labeling ──
    df_a = full_df.dropna(subset=['Label_Rank']).copy()
    avail_a = [f for f in v36_features if f in df_a.columns]
    X_a = df_a[avail_a].fillna(0).astype(np.float32)
    y_a = df_a['Label_Rank'].astype(int)
    dates_a = df_a.index
    auc_a, std_a, p20_a = run_experiment(
        X_a, y_a, dates_a,
        f"A: v3.6 ({len(avail_a)}f) + rank labels",
    )
    results['A'] = (auc_a, std_a, p20_a)

    # ── Config B: v3.6 features + fixed 5% threshold ──
    df_b = full_df.copy()
    avail_b = [f for f in v36_features if f in df_b.columns]
    X_b = df_b[avail_b].fillna(0).astype(np.float32)
    y_b = df_b['Label_Fixed'].astype(int)
    dates_b = df_b.index
    auc_b, std_b, p20_b = run_experiment(
        X_b, y_b, dates_b,
        f"B: v3.6 ({len(avail_b)}f) + fixed 5% threshold",
    )
    results['B'] = (auc_b, std_b, p20_b)

    # ── Config C: v3.6 features + rank labeling + tuned hyperparams ──
    # Try: lower depth (6), more regularization, slower LR
    tuned_params = dict(
        max_iter=800, learning_rate=0.03, max_depth=6,
        min_samples_leaf=25, l2_regularization=0.5,
        class_weight='balanced', early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=15,
        random_state=42,
    )
    auc_c, std_c, p20_c = run_experiment(
        X_a, y_a, dates_a,
        f"C: v3.6 ({len(avail_a)}f) + rank + tuned HP (lr=0.03, d=6, l2=0.5)",
        hgb_params=tuned_params,
    )
    results['C'] = (auc_c, std_c, p20_c)

    # ── Config D: v3.7 features (16) + rank labeling ──
    df_d = full_df.dropna(subset=['Label_Rank']).copy()
    avail_d = [f for f in v37_features if f in df_d.columns]
    X_d = df_d[avail_d].fillna(0).astype(np.float32)
    y_d = df_d['Label_Rank'].astype(int)
    dates_d = df_d.index
    auc_d, std_d, p20_d = run_experiment(
        X_d, y_d, dates_d,
        f"D: v3.7 ({len(avail_d)}f) + rank labels",
    )
    results['D'] = (auc_d, std_d, p20_d)

    # ── Config E: v3.1 features (39, full OHLCV) + rank labeling ──
    df_e = full_df.dropna(subset=['Label_Rank']).copy()
    avail_e = [f for f in v31_features if f in df_e.columns]
    X_e = df_e[avail_e].fillna(0).astype(np.float32)
    y_e = df_e['Label_Rank'].astype(int)
    dates_e = df_e.index
    auc_e, std_e, p20_e = run_experiment(
        X_e, y_e, dates_e,
        f"E: v3.1 ({len(avail_e)}f, full OHLCV) + rank labels",
    )
    results['E'] = (auc_e, std_e, p20_e)

    # ── Config F: v3.1 features (39) + fixed 5% threshold ──
    avail_f = [f for f in v31_features if f in full_df.columns]
    X_f = full_df[avail_f].fillna(0).astype(np.float32)
    y_f = full_df['Label_Fixed'].astype(int)
    dates_f = full_df.index
    auc_f, std_f, p20_f = run_experiment(
        X_f, y_f, dates_f,
        f"F: v3.1 ({len(avail_f)}f, full OHLCV) + fixed 5% threshold",
    )
    results['F'] = (auc_f, std_f, p20_f)

    # ── Config G: v3.6 features + rank + strong regularization ──
    more_reg = dict(
        max_iter=1000, learning_rate=0.02, max_depth=5,
        min_samples_leaf=30, l2_regularization=1.0,
        class_weight='balanced', early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=20,
        random_state=42,
    )
    auc_g, std_g, p20_g = run_experiment(
        X_a, y_a, dates_a,
        f"G: v3.6 ({len(avail_a)}f) + rank + strong reg (lr=0.02, d=5, l2=1.0)",
        hgb_params=more_reg,
    )
    results['G'] = (auc_g, std_g, p20_g)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Production v3.6 baseline (Polygon): AUC=0.6144")
    print(f"  v4.0 with fundamentals:             AUC=0.5395")
    print("-" * 60)
    for key in sorted(results.keys()):
        auc, std, p20 = results[key]
        print(f"  Config {key}: AUC={auc:.4f} ± {std:.4f}, P@20={p20:.1%}")
    print("=" * 60)

    best = max(results.items(), key=lambda x: x[1][0])
    print(f"\n🏆 Best: Config {best[0]} with AUC={best[1][0]:.4f}")


if __name__ == "__main__":
    main()
