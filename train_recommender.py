"""Train a probabilistic recommender from historical signals.

Workflow:
 1. Load backtest signals CSV (produced by `backtest_recommendations.py`).
 2. Build features from the signal row (RSI, ATR_Pct, Overext, RR, MomCons, VolSurge).
 3. Label using forward return horizon (default 20d) -> positive if forward return > 0
    or > benchmark (SPY) optionally.
 4. Train a logistic regression and evaluate ROC/AUC, calibration, reliability.
 5. Save model and produce simple diagnostics CSV.

Usage:
    python train_recommender.py --signals backtest_signals_YYYYMMDD_*.csv --horizon 20

Notes:
    - Uses scikit-learn (LogisticRegression). If unavailable, installs are required.
    - This is intentionally simple; can be swapped with XGBoost or LightGBM later.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--signals', type=str, required=True, help='CSV file path with backtest signals')
    p.add_argument('--horizon', type=int, default=20, help='Forward horizon to label (days)')
    p.add_argument('--target', type=str, default='pos', choices=['pos', 'beat_bench'], help='Label target')
    p.add_argument('--out', type=str, default='recommender_model.pkl', help='Output model file')
    return p.parse_args()


def load_signals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df


def build_dataset(df: pd.DataFrame, horizon: int, target: str = 'pos') -> pd.DataFrame:
    # Expect columns like 'R_{h}d' and 'Excess_{h}d'
    rcol = f'R_{horizon}d'
    ecol = f'Excess_{horizon}d'
    if rcol not in df.columns:
        raise SystemExit(f"Signals missing column {rcol}")

    feature_cols = [c for c in ['RSI', 'ATR_Pct', 'Overext', 'RR', 'MomCons', 'VolSurge'] if c in df.columns]
    X = df[feature_cols].copy()
    # Fill NaNs conservatively
    X = X.fillna(X.median())

    if target == 'pos':
        y = pd.to_numeric(df[rcol], errors='coerce') > 0
    else:
        y = pd.to_numeric(df[ecol], errors='coerce') > 0

    # drop rows without label
    mask = y.notna()
    X = X.loc[mask].astype(float)
    y = y.loc[mask].astype(int)
    out = X.copy()
    out['y'] = y.values
    out['Ticker'] = df.loc[mask, 'Ticker'].values
    out['Date'] = df.loc[mask, 'Date'].values
    return out


def train_and_eval(df: pd.DataFrame, out_path: str):
    y = df['y'].values
    X = df.drop(columns=['y', 'Ticker', 'Date'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    auc = roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else float('nan')
    brier = brier_score_loss(y_test, prob)

    print("=== Model Evaluation ===")
    print(f"AUC: {auc:.4f}")
    print(f"Brier score: {brier:.4f}")
    print(classification_report(y_test, preds))

    # calibration curve
    prob_true, prob_pred = calibration_curve(y_test, prob, n_bins=10)
    calib = pd.DataFrame({'prob_pred': prob_pred, 'prob_true': prob_true})
    calib.to_csv('calibration_curve.csv', index=False)

    # save model
    with open(out_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {out_path}")


def main():
    args = parse_args()
    df = load_signals(args.signals)
    ds = build_dataset(df, args.horizon, args.target)
    print(f"Dataset rows: {len(ds)}; positives: {ds['y'].sum()} / {len(ds)}")
    train_and_eval(ds, args.out)


if __name__ == '__main__':
    main()
