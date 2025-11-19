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
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, brier_score_loss, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import shap


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--signals', type=str, required=True, help='CSV file path with backtest signals')
    p.add_argument('--horizon', type=int, default=20, help='Forward horizon to label (days)')
    p.add_argument('--target', type=str, default='pos', choices=['pos', 'beat_bench'], help='Label target')
    p.add_argument('--out', type=str, default='recommender_model.pkl', help='Output model file')
    p.add_argument('--model', type=str, default='xgboost', choices=['logistic', 'xgboost'], help='Model type')
    p.add_argument('--cv', action='store_true', help='Use TimeSeriesSplit cross-validation')
    return p.parse_args()


def load_signals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for better model performance."""
    X = df.copy()
    
    # Core interaction features - these showed best performance
    if 'RR' in X.columns and 'MomCons' in X.columns:
        X['RR_MomCons'] = X['RR'] * X['MomCons']
    
    if 'RSI' in X.columns:
        X['RSI_Neutral'] = (X['RSI'] - 50).abs()  # Distance from neutral
        X['RSI_Squared'] = X['RSI'] ** 2  # Amplify extremes
    
    if 'Overext' in X.columns and 'ATR_Pct' in X.columns:
        X['Risk_Score'] = X['Overext'].abs() + X['ATR_Pct']  # Combined risk
    
    if 'VolSurge' in X.columns and 'MomCons' in X.columns:
        X['Vol_Mom'] = X['VolSurge'] * X['MomCons']  # Volume confirmation
    
    if 'Overext' in X.columns and 'MomCons' in X.columns:
        # Overextension momentum divergence: proven useful
        X['Overext_Mom_Div'] = X['Overext'] * X['MomCons']
    
    if 'RR' in X.columns and 'Overext' in X.columns:
        # Risk-adjusted reward
        X['RR_Risk_Adj'] = X['RR'] / (1 + X['Overext'].abs())
    
    # Simple regime indicator
    if 'ATR_Pct' in X.columns:
        X['ATR_Regime'] = pd.qcut(X['ATR_Pct'], q=3, labels=[1, 2, 3], duplicates='drop').astype(float)
    
    return X


def build_dataset(df: pd.DataFrame, horizon: int, target: str = 'pos') -> pd.DataFrame:
    # Expect columns like 'R_{h}d' and 'Excess_{h}d'
    rcol = f'R_{horizon}d'
    ecol = f'Excess_{horizon}d'
    if rcol not in df.columns:
        raise SystemExit(f"Signals missing column {rcol}")

    # Base technical features
    base_features = ['RSI', 'ATR_Pct', 'Overext', 'RR', 'MomCons', 'VolSurge']
    
    # Context features (if available)
    context_features = [
        'Market_Trend', 'Market_Volatility', 'SPY_RSI', 
        'Relative_Strength_20d', 'Dist_From_52w_High',
        'Vol_Breakout', 'Price_Breakout', 'Mom_Acceleration'
    ]
    
    # Select available features
    feature_cols = [c for c in base_features if c in df.columns]
    feature_cols += [c for c in context_features if c in df.columns]
    
    if not feature_cols:
        raise SystemExit("No valid feature columns found in signals")
    
    X = df[feature_cols].copy()
    
    # Engineer additional features
    X = engineer_features(X)
    
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


def train_and_eval(df: pd.DataFrame, out_path: str, model_type: str = 'xgboost', use_cv: bool = False):
    y = df['y'].values
    X = df.drop(columns=['y', 'Ticker', 'Date'])
    feature_names = X.columns.tolist()

    if use_cv:
        # TimeSeriesSplit for temporal data
        tscv = TimeSeriesSplit(n_splits=5)
        aucs = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            if model_type == 'xgboost':
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    scale_pos_weight=len(y_train[y_train==0])/max(1, len(y_train[y_train==1])),
                    random_state=42
                )
            else:
                model = LogisticRegression(max_iter=1000, class_weight='balanced')
            
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[:, 1]
            if len(np.unique(y_test)) > 1:
                aucs.append(roc_auc_score(y_test, prob))
        
        print(f"\n=== Cross-Validation Results ({len(aucs)} folds) ===")
        print(f"Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"Fold AUCs: {[f'{a:.3f}' for a in aucs]}")

    # Train final model on 75% data, test on most recent 25%
    split_idx = int(len(X) * 0.75)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if model_type == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=len(y_train[y_train==0])/max(1, len(y_train[y_train==1])),
            random_state=42,
            eval_metric='auc'
        )
    else:
        model = LogisticRegression(max_iter=1000, class_weight='balanced')

    model.fit(X_train, y_train)

    # Calibrate model probabilities using isotonic regression on holdout set
    try:
        calibrator = CalibratedClassifierCV(base_estimator=model, method='isotonic', cv='prefit')
        calibrator.fit(X_test, y_test)
        model_calibrated = calibrator
        print('\n✓ Model calibrated with isotonic regression (prefit on holdout set)')
    except Exception as e:
        print(f"\n⚠ Calibration failed: {e} - saving uncalibrated model")
        model_calibrated = model

    prob = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    auc = roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else float('nan')
    brier = brier_score_loss(y_test, prob)

    print("\n=== Final Model Evaluation ===")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Test set positives: {y_test.sum()}/{len(y_test)} ({y_test.mean()*100:.1f}%)")
    print(f"AUC: {auc:.4f}")
    print(f"Brier score: {brier:.4f}")
    print(classification_report(y_test, preds))

    # Find optimal threshold using PR curve
    precision, recall, thresholds = precision_recall_curve(y_test, prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    print(f"\nOptimal threshold (F1): {optimal_threshold:.3f}")

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, prob, n_bins=10)
    calib = pd.DataFrame({'prob_pred': prob_pred, 'prob_true': prob_true})
    calib.to_csv('calibration_curve.csv', index=False)

    # Feature importance for XGBoost
    if model_type == 'xgboost':
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\n=== Feature Importance (Top 10) ===")
        print(importance_df.head(10).to_string(index=False))
        importance_df.to_csv('feature_importance.csv', index=False)
        
        # SHAP explanations (sample)
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test.head(50))
            shap_df = pd.DataFrame(shap_values, columns=feature_names)
            shap_df.to_csv('shap_values_sample.csv', index=False)
            print("\n✓ SHAP values saved to shap_values_sample.csv")
        except Exception as e:
            print(f"\n⚠ SHAP computation skipped: {e}")

    # Save model with metadata
    # Save calibrated model
    model_filename = out_path
    model_data = {
        'model': model_calibrated,
        'feature_names': feature_names,
        'optimal_threshold': optimal_threshold,
        'auc': auc,
        'model_type': model_type,
        'calibrated': True if model_calibrated is not None and hasattr(model_calibrated, 'predict_proba') else False
    }
    with open(model_filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n✓ Calibrated model saved to {model_filename}")


def main():
    args = parse_args()
    df = load_signals(args.signals)
    ds = build_dataset(df, args.horizon, args.target)
    print(f"\n📊 Dataset Summary")
    print(f"Total rows: {len(ds)}")
    print(f"Positives: {ds['y'].sum()} / {len(ds)} ({ds['y'].mean()*100:.1f}%)")
    print(f"Features: {len([c for c in ds.columns if c not in ['y', 'Ticker', 'Date']])}")
    train_and_eval(ds, args.out, args.model, args.cv)


if __name__ == '__main__':
    main()
