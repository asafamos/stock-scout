"""
Train a stronger 20d ML model (v2)
- Loads experiments/training_dataset_20d_v2.csv
- Hyperparameter tuning via GridSearchCV with GradientBoostingClassifier
- Ranking-oriented evaluation: deciles, top-k buckets, multi-threshold metrics
- Saves bundle at models/model_20d_v2.pkl with keys: {"model": estimator, "feature_names": [..]}
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FEATURES: List[str] = [
    "TechScore_20d",
    "RSI",
    "ATR_Pct",
    "RR",
    "MomCons",
    "VolSurge",
]

DATASET_DEFAULT = "experiments/training_dataset_20d_v2.csv"
MODEL_OUT = Path("models/model_20d_v2.pkl")


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Validate required columns
    required = set(FEATURES + ["Label_20d", "Forward_Return_20d"])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    # Drop rows with invalid labels
    df = df[df["Label_20d"].isin([0, 1])].copy()
    # Drop rows with missing features
    df = df.dropna(subset=FEATURES + ["Label_20d"]).copy()
    # Clip features similar to inference rules
    df["ATR_Pct"] = np.clip(df["ATR_Pct"], 0.0, 0.2)
    df["RR"] = np.clip(df["RR"], 0.0, 10.0)
    df["RSI"] = np.clip(df["RSI"], 5.0, 95.0)
    df["TechScore_20d"] = np.clip(df["TechScore_20d"], 0.0, 100.0)
    return df


def train_and_evaluate(df: pd.DataFrame):
    X = df[FEATURES].astype(float).values
    y = df["Label_20d"].astype(int).values
    ret = df["Forward_Return_20d"].astype(float).values

    # Stratified split to maintain class balance
    X_train, X_test, y_train, y_test, ret_train, ret_test = train_test_split(
        X, y, ret, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[TRAIN] Train/test split: {len(X_train)} / {len(X_test)}")
    print(f"[TRAIN] Class distribution train: {np.bincount(y_train)}, test: {np.bincount(y_test)}")

    # GridSearchCV for hyperparameter tuning
    param_grid = {
        "gb__n_estimators": [100, 200],
        "gb__learning_rate": [0.05, 0.1],
        "gb__max_depth": [2, 3],
        "gb__subsample": [0.7, 1.0],
    }

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingClassifier(random_state=42, validation_fraction=0.1, n_iter_no_change=10))
    ])

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=2, verbose=1)
    grid.fit(X_train, y_train)

    print(f"[TRAIN] Best params: {grid.best_params_}")
    print(f"[TRAIN] Best CV ROC-AUC: {grid.best_score_:.3f}")

    # Best model: refit on full training set
    best_model = grid.best_estimator_
    best_model.fit(X_train, y_train)

    # Predict on test set
    proba = best_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    print(f"\n[TEST] ROC-AUC={roc_auc:.3f} | Average Precision={ap:.3f}")

    # Multi-threshold metrics
    print("\n[TEST] Threshold sweep (Precision/Recall/F1):")
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        y_pred = (proba >= thresh).astype(int)
        if y_pred.sum() == 0:
            print(f"  threshold={thresh:.1f}: No positive predictions")
            continue
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
        print(f"  threshold={thresh:.1f}: P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")

    # Ranking metrics: deciles and top-k buckets
    test_df = pd.DataFrame({
        "proba": proba,
        "ret": ret_test,
    })
    test_df["label"] = (test_df["ret"] >= 0.15).astype(int)

    # Baseline
    baseline_ret = test_df["ret"].mean()
    baseline_hit = test_df["label"].mean()
    print(f"\n[BASELINE] Avg 20d return={baseline_ret:.4f}, Hit rate (≥15%)={baseline_hit:.3f}")

    # Sort by proba descending
    test_df = test_df.sort_values(by=["proba"], ascending=False).reset_index(drop=True)
    n = len(test_df)

    # Deciles
    print("\n[DECILES] (sorted by probability, descending):")
    for decile in range(10):
        start_idx = int(decile * n / 10)
        end_idx = int((decile + 1) * n / 10)
        sub = test_df.iloc[start_idx:end_idx]
        avg_ret = sub["ret"].mean()
        hit_rate = sub["label"].mean()
        print(f"  Decile {decile}: n={len(sub)}, avg_ret={avg_ret:.4f}, hit_rate={hit_rate:.3f}")

    # Top-k buckets
    print("\n[TOP-K BUCKETS]:")
    for pct in [0.05, 0.10, 0.20]:
        k = int(pct * n)
        sub = test_df.head(k)
        avg_ret = sub["ret"].mean()
        hit_rate = sub["label"].mean()
        print(f"  Top {pct*100:.0f}% (n={k}): avg_ret={avg_ret:.4f}, hit_rate={hit_rate:.3f}")

    return best_model, roc_auc, ap, baseline_ret, baseline_hit, test_df


def save_bundle(model, feature_names: List[str]):
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"model": model, "feature_names": feature_names}
    joblib.dump(bundle, MODEL_OUT)
    print(f"[MODEL v2] Saved to {MODEL_OUT}")


def main():
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else DATASET_DEFAULT
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    df = load_dataset(dataset_path)
    print(f"[DATASET v2] Loaded {len(df)} rows from {dataset_path} | Unique tickers: {df['Ticker'].nunique()}")
    model, roc_auc, ap, baseline_ret, baseline_hit, test_df = train_and_evaluate(df)
    save_bundle(model, FEATURES)
    print("\n[SUMMARY]")
    print(f"  ROC-AUC: {roc_auc:.3f}")
    print(f"  Avg Precision: {ap:.3f}")
    print(f"  Baseline avg 20d return: {baseline_ret:.4f}")
    print(f"  Baseline hit rate (≥15%): {baseline_hit:.3f}")



if __name__ == "__main__":
    main()
