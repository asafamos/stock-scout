#!/usr/bin/env python3
"""
Train ML model from existing CSV data.
No API keys needed - uses pre-downloaded training data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from core.feature_registry import get_feature_names, get_feature_defaults

# Paths
MODELS_DIR = Path("models")
DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")

# Feature list from registry (34 features)
FEATURE_NAMES_V3 = get_feature_names("v3")


def train_from_csv():
    """Train model from existing CSV data."""
    print("🚀 Training from Existing CSV Data")
    print("=" * 50)

    # Find latest training CSV
    csv_candidates = [
        DATA_DIR / "training_dataset_20d_v3_with_overlay_v2_with_meta.csv",
        DATA_DIR / "training_dataset_20d_v3_with_overlay_v2.csv",
        DATA_DIR / "training_dataset_20d_v3_with_scores.csv",
        DATA_DIR / "training_dataset_20d_v3.csv",
    ]

    csv_path = None
    for candidate in csv_candidates:
        if candidate.exists():
            csv_path = candidate
            break

    if csv_path is None:
        print("❌ No training CSV found!")
        return

    print(f"📂 Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Rows: {len(df)}")

    # Check for required columns
    print(f"\n📋 Available columns: {len(df.columns)}")

    # Determine label column
    label_col = None
    for candidate in ['Label', 'label', 'Forward_Return_20d_Label', 'Winner']:
        if candidate in df.columns:
            label_col = candidate
            break

    # If no label, create from Forward_Return_20d
    if label_col is None and 'Forward_Return_20d' in df.columns:
        print("   Creating labels from Forward_Return_20d (top 20% = winners)")
        threshold = df['Forward_Return_20d'].quantile(0.80)
        df['Label'] = (df['Forward_Return_20d'] >= threshold).astype(int)
        label_col = 'Label'
        print(f"   Winner threshold: {threshold*100:.1f}% return")

    if label_col is None:
        print("❌ No label column found and can't create one!")
        return

    print(f"   Label column: {label_col}")

    # Map existing columns to expected feature names
    column_mapping = {
        'Dist_52w_High': 'Dist_From_52w_High',
        'MomCons': 'Momentum_Consistency',
        'VolSurge': 'Volume_Surge',
        'RS_SPY_20d': 'RS_vs_SPY_20d',
        'Volume_Surge_Ratio': 'Volume_Surge',
    }

    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]

    # Check which features we have
    available_features = [f for f in FEATURE_NAMES_V3 if f in df.columns]
    missing_features = [f for f in FEATURE_NAMES_V3 if f not in df.columns]

    print(f"\n📊 Features available: {len(available_features)}/{len(FEATURE_NAMES_V3)}")
    if missing_features:
        print(f"   Missing: {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}")

    # Fill missing features with defaults
    defaults = get_feature_defaults("v3")
    for feat in missing_features:
        df[feat] = defaults.get(feat, 0.0)

    # Prepare X and y
    X = df[FEATURE_NAMES_V3].copy()
    y = df[label_col].copy()

    # Handle NaN/inf
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        X[col] = X[col].fillna(defaults.get(col, 0.0))

    print(f"\n🎯 Label distribution:")
    print(f"   Winners (1): {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"   Losers (0): {len(y) - y.sum()} ({(1-y.mean())*100:.1f}%)")

    # Time-series cross-validation
    print("\n📊 Running Time-Series Cross-Validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    aucs = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if y_train.sum() < 10 or y_val.sum() < 5:
            continue

        model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            class_weight='balanced',
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        aucs.append(auc)
        print(f"   Fold {fold}: AUC = {auc:.4f}")

    mean_auc = np.mean(aucs)
    print(f"\n✅ Mean CV AUC: {mean_auc:.4f}")

    # Train final model on all data
    print("\n🔧 Training final model on all data...")
    final_model = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        class_weight='balanced',
    )
    final_model.fit(X, y)

    # Save model bundle
    MODELS_DIR.mkdir(exist_ok=True)

    bundle = {
        "model": final_model,
        "feature_names": FEATURE_NAMES_V3,
    }

    bundle_path = MODELS_DIR / "model_20d_v3.pkl"
    joblib.dump(bundle, bundle_path)
    print(f"💾 Saved model bundle: {bundle_path}")

    # Save metadata
    import sklearn
    metadata = {
        "sklearn_version": sklearn.__version__,
        "feature_list": FEATURE_NAMES_V3,
        "training_timestamp_utc": datetime.utcnow().isoformat(),
        "label_spec": {
            "horizon_days": 20,
            "threshold_return": 0.2,
            "label_name": "Forward_Return_20d>threshold",
        },
        "model_type": "HistGradientBoostingClassifier",
        "model_name": "ml_20d_v3_hgb",
        "cv_auc": float(mean_auc),
        "training_samples": len(X),
        "source_csv": str(csv_path),
        "note": "Trained from existing CSV data with 34-feature registry",
    }

    meta_path = MODELS_DIR / "model_20d_v3.metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Saved metadata: {meta_path}")

    # Feature importance
    print("\n📊 Top 10 Feature Importances:")
    try:
        from sklearn.inspection import permutation_importance
        perm_imp = permutation_importance(final_model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
        imp_df = pd.DataFrame({
            'feature': FEATURE_NAMES_V3,
            'importance': perm_imp.importances_mean,
        }).sort_values('importance', ascending=False)
        for _, row in imp_df.head(10).iterrows():
            print(f"   {row['feature']:30s} {row['importance']:.4f}")
    except Exception as e:
        print(f"   (permutation importance failed: {e})")

    print("\n✅ Training complete!")
    print(f"   Model: {bundle_path}")
    print(f"   Metadata: {meta_path}")
    print(f"   CV AUC: {mean_auc:.4f}")


if __name__ == "__main__":
    train_from_csv()
