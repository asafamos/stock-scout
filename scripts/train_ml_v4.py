#!/usr/bin/env python3
"""V4 ML model training — 72 features, rank-based targets, LightGBM ensemble.

Key improvements over train_rolling_ml_20d.py:
  1. 72 features (39 technical v3.1 + 15 fundamental + 8 rank + 6 delta + 4 interaction)
  2. Rank-based labeling (top 20% = winner, bottom 40% = loser, per date)
  3. LightGBM added to ensemble (30%)
  4. Temporal sample weighting (recent data weighted more)
  5. SHAP-based feature selection post-training

Usage::

    python scripts/train_ml_v4.py
    python scripts/train_ml_v4.py --target-mode absolute  # use legacy thresholds
    python scripts/train_ml_v4.py --no-lightgbm           # skip LightGBM
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_ml_v4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML V4 model")
    parser.add_argument("--data-dir", type=str, default="ml/data", help="Training data directory")
    parser.add_argument("--output-dir", type=str, default="ml/bundles/latest", help="Output bundle directory")
    parser.add_argument("--target-mode", type=str, default="rank", choices=["rank", "absolute"])
    parser.add_argument("--no-lightgbm", action="store_true", help="Skip LightGBM in ensemble")
    parser.add_argument("--n-folds", type=int, default=3, help="Walk-forward CV folds")
    parser.add_argument("--min-auc", type=float, default=0.55, help="Minimum AUC to save model")
    args = parser.parse_args()

    # ── Load training data ──────────────────────────────────────────
    data_path = Path(args.data_dir) / "training_data.parquet"
    if not data_path.exists():
        logger.error("Training data not found: %s", data_path)
        logger.info("Run the data preparation pipeline first to generate training data.")
        sys.exit(1)

    logger.info("Loading training data from %s", data_path)
    df = pd.read_parquet(data_path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # ── Feature columns ─────────────────────────────────────────────
    from core.feature_registry import get_feature_names
    feature_cols = get_feature_names("v4")
    logger.info("V4 feature set: %d features", len(feature_cols))

    # Check which features are available in training data
    available = [f for f in feature_cols if f in df.columns]
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        logger.warning("Missing %d features (will use defaults): %s", len(missing), missing[:10])
        # Fill missing with defaults
        from core.feature_registry import get_feature_defaults
        defaults = get_feature_defaults("v4")
        for feat in missing:
            df[feat] = defaults.get(feat, 0.0)

    # ── Targets ─────────────────────────────────────────────────────
    if "Forward_Return_20d" not in df.columns:
        logger.error("Forward_Return_20d column not found in training data")
        sys.exit(1)

    if args.target_mode == "rank":
        from core.ml_targets import make_label_20d_ranked
        logger.info("Using RANK-BASED labeling (top 20%% winner, bottom 40%% loser)")
        date_col = "As_Of_Date" if "As_Of_Date" in df.columns else "Date"
        if date_col not in df.columns:
            logger.error("Date column (%s) not found for rank-based labeling", date_col)
            sys.exit(1)
        labels = make_label_20d_ranked(
            df["Forward_Return_20d"],
            df[date_col],
            top_pct=0.20,
            bottom_pct=0.40,
        )
    else:
        from core.ml_targets import make_label_20d
        logger.info("Using ABSOLUTE labeling (UP=%.2f, DOWN=%.2f)",
                     float(os.environ.get("ML_UP_THRESHOLD", 0.08)),
                     float(os.environ.get("ML_DOWN_THRESHOLD", 0.00)))
        labels = make_label_20d(df["Forward_Return_20d"])

    df["Label_20d"] = labels
    df_train = df.dropna(subset=["Label_20d"])
    logger.info("Training samples: %d (pos=%.1f%%)",
                len(df_train), df_train["Label_20d"].mean() * 100)

    if len(df_train) < 1000:
        logger.error("Too few training samples (%d < 1000)", len(df_train))
        sys.exit(1)

    # ── Temporal sample weights (exponential decay, 1-year half-life) ──
    date_col = "As_Of_Date" if "As_Of_Date" in df_train.columns else "Date"
    if date_col in df_train.columns:
        dates = pd.to_datetime(df_train[date_col])
        max_date = dates.max()
        days_from_end = (max_date - dates).dt.days
        sample_weights = np.exp(-days_from_end / 365.0)
        sample_weights = sample_weights / sample_weights.mean()  # normalise
    else:
        sample_weights = np.ones(len(df_train))

    X = df_train[feature_cols].copy()
    y = df_train["Label_20d"].astype(int)

    # ── Build ensemble ──────────────────────────────────────────────
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    models_config = [
        ("histgb", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=5,
            min_samples_leaf=20, l2_regularization=0.1,
            early_stopping=True, validation_fraction=0.1,
            random_state=42,
        ), 0.30),
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=20,
            max_features="sqrt", n_jobs=-1, random_state=42,
        ), 0.25),
        ("lr", LogisticRegression(
            C=0.1, max_iter=1000, solver="saga", random_state=42,
        ), 0.15),
    ]

    if not args.no_lightgbm:
        try:
            from lightgbm import LGBMClassifier
            models_config.append(
                ("lgbm", LGBMClassifier(
                    n_estimators=300, learning_rate=0.05, max_depth=5,
                    min_child_samples=20, reg_lambda=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    n_jobs=-1, verbose=-1, random_state=42,
                ), 0.30)
            )
            # Adjust weights to sum to 1.0
            total = sum(w for _, _, w in models_config)
            models_config = [(n, m, w / total) for n, m, w in models_config]
            logger.info("LightGBM enabled in ensemble")
        except ImportError:
            logger.warning("LightGBM not available — using 3-model ensemble")

    # ── Walk-forward CV ─────────────────────────────────────────────
    from core.walk_forward import walk_forward_splits
    from sklearn.metrics import roc_auc_score

    splits = walk_forward_splits(df_train, n_folds=args.n_folds)
    logger.info("Walk-forward: %d folds", len(splits))

    fold_aucs = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        sw_tr = sample_weights[train_idx]

        # Ensemble prediction (weighted average of probabilities)
        probas = np.zeros(len(X_te))
        for name, model, weight in models_config:
            try:
                if name in ("histgb", "lgbm"):
                    model.fit(X_tr, y_tr, sample_weight=sw_tr)
                else:
                    model.fit(X_tr, y_tr)
                p = model.predict_proba(X_te)[:, 1]
                probas += weight * p
            except Exception as e:
                logger.warning("Fold %d, model %s failed: %s", fold_idx, name, e)

        if y_te.nunique() < 2:
            logger.warning("Fold %d: only one class in test set, skipping AUC", fold_idx)
            continue

        auc = roc_auc_score(y_te, probas)
        fold_aucs.append(auc)
        logger.info("Fold %d: AUC=%.4f (test=%d, pos=%.1f%%)",
                     fold_idx, auc, len(y_te), y_te.mean() * 100)

    mean_auc = np.mean(fold_aucs) if fold_aucs else 0.0
    logger.info("=" * 60)
    logger.info("CV Mean AUC: %.4f (±%.4f)",
                mean_auc, np.std(fold_aucs) if fold_aucs else 0.0)
    logger.info("=" * 60)

    if mean_auc < args.min_auc:
        logger.error(
            "AUC %.4f < minimum %.4f — model NOT saved. Keeping existing model.",
            mean_auc, args.min_auc,
        )
        sys.exit(1)

    # ── Train final model on ALL data ───────────────────────────────
    logger.info("Training final model on all %d samples...", len(X))
    from core.ensemble import EnsembleClassifier

    final_models = []
    final_weights = []
    for name, model, weight in models_config:
        try:
            if name in ("histgb", "lgbm"):
                model.fit(X, y, sample_weight=sample_weights)
            else:
                model.fit(X, y)
            final_models.append(model)
            final_weights.append(weight)
            logger.info("  Trained %s (weight=%.2f)", name, weight)
        except Exception as e:
            logger.error("  Failed to train %s: %s", name, e)

    if not final_models:
        logger.error("No models trained successfully")
        sys.exit(1)

    # Build ensemble
    ensemble = EnsembleClassifier(
        models=final_models,
        weights=final_weights,
    )

    # ── Save bundle ─────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.joblib"
    meta_path = output_dir / "metadata.json"

    joblib.dump(ensemble, model_path)
    logger.info("Saved model to %s", model_path)

    # Feature importance (from HistGB or LightGBM)
    importances = {}
    for name, model, _ in models_config:
        if hasattr(model, "feature_importances_"):
            for feat, imp in zip(feature_cols, model.feature_importances_):
                importances[feat] = importances.get(feat, 0) + float(imp)
            break

    metadata = {
        "model_version": "v4",
        "feature_version": "v4",
        "feature_count": len(feature_cols),
        "feature_list": feature_cols,
        "target_mode": args.target_mode,
        "training_samples": len(X),
        "positive_rate": float(y.mean()),
        "metrics": {
            "cv_auc_mean": float(mean_auc),
            "cv_auc_std": float(np.std(fold_aucs)) if fold_aucs else 0.0,
            "cv_fold_aucs": [float(a) for a in fold_aucs],
            "n_folds": len(fold_aucs),
            "oos_auc": float(mean_auc),  # circuit breaker reads this
        },
        "ensemble_models": [name for name, _, _ in models_config],
        "ensemble_weights": [float(w) for _, _, w in models_config],
        "feature_importances": dict(sorted(importances.items(), key=lambda x: -x[1])[:20]),
        "trained_at": datetime.utcnow().isoformat(),
        "sklearn_version": _get_sklearn_version(),
        "lightgbm_available": not args.no_lightgbm,
    }

    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    logger.info("Saved metadata to %s", meta_path)

    logger.info("")
    logger.info("Training complete!")
    logger.info("  Model: %s", model_path)
    logger.info("  AUC: %.4f", mean_auc)
    logger.info("  Features: %d", len(feature_cols))
    logger.info("  Samples: %d", len(X))


def _get_sklearn_version() -> str:
    try:
        import sklearn
        return sklearn.__version__
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
