import argparse
from datetime import datetime
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
import joblib

def main():
    parser = argparse.ArgumentParser(description="Train 20d technical-only winner classifier (v3 with enriched features)")
    parser.add_argument("--input", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--output-model", type=str, required=True, help="Path to output model pickle")
    parser.add_argument("--test-split", type=float, default=0.25, help="Test set fraction (default: 0.25)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--min-return", type=float, default=0.15, help="Label threshold (for documentation)")
    args = parser.parse_args()

    # V3 Feature set: original + multi-period returns + sequential patterns + volatility context + relative strength
    feature_cols = [
        # Original technical indicators
        "TechScore_20d", "RSI", "ATR_Pct", "RR", "MomCons", "VolSurge",
        # Multi-period returns
        "Return_5d", "Return_10d", "Return_20d", "Return_60d", "Return_120d",
        # Sequential pattern features
        "UpStreak_Days", "DownStreak_Days", "PullbackFromHigh_20d", "DistanceFromLow_20d", "Range_Pct",
        # Relative strength vs market
        "RS_SPY_20d", "RS_SPY_60d",
        # Volatility context
        "ATR_Pct_percentile", "Vol_Class", "Vol_SweetSpot_Flag", "Vol_Extreme_Flag",
        # Big winner signal
        "BigWinnerScore_20d", "BigWinnerFlag_20d",
    ]
    label_col = "Label_20d"
    date_col = "As_Of_Date"
    ticker_col = "Ticker"
    price_col = "Price_As_Of_Date"

    # Load data
    df = pd.read_csv(args.input)
    
    # Check which features are available (handle missing features gracefully)
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        print(f"[WARN] Missing features in dataset: {missing_features}. Removing from training.")
        feature_cols = [f for f in feature_cols if f in df.columns]
    
    # Check for features with all NaN values
    features_with_all_nan = [f for f in feature_cols if df[f].isna().all()]
    if features_with_all_nan:
        print(f"[WARN] Features with all NaN values: {features_with_all_nan}. Removing from training.")
        feature_cols = [f for f in feature_cols if f not in features_with_all_nan]
    
    # Drop rows with missing label or features
    df = df.dropna(subset=[label_col])
    df = df[df[label_col].isin([0, 1])]
    
    # Fill missing feature values with 0 (safe default for most features)
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    print(f"[INFO] Training with {len(feature_cols)} features after cleanup")

    # Clip outliers for stability
    df["ATR_Pct"] = df["ATR_Pct"].clip(lower=0, upper=0.2)
    if "RR" in df.columns:
        df["RR"] = df["RR"].clip(lower=0, upper=10)
    if "RSI" in df.columns:
        df["RSI"] = df["RSI"].clip(lower=5, upper=95)
    if "Return_60d" in df.columns:
        df["Return_60d"] = df["Return_60d"].clip(lower=-0.5, upper=2.0)
    if "Return_120d" in df.columns:
        df["Return_120d"] = df["Return_120d"].clip(lower=-0.6, upper=3.0)

    # Convert date
    df[date_col] = pd.to_datetime(df[date_col])
    cutoff = pd.to_datetime("2024-01-01")
    df_train = df[df[date_col] < cutoff].copy()
    df_test = df[df[date_col] >= cutoff].copy()

    # Fallback to random split if needed
    if len(df_train) < 500 or len(df_test) < 200:
        df_train, df_test = train_test_split(
            df, test_size=args.test_split, random_state=args.random_state, stratify=df[label_col]
        )
        print("[INFO] Used random split due to small train/test sets.")

    X_train = df_train[feature_cols].values
    y_train = df_train[label_col].values.astype(int)
    X_test = df_test[feature_cols].values
    y_test = df_test[label_col].values.astype(int)

    print(f"TRAIN class balance: 0={np.sum(y_train==0)}, 1={np.sum(y_train==1)} / {len(y_train)}")
    print(f"TEST  class balance: 0={np.sum(y_test==0)}, 1={np.sum(y_test==1)} / {len(y_test)}")

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model: GradientBoostingClassifier for better handling of non-linear interactions
    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=100,
        min_samples_leaf=50,
        subsample=0.8,
        random_state=args.random_state,
        verbose=1,
    )
    print(f"\n[INFO] Training GradientBoostingClassifier with {len(feature_cols)} features...")
    clf.fit(X_train_scaled, y_train)

    # Evaluation
    def evaluate_split(name, y_true, y_proba):
        y_pred = (y_proba >= 0.5).astype(int)
        print(f"\n=== {name} ===")
        print("Positives:", int((y_true == 1).sum()), "total:", len(y_true))
        print("ROC AUC:", roc_auc_score(y_true, y_proba))
        print("PR AUC:", average_precision_score(y_true, y_proba))
        print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
        print("Classification report:\n", classification_report(y_true, y_pred, digits=3))

    y_train_proba = clf.predict_proba(X_train_scaled)[:, 1]
    y_test_proba = clf.predict_proba(X_test_scaled)[:, 1]
    evaluate_split("TRAIN", y_train, y_train_proba)
    evaluate_split("TEST", y_test, y_test_proba)

    # Feature importance analysis (for GradientBoosting)
    if hasattr(clf, "feature_importances_"):
        feature_importance = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": clf.feature_importances_
        }).sort_values("Importance", ascending=False)
        print("\n=== FEATURE IMPORTANCE (Top 15) ===")
        print(feature_importance.head(15).to_string(index=False))
        
        # Save feature importance to CSV
        importance_path = args.output_model.replace(".pkl", "_feature_importance.csv")
        feature_importance.to_csv(importance_path, index=False)
        print(f"\n[SAVED] Feature importance saved to {importance_path}")

    # Auto-select best scoring policy based on historical performance
    preferred_scoring_mode = "hybrid"  # default fallback
    scoring_policy_metrics = {}
    
    # Try to load enriched dataset with all scoring columns
    enriched_dataset_path = args.input.replace(".csv", "_with_adjusted_score.csv")
    if not os.path.exists(enriched_dataset_path):
        enriched_dataset_path = "data/training_dataset_20d_v3_with_adjusted_score.csv"
    
    if os.path.exists(enriched_dataset_path):
        try:
            print(f"\n[INFO] Auto-selecting best scoring policy from {enriched_dataset_path}...")
            from validate_ml_improvements import choose_best_scoring_policy_20d
            
            df_scores = pd.read_csv(enriched_dataset_path)
            policy_result = choose_best_scoring_policy_20d(df_scores, lookback_days=365)
            preferred_scoring_mode = policy_result.get("best_policy", "hybrid")
            scoring_policy_metrics = policy_result.get("policies", {})
            
            # Write scoring policy report
            reports_dir = os.path.dirname(args.output_model).replace("models", "reports")
            os.makedirs(reports_dir, exist_ok=True)
            report_path = os.path.join(reports_dir, "scoring_policy_20d.txt")
            
            with open(report_path, "w") as f:
                f.write("="*70 + "\n")
                f.write("  20d SCORING POLICY AUTO-SELECTION REPORT\n")
                f.write("="*70 + "\n\n")
                f.write(f"Evaluation Date: {datetime.utcnow().isoformat()}\n")
                f.write(f"Dataset: {enriched_dataset_path}\n")
                f.write(f"Lookback Period: 365 days\n\n")
                f.write("="*70 + "\n")
                f.write("  POLICY EVALUATION RESULTS\n")
                f.write("="*70 + "\n\n")
                
                for policy_name, metrics in scoring_policy_metrics.items():
                    f.write(f"{policy_name.upper()}:\n")
                    f.write(f"  Baseline Positive Rate: {metrics.get('baseline_pos_rate', 0):.4f}\n")
                    f.write(f"  Top Decile Positive Rate: {metrics.get('top_pos_rate', 0):.4f}\n")
                    f.write(f"  Top Decile Avg Return: {metrics.get('top_avg_return', 0):.4f}\n")
                    f.write(f"  Utility Score: {metrics.get('utility', 0):.4f}\n")
                    f.write(f"  Top Decile Count: {metrics.get('top_decile_count', 0)}\n")
                    f.write(f"  Total Rows Evaluated: {metrics.get('total_rows', 0)}\n\n")
                
                f.write("="*70 + "\n")
                f.write(f"  SELECTED POLICY: {preferred_scoring_mode.upper()}\n")
                f.write("="*70 + "\n\n")
                f.write("INTERPRETATION:\n")
                f.write("- ml_only: Sort by ML_20d_Prob (pure ML predictions)\n")
                f.write("- hybrid: Sort by HybridFinalScore_20d (0.20 Tech + 0.80 ML)\n")
                f.write("- hybrid_overlay: Sort by AdjustedScore_20d (Hybrid + overlay adjustments)\n\n")
                f.write("The selected policy will be automatically used in stock_scout.py.\n")
                f.write("No manual scoring mode selection is required.\n")
            
            print(f"[SAVED] Scoring policy report saved to {report_path}")
            
        except Exception as e:
            print(f"[WARN] Could not auto-select scoring policy: {e}")
            print("[INFO] Falling back to default 'hybrid' mode")
    else:
        print(f"[WARN] Enriched dataset not found at {enriched_dataset_path}")
        print("[INFO] Using default 'hybrid' scoring mode")

    # Save model + metadata
    model_bundle = {
        "model": clf,
        "scaler": scaler,
        "feature_names": feature_cols,
        "label_col": label_col,
        "preferred_scoring_mode_20d": preferred_scoring_mode,
        "preferred_scoring_policy_metrics_20d": scoring_policy_metrics,
        "meta": {
            "created_at": datetime.utcnow().isoformat(),
            "train_rows": int(len(df_train)),
            "test_rows": int(len(df_test)),
            "min_return_for_label": float(args.min_return),
            "model_type": "GradientBoostingClassifier",
            "n_features": len(feature_cols),
            "train_roc_auc": float(roc_auc_score(y_train, y_train_proba)),
            "test_roc_auc": float(roc_auc_score(y_test, y_test_proba)),
            "note": "20d ML v3 with enriched features: multi-period returns, sequential patterns, RS_SPY, volatility context",
        },
    }
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    joblib.dump(model_bundle, args.output_model)
    print(f"\n[SAVED] Model bundle saved to {args.output_model}")
    print(f"[INFO] Preferred scoring mode: {preferred_scoring_mode}")

if __name__ == "__main__":
    main()
