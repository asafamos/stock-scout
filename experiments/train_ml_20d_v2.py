from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from typing import List
from pathlib import Path
import pandas as pd
from core.ml_targets import make_label_20d
import numpy as np
import joblib
from core.ml_targets import make_label_20d_strict, make_label_20d_soft
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
Train a stronger 20d ML model (v2)
- Loads experiments/training_dataset_20d_v2.csv
- Hyperparameter tuning via GridSearchCV with GradientBoostingClassifier
- Ranking-oriented evaluation: deciles, top-k buckets, multi-threshold metrics
- Saves bundle at models/model_20d_v2.pkl with keys: {"model": estimator, "feature_names": [..]}
"""


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
        # --- Ticker coverage sanity check ---
        min_tickers = int(os.environ.get("MIN_TICKERS", 500))
        tickers = df["Ticker"].unique() if "Ticker" in df.columns else []
        n_tickers = len(tickers)
        if n_tickers < min_tickers:
            print(
                f"[WARNING] Dataset too small for stable walk-forward evaluation: only {n_tickers} tickers (min required: {min_tickers})")
        else:
            print(
                f"[TICKER COVERAGE] {n_tickers} tickers included (min required: {min_tickers})")

        # --- Log tickers included and reasons for exclusion ---
        # If available, log exclusion reasons (provider errors, missing
        # history, filters)
        if "Ticker_Included" in df.columns:
            included = df[df["Ticker_Included"] == 1]["Ticker"].unique()
            excluded = df[df["Ticker_Included"] == 0][[
                "Ticker", "Exclusion_Reason"]].drop_duplicates()
            print(
                f"[TICKERS INCLUDED] {len(included)}: {sorted(list(included))[:10]}{' ...' if len(included) > 10 else ''}")
            print(f"[TICKERS EXCLUDED] {len(excluded)}")
            for _, row in excluded.iterrows():
                print(f"  {row['Ticker']}: {row['Exclusion_Reason']}")
        else:
            print(
                "[TICKER LOG] No Ticker_Included/Exclusion_Reason columns found for detailed logging.")
    df = pd.read_csv(path)
    # Validate required columns
    required = set(FEATURES + ["Label_20d", "Forward_Return_20d"])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # --- Forward_Return_20d sanity checks ---
    if "Forward_Return_20d" in df.columns:
        ret = df["Forward_Return_20d"]
        print("[FORWARD RETURN] describe:")
        print(ret.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))
        print(f"[FORWARD RETURN] >+0.15: {(ret > 0.15).sum()} | >+0.08: {(ret > 0.08).sum()} | <-0.08: {(ret < -0.08).sum()}")
        n_nan = ret.isna().sum()
        print(f"[FORWARD RETURN] NaNs: {n_nan} / {len(ret)}")
        if ret.max() < 0.05:
            raise ValueError("[FAIL] Max forward return < 0.05. Check data pipeline.")
        if n_nan > 0.1 * len(ret):
            raise ValueError("[FAIL] Too many NaNs in Forward_Return_20d.")

            df = pd.read_csv(path)
        # Print 5 random samples: ticker, date, close[t], close[t+20], forward return
        import random
        if "Ticker" in df.columns and "As_Of_Date" in df.columns and "Price_As_Of_Date" in df.columns:
            sample_idx = random.sample(list(df.index), min(5, len(df)))
            print("[FORWARD RETURN] Sample calculations:")
            for idx in sample_idx:
                row = df.loc[idx]
                ticker = row["Ticker"]
                date = row["As_Of_Date"]
                close_t = row["Price_As_Of_Date"]
                # Try to find close[t+20] for this ticker
                future = df[(df["Ticker"] == ticker) & (df["As_Of_Date"] > date)].sort_values("As_Of_Date").head(20)
                if len(future) >= 20:
                    close_t20 = future.iloc[19]["Price_As_Of_Date"]
                    fwd_ret = (close_t20 / close_t) - 1 if close_t else float('nan')
                    print(f"  {ticker} {date}: close[t]={close_t:.2f}, close[t+20]={close_t20:.2f}, fwd_ret={fwd_ret:.4f}")
                else:
                    print(f"  {ticker} {date}: insufficient future data for t+20")

    # Drop rows with invalid labels (unified logic)
    df = df.dropna(subset=["Label_20d"]).copy()
    df["Label_20d"] = df["Label_20d"].astype(int)
    # Drop rows with missing features
    df = df.dropna(subset=FEATURES + ["Label_20d"]).copy()
    # Clip features similar to inference rules
    df["ATR_Pct"] = np.clip(df["ATR_Pct"], 0.0, 0.2)
    df["RR"] = np.clip(df["RR"], 0.0, 10.0)
    df["RSI"] = np.clip(df["RSI"], 5.0, 95.0)
    df["TechScore_20d"] = np.clip(df["TechScore_20d"], 0.0, 100.0)

    # 1) Overall positive rate
    total = len(df)
    pos = (df["Label_20d"] == 1).sum()
    neg = (df["Label_20d"] == 0).sum()
    pos_rate = 100.0 * pos / total if total > 0 else 0.0
    print(f"[LABEL STATS] Total rows: {total}")
    print(f"[LABEL STATS] Positives: {pos}, Negatives: {neg}, Positive rate: {pos_rate:.2f}%")

    # 2) Positive rate by date (by month if many dates)
    if 'As_Of_Date' in df.columns:
        df['As_Of_Date'] = pd.to_datetime(df['As_Of_Date'])
        if df['As_Of_Date'].nunique() > 30:
            df['Period'] = df['As_Of_Date'].dt.to_period('M')
        else:
            df['Period'] = df['As_Of_Date']
        by_period = df.groupby('Period')['Label_20d'].sum()
        min_pos = by_period.min()
        max_pos = by_period.max()
        zero_pos = (by_period == 0).sum()
        print(f"[LABEL STATS] By period: min positives={min_pos}, max positives={max_pos}, periods with 0 positives={zero_pos}")
    else:
        print("[LABEL STATS] As_Of_Date column not found for period stats.")
    return df
def train_and_evaluate(df: pd.DataFrame):
    from core.walk_forward import walk_forward_splits
    X = df[FEATURES].astype(float).values
    y = df["Label_20d"].astype(int).values
    ret = df["Forward_Return_20d"].astype(float).values
    dates = df["As_Of_Date"].values


    splits = walk_forward_splits(df, date_col="As_Of_Date", embargo=20, n_folds=5, min_train_periods=252)
    if not splits:
        print("[WARNING] No valid walk-forward splits found. Using fallback single split: train=first 80% by date, embargo=20 days, test=last 20% by date.")
        df_sorted = df.sort_values("As_Of_Date").reset_index(drop=True)
        n = len(df_sorted)
        embargo = 20
        train_end = int(n * 0.8)
        test_start = train_end + embargo
        if test_start >= n:
            print("[ERROR] Not enough data for fallback split after embargo.")
            return None, None, None, None, None, None
        train_idx = df_sorted.iloc[:train_end].index.values
        test_idx = df_sorted.iloc[test_start:].index.values
        splits = [(train_idx, test_idx)]

    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        train_dates = df.loc[train_idx, "As_Of_Date"]
        test_dates = df.loc[test_idx, "As_Of_Date"]
        train_start = train_dates.min() if not train_dates.empty else None
        train_end = train_dates.max() if not train_dates.empty else None
        test_start = test_dates.min() if not test_dates.empty else None
        test_end = test_dates.max() if not test_dates.empty else None
        # Embargo days = gap between train_end and test_start (in trading days)
        embargo_days = None
        if train_end is not None and test_start is not None:
            embargo_days = (pd.to_datetime(test_start) - pd.to_datetime(train_end)).days
        unique_test_dates = test_dates.nunique()
        print(f"\n[WALK-FORWARD] Fold {fold+1}/{len(splits)}: train={len(train_idx)}, test={len(test_idx)}")
        print(f"  Train: {train_start} → {train_end}")
        print(f"  Test:  {test_start} → {test_end}")
        print(f"  Embargo days applied: {embargo_days if embargo_days is not None else 'N/A'}")
        print(f"  Unique As_Of_Date in test: {unique_test_dates}")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test, ret_test = X[test_idx], y[test_idx], ret[test_idx]

        train_pos = (y_train == 1).sum()
        train_neg = (y_train == 0).sum()
        test_pos = (y_test == 1).sum()
        test_neg = (y_test == 0).sum()
        print(f"  Train positives: {train_pos}, negatives: {train_neg}")
        print(f"  Test positives: {test_pos}, negatives: {test_neg}")
        if train_pos < 20 or test_pos < 20:
            print(f"[ERROR] Fold {fold+1}: Not enough positives in train or test (train: {train_pos}, test: {test_pos}).\nConsider lowering UP_THRESHOLD or increasing data.")
            return None, None, None, None, None, None

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
        grid = GridSearchCV(pipe, param_grid, cv=3, scoring="roc_auc", n_jobs=2, verbose=0)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        proba = best_model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, proba)
        ap = average_precision_score(y_test, proba)
        baseline_ret = np.mean(ret_test)
        baseline_hit = np.mean(y_test)
        fold_metrics.append({
            "roc_auc": roc_auc,
            "ap": ap,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "proba": proba,
            "y_test": y_test,
            "ret_test": ret_test,
            "baseline_ret": baseline_ret,
            "baseline_hit": baseline_hit,
        })
        print(f"  ROC-AUC={roc_auc:.3f} | AP={ap:.3f} | Baseline avg_ret={baseline_ret:.4f}, hit_rate={baseline_hit:.3f}")

    # Aggregate metrics
    mean_roc_auc = np.mean([m["roc_auc"] for m in fold_metrics])
    std_roc_auc = np.std([m["roc_auc"] for m in fold_metrics])
    mean_ap = np.mean([m["ap"] for m in fold_metrics])
    std_ap = np.std([m["ap"] for m in fold_metrics])
    mean_baseline_ret = np.mean([m["baseline_ret"] for m in fold_metrics])
    mean_baseline_hit = np.mean([m["baseline_hit"] for m in fold_metrics])
    print(f"\n[SUMMARY] Walk-forward ROC-AUC: mean={mean_roc_auc:.3f} ± {std_roc_auc:.3f}, AP: mean={mean_ap:.3f} ± {std_ap:.3f}")
    print(f"[SUMMARY] Baseline avg 20d return: {mean_baseline_ret:.4f}, hit rate: {mean_baseline_hit:.3f}")

    # Decile/top-k reporting for last fold
    last = fold_metrics[-1]
    proba, ret_test, y_test = last["proba"], last["ret_test"], last["y_test"]
    test_df = pd.DataFrame({"proba": proba, "ret": ret_test, "label": y_test})
    test_df = test_df.sort_values(by=["proba"], ascending=False).reset_index(drop=True)
    n = len(test_df)
    print("\n[DECILES] (sorted by probability, descending):")
    for decile in range(10):
        start_idx = int(decile * n / 10)
        end_idx = int((decile + 1) * n / 10)
        sub = test_df.iloc[start_idx:end_idx]
        avg_ret = sub["ret"].mean()
        hit_rate = sub["label"].mean()
        print(f"  Decile {decile}: n={len(sub)}, avg_ret={avg_ret:.4f}, hit_rate={hit_rate:.3f}")
    print("\n[TOP-K BUCKETS]:")
    for pct in [0.05, 0.10, 0.20]:
        k = int(pct * n)
        sub = test_df.head(k)
        avg_ret = sub["ret"].mean()
        hit_rate = sub["label"].mean()
        print(f"  Top {pct*100:.0f}% (n={k}): avg_ret={avg_ret:.4f}, hit_rate={hit_rate:.3f}")

    return best_model, mean_roc_auc, mean_ap, mean_baseline_ret, mean_baseline_hit, test_df

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

    # Strict label
    df["Label_20d_strict"] = make_label_20d_strict(df)
    print("\n=== [STRICT LABEL: UP=0.15, DOWN=0.02] ===")
    model_s, roc_auc_s, ap_s, baseline_ret_s, baseline_hit_s, test_df_s = train_and_evaluate(df.rename(columns={"Label_20d": "Label_20d_strict"}))
    save_bundle(model_s, FEATURES)
    print("\n[SUMMARY - STRICT]")
    if roc_auc_s is not None:
        print(f"  ROC-AUC: {roc_auc_s:.3f}")
        print(f"  Avg Precision: {ap_s:.3f}")
        print(f"  Baseline avg 20d return: {baseline_ret_s:.4f}")
        print(f"  Baseline hit rate (≥15%): {baseline_hit_s:.3f}")
    else:
        print("  No valid splits or not enough positives to compute metrics.")

    # Soft label
    df["Label_20d_soft"] = make_label_20d_soft(df)
    print("\n=== [SOFT LABEL: UP=0.08, DOWN=0.00] ===")
    model_sf, roc_auc_sf, ap_sf, baseline_ret_sf, baseline_hit_sf, test_df_sf = train_and_evaluate(df.rename(columns={"Label_20d": "Label_20d_soft"}))
    save_bundle(model_sf, FEATURES)
    print("\n[SUMMARY - SOFT]")
    if roc_auc_sf is not None:
        print(f"  ROC-AUC: {roc_auc_sf:.3f}")
        print(f"  Avg Precision: {ap_sf:.3f}")
        print(f"  Baseline avg 20d return: {baseline_ret_sf:.4f}")
        print(f"  Baseline hit rate (≥8%): {baseline_hit_sf:.3f}")
    else:
        print("  No valid splits or not enough positives to compute metrics.")



if __name__ == "__main__":
    main()
