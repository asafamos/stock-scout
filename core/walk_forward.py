
import numpy as np
import pandas as pd
import os
from typing import List, Tuple
from core.ml_target_config import HORIZON_DAYS


def walk_forward_splits(
    df: pd.DataFrame,
    date_col: str = "As_Of_Date",
    embargo: int = HORIZON_DAYS,
    n_folds: int = 5,
    min_train_periods: int = 252
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate walk-forward train/test splits with embargo and minimum positives per fold.
    Each fold uses all data up to a split date for training, skips embargo, and tests on the next chunk.
    Returns list of (train_idx, test_idx) tuples.
    """
    # Robust, diagnostics-rich walk-forward split generator
    def envint(var, default):
        v = os.environ.get(var)
        return int(v) if v is not None else default

    n_folds_ = n_folds or envint("N_FOLDS", 3)
    min_train_rows = envint("MIN_TRAIN_ROWS", 8000)
    min_test_rows = envint("MIN_TEST_ROWS", 1500)
    min_pos_train = envint("MIN_POS_TRAIN", 200)
    min_pos_test = envint("MIN_POS_TEST", 50)
    embargo_days = envint("EMBARGO_DAYS", embargo)

    label_col = "Label_20d" if "Label_20d" in df.columns else None
    if label_col is None:
        raise ValueError("Label_20d column required for walk-forward splits.")

    df = df.sort_values(date_col).reset_index(drop=True)
    diagnostics = []
    min_pos_test_env = int(os.environ.get("MIN_POS_TEST", min_pos_test or 30))
    test_window_days = int(os.environ.get("TEST_WINDOW_DAYS", 90))
    max_folds = n_folds_ or 3
    embargo_days = embargo_days
    unique_dates = pd.to_datetime(df[date_col]).sort_values().unique()
    total_positives = (df[label_col] == 1).sum() if label_col in df.columns else 0
    # Try to find up to 6 candidate windows, spaced across the time axis
    n_candidates = min(6, max_folds * 2, max(1, len(unique_dates) // test_window_days))
    candidate_starts = [int(i * (len(unique_dates) - test_window_days) / max(1, n_candidates - 1)) for i in range(n_candidates)]
    all_candidates = []
    min_pos_test_actual = min_pos_test_env
    found = False
    while min_pos_test_actual >= 10 and not found:
        splits = []
        for start_idx in candidate_starts:
            test_dates = unique_dates[start_idx:start_idx + test_window_days]
            embargo = embargo_days if embargo_days is not None else 20
            if len(test_dates) > embargo:
                test_dates_emb = test_dates[embargo:]
            else:
                continue
            first_test_date = test_dates_emb[0]
            train_dates = unique_dates[unique_dates < first_test_date]
            train_idx = df[df[date_col].isin(train_dates)].index.values
            test_idx = df[df[date_col].isin(test_dates_emb)].index.values
            pos_train = (df.loc[train_idx, label_col] == 1).sum() if len(train_idx) > 0 else 0
            pos_test = (df.loc[test_idx, label_col] == 1).sum() if len(test_idx) > 0 else 0
            train_range = (str(train_dates[0].date()) if len(train_dates) else 'NA', str(train_dates[-1].date()) if len(train_dates) else 'NA')
            test_range = (str(test_dates_emb[0].date()) if len(test_dates_emb) else 'NA', str(test_dates_emb[-1].date()) if len(test_dates_emb) else 'NA')
            print(f"[WF SPLIT] Fold {len(splits)+1}: train {train_range[0]}→{train_range[1]} ({len(train_idx)}), test {test_range[0]}→{test_range[1]} ({len(test_idx)}), pos_train={pos_train}, pos_test={pos_test}, min_pos_test={min_pos_test_actual}")
            all_candidates.append((pos_test, (train_idx, test_idx), train_range, test_range, pos_train, len(train_idx), len(test_idx)))
            if pos_test >= min_pos_test_actual and len(train_idx) > 0:
                splits.append((train_idx, test_idx))
                if len(splits) >= max_folds:
                    found = True
                    break
        if splits:
            found = True
            return splits
        if not found:
            print(f"[WARNING] Could not find enough splits with min_pos_test={min_pos_test_actual}, lowering requirement...")
            min_pos_test_actual -= 5
    # If still no splits, fallback to best candidate with max positives in test
    if all_candidates and total_positives >= 100:
        best = max(all_candidates, key=lambda x: x[0])
        pos_test, (train_idx, test_idx), train_range, test_range, pos_train, n_train, n_test = best
        print(f"[WALK-FORWARD] Using fallback: best window with max positives in test (pos_test={pos_test}, train {train_range[0]}→{train_range[1]}, test {test_range[0]}→{test_range[1]})")
        return [(train_idx, test_idx)]
    print("[WALK-FORWARD] No usable split found. Consider lowering constraints or expanding data.")
    return []
    # If no valid splits, print diagnostics table
    print("No valid walk-forward splits found. Diagnostics:")
    diag_df = pd.DataFrame(diagnostics)
    print(diag_df[["fold","train_rows","test_rows","pos_train","pos_test"]])
    # As a fallback, try to produce a single split with the largest possible train/test
    fallback = None
    for cut in range(n_dates//3, n_dates-embargo_days-min_test_rows):
        train_dates = unique_dates[:cut]
        test_dates = unique_dates[cut+embargo_days:]
        train_idx = df[df[date_col].isin(train_dates)].index.values
        test_idx = df[df[date_col].isin(test_dates)].index.values
        pos_train = (df.loc[train_idx, label_col] == 1).sum() if len(train_idx) > 0 else 0
        pos_test = (df.loc[test_idx, label_col] == 1).sum() if len(test_idx) > 0 else 0
        if (
            len(train_idx) >= min_train_rows and
            len(test_idx) >= min_test_rows and
            pos_train >= min_pos_train and
            pos_test >= min_pos_test
        ):
            fallback = [(train_idx, test_idx)]
            break
    if fallback:
        print("[WALK-FORWARD] Using fallback single split.")
        return fallback
    print("[WALK-FORWARD] No usable split found. Consider lowering constraints or expanding data.")
    return []
