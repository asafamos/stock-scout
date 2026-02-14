import numpy as np
import pandas as pd
import re
from core.ml_targets import make_label_20d
from core.ml_target_config import UP_THRESHOLD, DOWN_THRESHOLD

def test_label_quantile_not_used():
    # Ensure no quantile/percentile label logic remains in codebase
    files = [
        'scripts/train_ml_smart.py',
        'scripts/train_rolling_ml_20d.py',
        'experiments/offline_recommendation_audit.py',
        'experiments/train_ml_20d_v2.py',
    ]
    for fname in files:
        with open(fname) as f:
            code = f.read()
            assert not re.search(r'quantile|WINNER_PERCENTILE', code), f"Quantile/WINNER_PERCENTILE logic found in {fname}"
            assert 'else -1' not in code, f"else -1 found in {fname}"

def test_make_label_20d_behavior():
    # Test correct label assignment using configured thresholds
    # UP_THRESHOLD=0.08, DOWN_THRESHOLD=0.00 (defaults)
    arr = np.array([-0.1, 0.0, 0.05, 0.08, 0.15, 0.2, np.nan])
    s = pd.Series(arr)
    label = make_label_20d(s)
    # Only 0/1 or NaN
    assert set(np.unique(label.dropna())) <= {0, 1}
    # Below or at DOWN_THRESHOLD → 0
    assert label.iloc[0] == 0  # -0.1 <= DOWN_THRESHOLD(0.00)
    assert label.iloc[1] == 0  # 0.0 <= DOWN_THRESHOLD(0.00)
    # Between thresholds → NaN (grey zone)
    assert np.isnan(label.iloc[2])  # 0.05: above 0.00, below 0.08
    # At or above UP_THRESHOLD → 1
    assert label.iloc[3] == 1  # 0.08 >= UP_THRESHOLD(0.08)
    assert label.iloc[4] == 1  # 0.15 >= UP_THRESHOLD(0.08)
    assert label.iloc[5] == 1  # 0.20 >= UP_THRESHOLD(0.08)
    # NaN input → NaN output
    assert np.isnan(label.iloc[6])

def test_label_20d_postprocessing():
    # Simulate postprocessing
    arr = np.array([-0.1, 0.0, 0.02, 0.05, 0.15, 0.2, np.nan])
    s = pd.Series(arr)
    label = make_label_20d(s)
    df = pd.DataFrame({'Label_20d': label})
    df = df.dropna(subset=["Label_20d"]).copy()
    df["Label_20d"] = df["Label_20d"].astype(int)
    assert set(df["Label_20d"].unique()) <= {0, 1}
