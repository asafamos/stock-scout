import numpy as np
import pandas as pd
import re
from core.ml_targets import make_label_20d

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
            assert not re.search(r'quantile|WINNER_PERCENTILE|percentile', code), f"Quantile/percentile logic found in {fname}"
            assert 'else -1' not in code, f"else -1 found in {fname}"

def test_make_label_20d_behavior():
    # Test correct label assignment and output
    arr = np.array([-0.1, 0.0, 0.02, 0.05, 0.15, 0.2, np.nan])
    s = pd.Series(arr)
    label = make_label_20d(s)
    # Only 0/1 or NaN
    assert set(np.unique(label.dropna())) <= {0, 1}
    # Check correct assignment
    assert label.iloc[0] == 0  # -0.1 <= 0.02
    assert label.iloc[2] == 0  # 0.02 == 0.02
    assert label.iloc[4] == 1  # 0.15 >= 0.15
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
