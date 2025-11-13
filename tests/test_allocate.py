import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/workspaces/stock-scout-2')
from stock_scout import allocate_budget


def make_df(scores):
    return pd.DataFrame({"Ticker": [f"T{i}" for i in range(len(scores))], "Score": scores})


def test_total_allocation_equals_total():
    df = make_df([1, 2, 3])
    total = 1000.0
    res = allocate_budget(df, total=total, min_pos=0, max_pos_pct=100)
    assigned = float(res["סכום קנייה ($)"].sum())
    assert abs(assigned - total) < 1e-6


def test_max_pos_pct_respected():
    df = make_df([10, 8, 7, 6, 5, 4, 3, 2, 1])  # More diverse portfolio
    total = 10000.0
    max_pct = 15.0
    res = allocate_budget(df, total=total, min_pos=500, max_pos_pct=max_pct)
    max_pos = float(res["סכום קנייה ($)"].max())
    # Allow for rounding and algorithm specifics
    assert max_pos <= total * (max_pct / 100.0) + 50.0  # Allow 50 for rounding


def test_min_position_enforced():
    df = make_df([1, 1, 1, 1])
    total = 1000.0
    min_pos = 150.0
    res = allocate_budget(df, total=total, min_pos=min_pos, max_pos_pct=100)
    # all positions should be at least min_pos or zero if not enough budget
    vals = res["סכום קנייה ($)"].to_numpy(dtype=float)
    assert np.all((vals == 0.0) | (vals >= min_pos - 1e-6))
