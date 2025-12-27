import pandas as pd
import numpy as np

from core.portfolio import allocate_budget


def test_allocation_flow_shares_and_leftover():
    # Synthetic tickers with unit prices and scores
    df = pd.DataFrame({
        "Ticker": ["AAA", "BBB", "CCC"],
        "Score": [90.0, 70.0, 50.0],
        "Unit_Price": [10.0, 25.0, 50.0],
    })

    total_budget = 1000.0
    min_pos = 100.0
    max_pos_pct = 50.0  # max $500 per position

    # Allocate budget using portfolio helper
    alloc = allocate_budget(df, total=total_budget, min_pos=min_pos, max_pos_pct=max_pos_pct, score_col="Score", dynamic_sizing=False)

    # Column exists
    assert "סכום קנייה ($)" in alloc.columns

    # Min position satisfied for the top-3 (sorted by Score desc)
    top3 = alloc.sort_values(["Score", "Ticker"], ascending=[False, True]).head(3)
    assert (top3["סכום קנייה ($)"] >= min_pos - 1e-6).all()

    # Max position cap respected
    assert (alloc["סכום קנייה ($)"] <= (max_pos_pct / 100.0) * total_budget + 1e-6).all()

    # Compute shares and leftover like the app
    shares = np.floor(np.where(alloc["Unit_Price"] > 0, alloc["סכום קנייה ($)"] / alloc["Unit_Price"], 0)).astype(int)
    position_value = alloc["Unit_Price"].fillna(0) * shares
    leftover = alloc["סכום קנייה ($)"] - position_value

    # Sanity checks
    assert (shares >= 0).all()
    assert (leftover >= -1e-6).all()
    assert position_value.sum() <= total_budget + 1e-6
