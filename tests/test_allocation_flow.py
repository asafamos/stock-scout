"""
Allocation guarantees with risk-aware sizing
- Ensures allocated dollars never exceed total budget
- Verifies risk factor from v2 risk engine affects position sizes (CORE > SPEC)
"""
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

    # Total allocated should not exceed total budget (allowing tiny epsilon)
    assert float(alloc["סכום קנייה ($)"].sum()) <= total_budget + 1e-6

    # Compute shares and leftover like the app
    shares = np.floor(np.where(alloc["Unit_Price"] > 0, alloc["סכום קנייה ($)"] / alloc["Unit_Price"], 0)).astype(int)
    position_value = alloc["Unit_Price"].fillna(0) * shares
    leftover = alloc["סכום קנייה ($)"] - position_value

    # Sanity checks
    assert (shares >= 0).all()
    assert (leftover >= -1e-6).all()
    assert position_value.sum() <= total_budget + 1e-6


def test_risk_factor_influences_allocation_core_vs_spec():
    """Risk-aware dynamic sizing should allocate more to CORE than SPEC all else equal."""
    df = pd.DataFrame({
        "Ticker": ["CORE1", "SPEC1"],
        "Score": [80.0, 80.0],            # same nominal score
        "Unit_Price": [50.0, 50.0],       # same price
        # Canonical classification/risk fields to drive the factor
        "RiskClass": ["CORE", "SPEC"],
        "SafetyBlocked": [False, False],
        "ReliabilityScore": [90.0, 50.0],  # higher reliability for CORE
        "RR": [2.0, 1.3],                  # better RR for CORE
        # Mild nuance fields (optional)
        "ATR_Price": [0.03, 0.06],
        "Beta": [1.2, 1.7],
    })

    total_budget = 1000.0
    # No min position, no cap, to observe pure proportionality by risk factor
    alloc = allocate_budget(
        df,
        total=total_budget,
        min_pos=0.0,
        max_pos_pct=100.0,
        score_col="Score",
        dynamic_sizing=True,
    )

    # Sum must not exceed budget
    assert float(alloc["סכום קנייה ($)"].sum()) <= total_budget + 1e-6

    # Extract allocations
    by_ticker = alloc.set_index("Ticker")["סכום קנייה ($)"]
    core_alloc = float(by_ticker.loc["CORE1"]) if "CORE1" in by_ticker.index else 0.0
    spec_alloc = float(by_ticker.loc["SPEC1"]) if "SPEC1" in by_ticker.index else 0.0

    # CORE should strictly exceed SPEC due to higher position_risk_factor
    assert core_alloc > spec_alloc, (
        f"Expected CORE allocation > SPEC (got CORE={core_alloc}, SPEC={spec_alloc})"
    )

    # All sizes non-negative
    assert (alloc["סכום קנייה ($)"] >= -1e-9).all()
