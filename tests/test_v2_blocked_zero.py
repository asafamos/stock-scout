import pandas as pd
from core.v2_risk_engine import score_ticker_v2_enhanced


def test_v2_blocked_results_in_zero_buy_and_shares():
    """A ticker with very low RR and no fund sources should be blocked and have zero buy/shares."""
    # Build a minimal row that triggers blocking: rr very low and no fund sources
    row = pd.Series({
        "Ticker": "TESTBLK",
        "Unit_Price": 10.0,
        "Price_Mean": 10.0,
        "Price_STD": 0.5,
        "Price_Sources_Count": 1,
        "Fundamental_Sources_Count": 0,
        "Fund_from_FMP": False,
        "Fund_from_Finnhub": False,
        "Fund_from_SimFin": False,
        "RR_Ratio": 0.2,
        "RewardRisk": 0.2,
        "Risk_Level": "core",
    })

    res = score_ticker_v2_enhanced("TESTBLK", row, budget_total=5000.0, min_position=50.0, enable_ml=False)

    assert res.get("risk_gate_status_v2") == "blocked", "Expected gate to be 'blocked' for low RR / no funds"
    assert float(res.get("buy_amount_v2", 0)) == 0.0, "Blocked tickers must have buy_amount_v2 == 0"
    assert int(res.get("shares_to_buy_v2", 0)) == 0, "Blocked tickers must have shares_to_buy_v2 == 0"
