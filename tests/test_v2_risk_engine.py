import pandas as pd
import numpy as np

from core.v2_risk_engine import (
    calculate_reliability_v2,
    calculate_risk_gate_v2,
    calculate_position_size_v2,
    apply_v2_conviction_adjustments,
    score_ticker_v2_enhanced,
)


def make_row(base=None):
    base = base or {}
    # minimal row with defaults
    row = {
        "Ticker": "TST",
        "Price_Mean": 100.0,
        "Price_STD": 1.0,
        "Price_Sources_Count": 2,
        "Fundamental_Sources_Count": 2,
        "Fund_from_FMP": True,
        "Fund_from_Finnhub": True,
        "Fund_from_SimFin": False,
        "PE_f": 15.0,
        "PS_f": 2.0,
        "ROE_f": 10.0,
        "ROIC_f": 8.0,
        "GM_f": 25.0,
        "ProfitMargin": 8.0,
        "DE_f": 0.5,
        "RevG_f": 5.0,
        "EPSG_f": 6.0,
        "RevenueGrowthYoY": 5.0,
        "EPSGrowthYoY": 6.0,
        "PBRatio": 1.5,
        "Fundamental_S": 60.0,
        "Quality_Score": 40.0,
        "Score": 60.0,
        "Score_Tech": 60.0,
        "Risk_Level": "core",
        "Unit_Price": 100.0,
        "סכום קנייה ($)": 500.0,
        "ML_Probability": 0.6,
        "RewardRisk": 2.0,
        "RR_Ratio": 2.0,
    }
    row.update(base)
    return pd.Series(row)


def test_rr_less_than_one_blocked():
    row = make_row({"RR_Ratio": 0.9, "RewardRisk": 0.9})
    res = score_ticker_v2_enhanced("TST", row, budget_total=5000.0, min_position=50.0, enable_ml=False)
    assert res["risk_gate_status_v2"] == "blocked"
    assert res["buy_amount_v2"] == 0.0
    assert res["shares_to_buy_v2"] == 0


def test_zero_fund_sources_blocked():
    row = make_row({"Fund_from_FMP": False, "Fund_from_Finnhub": False, "Fund_from_SimFin": False, "Fundamental_Sources_Count": 0})
    res = score_ticker_v2_enhanced("TST", row, budget_total=5000.0, min_position=50.0, enable_ml=False)
    assert res["risk_gate_status_v2"] == "blocked"
    assert res["buy_amount_v2"] == 0.0


def test_low_reliability_reduced():
    # remove many fundamentals to lower completeness but keep one fund source
    row = make_row({
        "PE_f": None, "PS_f": None, "ROE_f": None, "ROIC_f": None,
        "GM_f": None, "ProfitMargin": None, "DE_f": None,
        "RevG_f": None, "EPSG_f": None, "RevenueGrowthYoY": None,
        "EPSGrowthYoY": None, "PBRatio": None,
        "Fund_from_FMP": False, "Fund_from_Finnhub": True, "Fundamental_Sources_Count": 1,
        "RR_Ratio": 1.4
    })
    res = score_ticker_v2_enhanced("TST", row, budget_total=5000.0, min_position=50.0, enable_ml=False)
    # reliability should be low and gate should be reduced (or possibly blocked if extreme)
    assert res["reliability_v2"] < 60
    assert res["risk_gate_status_v2"] in ("reduced", "blocked")


def test_speculative_cap_applied():
    row = make_row({"Risk_Level": "speculative", "RR_Ratio": 1.6})
    res = score_ticker_v2_enhanced("TST", row, budget_total=5000.0, min_position=50.0, enable_ml=False)
    # Speculative cap should limit buy_amount to 3% of budget = 150
    assert res["buy_amount_v2"] <= 150.0


def test_core_full_allocation():
    row = make_row({"Risk_Level": "core", "RR_Ratio": 2.0, "Fund_from_FMP": True, "Fund_from_Finnhub": True, "Fund_from_SimFin": True, "Fundamental_Sources_Count": 3})
    res = score_ticker_v2_enhanced("TST", row, budget_total=5000.0, min_position=50.0, enable_ml=False)
    assert res["risk_gate_status_v2"] == "full"
    assert res["buy_amount_v2"] > 0.0
