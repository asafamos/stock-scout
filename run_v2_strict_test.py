#!/usr/bin/env python3
"""
Run a strict V2 test for a ticker and print a compact summary.

Usage:
    python run_v2_strict_test.py TICKER

This script creates three scenarios (high/medium/low quality) and calls
`score_ticker_v2_enhanced` from `core.v2_risk_engine` for each, printing a
concise JSON object with the relevant V2 fields for audit and quick checks.
"""

import sys
import json
from pathlib import Path
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from core.v2_risk_engine import score_ticker_v2_enhanced


def make_scenarios(ticker, price=100.0):
    return [
        (
            "HIGH",
            pd.Series({
                "Ticker": ticker,
                "Price_Mean": price,
                "Price_STD": price * 0.005,
                "Price_Sources_Count": 4,
                "Fundamental_Sources_Count": 3,
                "Fund_from_FMP": True,
                "Fund_from_Finnhub": True,
                "Fund_from_SimFin": True,
                "PE_f": 25.0,
                "PS_f": 5.0,
                "ROE_f": 20.0,
                "Quality_Score": 45.0,
                "RewardRisk": 2.5,
                "RR_Ratio": 2.5,
                "Score": 75.0,
                "Risk_Level": "core",
                "Unit_Price": price,
                "סכום קנייה ($)": 500.0,
                "ML_Probability": 0.75
            })
        ),
        (
            "MEDIUM",
            pd.Series({
                "Ticker": ticker,
                "Price_Mean": price,
                "Price_STD": price * 0.02,
                "Price_Sources_Count": 2,
                "Fundamental_Sources_Count": 1,
                "Fund_from_FMP": True,
                "PE_f": 30.0,
                "Quality_Score": 30.0,
                "RewardRisk": 1.6,
                "RR_Ratio": 1.6,
                "Score": 60.0,
                "Risk_Level": "speculative",
                "Unit_Price": price,
                "סכום קנייה ($)": 500.0,
                "ML_Probability": 0.6
            })
        ),
        (
            "LOW",
            pd.Series({
                "Ticker": ticker,
                "Price_Mean": price,
                "Price_STD": price * 0.05,
                "Price_Sources_Count": 1,
                "Fundamental_Sources_Count": 0,
                "RewardRisk": 0.23,
                "RR_Ratio": 0.23,
                "Score": 45.0,
                "Risk_Level": "speculative",
                "Unit_Price": price,
                "סכום קנייה ($)": 500.0,
                "ML_Probability": 0.45
            })
        )
    ]


def compact_result(result):
    """Return a compact dict with the key V2 fields requested by the user."""
    return {
        "ticker": result.get("ticker"),
        "rr_ratio": result.get("rr_ratio_v2") or result.get("rr_ratio_v2", 0),
        "reliability_v2": round(float(result.get("reliability_v2", 0.0)), 2),
        "fund_sources_count": int(result.get("fund_sources_count_v2", 0)),
        "risk_gate_status": result.get("risk_gate_status_v2"),
        "risk_gate_penalty": float(result.get("risk_gate_penalty_v2", 0.0)),
        "conviction_base": float(result.get("conviction_v2_base", 0.0)),
        "conviction_final": float(result.get("conviction_v2_final", 0.0)),
        "ml_probability": result.get("ml_probability_v2"),
        "ml_boost": float(result.get("ml_boost_v2", 0.0)),
        "buy_amount_v2": float(result.get("buy_amount_v2", 0.0)),
        "shares_v2": int(result.get("shares_to_buy_v2", 0)),
        "reason": result.get("risk_gate_status_v2") or "",
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_v2_strict_test.py TICKER")
        sys.exit(1)

    ticker = sys.argv[1].upper()

    # Use a sensible default mock price; we don't fetch live data here to keep
    # this script fast and offline-friendly. The user can adjust price if needed.
    scenarios = make_scenarios(ticker, price=100.0)

    outputs = []
    for name, row in scenarios:
        res = score_ticker_v2_enhanced(ticker, row, budget_total=5000.0, min_position=50.0, enable_ml=True)
        compact = compact_result(res)
        compact["scenario"] = name
        outputs.append(compact)

    # Print JSON array
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
