#!/usr/bin/env python3
"""
Run an end-to-end V2 strict-mode sample on 25 tickers and export CSV.

The script builds a 25-ticker universe via `build_universe`, fetches
a recent price for each ticker, then synthesizes rows that exercise the
strict-mode rules:
 - first 5: R<1.0 -> blocked
 - next 5: 0 fund sources -> blocked
 - next 5: low reliability (missing funds) -> reliability_v2 < 20 -> reduced/blocked
 - next 5: speculative -> ensure buy_amount capped (<= ~150 USD)
 - last 5: core with good RR & funds -> full

Outputs:
 - `v2_e2e_sample.csv` in repo root with the requested V2 fields.
"""

import json
from pathlib import Path
import sys
import time

import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))
from stock_scout import build_universe
from core.v2_risk_engine import score_ticker_v2_enhanced

OUT_CSV = Path("v2_e2e_sample.csv")

def fetch_price(ticker):
    try:
        t = yf.Ticker(ticker)
        h = t.history(period="1mo", actions=False)
        if h.empty:
            return None, None
        close = float(h['Close'].iloc[-1])
        std = float(h['Close'].pct_change().dropna().std() * close) if len(h) > 1 else close*0.01
        return close, std
    except Exception:
        return None, None


def make_row(ticker, price, std, scenario):
    # Base skeleton with many fields used by V2 engine
    row = {
        "Ticker": ticker,
        "Price_Mean": price,
        "Price_STD": std if std is not None else (price*0.01 if price else 1.0),
        "Price_Sources_Count": 3,
        "Fundamental_Sources_Count": 2,
        "Fund_from_FMP": True,
        "Fund_from_Finnhub": True,
        "Fund_from_SimFin": True,
        "PE_f": 20.0,
        "PS_f": 3.5,
        "ROE_f": 12.0,
        "ROIC_f": 8.0,
        "GM_f": 30.0,
        "ProfitMargin": 12.0,
        "DE_f": 0.6,
        "RevG_f": 8.0,
        "EPSG_f": 10.0,
        "RevenueGrowthYoY": 8.0,
        "EPSGrowthYoY": 10.0,
        "PBRatio": 2.2,
        "Fundamental_S": 65.0,
        "Quality_Score": 40.0,
        "Score": 60.0,
        "Score_Tech": 60.0,
        "Risk_Level": "core",
        "Unit_Price": price or 100.0,
        "סכום קנייה ($)": 500.0,
        "ML_Probability": 0.6
    }

    if scenario == 'RR_LT_1':
        row['RewardRisk'] = 0.9
        row['RR_Ratio'] = 0.9
    elif scenario == 'NO_FUNDS':
        # set no fund sources
        row['Fund_from_FMP'] = False
        row['Fund_from_Finnhub'] = False
        row['Fund_from_SimFin'] = False
        row['Fundamental_Sources_Count'] = 0
        row['RR_Ratio'] = 1.5
    elif scenario == 'LOW_REL':
        # Remove fundamental fields to reduce completeness
        for k in ['PE_f','PS_f','ROE_f','ROIC_f','GM_f','ProfitMargin','DE_f','RevG_f','EPSG_f','RevenueGrowthYoY','EPSGrowthYoY','PBRatio']:
            row[k] = None
        row['Fundamental_Sources_Count'] = 1
        row['Fund_from_FMP'] = False
        row['Fund_from_Finnhub'] = True
        row['RR_Ratio'] = 1.2
    elif scenario == 'SPEC_CAP':
        row['Risk_Level'] = 'speculative'
        row['RR_Ratio'] = 1.5
    elif scenario == 'CORE_GOOD':
        row['Risk_Level'] = 'core'
        row['Fundamental_Sources_Count'] = 3
        row['Fund_from_FMP'] = True
        row['Fund_from_Finnhub'] = True
        row['Fund_from_SimFin'] = True
        row['RR_Ratio'] = 2.0
    return pd.Series(row)


def main():
    tickers = build_universe(25)
    print(f"Running V2 E2E sample on {len(tickers)} tickers")

    scenarios = []
    # Assign scenarios deterministically
    for i, t in enumerate(tickers):
        if i < 5:
            scenarios.append('RR_LT_1')
        elif i < 10:
            scenarios.append('NO_FUNDS')
        elif i < 15:
            scenarios.append('LOW_REL')
        elif i < 20:
            scenarios.append('SPEC_CAP')
        else:
            scenarios.append('CORE_GOOD')

    results = []
    for tkr, scen in zip(tickers, scenarios):
        price, std = fetch_price(tkr)
        if price is None:
            price = 100.0
            std = 1.0
        row = make_row(tkr, price, std, scen)
        res = score_ticker_v2_enhanced(tkr, row, budget_total=5000.0, min_position=50.0, enable_ml=True)
        out = {
            'Ticker': tkr,
            'scenario': scen,
            'rr_ratio_v2': res.get('rr_ratio_v2'),
            'reliability_v2': res.get('reliability_v2'),
            'fund_sources_count_v2': res.get('fund_sources_count_v2'),
            'risk_gate_status_v2': res.get('risk_gate_status_v2'),
            'risk_gate_reason_v2': res.get('risk_gate_reason_v2'),
            'conviction_v2_base': res.get('conviction_v2_base'),
            'conviction_v2_final': res.get('conviction_v2_final'),
            'buy_amount_v2': res.get('buy_amount_v2'),
            'shares_to_buy_v2': res.get('shares_to_buy_v2'),
            'fund_sources_used_v2': json.dumps(res.get('fund_sources_used_v2', []), ensure_ascii=False),
            'price_sources_used_v2': json.dumps(res.get('price_sources_used_v2', []), ensure_ascii=False),
            'fund_disagreement_score_v2': res.get('fund_disagreement_score_v2'),
            'price_variance_score_v2': res.get('price_variance_score_v2')
        }
        results.append(out)
        print(f"{tkr}: scenario={scen} status={out['risk_gate_status_v2']} buy={out['buy_amount_v2']}")
        time.sleep(0.1)

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote sample CSV: {OUT_CSV.resolve()}")

    # Quick summary checks
    print('\nSummary:')
    print(df['risk_gate_status_v2'].value_counts(dropna=False))
    # Validate rules
    # R<1 -> blocked
    r_lt1 = df[df['scenario']=='RR_LT_1']
    print('\nR<1 sample statuses:')
    print(r_lt1[['Ticker','risk_gate_status_v2','buy_amount_v2']])

    no_funds = df[df['scenario']=='NO_FUNDS']
    print('\nNo fund sources sample statuses:')
    print(no_funds[['Ticker','risk_gate_status_v2','buy_amount_v2']])

    low_rel = df[df['scenario']=='LOW_REL']
    print('\nLow reliability sample statuses:')
    print(low_rel[['Ticker','risk_gate_status_v2','reliability_v2','buy_amount_v2']])

    spec = df[df['scenario']=='SPEC_CAP']
    print('\nSpeculative sample buy amounts (should be <= ~150):')
    print(spec[['Ticker','buy_amount_v2']])

    core = df[df['scenario']=='CORE_GOOD']
    print('\nCore good sample statuses (should be full):')
    print(core[['Ticker','risk_gate_status_v2','buy_amount_v2']])


if __name__ == '__main__':
    main()
