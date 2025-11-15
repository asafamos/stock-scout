"""Time-test validation: check if model/filters would flag stocks before a known move.

Usage:
  python time_test_validation.py --model model_5d.pkl --cases cases.csv

cases.csv format:
  Ticker,EventDate,PreDays
  NVDA,2024-05-20,5
  AAPL,2023-07-25,10
"""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

from backtest_recommendations import (
    build_indicator_frame,
    FilterThresholds,
    apply_filters,
)


def parse_args():
    p = argparse.ArgumentParser(description="Time-test validation for specific tickers/events")
    p.add_argument('--model', type=str, required=True, help='Path to trained model pickle')
    p.add_argument('--cases', type=str, required=True, help='CSV with Ticker,EventDate,PreDays')
    return p.parse_args()


def load_cases(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'Ticker' not in df.columns or 'EventDate' not in df.columns:
        raise SystemExit('cases CSV must include Ticker,EventDate[,PreDays]')
    if 'PreDays' not in df.columns:
        df['PreDays'] = 5
    return df


def main():
    args = parse_args()
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    cases = load_cases(args.cases)
    rows = []
    thresholds = FilterThresholds()

    for _, r in cases.iterrows():
        tk = r['Ticker']
        event = pd.to_datetime(r['EventDate']).tz_localize(None)
        pre_days = int(r.get('PreDays', 5))
        end = event
        start = event - timedelta(days=400)
        try:
            df = yf.download(tk, start=start, end=end, auto_adjust=True, progress=False)
            if df.empty:
                rows.append({'Ticker': tk, 'EventDate': event.date(), 'Status': 'NoData'})
                continue
            ind = build_indicator_frame(df)
            # pick evaluation date = last available day minus pre_days
            if len(ind) < 220:
                rows.append({'Ticker': tk, 'EventDate': event.date(), 'Status': 'InsufficientHistory'})
                continue
            eval_idx = max(0, len(ind) - pre_days - 1)
            eval_date = ind.index[eval_idx]
            vec = ind.loc[eval_date]
            # Require indicators present
            feat_cols = [c for c in ['RSI', 'ATR_Pct', 'Overext', 'RR', 'MomCons', 'VolSurge'] if c in ind.columns]
            if vec[feat_cols].isna().any():
                rows.append({'Ticker': tk, 'EventDate': event.date(), 'Status': 'NaNFeatures'})
                continue
            prob = float(model.predict_proba([vec[feat_cols].astype(float).values])[0][1])
            passed = apply_filters(vec, thresholds)
            rows.append({
                'Ticker': tk,
                'EventDate': event.date(),
                'EvalDate': pd.to_datetime(eval_date).date(),
                'Prob': prob,
                'PassedFilters': bool(passed),
                'RSI': float(vec['RSI']),
                'ATR_Pct': float(vec['ATR_Pct']),
                'Overext': float(vec['Overext']),
                'RR': float(vec['RR']),
                'MomCons': float(vec['MomCons']),
            })
        except Exception as e:
            rows.append({'Ticker': tk, 'EventDate': event.date(), 'Status': f'Error: {e}'})

    out = pd.DataFrame(rows)
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    path = f'time_test_{ts}.csv'
    out.to_csv(path, index=False)
    print(out)
    print(f"Saved time test results to {path}")


if __name__ == '__main__':
    main()
