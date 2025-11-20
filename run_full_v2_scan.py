#!/usr/bin/env python3
"""
Run a full V2 scan (best-effort).
- If API keys are present, try up to 500 tickers.
- Otherwise, fall back to configured `universe_limit` to avoid heavy external API use.
Writes `full_v2_scan.csv` with key V2 fields.
"""
import time
from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from stock_scout import build_universe
from core.config import get_api_keys, get_config
from core.v2_risk_engine import score_ticker_v2_enhanced

OUT = Path("full_v2_scan.csv")

try:
    api_keys = get_api_keys()
    cfg = get_config()
except Exception as e:
    print("Failed to load config/API keys:", e)
    api_keys = None
    cfg = None

use_limit = cfg.universe_limit if cfg is not None else 50
if api_keys and (api_keys.has_alpha_vantage() or api_keys.has_finnhub() or api_keys.has_polygon() or api_keys.has_tiingo()):
    limit = 500
    print("API keys detected — attempting larger universe (500)")
else:
    limit = use_limit
    print(f"No API keys detected — using configured limit ({limit}) to avoid heavy external calls")

print(f"Building universe (limit={limit})...")
t0 = time.time()
try:
    tickers = build_universe(limit=limit)
except Exception as e:
    print("build_universe failed:", e)
    tickers = []
print(f"Built universe: {len(tickers)} tickers (took {time.time()-t0:.1f}s)")

# lightweight price fetch via yfinance
import yfinance as yf

def fetch_price(ticker):
    try:
        t = yf.Ticker(ticker)
        # Fetch 1y history to allow ATR / 52-week high estimation
        h = t.history(period="1y", actions=False)
        if h.empty:
            return None, None
        close = float(h['Close'].iloc[-1])
        std = float(h['Close'].pct_change().dropna().std() * close) if len(h) > 1 else close * 0.01
        return close, std, h
    except Exception:
        return None, None, None

rows = []
for i, tkr in enumerate(tickers, 1):
    price, std, history = fetch_price(tkr)
    if price is None:
        price = 100.0
        std = 1.0
        history = None
    # Minimal row expected by V2 engine
    row = {
        "Ticker": tkr,
        "Price_Mean": price,
        "Price_STD": std,
        "Price_Sources_Count": 2,
        "Fundamental_Sources_Count": 2,
        "Fund_from_FMP": True,
        "Fund_from_Finnhub": True,
        "Fund_from_SimFin": False,
        "Unit_Price": price,
        "Risk_Level": "core",
        "ML_Probability": 0.5,
        "PERatio": 20.0,
        "PBRatio": 2.0,
        "ProfitMargin": 10.0,
        "RevenueGrowthYoY": 5.0,
        "Score": 50.0,
    }
    # Estimate ATR and 52-week high to compute RewardRisk if possible
    try:
        atr_est = None
        target_price = None
        if history is not None and not history.empty:
            # ATR ~ mean(high-low) over last 14 days
            if 'High' in history.columns and 'Low' in history.columns:
                last = history.tail(14)
                atr_est = float((last['High'] - last['Low']).abs().dropna().mean())
            # 52-week high
            if 'High' in history.columns:
                window = history.tail(252)
                if not window.empty:
                    target_price = float(window['High'].max())
        # Fall back conservative target if missing
        if target_price is None:
            target_price = price * 1.10
        # Use ATR fallback to 1% of price if missing
        if not (isinstance(atr_est, (int, float)) and not np.isnan(atr_est) and atr_est > 0):
            atr_est = max(price * 0.01, 0.01)
        # RewardRisk approx
        reward = max(0.0, target_price - price)
        risk = max(atr_est * 2.0, price * 0.01)
        rr = 0.0 if risk <= 0 else reward / risk
        rr = float(np.clip(rr, 0.0, 5.0))
        row['RewardRisk'] = rr
    except Exception:
        row['RewardRisk'] = 0.0
    try:
        res = score_ticker_v2_enhanced(tkr, pd.Series(row), budget_total=5000.0, min_position=50.0, enable_ml=True)
    except Exception as e:
        print(f"scoring failed for {tkr}: {e}")
        continue
    out = {
        'Ticker': tkr,
        'risk_gate_status_v2': res.get('risk_gate_status_v2'),
        'risk_gate_reason_v2': res.get('risk_gate_reason_v2'),
        'reward_risk_v2': res.get('rr_ratio_v2') or res.get('rr_ratio'),
        'conviction_v2_base': res.get('conviction_v2_base'),
        'conviction_v2_final': res.get('conviction_v2_final'),
        'buy_amount_v2': res.get('buy_amount_v2'),
        'shares_to_buy_v2': res.get('shares_to_buy_v2'),
        'reliability_score_v2': res.get('reliability_v2') or res.get('reliability_score_v2'),
        'fund_sources_used_v2': json.dumps(res.get('fund_sources_used_v2', []), ensure_ascii=False),
        'price_sources_used_v2': json.dumps(res.get('price_sources_used_v2', []), ensure_ascii=False),
        'fund_disagreement_score_v2': res.get('fund_disagreement_score_v2'),
        'price_variance_score_v2': res.get('price_variance_score_v2')
    }
    rows.append(out)
    if i % 25 == 0:
        print(f"Scored {i}/{len(tickers)}")

if not rows:
    print("No results to write")
    sys.exit(1)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"Wrote {OUT.resolve()} ({len(df)} rows)")
print(df['risk_gate_status_v2'].value_counts(dropna=False))
print('Done')
