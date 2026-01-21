#!/usr/bin/env python3
"""Legacy simple flow script (not collected by pytest)."""
from core.unified_logic import build_technical_indicators
import yfinance as yf
from datetime import datetime, timedelta

ticker = "AAPL"
end = datetime.utcnow()
start = end - timedelta(days=250)
df = yf.download(ticker, start=start, end=end, progress=False)
tech = build_technical_indicators(df)
print(f"Rows: {len(tech)}")
