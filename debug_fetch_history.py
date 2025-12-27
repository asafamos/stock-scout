#!/usr/bin/env python
"""Debug fetch_history_bulk"""

import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

tickers = ["AAPL", "MSFT", "JPM"]
period_days = 90
ma_long = 50

print(f"[1] Parameters:")
print(f"    tickers: {tickers}")
print(f"    period_days: {period_days}")
print(f"    ma_long: {ma_long}")

end = datetime.utcnow()
start = end - timedelta(days=period_days + 50)
min_rows = ma_long + 40

print(f"\n[2] Time window:")
print(f"    Start: {start}")
print(f"    End: {end}")
print(f"    Min rows required: {min_rows}")

print(f"\n[3] Downloading data...")
try:
    df_all = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False, threads=True)
    print(f"    ✅ Downloaded, type: {type(df_all)}")
    print(f"    Shape: {df_all.shape if hasattr(df_all, 'shape') else 'N/A'}")
    print(f"    Columns: {df_all.columns.tolist() if hasattr(df_all, 'columns') else 'N/A'}")
except Exception as e:
    print(f"    ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print(f"\n[4] Processing tickers...")
data_map = {}
if len(tickers) == 1:
    print(f"    Single ticker mode")
    tkr = tickers[0]
    if not df_all.empty and len(df_all) >= min_rows:
        data_map[tkr] = df_all
        print(f"    ✅ Added {tkr}: {len(df_all)} rows")
    else:
        print(f"    ❌ {tkr}: empty or < {min_rows} rows (has {len(df_all)})")
else:
    print(f"    Multi-ticker mode")
    for tkr in tickers:
        try:
            df = df_all[tkr].dropna(how='all')
            print(f"    {tkr}: {len(df)} rows", end='')
            if len(df) >= min_rows:
                data_map[tkr] = df
                print(f" ✅ ADDED")
            else:
                print(f" ❌ (< {min_rows})")
        except KeyError:
            print(f"    {tkr}: ❌ KeyError (not found in df_all)")
        except Exception as e:
            print(f"    {tkr}: ❌ Exception: {e}")

print(f"\n[5] Result:")
print(f"    data_map has {len(data_map)} tickers: {list(data_map.keys())}")
