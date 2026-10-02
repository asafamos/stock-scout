"""Out-of-sample validation of the CoreTrend rule on long index history.

Rule (fixed BEFORE looking at this data, from beat_spy_search.py): hold the index while its
month-end close > SMA of the last N month-end closes (N=10), else hold cash. Decided at
month-end, executed from the next month (no look-ahead).

Why this is a real OOS test: the rule was chosen on QQQ/SPY data 2005-2026. Here we run it on the
Nasdaq Composite / S&P 500 price indexes BEFORE that period (1971-2004: 1973-74 bear, 1987 crash,
2000-02 bust). Price indexes only (no dividends) for BOTH the strategy and buy&hold, so the
comparison is like-for-like; cash earns 0% (conservative: understates the strategy).

N is NOT tuned: the sensitivity grid (6..12 months) is reported only to show robustness.
Costs: 0.1% per switch (round trip counted as two switches).
"""
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

KEY = os.environ.get("FMP_API_KEY") or sys.exit("FMP_API_KEY missing")
COST = 0.001


def fetch(sym, a, b):
    url = ("https://financialmodelingprep.com/stable/historical-price-eod/light?symbol=%s&from=%s&to=%s&apikey=%s"
           % (sym.replace("^", "%5E"), a, b, KEY))
    d = json.load(urllib.request.urlopen(url, timeout=60))
    return pd.Series({r["date"]: r["price"] for r in d})


def history(sym):
    parts = []
    # the API caps a response at 5000 rows (~20y): fetch in <=15y chunks
    for a, b in (("1960-01-01", "1974-12-31"), ("1975-01-01", "1989-12-31"), ("1990-01-01", "2004-12-31"),
                 ("2005-01-01", "2019-12-31"), ("2020-01-01", "2026-10-02")):
        try:
            parts.append(fetch(sym, a, b))
        except Exception as e:
            print("fetch fail", sym, a, e)
    s = pd.concat(parts).sort_index()
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated()]
    gaps = s.index.to_series().diff().dt.days
    assert gaps.max() < 10, "data gap of %d days at %s" % (gaps.max(), gaps.idxmax().date())
    return s


def run(daily, n=10, lo=None, hi=None):
    m = daily.resample("ME").last().dropna()
    sma = m.rolling(n).mean()
    on = (m > sma)                      # decided at month-end t
    ret = m.pct_change()                # return of month t
    pos = on.shift(1).fillna(False)     # held during month t = signal from end of t-1
    switches = pos.astype(int).diff().abs().fillna(0)
    strat = np.where(pos, ret, 0.0) - switches * COST
    strat = pd.Series(strat, index=m.index)
    bh = ret
    df = pd.DataFrame({"strat": strat, "bh": bh, "inmkt": pos.astype(float)}).dropna()
    if lo: df = df[df.index >= lo]
    if hi: df = df[df.index <= hi]
    return df


def stats(r):
    eq = (1 + r).cumprod()
    yrs = len(r) / 12
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    vol = r.std() * np.sqrt(12)
    return cagr, dd, vol


def report(name, daily, lo, hi, n=10):
    df = run(daily, n, lo, hi)
    cs, ds, vs = stats(df.strat)
    cb, db, vb = stats(df.bh)
    print(f"{name:26s} {lo[:4]}-{hi[:4]}  strat CAGR {cs*100:5.1f}% maxDD {ds*100:6.1f}% vol {vs*100:4.1f}% | "
          f"B&H CAGR {cb*100:5.1f}% maxDD {db*100:6.1f}% vol {vb*100:4.1f}% | invested {df.inmkt.mean()*100:3.0f}% "
          f"| CAGR/|DD| strat {cs/abs(ds):.2f} vs B&H {cb/abs(db):.2f}")
    return df


if __name__ == "__main__":
    for sym, nm in (("^IXIC", "Nasdaq Composite"), ("^GSPC", "S&P 500")):
        d = history(sym)
        print(f"\n=== {nm}: {d.index[0].date()} -> {d.index[-1].date()} ({len(d)} days)")
        for lo, hi in (("1971-01-01", "1998-12-31"), ("1971-01-01", "2004-12-31"), ("1999-01-01", "2004-12-31"),
                       ("2005-01-01", "2026-10-02"), ("1971-01-01", "2026-10-02")):
            report(nm, d, lo, hi)
        print("-- sensitivity of N (1971-2004, NOT used to select):")
        for n in (6, 8, 10, 12):
            report(f"  SMA{n}", d, "1971-01-01", "2004-12-31", n)
