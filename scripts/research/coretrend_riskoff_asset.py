"""CoreTrend risk-off asset test: QQQ 10m-trend, risk-off = IEF vs BIL (T-bill) vs 0% cash.

Pre-specified (3 variants only, nothing tuned): same signal, only the risk-off holding differs.
Total-return (dividend-adjusted) prices, 2007-2026, 0.05%/side cost on switches.
Reports full period, each calendar year, and the years where bonds hurt (2022).
"""
import json, os, sys, urllib.request
import numpy as np, pandas as pd

KEY = os.environ.get("FMP_API_KEY") or sys.exit("FMP_API_KEY missing")
COST = 0.0005


def adj(sym):
    u = ("https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted?symbol=%s"
         "&from=2007-01-01&to=2026-10-02&apikey=%s" % (sym, KEY))
    d = json.load(urllib.request.urlopen(u, timeout=60))
    s = pd.Series({r["date"]: r["adjClose"] for r in d}); s.index = pd.to_datetime(s.index)
    return s.sort_index()


px = pd.DataFrame({k: adj(k) for k in ("QQQ", "IEF", "BIL")}).dropna()
gaps = px.index.to_series().diff().dt.days
assert gaps.max() < 10, "data gap"
m = px.resample("ME").last()
r = m.pct_change()
on = (m["QQQ"] > m["QQQ"].rolling(10).mean())
pos = on.shift(1).fillna(False).astype(bool)            # decided at month-end t-1, earned in month t
sw = pos.astype(int).diff().abs().fillna(0)
cost = sw * COST * 2                        # sell one + buy other


def strat(off):
    off_ret = r[off] if off else 0.0
    return pd.Series(np.where(pos, r["QQQ"], off_ret) - cost, index=m.index)


V = {"risk-off = IEF (live rule)": strat("IEF"), "risk-off = BIL (T-bills)": strat("BIL"),
     "risk-off = 0% cash": strat(None), "QQQ buy&hold": r["QQQ"], "BIL only": r["BIL"], "IEF only": r["IEF"]}
df = pd.DataFrame(V).iloc[11:].dropna()


def st(x):
    eq = (1 + x).cumprod(); y = len(x) / 12
    return eq.iloc[-1] ** (1 / y) - 1, (eq / eq.cummax() - 1).min(), x.mean() / x.std() * np.sqrt(12)


print(f"{df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} months, {int(pos.reindex(df.index).sum())} risk-on)")
print(f"{'variant':30s} {'CAGR':>7s} {'maxDD':>8s} {'Sharpe':>7s}")
for c in df.columns:
    a, b, s = st(df[c]); print(f"{c:30s} {a*100:6.1f}% {b*100:7.1f}% {s:7.2f}")
print("\ncalendar-year returns %:")
yr = (1 + df).groupby(df.index.year).prod() - 1
print((yr[["risk-off = IEF (live rule)", "risk-off = BIL (T-bills)", "QQQ buy&hold", "IEF only"]] * 100).round(1).to_string())
off = ~pos.reindex(df.index).astype(bool)
print(f"\nrisk-off months: {int(off.sum())};  IEF beat BIL in {(r['IEF'].reindex(df.index)[off] > r['BIL'].reindex(df.index)[off]).sum()} of them; "
      f"mean IEF {r['IEF'].reindex(df.index)[off].mean()*100:.2f}%/mo vs BIL {r['BIL'].reindex(df.index)[off].mean()*100:.2f}%/mo")
