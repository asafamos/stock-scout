"""H6 — 12-1 momentum, long-only top decile of U1, monthly. Implements docs/research_prereg_H6_momentum.md.
  python scripts/research/h6_momentum.py dev
  python scripts/research/h6_momentum.py holdout --confirm-holdout-once"""
import sys
import numpy as np, pandas as pd

RD = "/Users/asafamos/StockScout/research_data"; HOLD = 21
PERIODS = {"dev": ("2019-06-01", "2021-12-31"), "holdout": ("2022-01-01", "2026-09-15")}
mode = sys.argv[1] if len(sys.argv) > 1 else "dev"; assert mode in PERIODS
if mode == "holdout": assert "--confirm-holdout-once" in sys.argv, "holdout is evaluated ONCE"
lo, hi = map(pd.Timestamp, PERIODS[mode])
px = pd.read_pickle(f"{RD}/univ_px.pkl"); mc = pd.read_pickle(f"{RD}/univ_mcap.pkl")
cal = pd.DatetimeIndex(sorted(set().union(*[set(v.index) for k, v in list(px.items())[:400]]))); cal = cal[cal >= "2018-06-01"]
OPEN = pd.DataFrame({k: v["Open"] for k, v in px.items()}).reindex(cal)
CLOSE = pd.DataFrame({k: v["Close"] for k, v in px.items()}).reindex(cal)
DVOL = pd.DataFrame({k: v["Close"] * v["Volume"] for k, v in px.items()}).reindex(cal).rolling(20, min_periods=15).mean()
MCAP = pd.DataFrame({k: v.set_index(pd.to_datetime(v["date"]))["marketCap"] for k, v in mc.items()}).reindex(cal).ffill(limit=10)
U1 = (CLOSE >= 5) & (DVOL >= 1e6) & (MCAP >= 3e8)
M = CLOSE.shift(21) / CLOSE.shift(252) - 1
FWD = OPEN.ffill(limit=HOLD).shift(-HOLD) / OPEN - 1            # row i: entry Open(i) -> Open(i+HOLD)
month_end = pd.Series(cal, index=cal).groupby(cal.to_period("M")).last().values
rows = []
for d in pd.DatetimeIndex(month_end):
    di = cal.get_loc(d)
    if not (lo <= d <= hi) or di < 252 or di + 1 + HOLD >= len(cal): continue
    ei = di + 1
    elig = U1.iloc[di] & M.iloc[di].notna() & FWD.iloc[ei].notna() & OPEN.iloc[ei].notna()
    names = elig.index[elig.values]
    if len(names) < 200: continue
    m = M.iloc[di][names]; r = FWD.iloc[ei][names]; cap = MCAP.iloc[di][names]
    allm = r.mean(); n10 = len(names) // 10; n20 = len(names) // 20
    order = m.sort_values(ascending=False).index
    top, top5, bot = order[:n10], order[:n20], order[-n10:]
    cost = lambda c: np.where(c >= 1e10, 0.0005, np.where(c >= 2e9, 0.0010, 0.0025)) + 0.0026 + 0.0015
    rows.append({"date": d, "n": len(names), "top": r[top].mean() - allm, "top5pct": r[top5].mean() - allm, "bottom": r[bot].mean() - allm,
                 "net": r[top].mean() - allm - float(np.mean(cost(cap[top].values))), "all": allm})
df = pd.DataFrame(rows).set_index("date"); print(f"mode={mode} months={len(df)} {df.index[0].date()} -> {df.index[-1].date()}  avg universe {df.n.mean():.0f}")
def nw(x, L=1):
    x = np.asarray(x, float); n = len(x); e = x - x.mean(); v = e @ e / n
    for l in range(1, L + 1): v += 2 * (1 - l / (L + 1)) * (e[l:] @ e[:-l]) / n
    return x.mean() / np.sqrt(v / n)
ex = df.top; se = ex.std() / np.sqrt(len(ex)); nse = df.net.std() / np.sqrt(len(df))
print(f"\n[P1 top-decile 12-1 momentum, long-only]  mean excess {ex.mean()*100:+.2f}%/month (median {ex.median()*100:+.2f}%)  t={ex.mean()/se:+.2f}  NW-t={nw(ex):+.2f}  months>0: {(ex>0).mean()*100:.0f}%")
print(f"   net of costs {df.net.mean()*100:+.2f}%  [95% CI {(df.net.mean()-1.96*nse)*100:+.2f}, {(df.net.mean()+1.96*nse)*100:+.2f}]")
by = ex.groupby(ex.index.year).agg(["mean", "count"]); print("   by year:", "  ".join(f"{y}:{v['mean']*100:+.2f}%(n={int(v['count'])})" for y, v in by.iterrows()))
mid = ex.index[len(ex) // 2]; print(f"   halves: first {ex[ex.index<=mid].mean()*100:+.2f}%  second {ex[ex.index>mid].mean()*100:+.2f}%   excluding best month: {ex.drop(ex.idxmax()).mean()*100:+.2f}%")
print(f"   secondary (not a test): top 5% {df.top5pct.mean()*100:+.2f}%  |  bottom decile {df.bottom.mean()*100:+.2f}%  |  top-bottom spread {(df.top-df.bottom).mean()*100:+.2f}%")
