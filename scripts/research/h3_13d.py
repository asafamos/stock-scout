"""H3 analysis — initial Schedule 13D filings, post-filing excess return at 20 and 60 trading days.
Implements docs/research_prereg_H3_13d_activist.md (with amendment A1: recent holdout evaluated once).

  python scripts/research/h3_13d.py dev
  python scripts/research/h3_13d.py holdout --confirm-holdout-once
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

RD = "/Users/asafamos/StockScout/research_data"
PERIODS = {"dev": ("2018-06-01", "2021-12-31"), "holdout": ("2022-01-01", "2026-09-15")}
HORIZONS = (20, 60)
mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
assert mode in PERIODS
if mode == "holdout":
    assert "--confirm-holdout-once" in sys.argv, "holdout is evaluated ONCE per the prereg"
lo, hi = map(pd.Timestamp, PERIODS[mode])

px = pd.read_pickle(f"{RD}/univ_px.pkl"); mc = pd.read_pickle(f"{RD}/univ_mcap.pkl")
cal = pd.DatetimeIndex(sorted(set().union(*[set(v.index) for k, v in list(px.items())[:400]])))
cal = cal[cal >= "2018-06-01"]
OPEN = pd.DataFrame({k: v["Open"] for k, v in px.items()}).reindex(cal)
CLOSE = pd.DataFrame({k: v["Close"] for k, v in px.items()}).reindex(cal)
DVOL = pd.DataFrame({k: v["Close"] * v["Volume"] for k, v in px.items()}).reindex(cal).rolling(20, min_periods=15).mean()
MCAP = pd.DataFrame({k: v.set_index(pd.to_datetime(v["date"]))["marketCap"] for k, v in mc.items()}).reindex(cal).ffill(limit=10)
U1 = ((CLOSE >= 5) & (DVOL >= 1e6) & (MCAP >= 3e8)).shift(1, fill_value=False)
FWD, UM = {}, {}
for h in HORIZONS:
    FWD[h] = OPEN.ffill(limit=h).shift(-h) / OPEN - 1
    UM[h] = FWD[h].where(U1).mean(axis=1)

# ---- events: classification rule per prereg AMENDMENT A2
import re
D_RE = re.compile(r"(?<!\d)13d")
rows13 = []
n_rows = n_g = 0
for f in glob.glob(f"{RD}/ownership13/*.json"):
    s = os.path.basename(f)[:-5]
    if s not in OPEN.columns: continue
    for r in json.load(open(f)):
        n_rows += 1
        u = (r.get("url") or "").lower()
        if D_RE.search(u):
            rows13.append((s, r["filingDate"], re.sub(r"[^a-z0-9]", "", (r.get("nameOfReportingPerson") or "").lower()), r.get("percentOfClass")))
        elif "13g" in u: n_g += 1
R = pd.DataFrame(rows13, columns=["sym", "date", "person", "pct"]); R["date"] = pd.to_datetime(R["date"])
R = R.sort_values(["sym", "person", "date"])
first = R.groupby(["sym", "person"], as_index=False).head(1)         # first 13D-type row ever for (stock, person)
ev = [(r.sym, r.date, r.pct) for r in first.itertuples()]
print(f"rows {n_rows:,}: 13D-type {len(R):,}  13G {n_g:,}  first-ever per (stock,person) {len(first):,}")
E = pd.DataFrame(ev, columns=["sym", "date", "pct"]); E["date"] = pd.to_datetime(E["date"])
E = E.drop_duplicates(["sym", "date"])
E["ei"] = cal.searchsorted(E["date"].values, side="right")
E = E[E.ei < len(cal) - max(HORIZONS) - 1]
E["edate"] = cal[E.ei.values]
E = E[(E.edate >= lo) & (E.edate <= hi)].sort_values(["sym", "ei"])
keep, last = [], {}
for i, r in zip(E.index, E.itertuples()):
    if r.sym in last and r.ei - last[r.sym] <= 60: continue
    last[r.sym] = r.ei; keep.append(i)
E = E.loc[keep]
print(f"mode={mode}  13D-initial events after dedupe {len(E):,}  entry {E.edate.min().date()} -> {E.edate.max().date()}")


def cost(m):
    return np.where(m >= 1e10, 0.0005, np.where(m >= 2e9, 0.0010, 0.0025)) + 0.0026 + 0.0015


def nw_t(x, lags=3):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m
    v = (e @ e) / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return m / np.sqrt(v / n)


for h in HORIZONS:
    d = E.copy()
    cols = [OPEN.columns.get_loc(s) for s in d.sym]
    ok = np.array([bool(U1.iat[i, c]) and np.isfinite(FWD[h].iat[i, c]) for i, c in zip(d.ei, cols)])
    d, cols = d[ok], [c for c, k in zip(cols, ok) if k]
    d["ret"] = [FWD[h].iat[i, c] for i, c in zip(d.ei, cols)]
    d["ex"] = d.ret.values - UM[h].values[d.ei.values]
    d["mcap"] = [MCAP.iat[i - 1, c] for i, c in zip(d.ei, cols)]
    d["net"] = d.ex - cost(d.mcap.values)
    d["month"] = d.edate.dt.strftime("%Y-%m")
    g = d.groupby("month").agg(ex=("ex", "mean"), net=("net", "mean"), n=("ex", "count"), d0=("edate", "min")).sort_values("d0")
    m, se = g.ex.mean(), g.ex.std() / np.sqrt(len(g))
    mn, sen = g.net.mean(), g.net.std() / np.sqrt(len(g))
    print(f"\n[13D initial, U1, h={h}d] events={len(d):,} months={len(g)}  mean excess {d.ex.mean()*100:+.2f}% (median {d.ex.median()*100:+.2f}%)  "
          f"month-cluster mean {m*100:+.2f}%  t={m/se:+.2f} NW-t={nw_t(g.ex):+.2f}")
    print(f"   net of costs {mn*100:+.2f}%  [95% CI {(mn-1.96*sen)*100:+.2f}, {(mn+1.96*sen)*100:+.2f}]   events/month {len(d)/len(g):.1f}")
    yr = g.groupby(g.d0.dt.year).ex.agg(["mean", "count"]); ne = d.groupby(d.edate.dt.year).size()
    print("   by year:", "  ".join(f"{y}:{v['mean']*100:+.2f}%(n={int(ne.get(y,0))})" for y, v in yr.iterrows()))
    mid = g.d0.median()
    print(f"   halves: first {g[g.d0<=mid].ex.mean()*100:+.2f}%  second {g[g.d0>mid].ex.mean()*100:+.2f}%")
print("\nK=2 tests. Prereg bar on the HOLDOUT: mean excess > 0 and |t| >= 3.0, positive in both halves and >=70% of years, net CI lower bound > 0, >=4 events/month.")
