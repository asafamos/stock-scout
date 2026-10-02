"""H2 analysis — analyst upgrades/downgrades, 10-day post-event excess return. Implements docs/research_prereg_H2_analyst_grades.md
(+ amendment A1: the recent holdout is always evaluated, once).

  python scripts/research/h2_analyst_grades.py dev
  python scripts/research/h2_analyst_grades.py holdout --confirm-holdout-once
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

RD = "/Users/asafamos/StockScout/research_data"
PERIODS = {"dev": ("2018-06-01", "2021-12-31"), "holdout": ("2022-01-01", "2026-09-15")}
HOLD = 10
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
O_ff = OPEN.ffill(limit=HOLD)                       # delisted before exit -> last available Open (prereg)
FWD = O_ff.shift(-HOLD) / OPEN - 1                  # entry at row i Open -> Open 10 days later
U1 = ((CLOSE >= 5) & (DVOL >= 1e6) & (MCAP >= 3e8)).shift(1, fill_value=False)   # decided with PRIOR-day data
U2 = U1 & (MCAP.shift(1) >= 2e9)
UM = {"U1": FWD.where(U1).mean(axis=1), "U2": FWD.where(U2).mean(axis=1)}

ev = []
for f in glob.glob(f"{RD}/grades/*.json"):
    s = os.path.basename(f)[:-5]
    if s not in OPEN.columns: continue
    for r in json.load(open(f)):
        if r.get("action") in ("upgrade", "downgrade"):
            ev.append((s, r["date"], 1 if r["action"] == "upgrade" else -1))
E = pd.DataFrame(ev, columns=["sym", "date", "v"]); E["date"] = pd.to_datetime(E["date"])
print(f"raw upgrade/downgrade rows {len(E):,} over {E.sym.nunique()} symbols")
E = E.groupby(["sym", "date"], as_index=False)["v"].sum()
E = E[E.v != 0]; E["dir"] = np.where(E.v > 0, "Up", "Down")
E["ei"] = cal.searchsorted(E["date"].values, side="right")      # first trading day strictly AFTER the event date = entry
E = E[(E.ei < len(cal) - HOLD - 1)]
E["edate"] = cal[E.ei.values]
E = E[(E.edate >= lo) & (E.edate <= hi)].sort_values(["sym", "dir", "ei"])
# drop same-direction repeats within 10 trading days for the same stock
keep = []
last = {}
for i, r in zip(E.index, E.itertuples()):
    k = (r.sym, r.dir)
    if k in last and r.ei - last[k] <= 10: continue
    last[k] = r.ei; keep.append(i)
E = E.loc[keep]
print(f"mode={mode}  events after netting/dedupe {len(E):,}  entry {E.edate.min().date()} -> {E.edate.max().date()}")


def cost(m):  # round trip, fraction
    spread = np.where(m >= 1e10, 0.0005, np.where(m >= 2e9, 0.0010, 0.0025))
    return spread + 0.0026 + 0.0015


def nw_t(x, lags=2):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m
    v = (e @ e) / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return m / np.sqrt(v / n)


def run(dirn, uni):
    U = {"U1": U1, "U2": U2}[uni]
    d = E[E["dir"] == dirn].copy()
    ok = [bool(U.iat[i, OPEN.columns.get_loc(s)]) and np.isfinite(FWD.iat[i, OPEN.columns.get_loc(s)]) for i, s in zip(d.ei, d.sym)]
    d = d[ok]
    d["ret"] = [FWD.iat[i, OPEN.columns.get_loc(s)] for i, s in zip(d.ei, d.sym)]
    d["ex"] = d.ret.values - UM[uni].values[d.ei.values]
    d["mcap"] = [MCAP.iat[i - 1, OPEN.columns.get_loc(s)] for i, s in zip(d.ei, d.sym)]
    d["net"] = d.ex - cost(d.mcap.values) if dirn == "Up" else -d.ex - cost(d.mcap.values)
    cnt = d.groupby("edate").ex.transform("count")
    d["key"] = np.where(cnt >= 3, d.edate.dt.strftime("%Y-%m-%d"), "M" + d.edate.dt.strftime("%Y-%m"))
    g = d.groupby("key").agg(ex=("ex", "mean"), net=("net", "mean"), n=("ex", "count"), d0=("edate", "min")).sort_values("d0")
    m, se = g.ex.mean(), g.ex.std() / np.sqrt(len(g))
    mn, sen = g.net.mean(), g.net.std() / np.sqrt(len(g))
    exp_sign = 1 if dirn == "Up" else -1
    print(f"\n[{dirn} {uni}] events={len(d):,} clusters={len(g)}  mean excess {m*100:+.3f}%/10d  t={m/se:+.2f} NW-t={nw_t(g.ex):+.2f}  "
          f"(expected sign {'+' if exp_sign>0 else '-'}); {'tradable-side net' if dirn=='Up' else 'short-side net'} {mn*100:+.3f}% "
          f"[95% CI {(mn-1.96*sen)*100:+.3f}, {(mn+1.96*sen)*100:+.3f}]")
    yr = g.groupby(g.d0.dt.year).ex.agg(["mean", "count"])
    print("   by year:", "  ".join(f"{y}:{v['mean']*100:+.2f}%(c={int(v['count'])})" for y, v in yr.iterrows()))
    mid = g.d0.median()
    print(f"   halves: first {g[g.d0<=mid].ex.mean()*100:+.3f}%  second {g[g.d0>mid].ex.mean()*100:+.3f}%")
    if dirn == "Up":
        weeks = max((d.edate.max() - d.edate.min()).days / 7, 1)
        print(f"   capacity: {len(d)/weeks:.1f} qualifying Up events / week")
    return g


for dirn in ("Up", "Down"):
    for uni in ("U1", "U2"):
        run(dirn, uni)
print("\nK=4 tests. Prereg bar on the HOLDOUT: right sign and |t|>=3.0 (Up, tradable form), same sign in both halves and >=70% of years, net CI lower bound > 0, capacity >= 3/week.")
