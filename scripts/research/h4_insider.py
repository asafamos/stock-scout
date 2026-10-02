"""H4 analysis — insider open-market purchase clusters (E1) and large purchases (E2), 60-trading-day excess return.
Implements docs/research_prereg_H4_insider_purchases.md (amendment A1: recent holdout evaluated once).

  python scripts/research/h4_insider.py dev
  python scripts/research/h4_insider.py holdout --confirm-holdout-once
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

RD = "/Users/asafamos/StockScout/research_data"
PERIODS = {"dev": ("2018-06-01", "2021-12-31"), "holdout": ("2022-01-01", "2026-09-15")}
H = 60
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
FWD = OPEN.ffill(limit=H).shift(-H) / OPEN - 1
UM = FWD.where(U1).mean(axis=1)

# ---- valid purchase rows (prereg data rules)
rows, n_raw = [], 0
for f in glob.glob(f"{RD}/insider_buys/*.json"):
    s = os.path.basename(f)[:-5]
    if s not in OPEN.columns: continue
    for r in json.load(open(f)):
        n_raw += 1
        try:
            if r.get("transactionType") != "P-Purchase": continue
            if not str(r.get("formType", "")).startswith("4"): continue
            px_, sh = float(r.get("price") or 0), float(r.get("securitiesTransacted") or 0)
            if px_ < 1 or sh <= 0: continue
            tx, fl = pd.Timestamp(r["transactionDate"]), pd.Timestamp(r["filingDate"])
            if not (0 <= (fl - tx).days <= 90): continue
            who = (r.get("typeOfOwner") or "").lower()
            if "officer" not in who and "director" not in who: continue
            rows.append((s, r["reportingCik"], tx, fl, px_ * sh))
        except Exception:
            continue
P = pd.DataFrame(rows, columns=["sym", "cik", "tx", "fl", "val"])
print(f"raw rows {n_raw:,}  valid purchases {len(P):,}  symbols {P.sym.nunique():,}")
A = P.groupby(["sym", "cik", "tx"], as_index=False).agg(val=("val", "sum"), fl=("fl", "max"))   # person-day

ev1, ev2 = [], []
for s, g in A.groupby("sym"):
    g = g.sort_values("tx")
    big = g[g.val >= 10_000]
    for r in big.itertuples():
        w = big[(big.tx >= r.tx - pd.Timedelta(days=30)) & (big.tx <= r.tx) & (big.cik != r.cik)]
        if len(w):
            ev1.append((s, max(r.fl, w.fl.min())))      # earliest date at which a cluster of >=2 distinct insiders is public
    for r in g[g.val >= 250_000].itertuples():
        ev2.append((s, r.fl))


def build(ev):
    E = pd.DataFrame(ev, columns=["sym", "date"]).drop_duplicates()
    E["ei"] = cal.searchsorted(E["date"].values, side="right")       # first trading day strictly after the filing date
    E = E[E.ei < len(cal) - H - 1]
    E["edate"] = cal[E.ei.values]
    E = E[(E.edate >= lo) & (E.edate <= hi)].sort_values(["sym", "ei"])
    keep, last = [], {}
    for i, r in zip(E.index, E.itertuples()):
        if r.sym in last and r.ei - last[r.sym] <= 60: continue
        last[r.sym] = r.ei; keep.append(i)
    return E.loc[keep]


def cost(m):
    return np.where(m >= 1e10, 0.0005, np.where(m >= 2e9, 0.0010, 0.0025)) + 0.0026 + 0.0015


def nw_t(x, lags=3):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m
    v = (e @ e) / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return m / np.sqrt(v / n)


print(f"mode={mode}")
for name, ev in (("E1 cluster", ev1), ("E2 large", ev2)):
    E = build(ev)
    cols = [OPEN.columns.get_loc(s) for s in E.sym]
    ok = np.array([bool(U1.iat[i, c]) and np.isfinite(FWD.iat[i, c]) for i, c in zip(E.ei, cols)])
    d, cols = E[ok].copy(), [c for c, k in zip(cols, ok) if k]
    d["ret"] = [FWD.iat[i, c] for i, c in zip(d.ei, cols)]
    d["ex"] = d.ret.values - UM.values[d.ei.values]
    d["mcap"] = [MCAP.iat[i - 1, c] for i, c in zip(d.ei, cols)]
    d["net"] = d.ex - cost(d.mcap.values)
    d["month"] = d.edate.dt.strftime("%Y-%m")
    g = d.groupby("month").agg(ex=("ex", "mean"), net=("net", "mean"), n=("ex", "count"), d0=("edate", "min")).sort_values("d0")
    m, se = g.ex.mean(), g.ex.std() / np.sqrt(len(g))
    mn, sen = g.net.mean(), g.net.std() / np.sqrt(len(g))
    print(f"\n[{name}, U1, h=60d] events(raw {len(E):,}, in U1 {len(d):,}) months={len(g)}  mean excess {d.ex.mean()*100:+.2f}% "
          f"(median {d.ex.median()*100:+.2f}%)  month-cluster {m*100:+.2f}%  t={m/se:+.2f} NW-t={nw_t(g.ex):+.2f}")
    print(f"   net of costs {mn*100:+.2f}% [95% CI {(mn-1.96*sen)*100:+.2f}, {(mn+1.96*sen)*100:+.2f}]   events/month {len(d)/len(g):.1f}")
    yr = g.groupby(g.d0.dt.year).ex.agg(["mean", "count"]); ne = d.groupby(d.edate.dt.year).size()
    print("   by year:", "  ".join(f"{y}:{v['mean']*100:+.2f}%(n={int(ne.get(y,0))})" for y, v in yr.iterrows()))
    mid = g.d0.median()
    print(f"   halves: first {g[g.d0<=mid].ex.mean()*100:+.2f}%  second {g[g.d0>mid].ex.mean()*100:+.2f}%")
    sm, lg = d[d.mcap < 2e9], d[d.mcap >= 2e9]
    print(f"   secondary (not a test): $0.3-2B n={len(sm)} mean {sm.ex.mean()*100:+.2f}%  |  >=$2B n={len(lg)} mean {lg.ex.mean()*100:+.2f}%")
print("\nK=2 tests (E1,E2). Prereg bar on the HOLDOUT: mean>0, |t|>=3.0, positive in both halves and >=70% of years, net CI lower bound>0, >=4 events/month.")
