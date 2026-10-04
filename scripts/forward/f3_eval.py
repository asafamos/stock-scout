"""Evaluate prereg F3 (arms D 20d, H 5d) next to F2R's A and B (20d). Run ONLY after all pick files are committed."""
import json, os, urllib.parse, urllib.request
import numpy as np, pandas as pd
from dotenv import load_dotenv
load_dotenv("/Users/asafamos/StockScout/stock-scout-2/.env")
KEY = os.environ["FMP_API_KEY"]; RD = "/Users/asafamos/StockScout/research_data"; D = "data/forward_llm_retro"
rng = np.random.default_rng(777)
ROUNDS = ["2026-07-05","2026-07-12","2026-07-19","2026-07-26","2026-08-02","2026-08-09","2026-08-16","2026-08-23","2026-08-30"]
ARMS = [("A", 20), ("B", 20), ("D", 20), ("H", 5)]
pools = {d: json.load(open(f"{D}/round_{d}_pool.json"))["pool"] for d in ROUNDS}
names = sorted({r["ticker"] for p in pools.values() for r in p}); px = pd.read_pickle(f"{RD}/univ_px.pkl")
OPEN = pd.DataFrame({s: px[s]["Open"] for s in names if s in px}); LASTP = OPEN.index.max()
def fmp(sym, a, b):
    try:
        with urllib.request.urlopen(f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={urllib.parse.quote(sym)}&from={a}&to={b}&apikey={KEY}", timeout=40) as r:
            d = json.loads(r.read())
        return pd.Series({pd.Timestamp(x["date"]): float(x["open"]) for x in d})
    except Exception: return None
ext = {}
for s in names + ["SPY"]:
    if s == "SPY" or (s in OPEN.columns and OPEN[s].dropna().index.max() >= LASTP - pd.Timedelta(days=3)):
        x = fmp(s, "2026-06-25" if s == "SPY" else (LASTP - pd.Timedelta(days=5)).date().isoformat(), "2026-10-02")
        if x is not None and len(x): ext[s] = x
EXT = pd.DataFrame(ext); cal = pd.DatetimeIndex(sorted(set(OPEN.index) | set(EXT.index)))
ALL = OPEN.reindex(cal)
for s in EXT.columns:
    if s in ALL.columns: ALL[s] = ALL[s].combine_first(EXT[s].reindex(cal))
ALL = ALL.drop(columns=[c for c in ["SPY"] if c in ALL.columns]); SPY = EXT["SPY"].reindex(cal)
out = {}
for arm, H in ARMS:
    exs, nulls, rows = [], [], []
    for d in ROUNDS:
        rd = pd.Timestamp(d); ei = cal.searchsorted(rd, side="right"); xi = ei + H
        ret = (ALL.iloc[ei:xi + 1].ffill().iloc[-1] / ALL.iloc[ei] - 1)
        valid = [r["ticker"] for r in pools[d] if r["ticker"] in ret.index and np.isfinite(ret[r["ticker"]])]
        pm = float(ret[valid].mean()); arr = ret[valid].values
        picks = [p["ticker"] for p in json.load(open(f"{D}/round_{d}_picks_{arm}.json"))["picks"]]; got = [t for t in picks if t in valid]
        idx = np.stack([rng.choice(len(arr), 5, replace=False) for _ in range(10000)])
        nulls.append(arr[idx].mean(axis=1) - pm); ex = float(ret[got].mean()) - pm; exs.append(ex)
        rows.append((d, float(ret[got].mean()), pm, ex, sum(ret[t] > pm for t in got)))
    ex = np.array(exs); obs = ex.mean(); p = float((np.mean(np.stack(nulls), axis=0) >= obs).mean()); se = ex.std(ddof=1) / np.sqrt(9)
    out[arm] = (H, obs, se, int((ex > 0).sum()), p, sum(r[4] for r in rows), np.mean([r[1] for r in rows]), np.mean([r[2] for r in rows]))
print("arm  horizon  mean excess vs pool   SE     rounds+   perm p(1-sided)  hit-rate  picks ret / pool ret")
for arm, (H, obs, se, pos, p, hits, pr, pm) in out.items():
    print(f" {arm}    {H:>2}d     {obs*100:+6.2f}%          {se*100:4.2f}%   {pos}/9      {p:.3f}           {hits}/45    {pr*100:+.2f}% / {pm*100:+.2f}%")
ov = {a: [set(p["ticker"] for p in json.load(open(f"{D}/round_{d}_picks_{a}.json"))["picks"]) for d in ROUNDS] for a in "ABDH"}
print("\npick overlap with arm A (of 45):", {a: sum(len(x & y) for x, y in zip(ov[a], ov["A"])) for a in "BDH"})
