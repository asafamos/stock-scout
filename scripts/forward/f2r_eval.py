"""Evaluate prereg F2R (retro post-cutoff rounds). Run ONLY after all 18 pick files are committed.
Entry = Open of next trading day after the (Sunday) round date; exit = Open 20 trading days later. Controls: pool mean, 10,000 random-5, SPY."""
import glob, json, os, sys, urllib.parse, urllib.request
import numpy as np, pandas as pd
from dotenv import load_dotenv
load_dotenv("/Users/asafamos/StockScout/stock-scout-2/.env")
KEY = os.environ["FMP_API_KEY"]; RD = "/Users/asafamos/StockScout/research_data"; D = "data/forward_llm_retro"
H = 20; rng = np.random.default_rng(12345)
ROUNDS = ["2026-07-05","2026-07-12","2026-07-19","2026-07-26","2026-08-02","2026-08-09","2026-08-16","2026-08-23","2026-08-30"]
pools = {d: json.load(open(f"{D}/round_{d}_pool.json"))["pool"] for d in ROUNDS}
names = sorted({r["ticker"] for p in pools.values() for r in p})
px = pd.read_pickle(f"{RD}/univ_px.pkl")
OPEN = pd.DataFrame({s: px[s]["Open"] for s in names if s in px})
LASTP = OPEN.index.max()


def fmp(sym, a, b):
    try:
        with urllib.request.urlopen(f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={urllib.parse.quote(sym)}&from={a}&to={b}&apikey={KEY}", timeout=40) as r:
            d = json.loads(r.read())
        return pd.Series({pd.Timestamp(x["date"]): float(x["open"]) for x in d})
    except Exception:
        return None


# extend with FMP opens after the panel's last day (names in the last rounds), plus SPY
ext = {}
for s in names + ["SPY"]:
    need = s == "SPY" or (s in OPEN.columns and OPEN[s].dropna().index.max() >= LASTP - pd.Timedelta(days=3))
    if need:
        x = fmp(s, "2026-06-25" if s == "SPY" else (LASTP - pd.Timedelta(days=5)).date().isoformat(), "2026-10-02")
        if x is not None and len(x): ext[s] = x
EXT = pd.DataFrame(ext)
spy_open = pd.concat([pd.Series(px["SPY"]["Open"]) if "SPY" in px else pd.Series(dtype=float), EXT["SPY"]]) if "SPY" in EXT else None
cal = sorted(set(OPEN.index) | set(EXT.index)); cal = pd.DatetimeIndex(cal)
ALL = OPEN.reindex(cal)
for s in EXT.columns:
    if s in ALL.columns: ALL[s] = ALL[s].combine_first(EXT[s].reindex(cal))
ALL = ALL.drop(columns=[c for c in ["SPY"] if c in ALL.columns])
SPY = EXT["SPY"].reindex(cal)
if "SPY" in px: SPY = SPY.combine_first(px["SPY"]["Open"].reindex(cal))
print("price calendar:", cal[0].date(), "->", cal[-1].date(), " names:", ALL.shape[1])

res = {"A": [], "B": []}; per_round = []
momentum_ex, null_draws = [], {"A": [], "B": []}
for d in ROUNDS:
    rd = pd.Timestamp(d); ei = cal.searchsorted(rd, side="right"); xi = ei + H
    assert xi < len(cal), f"{d}: exit beyond data ({cal[-1].date()})"
    entry_open = ALL.iloc[ei]; ff = ALL.iloc[ei:xi + 1].ffill().iloc[-1]
    ret = (ff / entry_open - 1)
    pool = pools[d]; tick = [r["ticker"] for r in pool]
    valid = [t for t in tick if t in ret.index and np.isfinite(ret[t])]
    pm = float(ret[valid].mean()); spy = float(SPY.iloc[xi] / SPY.iloc[ei] - 1)
    meta = {r["ticker"]: r for r in pool}
    row = {"round": d, "entry": str(cal[ei].date()), "exit": str(cal[xi].date()), "valid": len(valid), "pool_mean": pm, "spy": spy}
    # random-5 null for this round (shared draws across arms)
    arr = ret[valid].values
    idx = np.stack([rng.choice(len(arr), 5, replace=False) for _ in range(10000)])
    null_ex = arr[idx].mean(axis=1) - pm
    for arm in ("A", "B"):
        picks = [p["ticker"] for p in json.load(open(f"{D}/round_{d}_picks_{arm}.json"))["picks"]]
        got = [t for t in picks if t in valid]
        r5 = float(ret[got].mean()); ex = r5 - pm
        row[f"{arm}_ret"] = r5; row[f"{arm}_ex"] = ex; row[f"{arm}_n"] = len(got)
        row[f"{arm}_beats_pool"] = int(sum(ret[t] > pm for t in got)); res[arm].append(ex); null_draws[arm].append(null_ex)
        bp, bpool = np.nanmean([meta[t]["beta"] or np.nan for t in got]), np.nanmean([meta[t]["beta"] or np.nan for t in valid])
        row[f"{arm}_beta_adj_ex"] = ex - (bp - bpool) * spy
        row[f"{arm}_mcap_b"] = float(np.mean([meta[t]["mcap_b"] for t in got]))
    # pure-momentum and pure near-52w-high baselines inside the same pool (diagnostic)
    mom = sorted(valid, key=lambda t: -((meta[t]["r12m"] or 0) - (meta[t]["r1m"] or 0)))[:5]
    hi = sorted(valid, key=lambda t: -(meta[t]["below_52w_high"]))[:5]
    row["mom_ex"] = float(ret[mom].mean() - pm); row["hi52_ex"] = float(ret[hi].mean() - pm)
    row["pool_mcap_b"] = float(np.mean([meta[t]["mcap_b"] for t in valid]))
    per_round.append(row)

df = pd.DataFrame(per_round)
pd.set_option("display.width", 220)
print("\nper round (20-day returns, entry Mon open -> +20 trading days open):")
print(df[["round", "entry", "exit", "valid", "pool_mean", "spy", "A_ret", "A_ex", "B_ret", "B_ex", "mom_ex", "hi52_ex"]].round(4).to_string(index=False))
print()
for arm, nm in (("A", "A numbers"), ("B", "B news-only")):
    ex = np.array(res[arm]); obs = ex.mean()
    nulls = np.mean(np.stack(null_draws[arm]), axis=0)          # per-draw mean over rounds
    p = float((nulls >= obs).mean())
    se = ex.std(ddof=1) / np.sqrt(len(ex))
    print(f"[{nm}] mean excess vs pool = {obs*100:+.2f}% (SE {se*100:.2f}%)  positive in {int((ex>0).sum())}/9 rounds  permutation p (one-sided) = {p:.3f}  "
          f"| mean picks' ret {df[arm+'_ret'].mean()*100:+.2f}% vs pool {df.pool_mean.mean()*100:+.2f}% vs SPY {df.spy.mean()*100:+.2f}%")
    print(f"      hit-rate (picks beating pool mean): {df[arm+'_beats_pool'].sum()}/45;  beta-adjusted excess {df[arm+'_beta_adj_ex'].mean()*100:+.2f}%;  "
          f"picks mean mcap ${df[arm+'_mcap_b'].mean():.1f}B vs pool ${df.pool_mcap_b.mean():.1f}B")
print(f"\n[diagnostic baselines in the same pools] top-5 by (12m-1m) momentum: {df.mom_ex.mean()*100:+.2f}% ; top-5 nearest 52w-high: {df.hi52_ex.mean()*100:+.2f}%")
df.to_csv(f"{D}/f2r_results.csv", index=False)
