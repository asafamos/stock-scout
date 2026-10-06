"""Resolve prereg F2 rounds: 20-trading-day excess of each arm's 5 picks vs the logged pool (plus random-5 permutation, SPY, naive-momentum baseline).

  python scripts/forward/f2_resolve.py                 # resolved rounds only
  python scripts/forward/f2_resolve.py --interim       # also show unresolved rounds marked to the last available open (NOT for decisions)
  python scripts/forward/f2_resolve.py --dir DIR       # alternative folder of round_<date>_picks.json (used for self-tests)

Entry = `entry_open_date` if present in the picks file, else the first trading day AFTER the round date. Exit = Open 20 trading days later.
No verdict is printed before 20 resolved rounds (prereg: evaluate ONCE at 26 rounds, minimum 20). FMP key from .env (never printed)."""
import argparse, glob, json, os, sys, urllib.parse, urllib.request
import numpy as np, pandas as pd
from dotenv import load_dotenv
load_dotenv("/Users/asafamos/StockScout/stock-scout-2/.env")
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="data/forward_llm"); ap.add_argument("--interim", action="store_true")
args = ap.parse_args()
KEY = os.environ["FMP_API_KEY"]; H = 20; rng = np.random.default_rng(2026)
files = sorted(glob.glob(f"{args.dir}/round_*_picks.json"))
if not files: sys.exit("no picks files")


def fmp(sym, a, b):
    try:
        with urllib.request.urlopen(f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={urllib.parse.quote(sym)}&from={a}&to={b}&apikey={KEY}", timeout=40) as r:
            d = json.loads(r.read())
        return pd.Series({pd.Timestamp(x["date"]): float(x["open"]) for x in d}).sort_index()
    except Exception:
        return None


rounds = [json.load(open(f)) for f in files]
first = min(pd.Timestamp(r["round_date"]) for r in rounds) - pd.Timedelta(days=3)
today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
spy = fmp("SPY", first.date().isoformat(), today.date().isoformat())
cal = spy.index; arms_all = sorted({a for r in rounds for a in r["arms"]})
rows, res, nulls, mom_ex = [], {a: [] for a in arms_all}, {a: [] for a in arms_all}, []
for r in rounds:
    d = r["round_date"]; pool = json.load(open(f"{args.dir}/{r['pool_file']}"))["pool"]
    ei = cal.searchsorted(pd.Timestamp(r["entry_open_date"])) if r.get("entry_open_date") else cal.searchsorted(pd.Timestamp(d), side="right")
    if ei >= len(cal):
        print(f"{d}: entry not reached yet"); continue
    xi = ei + H; resolved = xi < len(cal)
    if not resolved and not args.interim:
        print(f"{d}: unresolved (entry {cal[ei].date()}, exit {H} trading days later not reached)"); continue
    xi = min(xi, len(cal) - 1)
    ret = {}
    for p in pool:
        s = fmp(p["ticker"], cal[ei].date().isoformat(), cal[xi].date().isoformat())
        if s is not None and len(s) and s.index[0] == cal[ei]: ret[p["ticker"]] = float(s.iloc[-1] / s.iloc[0] - 1)   # last available open (delisted -> last open)
    tick = list(ret); arr = np.array([ret[t] for t in tick]); pm = float(arr.mean())
    idx = np.stack([rng.choice(len(arr), 5, replace=False) for _ in range(10000)]); nu = arr[idx].mean(axis=1) - pm
    row = {"round": d, "entry": str(cal[ei].date()), "exit": str(cal[xi].date()) + ("" if resolved else " (INTERIM)"), "n": len(tick), "pool": pm,
           "spy": float(spy.iloc[xi] / spy.iloc[ei] - 1)}
    for a, v in r["arms"].items():
        got = [p["ticker"] for p in v["picks"] if p["ticker"] in ret]; ex = float(np.mean([ret[t] for t in got])) - pm
        row[a] = ex
        if resolved: res[a].append(ex); nulls[a].append(nu)
    meta = {p["ticker"]: p for p in pool}
    top = sorted(tick, key=lambda t: -((meta[t]["r12m"] or 0) - (meta[t]["r1m"] or 0)))[:5]
    row["naive_mom"] = float(np.mean([ret[t] for t in top]) - pm)
    if resolved: mom_ex.append(row["naive_mom"])
    rows.append(row)
df = pd.DataFrame(rows)
if df.empty: sys.exit("nothing to show yet")
pd.set_option("display.width", 200)
print("\nper-round excess vs pool (20 trading days):"); print(df.round(4).to_string(index=False))
nres = len(res[arms_all[0]]) if res[arms_all[0]] else 0
print(f"\nresolved rounds: {nres}")
for a in arms_all:
    ex = np.array(res[a])
    if len(ex) < 2: continue
    obs = ex.mean(); p = float((np.mean(np.stack(nulls[a][:len(ex)]), axis=0) >= obs).mean()) if len(nulls[a]) == len(ex) else float("nan")
    print(f"  {a:12s} n={len(ex):2d} mean excess {obs*100:+.2f}% (SE {ex.std(ddof=1)/np.sqrt(len(ex))*100:.2f}%) positive {int((ex>0).sum())}/{len(ex)}  permutation p (1-sided) {p:.3f}")
if mom_ex: print(f"  naive 12-1 momentum top-5 baseline: mean excess {np.mean(mom_ex)*100:+.2f}%")
print("\nVERDICT:", "evaluate per prereg (needs a human read: p<0.0125/0.025 AND excess>=+1.0% AND >=6/9 style rule)" if nres >= 20 else f"none — only {nres} resolved rounds; the prereg requires >= 20 (evaluated once at 26). Numbers above are descriptive.")
