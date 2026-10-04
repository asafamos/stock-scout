"""Point-in-time pool+pack for RETRO rounds (prereg F2R). Uses ONLY data dated <= the round date.
  python scripts/forward/f2r_build.py 2026-07-05 2026-07-12 ...
Writes data/forward_llm_retro/round_<d>_pool.json and _pack.txt. FMP key from .env (profile sector/industry/beta only)."""
import glob, hashlib, json, os, random, sys, time, urllib.parse, urllib.request
from datetime import date, datetime, timedelta
import numpy as np, pandas as pd
from dotenv import load_dotenv
load_dotenv("/Users/asafamos/StockScout/stock-scout-2/.env")
KEY = os.environ["FMP_API_KEY"]; RD = "/Users/asafamos/StockScout/research_data"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = f"{ROOT}/data/forward_llm_retro"; os.makedirs(OUT, exist_ok=True)
EXCLUDE = set("APH VG DELL PACS HNGE CGAU AAL FRSH FTNT TEO IVZ ARCB MRX STUB ARWR QQQM QQQ IEF SPY".split())
px = pd.read_pickle(f"{RD}/univ_px.pkl"); mc = pd.read_pickle(f"{RD}/univ_mcap.pkl")
MC = {k: v.assign(date=pd.to_datetime(v["date"])).set_index("date")["marketCap"] for k, v in mc.items()}
SI = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{RD}/short_interest/part_*.parquet"))], ignore_index=True)
SI["settlement_date"] = pd.to_datetime(SI["settlement_date"])
_prof = {}


def profile(s):
    if s in _prof: return _prof[s]
    for a in range(3):
        try:
            with urllib.request.urlopen(f"https://financialmodelingprep.com/stable/profile?symbol={urllib.parse.quote(s)}&apikey={KEY}", timeout=30) as r:
                d = json.loads(r.read())
            _prof[s] = d[0] if d else None; return _prof[s]
        except Exception:
            time.sleep(1.5 * (a + 1))
    _prof[s] = None; return None


def f(x, p=1): return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"


for ds in sys.argv[1:]:
    rd = date.fromisoformat(ds); R = pd.Timestamp(rd); seed = int(rd.strftime("%Y%m%d"))
    cutoff60, cutoff30 = (rd - timedelta(days=60)).isoformat(), (rd - timedelta(days=30)).isoformat()
    si_use = SI[SI.settlement_date <= R - pd.Timedelta(days=13)]
    si_last = si_use[si_use.settlement_date == si_use.settlement_date.max()].set_index("ticker")
    cands = []
    for s, v in px.items():
        if s in EXCLUDE or not s.isalpha(): continue
        h = v[v.index <= R]
        if len(h) < 260 or h.index[-1] < R - pd.Timedelta(days=7): continue          # alive at the round date, enough history
        c = h["Close"]; last = float(c.iloc[-1])
        adv = float((c * h["Volume"]).tail(20).mean())
        m = MC.get(s); mcap = float(m[m.index <= R].iloc[-1]) if m is not None and (m.index <= R).any() else 0.0
        if last >= 5 and adv >= 1e6 and mcap >= 3e8: cands.append((s, h, last, adv, mcap))
    cands.sort(key=lambda t: t[0]); random.Random(seed).shuffle(cands)
    pool = []
    for s, h, last, adv, mcap in cands:
        if len(pool) >= 100: break
        p = profile(s)
        if not p or not p.get("sector"): continue
        c = h["Close"].astype(float)
        ret = lambda n: float(last / c.iloc[-1 - n] - 1)
        row = {"ticker": s, "close": round(last, 2), "r1m": ret(21), "r3m": ret(63), "r6m": ret(126), "r12m": ret(252),
               "below_52w_high": float(last / c.tail(252).max() - 1), "vol20_ann": float(c.pct_change().tail(20).std() * np.sqrt(252)),
               "adv_usd_m": adv / 1e6, "mcap_b": mcap / 1e9, "sector": p.get("sector"), "industry": p.get("industry"), "beta": p.get("beta"),
               "last_date": str(h.index[-1].date())}
        if s in si_last.index and last > 0:
            sh = mcap / last
            row["short_int_pct"] = float(si_last.loc[s, "short_interest"] / sh * 100) if sh > 0 else None
            row["days_to_cover"] = float(si_last.loc[s, "days_to_cover"])
        try:
            g = json.load(open(f"{RD}/grades/{s}.json"))
            row["net_upgrades_60d"] = sum((1 if x["action"] == "upgrade" else -1) for x in g if x.get("action") in ("upgrade", "downgrade") and cutoff60 <= x["date"] <= rd.isoformat())
        except Exception: row["net_upgrades_60d"] = 0
        try:
            ib = json.load(open(f"{RD}/insider_buys/{s}.json"))
            row["insider_buys_30d"] = sum(1 for x in ib if x.get("transactionType") == "P-Purchase" and cutoff30 <= str(x.get("filingDate")) <= rd.isoformat()
                                          and any(k in (x.get("typeOfOwner") or "").lower() for k in ("officer", "director")))
        except Exception: row["insider_buys_30d"] = 0
        try:
            cg = json.load(open(f"{RD}/congress/{s}.json"))
            row["congress_buys_30d"] = sum(1 for ch in ("senate", "house") for x in cg.get(ch, []) if "purchase" in (x.get("type") or "").lower()
                                           and cutoff30 <= str(x.get("disclosureDate")) <= rd.isoformat())
        except Exception: row["congress_buys_30d"] = 0
        pool.append(row)
    assert len(pool) == 100, f"{ds}: only {len(pool)}"
    lines = ["ticker | sector | industry | mcap$B | close | ret1m% | ret3m% | ret6m% | ret12m% | vs52wHigh% | vol20ann% | ADV$M | beta | shortInt%sh | daysToCover | netUpgrades60d | insiderBuys30d | congressBuys30d"]
    for r in pool:
        lines.append(" | ".join([r["ticker"], str(r["sector"]), str(r["industry"])[:28], f(r["mcap_b"], 1), f(r["close"], 2), f(r["r1m"] * 100), f(r["r3m"] * 100), f(r["r6m"] * 100),
                                  f(r["r12m"] * 100), f(r["below_52w_high"] * 100), f(r["vol20_ann"] * 100, 0), f(r["adv_usd_m"], 1), f(r["beta"], 2),
                                  f(r.get("short_int_pct")), f(r.get("days_to_cover")), str(r["net_upgrades_60d"]), str(r["insider_buys_30d"]), str(r["congress_buys_30d"])]))
    pack = "\n".join(lines)
    json.dump({"round_date": ds, "seed": seed, "built_utc": datetime.utcnow().isoformat(timespec="seconds"),
               "pack_sha256": hashlib.sha256(pack.encode()).hexdigest(), "pool": pool}, open(f"{OUT}/round_{ds}_pool.json", "w"), indent=1)
    open(f"{OUT}/round_{ds}_pack.txt", "w").write(pack)
    print(ds, "pool ok; last data date:", pd.Series([r["last_date"] for r in pool]).max(), "| SI settlement used:", si_last.index.size and str(si_use.settlement_date.max().date()), flush=True)
