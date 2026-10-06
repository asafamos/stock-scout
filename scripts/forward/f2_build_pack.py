"""Build the point-in-time data pack for a prereg-F2 round (docs/research_prereg_F2_llm_picker.md).
  python scripts/forward/f2_build_pack.py 2026-10-04     # round date (a Sunday)
Writes data/forward_llm/round_<date>_pool.json and prints the pack text. FMP key from .env (never printed)."""
import glob, hashlib, json, os, random, sys, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
import numpy as np, pandas as pd
from dotenv import load_dotenv
load_dotenv("/Users/asafamos/StockScout/stock-scout-2/.env")
KEY = os.environ["FMP_API_KEY"]; RD = "/Users/asafamos/StockScout/research_data"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
rd = date.fromisoformat(sys.argv[1]); seed = int(rd.strftime("%Y%m%d"))
ASOF = date.fromisoformat(os.environ.get("F2_ASOF", rd.isoformat()))      # data cutoff (last COMPLETED session); default = round date
FROM = (ASOF - timedelta(days=420)).isoformat(); TO = ASOF.isoformat()


def get(q):
    for a in range(4):
        try:
            with urllib.request.urlopen(f"https://financialmodelingprep.com/stable/{q}&apikey={KEY}", timeout=40) as r: return json.loads(r.read())
        except Exception as e:
            time.sleep(1.5 * (a + 1))
    return None


# 1) candidate symbols from the research panel (alive recently), shuffled with the round seed
px = pd.read_pickle(f"{RD}/univ_px.pkl")
cands = sorted(s for s, v in px.items() if len(v) and v.index[-1] >= pd.Timestamp(rd) - pd.Timedelta(days=30)
               and v["Close"].iloc[-1] >= 5 and (v["Close"] * v["Volume"]).tail(20).mean() >= 1e6 and s.replace("-", "").isalpha() and "-" not in s)
random.Random(seed).shuffle(cands)
print(f"candidates after panel filter: {len(cands)}", file=sys.stderr)

si_parts = sorted(glob.glob(f"{RD}/short_interest/part_*.parquet"))
SI = pd.concat([pd.read_parquet(f) for f in si_parts[-6:]], ignore_index=True)
SI["settlement_date"] = pd.to_datetime(SI["settlement_date"])
usable = SI[SI.settlement_date <= pd.Timestamp(rd) - pd.Timedelta(days=13)]       # ≥9 business days publication lag
SI = usable[usable.settlement_date == usable.settlement_date.max()].set_index("ticker")
print("short interest settlement used:", SI.index.size, usable.settlement_date.max().date(), file=sys.stderr)


def one(s):
    h = get(f"historical-price-eod/full?symbol={urllib.parse.quote(s)}&from={FROM}&to={TO}")
    p = get(f"profile?symbol={urllib.parse.quote(s)}")
    if not h or not p or not isinstance(h, list) or not isinstance(p, list) or not p: return None
    d = pd.DataFrame(h).sort_values("date").reset_index(drop=True)
    if len(d) < 200: return None
    c = d["close"].astype(float); v = d["volume"].astype(float)
    last = float(c.iloc[-1]); adv = float((c * v).tail(20).mean()); mcap = float(p[0].get("marketCap") or 0)
    if last < 5 or adv < 1e6 or mcap < 3e8: return None
    ret = lambda n: float(last / c.iloc[-1 - n] - 1) if len(c) > n else None
    vol20 = float(c.pct_change().tail(20).std() * np.sqrt(252))
    hi52 = float(c.tail(252).max())
    row = {"ticker": s, "close": round(last, 2), "r1m": ret(21), "r3m": ret(63), "r6m": ret(126), "r12m": ret(252),
           "below_52w_high": float(last / hi52 - 1), "vol20_ann": vol20, "adv_usd_m": adv / 1e6, "mcap_b": mcap / 1e9,
           "sector": p[0].get("sector"), "industry": p[0].get("industry"), "beta": p[0].get("beta"), "last_date": d["date"].iloc[-1]}
    if s in SI.index:
        sh = mcap / last
        row["short_int_pct"] = float(SI.loc[s, "short_interest"] / sh * 100) if sh > 0 else None
        row["days_to_cover"] = float(SI.loc[s, "days_to_cover"])
    # analyst net upgrades (60d), insider purchases (30d), congress purchases (30d) from the downloaded stores
    cutoff60, cutoff30 = (rd - timedelta(days=60)).isoformat(), (rd - timedelta(days=30)).isoformat()
    try:
        g = json.load(open(f"{RD}/grades/{s}.json"))
        row["net_upgrades_60d"] = sum((1 if x["action"] == "upgrade" else -1) for x in g if x.get("action") in ("upgrade", "downgrade") and cutoff60 <= x["date"] <= TO)
    except Exception: row["net_upgrades_60d"] = None
    try:
        ib = json.load(open(f"{RD}/insider_buys/{s}.json"))
        row["insider_buys_30d"] = sum(1 for x in ib if x.get("transactionType") == "P-Purchase" and cutoff30 <= str(x.get("filingDate")) <= TO
                                      and ("officer" in (x.get("typeOfOwner") or "").lower() or "director" in (x.get("typeOfOwner") or "").lower()))
    except Exception: row["insider_buys_30d"] = None
    try:
        cg = json.load(open(f"{RD}/congress/{s}.json"))
        row["congress_buys_30d"] = sum(1 for ch in ("senate", "house") for x in cg.get(ch, []) if "purchase" in (x.get("type") or "").lower() and cutoff30 <= str(x.get("disclosureDate")) <= TO)
    except Exception: row["congress_buys_30d"] = None
    return row


pool = []
with ThreadPoolExecutor(8) as ex:
    i = 0
    while len(pool) < 100 and i < len(cands):
        batch = cands[i:i + 40]; i += 40
        for r in ex.map(one, batch):
            if r and len(pool) < 100: pool.append(r)
pool = pool[:100]
assert len(pool) == 100, f"only {len(pool)} valid names"


def f(x, p=1): return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"
lines = ["ticker | sector | industry | mcap$B | close | ret1m% | ret3m% | ret6m% | ret12m% | vs52wHigh% | vol20ann% | ADV$M | beta | shortInt%sh | daysToCover | netUpgrades60d | insiderBuys30d | congressBuys30d"]
for r in pool:
    lines.append(" | ".join([r["ticker"], str(r["sector"]), str(r["industry"])[:28], f(r["mcap_b"], 1), f(r["close"], 2), f((r["r1m"] or 0) * 100), f((r["r3m"] or 0) * 100), f((r["r6m"] or 0) * 100),
                              f(None if r["r12m"] is None else r["r12m"] * 100), f(r["below_52w_high"] * 100), f(r["vol20_ann"] * 100, 0), f(r["adv_usd_m"], 1), f(r["beta"], 2),
                              f(r.get("short_int_pct")), f(r.get("days_to_cover")), str(r.get("net_upgrades_60d")), str(r.get("insider_buys_30d")), str(r.get("congress_buys_30d"))]))
pack = "\n".join(lines)
last_dates = pd.Series([r["last_date"] for r in pool]).value_counts().head(2).to_dict()
out = {"round_date": rd.isoformat(), "data_asof": ASOF.isoformat(), "seed": seed, "built_utc": datetime.utcnow().isoformat(timespec="seconds"), "data_as_of_counts": last_dates,
       "pack_sha256": hashlib.sha256(pack.encode()).hexdigest(), "pool": pool}
os.makedirs(f"{ROOT}/data/forward_llm", exist_ok=True)
json.dump(out, open(f"{ROOT}/data/forward_llm/round_{rd.isoformat()}_pool.json", "w"), indent=1)
open(f"{ROOT}/data/forward_llm/round_{rd.isoformat()}_pack.txt", "w").write(pack)
print(pack)
print("\ndata as-of dates in pool:", last_dates, file=sys.stderr)
