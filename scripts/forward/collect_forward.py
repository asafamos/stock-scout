"""Forward collector for pre-registered test F1 (docs/research_prereg_F1_forward_test.md). Collection only — no trading, no ranking.

Appends valid congress purchases and insider purchases to data/forward/*.jsonl with first_seen_utc. Idempotent (dedupe by key).
FMP key from FMP_API_KEY (never printed).   python scripts/forward/collect_forward.py [--out DIR] [--days 14]
"""
import argparse, json, os, re, sys, time, urllib.request
from datetime import datetime, timedelta, timezone

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "forward"))
ap.add_argument("--days", type=int, default=14, help="how far back (disclosure/filing date) to accept rows on each run")
args = ap.parse_args()
KEY = os.environ.get("FMP_API_KEY") or sys.exit("FMP_API_KEY missing")
OUT = os.path.abspath(args.out); os.makedirs(OUT, exist_ok=True)
NOW = datetime.now(timezone.utc); CUT = (NOW - timedelta(days=args.days)).strftime("%Y-%m-%d")
NUM = re.compile(r"[\d,]+")


def get(q):
    for a in range(4):
        try:
            with urllib.request.urlopen(f"https://financialmodelingprep.com/stable/{q}&apikey={KEY}", timeout=40) as r:
                return json.loads(r.read())
        except Exception as e:
            if any(c in str(e) for c in ("400", "402", "404")): return []
            time.sleep(2 * (a + 1))
    raise SystemExit(f"FMP failed: {q.split('?')[0]}")


def load_keys(path, keyf):
    ks = set()
    if os.path.exists(path):
        for line in open(path):
            try: ks.add(keyf(json.loads(line)))
            except Exception: pass
    return ks


def append(path, rows):
    with open(path, "a") as f:
        for r in rows: f.write(json.dumps(r) + "\n")


# ---- congress purchases (H5 validity rules)
cpath = os.path.join(OUT, "congress_events.jsonl")
ckey = lambda r: (r["symbol"], r["disclosureDate"], r["chamber"], r["name"], r["transactionDate"], r["amount"])
seen = load_keys(cpath, ckey); new = []
for chamber, ep, maxpages in (("senate", "senate-latest", 6), ("house", "house-latest", 25)):
    for p in range(maxpages):
        d = get(f"{ep}?page={p}&limit=100")
        if not d: break
        for r in d:
            try:
                if "purchase" not in (r.get("type") or "").lower(): continue
                if (r.get("assetType") or "Stock") not in ("Stock", ""): continue
                m = NUM.search(r.get("amount") or "")
                if not m or float(m.group().replace(",", "")) < 15001: continue
                tx, ds = r["transactionDate"], r["disclosureDate"]
                lag = (datetime.fromisoformat(ds) - datetime.fromisoformat(tx)).days
                if not (0 <= lag <= 400) or not r.get("symbol"): continue
                row = {"symbol": r["symbol"], "disclosureDate": ds, "transactionDate": tx, "chamber": chamber,
                       "name": f"{r.get('firstName','')} {r.get('lastName','')}".strip(), "amount": r.get("amount"),
                       "first_seen_utc": NOW.isoformat(timespec="seconds")}
                if ckey(row) not in seen: seen.add(ckey(row)); new.append(row)
            except Exception:
                continue
        if min(x.get("disclosureDate", "9") for x in d) < CUT: break
append(cpath, new); print(f"congress: +{len(new)} new valid purchases")

# ---- insider purchases (H4 validity rules)
ipath = os.path.join(OUT, "insider_purchases.jsonl")
ikey = lambda r: (r["symbol"], r["reportingCik"], r["transactionDate"], r["filingDate"], r["shares"], r["price"])
seen = load_keys(ipath, ikey); new = []
for p in range(40):
    d = get(f"insider-trading/latest?page={p}&limit=100&transactionType=P-Purchase")
    if not d: break
    for r in d:
        try:
            if r.get("transactionType") != "P-Purchase" or not str(r.get("formType", "")).startswith("4"): continue
            price, sh = float(r.get("price") or 0), float(r.get("securitiesTransacted") or 0)
            if price < 1 or sh <= 0: continue
            who = (r.get("typeOfOwner") or "").lower()
            if "officer" not in who and "director" not in who: continue
            tx, fl = r["transactionDate"], r["filingDate"]
            if not (0 <= (datetime.fromisoformat(fl) - datetime.fromisoformat(tx)).days <= 90): continue
            row = {"symbol": r["symbol"], "reportingCik": r["reportingCik"], "transactionDate": tx, "filingDate": fl,
                   "shares": sh, "price": price, "typeOfOwner": r.get("typeOfOwner"),
                   "first_seen_utc": NOW.isoformat(timespec="seconds")}
            if ikey(row) not in seen: seen.add(ikey(row)); new.append(row)
        except Exception:
            continue
    if min(x.get("filingDate", "9") for x in d) < CUT: break
append(ipath, new); print(f"insider: +{len(new)} new valid purchases")
