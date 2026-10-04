"""Arm B pack for prereg F2: recent headlines for the pool's names (no numeric data). Polygon key from .env (never printed).
  python scripts/forward/f2_build_news.py 2026-10-04"""
import json, os, sys, time, urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from dotenv import load_dotenv
load_dotenv("/Users/asafamos/StockScout/stock-scout-2/.env")
KEY = os.environ["POLYGON_API_KEY"]
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
rd = date.fromisoformat(sys.argv[1])
FD = os.environ.get("F2_DIR", "data/forward_llm")          # retro rounds use data/forward_llm_retro
pool = json.load(open(f"{ROOT}/{FD}/round_{rd}_pool.json"))["pool"]
tickers = {r["ticker"] for r in pool}
since = (datetime.combine(rd, datetime.min.time(), tzinfo=timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
until = datetime.combine(rd, datetime.min.time(), tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")   # sunday 00:00 UTC cutoff
url = ("https://api.polygon.io/v2/reference/news?" +
       f"published_utc.gte={since}&published_utc.lt={until}&order=desc&sort=published_utc&limit=1000&apiKey={KEY}")
news = defaultdict(list); pages = 0; n_art = 0; last = [0.0]
while url and pages < 40:
    w = 13.0 - (time.time() - last[0])
    if w > 0: time.sleep(w)
    last[0] = time.time()
    try:
        with urllib.request.urlopen(url, timeout=90) as r: d = json.loads(r.read())
    except Exception as e:
        print("retry:", str(e)[:60].replace(KEY, "***"), file=sys.stderr); time.sleep(20); continue
    pages += 1
    for a in d.get("results", []):
        n_art += 1
        for t in a.get("tickers", []):
            if t in tickers and len(news[t]) < 4:
                news[t].append((a["published_utc"][:10], (a.get("publisher") or {}).get("name", "?"), a["title"].strip()))
    nxt = d.get("next_url")
    url = (nxt + f"&apiKey={KEY}") if nxt else None
    print(f"page {pages}: total articles {n_art}, pool names with news {len(news)}", file=sys.stderr)
lines = []
for r in pool:
    t = r["ticker"]; head = f"{t} | {r['sector']} | {r['industry']}"
    if news.get(t):
        lines.append(head + "\n" + "\n".join(f"   - [{d}] ({pub}) {title}" for d, pub, title in news[t]))
    else:
        lines.append(head + "\n   (no news in the last 7 days)")
pack = "\n".join(lines)
open(f"{ROOT}/{FD}/round_{rd}_newspack.txt", "w").write(pack)
print(f"articles scanned {n_art}; pool names with news: {len(news)}/100", file=sys.stderr)
