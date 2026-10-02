"""Download FINRA short interest (Polygon /stocks/v1/short-interest) for all settlement dates since 2017-12.
Polygon plan: 5 calls/min, <=50k rows/call. Pages by settlement date range; resumable (skips existing parquet parts).
Key from .env (POLYGON_API_KEY) — never printed."""
import os, sys, time, json, urllib.request, urllib.parse
import pandas as pd
from dotenv import load_dotenv
load_dotenv("/Users/asafamos/StockScout/stock-scout-2/.env")
KEY = os.environ["POLYGON_API_KEY"]
OUT = "/Users/asafamos/StockScout/research_data/short_interest"
BASE = "https://api.polygon.io/stocks/v1/short-interest"
last_call = [0.0]

def get(url):
    for a in range(6):
        wait = 13.0 - (time.time() - last_call[0])
        if wait > 0: time.sleep(wait)
        last_call[0] = time.time()
        try:
            u = url + ("&" if "?" in url else "?") + "apiKey=" + KEY
            with urllib.request.urlopen(u, timeout=90) as r: return json.loads(r.read())
        except Exception as e:
            m = str(e)
            print("retry", a, m[:60].replace(KEY, "***"), flush=True)
            time.sleep(20 * (a + 1))
    raise SystemExit("giving up")

url = BASE + "?" + urllib.parse.urlencode({"settlement_date.gte": "2017-12-01", "limit": 50000, "sort": "settlement_date.asc"})
page = 0
while url:
    part = f"{OUT}/part_{page:03d}.parquet"
    if os.path.exists(part):
        # resume: reload next_url from sidecar
        url = json.load(open(part + ".next"))["next"]; page += 1; continue
    d = get(url)
    rows = d.get("results", [])
    if rows:
        df = pd.DataFrame(rows); df.to_parquet(part)
        print(f"page {page}: {len(rows)} rows  {df['settlement_date'].min()} -> {df['settlement_date'].max()}", flush=True)
    nxt = d.get("next_url")
    json.dump({"next": nxt}, open(part + ".next", "w"))
    url = nxt; page += 1
print("DONE", flush=True)
