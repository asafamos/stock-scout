"""Download FMP senate-trades and house-trades per symbol, paginated until empty (resumable, parallel). Key from .env, never printed."""
import os, json, time, urllib.request, urllib.parse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
load_dotenv("/Users/asafamos/StockScout/stock-scout-2/.env")
K = os.environ["FMP_API_KEY"]; RD = "/Users/asafamos/StockScout/research_data"
syms = sorted(pd.read_pickle(f"{RD}/univ_px.pkl").keys())
os.makedirs(f"{RD}/congress", exist_ok=True)

def page(ep, s, p):
    for a in range(5):
        try:
            u = f"https://financialmodelingprep.com/stable/{ep}?symbol={urllib.parse.quote(s)}&limit=250&page={p}&apikey={K}"
            with urllib.request.urlopen(u, timeout=40) as r: return json.loads(r.read())
        except Exception as e:
            m = str(e)
            if "404" in m or "402" in m or "400" in m: return []
            time.sleep(2 * (a + 1))
    return None

def get(s):
    path = f"{RD}/congress/{s}.json"
    if os.path.exists(path): return s, True
    out = {"senate": [], "house": []}
    for ep, key in (("senate-trades", "senate"), ("house-trades", "house")):
        for p in range(0, 30):
            d = page(ep, s, p)
            if d is None: return s, False
            if not d: break
            out[key] += d
            if len(d) < 250: break
    json.dump(out, open(path, "w")); return s, True

bad = 0; t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    for i, (s, ok) in enumerate(ex.map(get, syms)):
        bad += (not ok)
        if i % 500 == 0: print(i, len(syms), f"{time.time()-t0:.0f}s bad={bad}", flush=True)
print("DONE bad=", bad, flush=True)
