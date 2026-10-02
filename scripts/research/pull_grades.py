"""Download FMP analyst grades for every symbol in the PIT price panel (resumable, parallel). Key from .env, never printed."""
import os, json, time, urllib.request, urllib.parse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
load_dotenv("/Users/asafamos/StockScout/stock-scout-2/.env")
K = os.environ["FMP_API_KEY"]; RD = "/Users/asafamos/StockScout/research_data"
syms = sorted(pd.read_pickle(f"{RD}/univ_px.pkl").keys())
os.makedirs(f"{RD}/grades", exist_ok=True)
def get(s):
    p = f"{RD}/grades/{s}.json"
    if os.path.exists(p): return s, True
    for a in range(5):
        try:
            u = f"https://financialmodelingprep.com/stable/grades?symbol={urllib.parse.quote(s)}&apikey={K}"
            with urllib.request.urlopen(u, timeout=40) as r: d = json.loads(r.read())
            json.dump(d, open(p, "w")); return s, True
        except Exception as e:
            m = str(e)
            if "404" in m or "402" in m: json.dump([], open(p, "w")); return s, True
            time.sleep(2 * (a + 1))
    return s, False
bad = 0; t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    for i, (s, ok) in enumerate(ex.map(get, syms)):
        bad += (not ok)
        if i % 500 == 0: print(i, len(syms), f"{time.time()-t0:.0f}s bad={bad}", flush=True)
print("DONE bad=", bad, flush=True)
