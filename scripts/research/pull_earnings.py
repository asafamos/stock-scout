"""FMP earnings history (actual vs estimate) for every symbol of a PIT mcap pickle. usage: pull_earnings.py <mcap.pkl> <out.pkl>"""
import os, sys, json, time, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
k=os.environ["FMP_API_KEY"]; tick=list(pd.read_pickle(sys.argv[1]))
def get(s):
    for a in range(4):
        try:
            with urllib.request.urlopen(f"https://financialmodelingprep.com/stable/earnings?symbol={urllib.parse.quote(s)}&limit=60&apikey={k}",timeout=40) as r: return s,json.loads(r.read())
        except Exception as e:
            if "402" in str(e) or "404" in str(e): return s,[]
            time.sleep(1.5*(a+1))
    return s,None
out={}; t0=time.time()
with ThreadPoolExecutor(12) as ex:
    for i,(s,d) in enumerate(ex.map(get,tick)):
        if d: out[s]=d
        if i%1000==0: print("earn",i,len(out),round(time.time()-t0),"s",flush=True)
pd.to_pickle(out,sys.argv[2]); print("DONE",len(out),flush=True)
