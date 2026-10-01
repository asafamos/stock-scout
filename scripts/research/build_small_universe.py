"""Prices + earnings history for micro/small caps ($50M..$500M at some point since 2019-06) that the main PIT
universe (>= $500M) excluded. Output (in <workdir>): univ_px_small.pkl, univ_earn_small.pkl, univ_mcap_small.pkl."""
import os, sys, json, time, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
RD=sys.argv[1]; k=os.environ["FMP_API_KEY"]
allm=pd.read_pickle(RD+"/univ_mcap_all.pkl"); big=set(pd.read_pickle(RD+"/univ_mcap.pkl"))
cand=[s for s,df in allm.items() if s not in big and len(df[df.date>="2019-06-01"]) and df[df.date>="2019-06-01"].marketCap.max()>=5e7]
print("candidates",len(cand),flush=True)
def get(path):
    for a in range(4):
        try:
            with urllib.request.urlopen(f"https://financialmodelingprep.com/stable/{path}&apikey={k}",timeout=40) as r: return json.loads(r.read())
        except Exception as e:
            if "402" in str(e) or "404" in str(e): return []
            time.sleep(1.5*(a+1))
    return None
def one(s):
    q=urllib.parse.quote(s)
    p=get(f"historical-price-eod/full?symbol={q}&from=2018-06-01&to=2026-09-15")
    px=None
    if p:
        df=pd.DataFrame(p)[["date","open","high","low","close","volume"]]; df["date"]=pd.to_datetime(df["date"])
        px=df.sort_values("date").set_index("date").rename(columns=str.capitalize)
        if len(px)<260: px=None
    e=get(f"earnings?symbol={q}&limit=60") if px is not None else None
    return s,px,e
PX,EA={},{}; t0=time.time()
with ThreadPoolExecutor(12) as ex:
    for i,(s,px,e) in enumerate(ex.map(one,cand)):
        if px is not None: PX[s]=px
        if e: EA[s]=e
        if i%500==0: print("small",i,len(PX),round(time.time()-t0),"s",flush=True)
pd.to_pickle(PX,RD+"/univ_px_small.pkl"); pd.to_pickle(EA,RD+"/univ_earn_small.pkl"); pd.to_pickle({s:allm[s] for s in PX},RD+"/univ_mcap_small.pkl")
print("DONE",len(PX),len(EA),round(time.time()-t0),"s",flush=True)
