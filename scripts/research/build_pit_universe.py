import os, sys, json, time, csv, zipfile, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
SP=sys.argv[1]; k=os.environ["FMP_API_KEY"]
z=zipfile.ZipFile(SP+"/tiingo_tickers.zip"); rows=list(csv.DictReader(z.open("supported_tickers.csv").read().decode().splitlines()))
syms=sorted({r["ticker"] for r in rows if r["assetType"]=="Stock" and r["exchange"] in ("NASDAQ","NYSE","NYSE MKT","AMEX")
             and r["endDate"] and r["endDate"]>="2019-01-01" and r["startDate"]<="2025-12-31" and r["ticker"].replace("-","").isalpha() and len(r["ticker"])<=5})
print("candidate symbols",len(syms),flush=True)
def get(path):
    for a in range(4):
        try:
            with urllib.request.urlopen(f"https://financialmodelingprep.com/stable/{path}&apikey={k}",timeout=40) as r: return json.loads(r.read())
        except Exception as e:
            msg=str(e)
            if "402" in msg or "404" in msg: return []
            time.sleep(1.5*(a+1))
    return None
def mcap(s):
    d=get(f"historical-market-capitalization?symbol={urllib.parse.quote(s)}&from=2018-06-01&to=2026-09-15")
    if not d: return s,None
    df=pd.DataFrame(d)[["date","marketCap"]]; df["date"]=pd.to_datetime(df["date"]); return s,df.sort_values("date").reset_index(drop=True)
MC={}; t0=time.time()
with ThreadPoolExecutor(12) as ex:
    for i,(s,df) in enumerate(ex.map(mcap,syms)):
        if df is not None and len(df)>60: MC[s]=df
        if i%1000==0: print("mcap",i,len(MC),round(time.time()-t0),"s",flush=True)
pd.to_pickle(MC,SP+"/univ_mcap_all.pkl")
keep=[s for s,df in MC.items() if df[df.date>="2019-06-01"].marketCap.max()>=5e8]
print("mcap stage done:",len(MC),"with data; kept (>= $500M at some point):",len(keep),flush=True)
def px(s):
    d=get(f"historical-price-eod/full?symbol={urllib.parse.quote(s)}&from=2018-06-01&to=2026-09-15")
    if not d: return s,None
    df=pd.DataFrame(d)[["date","open","high","low","close","volume"]]; df["date"]=pd.to_datetime(df["date"])
    return s,df.sort_values("date").set_index("date").rename(columns=str.capitalize)
PX={}
with ThreadPoolExecutor(12) as ex:
    for i,(s,df) in enumerate(ex.map(px,keep)):
        if df is not None and len(df)>260: PX[s]=df
        if i%500==0: print("px",i,len(PX),round(time.time()-t0),"s",flush=True)
pd.to_pickle({s:MC[s] for s in PX},SP+"/univ_mcap.pkl"); pd.to_pickle(PX,SP+"/univ_px.pkl")
print("DONE symbols with price+mcap:",len(PX),round(time.time()-t0),"s",flush=True)
