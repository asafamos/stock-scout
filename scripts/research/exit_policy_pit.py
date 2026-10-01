import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/Users/asafamos/StockScout/stock-scout-2")
import numpy as np, pandas as pd
from core.trading import exit_sim as xs
SP=sys.argv[1]; TOPN=int(sys.argv[2]) if len(sys.argv)>2 else 2000; COST=0.7
PX=pd.read_pickle(SP+"/univ_px.pkl"); MC=pd.read_pickle(SP+"/univ_mcap.pkl")
import yfinance as yf
spy=yf.download("SPY",start="2018-06-01",auto_adjust=True,progress=False); spy.columns=[c[0] if isinstance(c,tuple) else c for c in spy.columns]; spy.index=pd.to_datetime(spy.index).tz_localize(None); cal=spy.index; D=len(cal); tick=list(PX); N=len(tick)
def mat(): return np.full((N,D),np.nan)
O,H,L,C,V,MCm=(mat() for _ in range(6)); last=np.zeros(N,int)
for n,t in enumerate(tick):
    b=PX[t]; b=b[~b.index.duplicated()].reindex(cal)
    o,h,l,c,v=(b[x].to_numpy(float) for x in ("Open","High","Low","Close","Volume"))
    if np.isfinite(c).sum()<260: continue
    last[n]=np.where(np.isfinite(c))[0][-1]
    m=MC[t].set_index("date").marketCap; MCm[n]=m[~m.index.duplicated()].reindex(cal,method="ffill",limit=10).to_numpy(float)
    O[n],H[n],L[n],C[n],V[n]=o,h,l,c,v
ADDV=pd.DataFrame((V*C).T).rolling(20,min_periods=20).mean().to_numpy().T
POL={"LEGACY 9%→5.5% ≤20":xs.POLICIES["LEGACY"],"HOLD 20":xs.POLICIES["HOLD20"],"HOLD 60":xs.POLICIES["HOLD60"],
     "trail 5% ≤20":{"kind":"pct","p":5.0,"max":20},"trail 9% ≤20":{"kind":"pct","p":9.0,"max":20},"trail 9% ≤60":{"kind":"pct","p":9.0,"max":60},
     "trail 15% ≤60":{"kind":"pct","p":15.0,"max":60},
     "CANARY ATR4 8-20 ≤30":xs.POLICIES["CANARY"],
     "ATR3 6-16 ≤30":{"kind":"atrpct","K":3.0,"lo":6.0,"hi":16.0,"max":30},
     "ATR2 5-12 ≤20":{"kind":"atrpct","K":2.0,"lo":5.0,"hi":12.0,"max":20},
     "ATR4 8-20 ≤60":xs.POLICIES["ATR4_60"],"ATR5 8-25 ≤60":xs.POLICIES["ATR5_60"]}
rng=np.random.default_rng(3); rows=[]
for j in range(300,D-70,5):
    fin=np.isfinite(C[:,j])&np.isfinite(MCm[:,j])&(MCm[:,j]>0)&(C[:,j]>=5)&np.isfinite(ADDV[:,j])&(ADDV[:,j]>=5e6)
    if fin.sum()<300: continue
    k=min(TOPN,fin.sum()); thr=np.partition(MCm[fin,j],-k)[-k]; idx=np.where(fin&(MCm[:,j]>=thr))[0]
    for n in rng.choice(idx,6,replace=False):
        e=j+1
        if not np.isfinite(O[n,e]) or last[n]<e+5: continue
        seg=slice(0,last[n]+1)
        for name,pol in POL.items():
            r=xs.simulate(O[n,seg],H[n,seg],L[n,seg],C[n,seg],e,pol,cost_pct=COST)
            if r is None: continue
            if not r.get("finished"): r={"ret_pct":(C[n,last[n]]/O[n,e]-1)*100-COST,"days":last[n]-e+1}
            rows.append((cal[j],tick[n],name,r["ret_pct"],r["days"]))
R=pd.DataFrame(rows,columns=["date","t","policy","ret","days"]); R.to_parquet(SP+f"/exit_pit_{TOPN}.parquet")
T=R.groupby("policy").agg(n=("ret","size"),mean=("ret","mean"),median=("ret","median"),sd=("ret","std"),WR=("ret",lambda x:100*(x>0).mean()),days=("days","mean"),p5=("ret",lambda x:x.quantile(.05)))
T["per20d"]=T["mean"]/T.days*20; T["mean/sd"]=T["mean"]/T.sd
base=R[R.policy=="LEGACY 9%→5.5% ≤20"].set_index(["date","t"]).ret
def paired(p):
    x=R[R.policy==p].set_index(["date","t"]).ret; d=(x-base).dropna().groupby(level=0).mean().values
    b=[rng.choice(d,len(d)).mean() for _ in range(1500)]; return d.mean(),np.percentile(b,2.5)
T["vs_LEGACY"]=[paired(p)[0] for p in T.index]; T["lo95"]=[paired(p)[1] for p in T.index]
pd.set_option("display.width",230)
print(f"PIT top-{TOPN}, random liquid entries (next open), net of {COST}% cost, 2019-2026, entries={T.n.iloc[0]}")
print(T.sort_values("per20d",ascending=False).round(2).to_string())
