"""What can be captured at different account sizes? Monthly-rebalanced, equal-weight top-K portfolios on the
point-in-time universe (incl. delisted), with order-level commissions (Fixed / Tiered / Lite) and band-specific
spread+slippage. usage: capital_scenarios.py <workdir>"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
RD=sys.argv[1]
PX=pd.read_pickle(RD+"/univ_px.pkl"); PX.update(pd.read_pickle(RD+"/univ_px_small.pkl"))
MC=pd.read_pickle(RD+"/univ_mcap.pkl"); MC.update(pd.read_pickle(RD+"/univ_mcap_small.pkl"))
spy=yf.download("SPY",start="2018-06-01",auto_adjust=True,progress=False); spy.columns=[c[0] if isinstance(c,tuple) else c for c in spy.columns]
spy.index=pd.to_datetime(spy.index).tz_localize(None); cal=spy.index; D=len(cal); tick=[t for t in PX if t in MC]; N=len(tick)
O=np.full((N,D),np.nan,dtype=np.float32); C=O.copy(); V=O.copy(); MCm=O.copy(); last=np.zeros(N,int)
for n,t in enumerate(tick):
    b=PX[t]; b=b[~b.index.duplicated()].reindex(cal); c=b["Close"].to_numpy(float)
    if np.isfinite(c).sum()<260: continue
    last[n]=np.where(np.isfinite(c))[0][-1]; O[n],C[n],V[n]=b["Open"].to_numpy(float),c,b["Volume"].to_numpy(float)
    m=MC[t].set_index("date").marketCap; MCm[n]=m[~m.index.duplicated()].reindex(cal,method="ffill",limit=10).to_numpy(float)
ADDV=pd.DataFrame((V.astype(float)*C.astype(float)).T).rolling(20,min_periods=20).mean().to_numpy().T
def spread(mc): return np.where(mc>=5e9,0.0005,np.where(mc>=1e9,0.0010,np.where(mc>=3e8,0.0025,0.006)))   # per side
def commission(model,sh,px):
    sh=np.maximum(sh,1)
    if model=="Fixed": return np.maximum(1.0,0.005*sh)
    if model=="Tiered": return np.maximum(0.35,0.0035*sh)+0.002*sh
    return 0.0*sh
# month-end signal dates
idx=pd.Series(range(D),index=cal); me=idx.groupby([cal.year,cal.month]).max().values; me=[j for j in me if j>=300 and j<D-25]
rng=np.random.default_rng(1)
def pick(j,strategy,K):
    c=C[:,j]; mc=MCm[:,j]; ok=np.isfinite(c)&np.isfinite(mc)&(mc>0)&np.isfinite(ADDV[:,j])&(ADDV[:,j]>=2e6)
    if strategy=="NH_small": ok&=(mc>=5e7)&(mc<3e8)&(c>=2)
    else: ok&=(mc>=3e8)&(c>=5)
    mom=C[:,j-21]/C[:,j-252]-1; nh=c/np.nanmax(C[:,j-251:j+1],axis=1)
    ok&=np.isfinite(mom)&np.isfinite(nh)
    idxs=np.where(ok)[0]
    if len(idxs)<max(60,3*K): return []
    if strategy=="RANDOM": return list(rng.choice(idxs,K,replace=False))
    if strategy=="MOM12_1": s=mom
    elif strategy=="NH_small" or strategy=="NH": s=nh
    elif strategy=="MOM+NH": s=pd.Series(mom[idxs]).rank(pct=True).to_numpy()+pd.Series(nh[idxs]).rank(pct=True).to_numpy(); full=np.full(N,-1e9); full[idxs]=s; s=full
    return list(idxs[np.argsort(-s[idxs])[:K]])
def px_open(n,k):
    v=O[n,k]
    return float(v) if np.isfinite(v) else float(C[n,min(last[n],k)])
def run(strategy,K,capital,comm):
    cash=float(capital); hold={}; curve=[]; cost_total=0.0
    for a,j in enumerate(me[:-1]):
        e=j+1; e2=me[a+1]+1
        eq=cash+sum(sh*px_open(n,e) for n,sh in hold.items())
        sel=pick(j,strategy,K); new=set(sel)
        for n,sh in list(hold.items()):                       # sell names that dropped out
            if n not in new:
                px=px_open(n,e); cst=commission(comm,sh,px)+sh*px*spread(MCm[n,j]); cash+=sh*px-cst; cost_total+=cst; del hold[n]
        target=eq/max(1,len(new))
        for n in sel:                                         # buy new names at the open
            if n in hold or not np.isfinite(O[n,e]): continue
            px=float(O[n,e]); sh=int(min(target,cash)//px)
            if sh<1: continue
            cst=commission(comm,sh,px)+sh*px*spread(MCm[n,j]); 
            if sh*px+cst>cash: sh=int((cash-cst)//px)
            if sh<1: continue
            cash-=sh*px+cst; cost_total+=cst; hold[n]=sh
        for n,sh in list(hold.items()):                       # delisted before next rebalance -> cash out at last close
            if last[n]<e2: cash+=sh*float(C[n,last[n]]); del hold[n]
        curve.append(cash+sum(sh*px_open(n,e2) for n,sh in hold.items()))
    c=np.array(curve); yrs=(cal[me[-1]+1]-cal[me[0]+1]).days/365.25; base=np.r_[capital,c]; r=np.diff(base)/base[:-1]
    dd=(c/np.maximum.accumulate(c)-1).min()
    return (c[-1]/capital)**(1/yrs)-1, r.std()*np.sqrt(12), dd, (r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan), cost_total/yrs/capital
spy_c=spy.Close.to_numpy(float); yrs=(cal[me[-1]]-cal[me[0]]).days/365.25
print(f"SPY buy&hold over the same span: CAGR {(spy_c[me[-1]]/spy_c[me[0]])**(1/yrs)-1:+.1%}\n")
SC=[(821,5),(2000,10),(5000,20),(10000,20)]
rows=[]
for strat in ("RANDOM","MOM12_1","MOM+NH","NH_small"):
    for cap,K in SC:
        for comm in ("Fixed","Tiered","Lite"):
            cagr,vol,dd,sh,cost=run(strat,K,cap,comm)
            rows.append((strat,cap,K,comm,cagr,vol,dd,sh,cost))
T=pd.DataFrame(rows,columns=["strategy","capital","K","commission","CAGR","vol","maxDD","Sharpe","costs_%/yr"])
T.to_csv(RD+"/capital_scenarios.csv",index=False)
pd.set_option("display.width",200); pd.set_option("display.max_rows",300)
f=T.copy()
for c in ("CAGR","vol","maxDD","costs_%/yr"): f[c]=(f[c]*100).round(1)
f["Sharpe"]=f.Sharpe.round(2)
print(f.to_string(index=False))
