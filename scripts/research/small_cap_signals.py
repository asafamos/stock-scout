"""Signals by market-cap band on the point-in-time universe (big + small), with band-specific round-trip costs.
usage: small_cap_signals.py <workdir>   needs univ_px/mcap/earn (>= $500M) and univ_px_small/mcap_small/earn_small."""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
RD=sys.argv[1]
PX=pd.read_pickle(RD+"/univ_px.pkl"); PX.update(pd.read_pickle(RD+"/univ_px_small.pkl"))
MC=pd.read_pickle(RD+"/univ_mcap.pkl"); MC.update(pd.read_pickle(RD+"/univ_mcap_small.pkl"))
EA=pd.read_pickle(RD+"/univ_earn.pkl"); EA.update(pd.read_pickle(RD+"/univ_earn_small.pkl"))
spy=yf.download("SPY",start="2018-06-01",auto_adjust=True,progress=False); spy.columns=[c[0] if isinstance(c,tuple) else c for c in spy.columns]
spy.index=pd.to_datetime(spy.index).tz_localize(None); cal=spy.index; D=len(cal); tick=[t for t in PX if t in MC]; N=len(tick)
def mat(): return np.full((N,D),np.nan,dtype=np.float32)
O,C,V,MCm,SURP,RSURP,EAR=(mat() for _ in range(7)); last=np.zeros(N,int)
for n,t in enumerate(tick):
    b=PX[t]; b=b[~b.index.duplicated()].reindex(cal)
    c=b["Close"].to_numpy(float)
    if np.isfinite(c).sum()<260: continue
    last[n]=np.where(np.isfinite(c))[0][-1]
    O[n],C[n],V[n]=b["Open"].to_numpy(float),c,b["Volume"].to_numpy(float)
    m=MC[t].set_index("date").marketCap; MCm[n]=m[~m.index.duplicated()].reindex(cal,method="ffill",limit=10).to_numpy(float)
    if t in EA:
        e=pd.DataFrame(EA[t]); e["date"]=pd.to_datetime(e["date"]); e=e.dropna(subset=["epsActual","epsEstimated"]).sort_values("date")
        est=e.epsEstimated.astype(float).to_numpy(); act=e.epsActual.astype(float).to_numpy(); s=np.clip((act-est)/np.maximum(np.abs(est),0.05),-3,3)
        rv=((e.revenueActual.astype(float)-e.revenueEstimated.astype(float))/e.revenueEstimated.astype(float).abs().replace(0,np.nan)).to_numpy(); rs=np.clip(rv,-1,1)
        for d,sv,rsv in zip(e.date.values,s,rs):
            i=int(cal.searchsorted(pd.Timestamp(d),side="left")); a=i+2
            if a>=D-1: continue
            end=min(D,a+45); SURP[n,a:end]=sv; RSURP[n,a:end]=rsv
            if i>=1 and i+1<D and np.isfinite(C[n,i-1]) and np.isfinite(C[n,i+1]): EAR[n,a:end]=C[n,i+1]/C[n,i-1]-1
ADDV=pd.DataFrame((V.astype(float)*C.astype(float)).T).rolling(20,min_periods=20).mean().to_numpy().T
BANDS=[("50-300M",5e7,3e8,2.0),("300M-1B",3e8,1e9,1.2),("1-5B",1e9,5e9,0.8),(">5B",5e9,1e13,0.7)]
def fwd_ret(j,h):
    e=j+1; x=j+1+h
    out=np.full(N,np.nan)
    for n in range(N):
        if not np.isfinite(O[n,e]): continue
        k=min(x,last[n]) if last[n]>e else None
        if k is None: continue
        out[n]=C[n,k]/O[n,e]-1
    return out
rows=[]; rk=lambda x: pd.Series(x).rank().to_numpy()
for j in range(300,D-70,5):
    f20=fwd_ret(j,20); f60=fwd_ret(j,60); spy20=spy.Close.iloc[min(j+21,D-1)]/spy.Open.iloc[j+1]-1; spy60=spy.Close.iloc[min(j+61,D-1)]/spy.Open.iloc[j+1]-1
    feats={"eps_surprise":SURP[:,j],"rev_surprise":RSURP[:,j],"earn_day_return":EAR[:,j],
           "mom_12_1":C[:,j-21]/C[:,j-252]-1,"mom_6_1":C[:,j-21]/C[:,j-126]-1,"rev_1m":-(C[:,j]/C[:,j-21]-1),
           "near_high52":C[:,j]/np.nanmax(C[:,j-251:j+1],axis=1),
           "vol_surge":np.nanmean(V[:,j-4:j+1],axis=1)/np.nanmean(V[:,j-60:j+1],axis=1)}
    for bname,lo,hi,cost in BANDS:
        ok=np.isfinite(C[:,j])&np.isfinite(MCm[:,j])&(MCm[:,j]>=lo)&(MCm[:,j]<hi)&(C[:,j]>=2)&np.isfinite(ADDV[:,j])&(ADDV[:,j]>=1e6)
        if ok.sum()<60: continue
        for hz,f,sp in (("20d",f20,spy20),("60d",f60,spy60)):
            for fn,x in feats.items():
                mm=ok&np.isfinite(x)&np.isfinite(f)
                if mm.sum()<60: continue
                ff=np.clip(f[mm],-0.9,3); xx=x[mm]
                ic=np.corrcoef(rk(xx),rk(ff))[0,1]
                top=xx>=np.quantile(xx,0.9); band_mean=ff.mean()
                rows.append((cal[j],bname,hz,fn,ic,ff[top].mean()-cost/100,ff[top].mean()-band_mean,ff[top].mean()-cost/100-sp,mm.sum()))
R=pd.DataFrame(rows,columns=["date","band","hz","feat","ic","top_net","top_vs_band","top_net_vs_spy","n"])
R.to_parquet(RD+"/small_cap_signals.parquet")
out=[]
for (b,h,f),g in R.groupby(["band","hz","feat"]):
    step=4 if h=="20d" else 12
    s=g.sort_values("date").iloc[::step]
    if len(s)<12: continue
    t=lambda z: z.mean()/z.std()*np.sqrt(len(z)) if z.std()>0 else np.nan
    out.append((b,h,f,s.ic.mean(),t(s.ic),s.top_vs_band.mean()*100,t(s.top_vs_band),s.top_net_vs_spy.mean()*100,t(s.top_net_vs_spy),len(s),int(g.n.mean())))
T=pd.DataFrame(out,columns=["band","hz","feature","IC","t_ic","top10_vs_band_%","t","top10_net_vs_SPY_%","t_spy","n_indep","avg_n"])
pd.set_option("display.width",250); pd.set_option("display.max_rows",400)
T.to_csv(RD+"/small_cap_signals.csv",index=False)
print("ranked by net-of-cost excess vs SPY (top decile, equal weight):")
print(T.sort_values("t_spy",ascending=False).head(25).round(3).to_string(index=False))
print("\nstrongest rank-IC overall:")
print(T.sort_values("t_ic",key=abs,ascending=False).head(15).round(3).to_string(index=False))
