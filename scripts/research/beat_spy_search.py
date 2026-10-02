"""Systematic, pre-specified search for low-turnover strategies vs SPY, with an in-sample / out-of-sample split.
Monthly decisions on ETF data, 0.05%/side trading cost. IS = 2005-2015, OOS = 2016-2026 (never used for selection)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
U=["SPY","QQQ","IWM","EFA","EEM","AGG","IEF","TLT","GLD","VNQ","RSP","XLK","XLV","XLF","XLY","XLP","XLE","XLI","XLU","XLB","MTUM","USMV","QUAL"]
d=yf.download(U,start="2003-06-01",auto_adjust=True,progress=False)["Close"]; d=d.dropna(subset=["SPY"]).ffill()
me=d.resample("ME").last(); r=me.pct_change(); COST=0.0005
def stats(ret,name,turn=None):
    ret=ret.dropna(); eq=(1+ret).cumprod(); yrs=len(ret)/12
    return dict(name=name,CAGR=eq.iloc[-1]**(1/yrs)-1,vol=ret.std()*np.sqrt(12),maxDD=(eq/eq.cummax()-1).min(),Sharpe=ret.mean()/ret.std()*np.sqrt(12) if ret.std()>0 else np.nan)
def apply_w(W):                       # W: DataFrame of month-end target weights (decided at t, earned over t+1)
    W=W.reindex(me.index).fillna(0.0); pr=r.shift(-1).fillna(0.0)
    gross=(W*pr[W.columns]).sum(axis=1); turn=W.diff().abs().sum(axis=1).fillna(0)
    return (gross-turn*COST).shift(1)   # shift so index = month earned
def single(sym,cond):
    w=pd.DataFrame(0.0,index=me.index,columns=[sym,"IEF","CASH"] if False else [sym]); w[sym]=cond.astype(float); return w
S={}
spy_m=me.SPY
for L in (6,10,12): S[f"SPY {L}m trend → cash"]=single("SPY",(spy_m>spy_m.rolling(L).mean()))
def trend_ief(sym,L):
    c=me[sym]>me[sym].rolling(L).mean(); w=pd.DataFrame(0.0,index=me.index,columns=[sym,"IEF"]); w[sym]=c.astype(float); w["IEF"]=(~c).astype(float); return w
for sym in ("SPY","QQQ"):
    for L in (10,):
        S[f"{sym} {L}m trend → IEF"]=trend_ief(sym,L)
# vol targeting on SPY daily returns
dr=d.SPY.pct_change(); rv=(dr.rolling(63).std()*np.sqrt(252)).resample("ME").last()
for tgt,cap in ((0.10,1.0),(0.15,1.0),(0.15,1.5)):
    w=pd.DataFrame({"SPY":(tgt/rv).clip(upper=cap)},index=me.index); S[f"SPY vol-target {int(tgt*100)}% cap {cap}x"]=w
# GEM dual momentum
m12=me.pct_change(12)
w=pd.DataFrame(0.0,index=me.index,columns=["SPY","EFA","AGG"])
for t in me.index:
    if pd.isna(m12.loc[t,"SPY"]): continue
    if m12.loc[t,"SPY"]>0: w.loc[t,"SPY" if m12.loc[t,"SPY"]>=m12.loc[t,"EFA"] else "EFA"]=1.0
    else: w.loc[t,"AGG"]=1.0
S["GEM dual momentum"]=w
# sector momentum top-3 (12m), abs filter
sec=["XLK","XLV","XLF","XLY","XLP","XLE","XLI","XLU","XLB"]
def secmom(K,L):
    w=pd.DataFrame(0.0,index=me.index,columns=sec+["IEF"])
    mm=me.pct_change(L)
    for t in me.index:
        x=mm.loc[t,sec].dropna()
        if len(x)<len(sec): continue
        top=x.sort_values(ascending=False).head(K)
        for s_,v in top.items():
            if v>0: w.loc[t,s_]=1.0/K
        w.loc[t,"IEF"]=1.0-w.loc[t,sec].sum()
    return w
S["Sector momentum top3 12m (+IEF)"]=secmom(3,12); S["Sector momentum top3 6m (+IEF)"]=secmom(3,6)
for sym in ("RSP","QQQ","MTUM","USMV","QUAL","IWM"):
    S[f"{sym} buy&hold"]=pd.DataFrame({sym:1.0},index=me.index)
S["60/40 SPY/IEF"]=pd.DataFrame({"SPY":0.6,"IEF":0.4},index=me.index)
S["SPY buy&hold"]=pd.DataFrame({"SPY":1.0},index=me.index)
rows=[]
for name,W in S.items():
    ret=apply_w(W)
    for per,(a,b) in {"FULL":("2005-01-01","2026-12-31"),"IS 05-15":("2005-01-01","2015-12-31"),"OOS 16-26":("2016-01-01","2026-12-31")}.items():
        x=ret.loc[a:b]
        if x.dropna().shape[0]<24: continue
        st=stats(x,name); st["period"]=per; rows.append(st)
T=pd.DataFrame(rows)
P=T.pivot(index="name",columns="period",values=["CAGR","Sharpe","maxDD"])
spy=T[T.name=="SPY buy&hold"].set_index("period")
print("SPY buy&hold:",{p:(round(spy.loc[p,'CAGR']*100,1),round(spy.loc[p,'Sharpe'],2),round(spy.loc[p,'maxDD']*100)) for p in spy.index})
out=pd.DataFrame({"CAGR_IS%":P[("CAGR","IS 05-15")]*100,"CAGR_OOS%":P[("CAGR","OOS 16-26")]*100,"Sharpe_IS":P[("Sharpe","IS 05-15")],"Sharpe_OOS":P[("Sharpe","OOS 16-26")],"maxDD_OOS%":P[("maxDD","OOS 16-26")]*100,"CAGR_FULL%":P[("CAGR","FULL")]*100})
out["beats_SPY_CAGR_IS"]=out["CAGR_IS%"]>spy.loc["IS 05-15","CAGR"]*100; out["beats_SPY_CAGR_OOS"]=out["CAGR_OOS%"]>spy.loc["OOS 16-26","CAGR"]*100
out["beats_SPY_Sharpe_OOS"]=out["Sharpe_OOS"]>spy.loc["OOS 16-26","Sharpe"]
pd.set_option("display.width",250); pd.set_option("display.max_rows",100)
print(out.sort_values("CAGR_OOS%",ascending=False).round(2).to_string())
out.to_csv("/Users/asafamos/StockScout/research_data/beat_spy_search.csv")
