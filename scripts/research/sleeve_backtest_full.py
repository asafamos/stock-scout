"""Survivorship-reduced sleeve backtest: universe = every US-listed stock (incl. since-delisted) with mcap >= $500M at some
point, top-2000 by market cap on each signal day (like the live scan), S3_v1 ranking, atr_wide/CANARY exit, risk cap, costs."""
import sys, os, json, time, urllib.request, urllib.parse, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/Users/asafamos/StockScout/stock-scout-2")
import numpy as np, pandas as pd
from core.trading import exit_sim as xs, exit_profile as xp, v2_selector as v2
SP=sys.argv[1]; UNIV=sys.argv[2]; EARNM=sys.argv[3]; MAXPOS=int(sys.argv[4]); MODE=sys.argv[5] if len(sys.argv)>5 else "S3"
TOPN=int(os.getenv("BT_TOPN","2000")); START=os.getenv("BT_START","2019-07-01")
PX=pd.read_pickle(SP+"/univ_px.pkl"); MC=pd.read_pickle(SP+"/univ_mcap.pkl")
surv=set(pd.read_pickle(SP+"/bars_vol.pkl"))
tick=[t for t in PX if t in MC and (UNIV=="all" or t in surv)]
spy=pd.read_pickle(SP+"/bars_vol.pkl")["SPY"]; cal=spy.index; D=len(cal); N=len(tick)
def mat(): return np.full((N,D),np.nan)
O,H,L,C,V,MCm,ATR,ADDV=(mat() for _ in range(8)); first=np.zeros(N,int); last=np.zeros(N,int)
for n,t in enumerate(tick):
    b=PX[t]; b=b[~b.index.duplicated()].reindex(cal)
    o,h,l,c,v=(b[x].to_numpy(float) for x in ("Open","High","Low","Close","Volume"))
    ok=np.isfinite(c); 
    if ok.sum()<260: continue
    idx=np.where(ok)[0]; first[n],last[n]=idx[0],idx[-1]
    cc=pd.Series(c).ffill().to_numpy(); pc=np.r_[cc[0],cc[:-1]]
    tr=np.maximum.reduce([h-l,np.abs(h-pc),np.abs(l-pc)])
    atr=pd.Series(tr).rolling(14,min_periods=14).mean().to_numpy(); addv=pd.Series(v*c).rolling(20,min_periods=20).mean().to_numpy()
    m=MC[t]; ms=m.set_index("date").marketCap[~m.set_index("date").index.duplicated()].reindex(cal,method="ffill",limit=10).to_numpy(float)
    O[n],H[n],L[n],C[n],V[n],MCm[n],ATR[n],ADDV[n]=o,h,l,c,v,ms,atr,addv
scl=spy.Close.to_numpy(float)
ED={}
def earn_dates(t):
    if t in ED: return ED[t]
    key=os.environ.get("FMP_API_KEY"); arr=np.array([],dtype="datetime64[ns]")
    try:
        with urllib.request.urlopen(f"https://financialmodelingprep.com/stable/earnings?symbol={urllib.parse.quote(t)}&limit=60&apikey={key}",timeout=30) as r: e=json.loads(r.read())
        arr=np.array(sorted(pd.to_datetime([x["date"] for x in e]).astype("datetime64[ns]")),dtype="datetime64[ns]")
    except Exception: pass
    ED[t]=arr; return arr
start=cal.searchsorted(pd.Timestamp(START)); end=D-70; rng=np.random.default_rng(42)
equity=821.0; openp=[]; trades=[]; eq_hist=[]
def cost(sh,px): return max(1.0,0.005*sh)+0.0010*sh*px
def rankpct(x): return (np.argsort(np.argsort(x))+1)/len(x)
for i in range(start,end):
    for p in [p for p in openp if p["exit_i"]<=i]:
        equity+=p["pnl"]; trades.append((p["date"],p["t"],p["sh"],p["entry"],p["ret"],p["days"],p["pnl"],equity,p["why"])); openp.remove(p)
    eq_hist.append(equity); free=MAXPOS-len(openp)
    if free<=0: continue
    j=i-1                                    # signal day = prior session
    c=C[:,j]; fin=np.isfinite(c)&np.isfinite(MCm[:,j])&(MCm[:,j]>0)
    if fin.sum()<200: continue
    thr=np.partition(MCm[fin,j],-min(TOPN,fin.sum()))[-min(TOPN,fin.sum())]
    a=ATR[:,j]/np.where(c>0,c,np.nan)
    ok=fin&(MCm[:,j]>=thr)&(c>=5)&np.isfinite(a)&(a>0)&np.isfinite(ADDV[:,j])&(ADDV[:,j]>=5e6)
    idx=np.where(ok)[0]
    if len(idx)<30: continue
    if MODE=="S3":
        sc=rankpct(a[idx])+rankpct(-MCm[idx,j]); order=idx[np.lexsort((np.array(tick)[idx],-sc))][:8]
    else: order=rng.permutation(idx)[:8]
    held={p["n"] for p in openp}; opened=0
    for n in order:
        if opened>=free: break
        if n in held or not np.isfinite(O[n,i]) or not np.isfinite(C[n,j]): continue
        px=O[n,i]
        if abs(px/C[n,j]-1)*100>v2.MAX_GAP_PCT: continue
        trail=xp.wide_trail_pct(ATR[n,j]/C[n,j])
        cash=equity-sum(p["sh"]*p["entry"] for p in openp)
        sh=min(int(450//px),int((0.04*equity)//(px*trail/100.0)),int(cash//px))
        if sh<1: continue
        pol=dict(xs.POLICIES["CANARY"])
        if EARNM!="noearn":
            ed=earn_dates(tick[n]); nxt=ed[ed>np.datetime64(cal[i])]
            if len(nxt):
                cd=(nxt[0]-np.datetime64(cal[i])).astype("timedelta64[D]").astype(int)
                if cd<=5: continue
                if EARNM=="earn":
                    bu=int(cal.searchsorted(pd.Timestamp(nxt[0]),side="left"))-i-1
                    pol["max"]=max(2,min(pol["max"],bu))
        seg=slice(0,last[n]+1)       # FULL history up to the last bar (ATR needs the bars BEFORE entry); entry index = i
        r=xs.simulate(O[n,seg],H[n,seg],L[n,seg],C[n,seg],i,pol)
        why="normal"
        if r is None: continue
        if not r.get("finished"):                       # series ended (delisting/merger) before the policy did
            lastc=C[n,last[n]]; r={"ret_pct":(lastc/px-1)*100,"days":last[n]-i+1,"finished":True,"reason":"delisted"}; why="delisted"
        exitp=px*(1+r["ret_pct"]/100); pnl=sh*(exitp-px)-cost(sh,px)-cost(sh,exitp)
        openp.append(dict(n=n,t=tick[n],sh=sh,entry=px,ret=r["ret_pct"],days=r["days"],pnl=pnl,exit_i=min(i+r["days"],D-1)+1,date=cal[i],why=(r.get("reason") or why))); held.add(n); opened+=1
for p in openp: equity+=p["pnl"]; trades.append((p["date"],p["t"],p["sh"],p["entry"],p["ret"],p["days"],p["pnl"],equity,p["why"]))
T=pd.DataFrame(trades,columns=["date","t","sh","entry","ret","days","pnl","equity","why"]).sort_values("date")
yrs=(cal[end]-cal[start]).days/365.25; eq=np.array(eq_hist); dd=((np.maximum.accumulate(eq)-eq)/np.maximum.accumulate(eq)).max()
si=cal.searchsorted(cal[start]); spy_cagr=(scl[end]/scl[start])**(1/yrs)-1
print(f"[{UNIV}/{EARNM}/{MODE}/maxpos{MAXPOS}] N={N} trades={len(T)} end ${equity:,.0f} CAGR {(equity/821)**(1/yrs)-1:+.1%} (SPY {spy_cagr:+.1%}) maxDD {dd:.0%} WR {100*(T.pnl>0).mean():.0f}% mean ret {T.ret.mean():+.1f}% stops {int((T.why=='stop').sum())} delisted {int((T.why=='delisted').sum())} avg days {T.days.mean():.1f}")
print("   yearly pnl:",{y:round(g.pnl.sum()) for y,g in T.groupby(T.date.dt.year)}, "| best 3 trades:",[(r.t,round(r.ret)) for r in T.nlargest(3,'ret').itertuples()], "| worst 3:",[(r.t,round(r.ret)) for r in T.nsmallest(3,'ret').itertuples()])
T.to_parquet(SP+f"/btfull_{UNIV}_{EARNM}_{MODE}_{MAXPOS}.parquet")
