"""H1 analysis — FINRA short interest vs forward returns. Implements docs/research_prereg_H1_short_interest.md EXACTLY.

  python scripts/research/h1_short_interest.py dev
  python scripts/research/h1_short_interest.py holdout --confirm-holdout-once   # only after dev passes; one run only

Rules baked in: 9-business-day publication lag; entry = next trading day's Open after the decision date; exit = Open 10
trading days later; excess vs equal-weight universe mean on the same dates; universes U1/U2; signals S1=SIR, S2=days_to_cover.
"""
import glob, sys
import numpy as np
import pandas as pd

RD = "/Users/asafamos/StockScout/research_data"
PERIODS = {"dev": ("2018-06-01", "2021-12-31"), "holdout": ("2022-01-01", "2026-09-15")}
LAG_BD, HOLD = 9, 10

mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
assert mode in PERIODS
if mode == "holdout":
    assert "--confirm-holdout-once" in sys.argv, "holdout is evaluated ONCE per the prereg; pass --confirm-holdout-once"
lo, hi = map(pd.Timestamp, PERIODS[mode])

px = pd.read_pickle(f"{RD}/univ_px.pkl"); mc = pd.read_pickle(f"{RD}/univ_mcap.pkl")
cal = pd.DatetimeIndex(sorted(set().union(*[set(v.index) for k, v in list(px.items())[:400]])))
cal = cal[(cal >= "2018-06-01")]
OPEN = pd.DataFrame({k: v["Open"] for k, v in px.items()}).reindex(cal)
CLOSE = pd.DataFrame({k: v["Close"] for k, v in px.items()}).reindex(cal)
DVOL = pd.DataFrame({k: (v["Close"] * v["Volume"]) for k, v in px.items()}).reindex(cal).rolling(20, min_periods=15).mean()
MCAP = pd.DataFrame({k: v.set_index(pd.to_datetime(v["date"]))["marketCap"] for k, v in mc.items()}).reindex(cal, method=None)
MCAP = MCAP.reindex(cal).ffill(limit=10)
alive = OPEN.notna()

si = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{RD}/short_interest/part_*.parquet"))], ignore_index=True)
si["settlement_date"] = pd.to_datetime(si["settlement_date"])
si = si.drop_duplicates(["ticker", "settlement_date"])
print(f"short interest rows {len(si):,}  dates {si.settlement_date.nunique()}  {si.settlement_date.min().date()} -> {si.settlement_date.max().date()}")


def trading_idx_on_or_after(d):
    i = cal.searchsorted(d)
    return i if i < len(cal) else None


rows = []
for sd, g in si.groupby("settlement_date"):
    pub = pd.Timestamp(np.busday_offset(sd.date(), LAG_BD, roll="forward"))
    di = trading_idx_on_or_after(pub)                      # decision day index
    if di is None: continue
    dd = cal[di]
    if not (lo <= dd <= hi): continue
    ei, xi = di + 1, di + 1 + HOLD                          # entry open next day; exit open 10 days later
    if xi >= len(cal): continue
    si_idx = cal.searchsorted(sd, side="right") - 1         # last trading day <= settlement date
    if si_idx < 0: continue
    t = g.set_index("ticker")
    tk = [x for x in t.index if x in OPEN.columns]
    t = t.loc[tk]
    shares = MCAP.iloc[si_idx][tk] / CLOSE.iloc[si_idx][tk]
    sir = t["short_interest"] / shares
    o_in = OPEN.iloc[ei][tk]
    o_out = OPEN.iloc[xi][tk].copy()
    # delisted before exit: last available Open (prereg); require entry to exist
    last_open = OPEN.iloc[ei:xi + 1][tk].ffill().iloc[-1]
    o_out = o_out.fillna(last_open)
    ret = o_out / o_in - 1
    df = pd.DataFrame({"sir": sir, "dtc": t["days_to_cover"], "ret": ret,
                       "px": CLOSE.iloc[di][tk], "dvol": DVOL.iloc[di][tk], "mcap": MCAP.iloc[di][tk]}).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["ret", "px", "dvol", "mcap"])
    df = df[(df.px >= 5) & (df.dvol >= 1e6) & (df.mcap >= 3e8)]   # U1
    df = df[(df.sir > 0) & (df.sir < 5)]                          # sanity: ratio of SI to shares, drop units garbage
    df["date"] = dd
    rows.append(df.reset_index().rename(columns={"index": "ticker"}))

D = pd.concat(rows, ignore_index=True)
print(f"mode={mode}  decision dates {D.date.nunique()}  rows {len(D):,}  {D.date.min().date()} -> {D.date.max().date()}")


def nw_t(x, lags=2):
    x = np.asarray(x, float); n = len(x); m = x.mean(); e = x - m
    v = (e @ e) / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return m / np.sqrt(v / n)


def analyse(df, sig, label):
    out = []
    for d, g in df.groupby("date"):
        if len(g) < 50: continue
        ex = g.ret - g.ret.mean()
        ic = g[sig].rank().corr(ex.rank())
        n10 = max(len(g) // 10, 5)
        s = g.assign(ex=ex).sort_values(sig)
        d1, d10 = s.head(n10).ex.mean(), s.tail(n10).ex.mean()
        out.append((d, ic, d1, d10, len(g)))
    r = pd.DataFrame(out, columns=["date", "ic", "D1_lowSI_excess", "D10_highSI_excess", "n"]).set_index("date")
    ic = r.ic.dropna()
    se = ic.std() / np.sqrt(len(ic))
    print(f"\n[{label}] signal={sig}  dates={len(ic)}  mean IC={ic.mean():+.4f}  t={ic.mean()/se:+.2f}  NW-t={nw_t(ic):+.2f}  "
          f"pos-frac={(ic>0).mean():.2f}")
    print(f"   D1(low) excess {r.D1_lowSI_excess.mean()*100:+.3f}%  D10(high) excess {r.D10_highSI_excess.mean()*100:+.3f}%  "
          f"D1-D10 spread {(r.D1_lowSI_excess-r.D10_highSI_excess).mean()*100:+.3f}% per 10d  "
          f"(long-only D1 excess t={r.D1_lowSI_excess.mean()/(r.D1_lowSI_excess.std()/np.sqrt(len(r))):+.2f}) avg names/date {r.n.mean():.0f}")
    by = ic.groupby(ic.index.year).agg(["mean", "count"])
    print("   by year IC:", "  ".join(f"{y}:{m:+.3f}(n={int(c)})" for y, (m, c) in by.iterrows()))
    return r


U2 = D[D.mcap >= 2e9]
res = {}
for lab, df in (("U1 tradable", D), ("U2 large", U2)):
    for sig in ("sir", "dtc"):
        res[(lab, sig)] = analyse(df.dropna(subset=[sig]), sig, lab)
print("\nK=4 tests; Bonferroni threshold on |t| ≈ 2.5 at p<0.05 two-sided; prereg requires |t|>=3.0 on the HOLDOUT.")
print("Direction hypothesised: NEGATIVE IC (high short interest -> underperformance).")
