"""Before/after replay for dropping the ML gate and decoupling the score from ML (plan docs/plan_oct2026_costs_and_scan.md, P2 option A).

Replays saved scan snapshots (git history of data/scans/latest_scan.parquet) through the REAL static gates (core/trading/policy.evaluate_static_gates)
in three configurations:
  C0  current:   ML gate [min,max] ON,  ML inside the score ON
  A1  ML gate OFF, ML inside the score ON            (what "drop the gate" alone would do)
  A2  ML gate OFF, ML removed from the score         (ML decoupled completely: ML_20d_Prob=0.5 → delta 0, no penalty)
Score for A2 = stored FinalScore_20d − [compute_final_score_20d(row) − compute_final_score_20d(row with ML=0.5)]  (the later W6 stage is assumed additive-neutral).
Read-only research. python scripts/research/ml_decoupling_replay.py [N_SCANS]"""
import io, subprocess, sys, copy
import numpy as np, pandas as pd
from core.trading.config import CONFIG
from core.trading.policy import evaluate_static_gates
from core.scoring_engine import compute_final_score_20d

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
revs = [l.split() for l in subprocess.run(["git", "log", "--format=%H %cI", "--", "data/scans/latest_scan.parquet"], capture_output=True, text=True).stdout.splitlines()]
revs = [r for r in revs if r[1] >= "2026-07-01"]; step = max(1, len(revs) // N); revs = revs[::step][:N]


def cfg_with(ml_on: bool):
    c = copy.copy(CONFIG)
    if not ml_on:
        object.__setattr__(c, "min_ml_prob", 0.0); object.__setattr__(c, "max_ml_prob", 1.0)
    return c


C0, C_OFF = cfg_with(True), cfg_with(False)
out, tickers = [], {"C0": set(), "A1": set(), "A2": set()}
for h, ts in revs:
    try:
        d = pd.read_parquet(io.BytesIO(subprocess.run(["git", "show", f"{h}:data/scans/latest_scan.parquet"], capture_output=True).stdout))
    except Exception:
        continue
    if "ML_20d_Prob" not in d.columns: continue
    n0 = n1 = n2 = 0; t0, t1, t2 = set(), set(), set()
    for _, r in d.iterrows():
        try:
            if evaluate_static_gates(r, cfg=C0).would_buy: n0 += 1; t0.add(r.get("Ticker"))
            if evaluate_static_gates(r, cfg=C_OFF).would_buy: n1 += 1; t1.add(r.get("Ticker"))
            r2 = r.copy(); r2["ML_20d_Prob"] = 0.5
            s_with, s_without = compute_final_score_20d(r), compute_final_score_20d(r2)
            delta = float(s_with) - float(s_without)
            for col in ("FinalScore_20d", "Score"):
                if col in r2.index: r2[col] = float(r[col]) - delta
            if evaluate_static_gates(r2, cfg=C_OFF).would_buy: n2 += 1; t2.add(r.get("Ticker"))
        except Exception:
            continue
    out.append((ts[:10], len(d), n0, n1, n2, len(t0 & t2), len(t0 & t1)))
    tickers["C0"] |= t0; tickers["A1"] |= t1; tickers["A2"] |= t2
df = pd.DataFrame(out, columns=["scan", "rows", "C0_pass", "A1_pass", "A2_pass", "A2∩C0", "A1∩C0"])
pd.set_option("display.width", 160); print(df.to_string(index=False))
print(f"\nscans {len(df)} | mean candidates passing ALL static gates per scan: C0 {df.C0_pass.mean():.2f}  A1 (gate off) {df.A1_pass.mean():.2f}  A2 (ML fully decoupled) {df.A2_pass.mean():.2f}")
print(f"scans with ≥1 eligible: C0 {(df.C0_pass>0).mean()*100:.0f}%  A1 {(df.A1_pass>0).mean()*100:.0f}%  A2 {(df.A2_pass>0).mean()*100:.0f}%")
print(f"distinct tickers ever eligible: C0 {len(tickers['C0'])}  A1 {len(tickers['A1'])}  A2 {len(tickers['A2'])} | A2 tickers also eligible under C0: {len(tickers['A2'] & tickers['C0'])}")
