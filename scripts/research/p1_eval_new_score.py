"""PREREG P1 evaluator — does the post-fix Score / the policy-eligible set select better 20-day outcomes? (docs/research_prereg_P1_new_score_eval.md)

  python scripts/research/p1_eval_new_score.py                       # coverage only until the prereg criteria are met
  python scripts/research/p1_eval_new_score.py --outcomes PATH       # scan_outcomes.jsonl copied from the VPS (data/outcomes/)
  python scripts/research/p1_eval_new_score.py --final               # the one allowed FINAL look (>= 2027-03-31), alpha halved

BLIND BY DESIGN: until >= MIN_DATES resolved scan dates AND >= MIN_ELIG eligible observations exist, only COVERAGE counts are printed —
no return statistic of any kind — so nobody can peek and re-tune. The thresholds below are copied from the prereg and must not change.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.outcome_stats import rank_ic  # noqa: E402

# ---- fixed by the prereg (2026-10-19) -------------------------------------------------------------------------------------------
SINCE = "2026-10-08"          # first scans of the fully fixed pipeline (ML out of decisions, real RR, RS_63d filled, D/E ratio, cap 99)
MIN_DATES = 25                # resolved scan dates with >= MIN_NAMES names each
MIN_NAMES = 30
MIN_ELIG = 150                # resolved eligible (ticker, scan_date) observations
IC_MIN = 0.03                 # T1: mean daily rank-IC(Score, excess 20d return) must be at least this ...
EXCESS_MIN = 1.0              # T2: mean daily excess of the eligible set (pct points) must be at least this (~ 2x the 0.5pp round-trip cost)
ALPHA_FIRST = 0.025           # two primary tests -> Bonferroni; the FINAL look uses ALPHA_FINAL
ALPHA_FINAL = 0.0125
N_PERM = 2000                 # ticker-permutation draws (see _perm_p)
BLOCK = 10                    # moving-block length (in scan dates): 20-day outcome windows overlap and the same names recur, so dates are NOT independent
N_BOOT = 4000
SEED = 20261019


def block_bootstrap_mean(per_date: Dict[str, float], block: int = BLOCK, n_boot: int = N_BOOT, seed: int = SEED) -> Dict[str, float]:
    """Mean of a per-date series with a MOVING-BLOCK bootstrap over consecutive dates (keeps serial dependence).
    lo/hi = 2.5/97.5 percentiles; p_le_0 = share of bootstrap means <= 0."""
    keys = sorted(per_date)
    v = np.array([per_date[k] for k in keys], float)
    n = len(v)
    if n == 0:
        return {"n_dates": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_le_0": float("nan")}
    bl = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / bl))
    starts = rng.integers(0, n - bl + 1, size=(n_boot, nb))
    means = np.array([np.concatenate([v[s:s + bl] for s in row])[:n].mean() for row in starts])
    return {"n_dates": n, "mean": float(v.mean()), "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5)),
            "p_le_0": float((means <= 0).mean())}


def load_outcomes(path: Path, since: str = SINCE) -> pd.DataFrame:
    rows = []
    for line in open(path):
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d = d[d["resolved"].fillna(False).astype(bool)]
    d = d[d["scan_date"].astype(str) >= since]
    d = d.assign(y=pd.to_numeric(d["final_return_pct"], errors="coerce"), score=pd.to_numeric(d["score"], errors="coerce"))
    d = d.dropna(subset=["y"]).sort_values("recorded_at").drop_duplicates(["ticker", "scan_date"], keep="first")
    d["scan_date"] = d["scan_date"].astype(str).str[:10]
    d["excess"] = d["y"] - d.groupby("scan_date")["y"].transform("mean")
    return d.reset_index(drop=True)


def eligibility_from_git(since: str = SINCE) -> Set[Tuple[str, str]]:
    """(ticker, scan_date) pairs that pass the FROZEN static gates (policy.evaluate_static_gates, CONFIG at the prereg date, ML gate off)
    in ANY committed scan of that date. Gate inputs such as SignalQuality are not in scan_outcomes, so they are replayed from the
    committed scan parquets."""
    os.environ["TRADE_ML_GATE_ENABLED"] = "0"
    from core.trading import policy
    from core.trading.config import CONFIG
    commits = subprocess.check_output(["git", "-C", str(ROOT), "log", f"--since={since}", "--format=%h", "--", "data/scans/latest_scan.parquet"]).decode().split()
    out: Set[Tuple[str, str]] = set()
    for c in commits:
        try:
            df = pd.read_parquet(io.BytesIO(subprocess.check_output(["git", "-C", str(ROOT), "show", f"{c}:data/scans/latest_scan.parquet"])))
        except Exception:
            continue
        if "As_Of_Date" not in df.columns:
            continue
        sd = pd.to_datetime(df["As_Of_Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for (_, row), day in zip(df.iterrows(), sd):
            if isinstance(day, str) and policy.evaluate_static_gates(row, cfg=CONFIG, state={}, held_tickers=set()).would_buy:
                out.add((str(row["Ticker"]), day))
    return out


def coverage(df: pd.DataFrame, elig: Set[Tuple[str, str]]) -> Dict[str, int]:
    per = df.groupby("scan_date").size()
    ok_dates = per[per >= MIN_NAMES].index
    n_el = int(sum(1 for t, d in zip(df["ticker"], df["scan_date"]) if (t, d) in elig and d in set(ok_dates)))
    return {"resolved_records": int(len(df)), "resolved_dates": int(len(per)), "dates_ok": int(len(ok_dates)), "eligible_obs": n_el}


def _mats(df: pd.DataFrame, col: str):
    dates = sorted(df["scan_date"].unique()); tick = sorted(df["ticker"].unique())
    di = {d: i for i, d in enumerate(dates)}; ti = {k: i for i, k in enumerate(tick)}
    X = np.full((len(dates), len(tick)), np.nan); E = np.full_like(X, np.nan)
    for d, k, x, e in zip(df["scan_date"], df["ticker"], pd.to_numeric(df[col], errors="coerce"), df["excess"]):
        X[di[d], ti[k]] = x; E[di[d], ti[k]] = e
    return dates, tick, X, E


def _ics(X: np.ndarray, E: np.ndarray) -> list:
    out = []
    for i in range(X.shape[0]):
        m = np.isfinite(X[i]) & np.isfinite(E[i])
        if m.sum() < MIN_NAMES:
            continue
        out.append(float(np.corrcoef(X[i][m].argsort().argsort(), E[i][m].argsort().argsort())[0, 1]))
    return out


def _perm_p(stat, obs: float, ncols: int, n_perm: int, rng) -> float:
    """TICKER-permutation null: randomly re-pair each ticker's whole signal path with another ticker's outcome path. The same names recur
    day after day and 20-day windows overlap, so permuting within a date (or bootstrapping dates) treats ~40 dates as ~40 independent looks
    when there are really only a couple; permuting whole paths keeps each name's persistence intact. One-sided."""
    null = np.array([stat(rng.permutation(ncols)) for _ in range(n_perm)], float)
    null = null[np.isfinite(null)]
    return float((null >= obs).mean()) if len(null) else float("nan")


def t1_ic(df: pd.DataFrame, col: str = "score", alpha: float = ALPHA_FIRST, n_perm: int = N_PERM) -> Dict[str, float]:
    dates, tick, X, E = _mats(df, col)
    ics = _ics(X, E)
    st = block_bootstrap_mean({str(i): v for i, v in enumerate(ics)})
    rng = np.random.default_rng(SEED)
    st["perm_p"] = _perm_p(lambda pi: float(np.mean(_ics(X[:, pi], E) or [np.nan])), st["mean"], len(tick), n_perm, rng) if ics else float("nan")
    st["passed"] = bool(ics and st["mean"] >= IC_MIN and st["perm_p"] < alpha)
    return st


def _elig_mean(F: np.ndarray, E: np.ndarray) -> float:
    vals = []
    for i in range(F.shape[0]):
        m = F[i] & np.isfinite(E[i])
        if m.any():
            vals.append(float(E[i][m].mean()))
    return float(np.mean(vals)) if vals else float("nan")


def t2_eligible(df: pd.DataFrame, elig: Set[Tuple[str, str]], alpha: float = ALPHA_FIRST, n_perm: int = N_PERM) -> Dict[str, float]:
    df = df.assign(_f=[1.0 if (t, d) in elig else 0.0 for t, d in zip(df["ticker"], df["scan_date"])])
    dates, tick, F, E = _mats(df, "_f")
    ok = np.array([(np.isfinite(E[i]).sum() >= MIN_NAMES) for i in range(len(dates))])
    F = np.nan_to_num(F) > 0.5
    F[~ok], E[~ok] = False, np.nan                                    # dates with fewer than MIN_NAMES resolved names are not used
    per_date = {dates[i]: float(E[i][F[i] & np.isfinite(E[i])].mean()) for i in range(len(dates)) if (F[i] & np.isfinite(E[i])).any()}
    st = block_bootstrap_mean(per_date)
    rng = np.random.default_rng(SEED)
    obs = _elig_mean(F, E)
    st["perm_p"] = _perm_p(lambda pi: _elig_mean(F[:, pi], E), obs, len(tick), n_perm, rng) if per_date else float("nan")
    st["passed"] = bool(per_date and st["mean"] >= EXCESS_MIN and st["lo"] > 0 and st["perm_p"] < alpha)
    return st


def decide(t1: Dict[str, float], t2: Dict[str, float]) -> str:
    if t1["passed"] and t2["passed"]:
        return "DEMONSTRATED EDGE (both primary tests pass)"
    if t1["passed"] or t2["passed"]:
        return "PARTIAL (only one primary test passes) — owner decision"
    if t1["mean"] <= 0.01 and t2["mean"] <= 0.5:
        return "NO DEMONSTRATED EDGE"
    return "INCONCLUSIVE"


def partial_ic_after_atr(df: pd.DataFrame, col: str = "score") -> Dict[str, float]:
    """Descriptive: IC of the score after removing its (rank) dependence on ATR_Pct within each date — 'does it add beyond volatility?'"""
    ics = {}
    for d, g in df.groupby("scan_date"):
        g = g.dropna(subset=[col, "atr_pct", "excess"])
        if len(g) < MIN_NAMES:
            continue
        rs, ra = g[col].rank().to_numpy(float), g["atr_pct"].rank().to_numpy(float)
        beta = np.polyfit(ra, rs, 1)
        resid = rs - np.polyval(beta, ra)
        ics[d] = rank_ic(resid, g["excess"])
    return block_bootstrap_mean({k: v for k, v in ics.items() if np.isfinite(v)})


def run(df: pd.DataFrame, elig: Set[Tuple[str, str]], final: bool = False) -> Dict[str, object]:
    cov = coverage(df, elig)
    ready = final or (cov["dates_ok"] >= MIN_DATES and cov["eligible_obs"] >= MIN_ELIG)
    if not ready:
        return {"coverage": cov, "verdict": f"NO VERDICT — need >= {MIN_DATES} resolved dates (have {cov['dates_ok']}) and >= {MIN_ELIG} eligible observations (have {cov['eligible_obs']}); no return statistics are shown before then"}
    alpha = ALPHA_FINAL if final else ALPHA_FIRST
    a, b = t1_ic(df, alpha=alpha), t2_eligible(df, elig, alpha=alpha)
    return {"coverage": cov, "alpha": alpha, "T1_score_IC": a, "T2_eligible_excess": b, "verdict": decide(a, b),
            "descriptive": {"IC_after_ATR": partial_ic_after_atr(df),
                            "IC_atr_rank_baseline": t1_ic(df.assign(atr=pd.to_numeric(df["atr_pct"], errors="coerce")).dropna(subset=["atr"]), col="atr", alpha=alpha),
                            "IC_fundamental": t1_ic(df.assign(f=pd.to_numeric(df["fundamental_score"], errors="coerce")).dropna(subset=["f"]), col="f", alpha=alpha),
                            "IC_technical": t1_ic(df.assign(t=pd.to_numeric(df["technical_score"], errors="coerce")).dropna(subset=["t"]), col="t", alpha=alpha)}}


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcomes", default=str(ROOT / "data" / "outcomes" / "scan_outcomes.jsonl"))
    ap.add_argument("--final", action="store_true", help="the one allowed final look (>= 2027-03-31); alpha halved")
    a = ap.parse_args(argv)
    if a.final and pd.Timestamp.now() < pd.Timestamp("2027-03-31"):
        print("--final is only allowed on/after 2027-03-31 (prereg)"); return 2
    df = load_outcomes(Path(a.outcomes))
    if df.empty:
        print("NO VERDICT — no resolved scan_outcomes records since", SINCE, "(a record resolves ~20 trading days after its scan)"); return 0
    res = run(df, eligibility_from_git(), final=a.final)
    print(json.dumps(res, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
