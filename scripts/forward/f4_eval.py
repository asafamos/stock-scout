"""F4 evaluator — the ONE pre-registered look (docs/research_prereg_F4_earnings_text.md, addendum A2 fixes the operational details below BEFORE any event exists).

  python scripts/forward/f4_eval.py            # refuses to compute anything until >= 1,000 resolved events over >= 30 entry dates

Inputs: data/forward_text/batches/*_scores.json (score per event) + prices (FMP historical-price-eod; Open of the entry date and Open 20 trading days later, same for SPY).
Everything statistical is a pure function of a DataFrame so it can be unit-tested on synthetic data (calibrated under H0 in tests/test_f4_eval.py).

Operational definitions (A2):
* ret = Open(entry+20 trading days) / Open(entry) - 1 ; excess = ret - SPY ret over the same two dates (percent).
* IC = Spearman correlation of score and excess over the pooled sample; p-value = one-sided permutation, scores permuted WITHIN entry-week (ISO week of the entry date).
* Top bucket = score >= +3. Criterion 2: mean(excess | top) - mean(excess | all scored) >= 1.0 pp.
* Criterion 3 (net): net = excess - cost; cost = spread (25 bp for mcap < $1B, 10 bp for $1-5B, 5 bp above) + round-trip commission on $270 (PRIMARY: IB Fixed $1.00/leg; Tiered $0.35/leg reported alongside) + 0.15% slippage.
  Mean net of the top bucket > 0 AND the 5th percentile of a week-cluster bootstrap (5,000 resamples of entry-weeks) > 0.
* Verdict labels: SUCCESS (all four); NOT TRADABLE AT FIXED COSTS (criteria 1, 2, 4 and the Tiered version of 3 pass, Fixed 3 fails); otherwise DEAD.
* Criterion 4: top-minus-all difference AND IC are > 0 in BOTH halves (entry dates split at the median date), and the top-minus-all difference stays > 0 after dropping the single entry-week that contributes most to the top bucket.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

TOP_SCORE, IC_MIN, P_MAX, EXCESS_MIN_PP = 3, 0.05, 0.05, 1.0
SLIP, TRADE_USD = 0.0015, 270.0
N_PERM, N_BOOT, SEED = 5000, 5000, 20261029


def spread_bp(mcap: float) -> float:
    return 25.0 if mcap < 1e9 else (10.0 if mcap < 5e9 else 5.0)


def cost_pct(mcap: float, commission_per_leg: float) -> float:
    return spread_bp(mcap) / 100.0 + 2 * commission_per_leg / TRADE_USD * 100.0 + SLIP * 100.0


def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def perm_p_ic(score: np.ndarray, excess: np.ndarray, week: np.ndarray, n_perm: int = N_PERM, seed: int = SEED) -> float:
    """One-sided permutation p for IC > 0, scores shuffled within entry-week."""
    rng = np.random.default_rng(seed)
    rs, re_ = _rank(score), _rank(excess)
    obs = np.corrcoef(rs, re_)[0, 1]
    groups = [np.where(week == w)[0] for w in np.unique(week)]
    ge = 0
    for _ in range(n_perm):
        p = rs.copy()
        for g in groups:
            if len(g) > 1:
                p[g] = rs[rng.permutation(g)]
        if np.corrcoef(p, re_)[0, 1] >= obs:
            ge += 1
    return (ge + 1) / (n_perm + 1)


def week_bootstrap_lower(values: np.ndarray, week: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED, q: float = 5.0) -> float:
    """q-th percentile of the mean of `values` when whole entry-weeks are resampled."""
    rng = np.random.default_rng(seed)
    ws = np.unique(week)
    sums = np.array([values[week == w].sum() for w in ws])
    cnt = np.array([(week == w).sum() for w in ws])
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(ws), len(ws))
        means[i] = sums[idx].sum() / max(cnt[idx].sum(), 1)
    return float(np.percentile(means, q))


def evaluate(df: pd.DataFrame, n_perm: int = N_PERM, n_boot: int = N_BOOT) -> Dict[str, object]:
    """df columns: entry_date (str/date), score (int), excess (pct), mcap (float)."""
    d = df.copy()
    d["entry_date"] = pd.to_datetime(d["entry_date"])
    d["week"] = d["entry_date"].dt.strftime("%G-W%V")
    score, ex, wk = d["score"].to_numpy(float), d["excess"].to_numpy(float), d["week"].to_numpy()
    top = score >= TOP_SCORE
    out: Dict[str, object] = {"n": int(len(d)), "n_dates": int(d["entry_date"].nunique()), "n_top": int(top.sum())}
    out["ic"] = spearman(score, ex)
    out["p_ic"] = perm_p_ic(score, ex, wk, n_perm)
    out["top_minus_all_pp"] = float(ex[top].mean() - ex.mean()) if top.any() else float("nan")
    for name, comm in (("fixed", 1.0), ("tiered", 0.35)):
        net = ex - np.array([cost_pct(m, comm) for m in d["mcap"]])
        out[f"net_top_mean_{name}"] = float(net[top].mean()) if top.any() else float("nan")
        out[f"net_top_lower5_{name}"] = week_bootstrap_lower(net[top], wk[top], n_boot) if top.sum() > 1 else float("nan")
    med = d["entry_date"].sort_values().iloc[len(d) // 2]
    halves = []
    for m in (d["entry_date"] < med, d["entry_date"] >= med):
        s, e, t = score[m.to_numpy()], ex[m.to_numpy()], top[m.to_numpy()]
        halves.append({"ic": spearman(s, e) if len(s) > 2 else float("nan"), "top_minus_all": float(e[t].mean() - e.mean()) if t.any() else float("nan")})
    out["halves"] = halves
    if top.any():
        contrib = pd.Series(ex[top]).groupby(wk[top]).sum()
        best = contrib.idxmax()
        keep = wk != best
        kt = top & keep
        out["top_minus_all_without_best_week"] = float(ex[kt].mean() - ex[keep].mean()) if kt.any() else float("nan")
    else:
        out["top_minus_all_without_best_week"] = float("nan")
    c1 = out["ic"] >= IC_MIN and out["p_ic"] < P_MAX
    c2 = out["top_minus_all_pp"] >= EXCESS_MIN_PP
    c3 = out["net_top_mean_fixed"] > 0 and out["net_top_lower5_fixed"] > 0
    c4 = all(h["ic"] > 0 and h["top_minus_all"] > 0 for h in halves) and out["top_minus_all_without_best_week"] > 0
    out["criteria"] = {"1_ic_and_p": bool(c1), "2_top_minus_all": bool(c2), "3_net_fixed": bool(c3), "4_halves_and_best_week": bool(c4)}
    c3t = out["net_top_mean_tiered"] > 0 and out["net_top_lower5_tiered"] > 0
    if c1 and c2 and c3 and c4:
        out["verdict"] = "SUCCESS"
    elif c1 and c2 and c4 and c3t and not c3:
        out["verdict"] = "NOT TRADABLE AT FIXED COSTS (passes with Tiered)"      # information is real; the owner's IB plan is the bottleneck
    else:
        out["verdict"] = "DEAD"
    return out


# ---------- IO (not unit-tested; thin) ----------
def load_scored() -> pd.DataFrame:
    from scripts.forward.f4_collect import data_dir
    rows = []
    for p in sorted((data_dir() / "batches").glob("*_scores.json")):
        d = json.load(open(p))
        ev = {e["id"]: e for e in json.load(open(str(p).replace("_scores", "_events")))["events"]}
        for s in d["scores"]:
            rows.append({"id": s["id"], "ticker": s["ticker"], "entry_date": d["entry_date"], "score": s["score"], "mcap": ev[s["id"]]["market_cap"]})
    return pd.DataFrame(rows)


def opens(symbol: str, a: str, b: str, key: str) -> pd.Series:
    import urllib.request
    u = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={symbol}&from={a}&to={b}&apikey={key}"
    d = json.load(urllib.request.urlopen(u, timeout=60))
    d = d if isinstance(d, list) else d.get("historical", [])
    return pd.Series({r["date"]: r["open"] for r in d}).sort_index()


def build_frame() -> pd.DataFrame:
    from dotenv import load_dotenv
    from scripts.forward.f4_collect import ROOT
    from scripts.forward.f4_resolve import trading_days_after, HORIZON
    load_dotenv(ROOT / ".env")
    key = os.environ["FMP_API_KEY"]
    sc = load_scored()
    ed = pd.to_datetime(sc["entry_date"]).min().date()
    spy = opens("SPY", str(ed), str(pd.Timestamp.today().date()), key)
    out = []
    for t, g in sc.groupby("ticker"):
        px = opens(t, str(ed), str(pd.Timestamp.today().date()), key)
        for _, r in g.iterrows():
            e0 = pd.Timestamp(r["entry_date"]).date()
            e1 = trading_days_after(e0, HORIZON)
            k0, k1 = str(e0), str(e1)
            if k0 in px.index and k1 in px.index and k0 in spy.index and k1 in spy.index:
                ret = px[k1] / px[k0] - 1
                out.append({**r.to_dict(), "excess": (ret - (spy[k1] / spy[k0] - 1)) * 100})
    return pd.DataFrame(out)


def main() -> None:
    from scripts.forward.f4_resolve import main as counts
    c = counts()
    if not c["stopping_rule_met"]:
        print("BLIND: stopping rule not met — nothing computed (prereg: single look).")
        return
    df = build_frame()
    if len(df) < 1000 or df["entry_date"].nunique() < 30:
        print(f"BLIND: only {len(df)} events have complete prices — nothing computed.")
        return
    print(json.dumps(evaluate(df), indent=1, default=str))


if __name__ == "__main__":
    main()
