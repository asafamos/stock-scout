"""Judge the shadow selector against the pre-registered criteria (docs/shadow_selector_prereg.md).

Arms (all measured entry=next open, exit=20th session close, gross of cost then COST applied):
  S1    picks of rule S1_v1 (fund-first, top 3 per day)
  LIVE  every row that passes today's static live gates that day (equal-weight) — an approximation
        of what the live bot could buy; the real ranker further narrows this to <=3
  ALL   every row of the scan (universe baseline)
Inference unit = scan date (cluster bootstrap). Verdicts are only issued once N_MIN dates have
resolved; before that the report says CONTINUE and no strategy change is justified by it.

Usage: python -m scripts.shadow_report [--cost 0.5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.outcome_stats import cluster_bootstrap_mean, date_means, paired_diff  # noqa: E402

OUT_DIR = ROOT / "data" / "outcomes"
N_MIN_DATES = 60


def _read(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def build_arms(picks: List[Dict], outcomes: List[Dict], cost_pct: float) -> Dict[str, Dict[str, float]]:
    """{arm: {scan_date: mean net excess-vs-SPY %}} over resolved rows."""
    res = {(o["scan_date"], o["ticker"]): o for o in outcomes}
    arms: Dict[str, List] = {"S1": ([], []), "LIVE": ([], []), "ALL": ([], [])}
    for p in picks:
        o = res.get((p["scan_date"], p["ticker"]))
        if not o:
            continue
        ex = o["ret_pct"] - o["spy_ret_pct"] - cost_pct
        arms["ALL"][0].append(p["scan_date"]); arms["ALL"][1].append(ex)
        if p.get("s1_rank"):
            arms["S1"][0].append(p["scan_date"]); arms["S1"][1].append(ex)
        if p.get("live_gate_pass"):
            arms["LIVE"][0].append(p["scan_date"]); arms["LIVE"][1].append(ex)
    return {k: date_means(d, v) for k, (d, v) in arms.items()}


def report(picks: List[Dict], outcomes: List[Dict], cost_pct: float = 0.5) -> str:
    arms = build_arms(picks, outcomes, cost_pct)
    lines = [f"Shadow selector report — cost {cost_pct:.2f}% per round trip, horizon 20 sessions, vs SPY",
             f"logged scan days: {len({p['scan_date'] for p in picks})}, "
             f"resolved days: {len({o['scan_date'] for o in outcomes})}", ""]
    for name in ("S1", "LIVE", "ALL"):
        r = cluster_bootstrap_mean(arms[name])
        lines.append(f"{name:5s} net excess/day  n_dates={r['n_dates']:3d}  mean={r['mean']:+6.2f}%  "
                     f"95% CI [{r['lo']:+.2f}, {r['hi']:+.2f}]  P(mean<=0)={r['p_le_0']:.3f}")
    d_all = paired_diff(arms["S1"], arms["ALL"])
    d_live = paired_diff(arms["S1"], arms["LIVE"])
    lines += ["",
              f"S1 - ALL   n={d_all['n_dates']}  mean={d_all['mean']:+.2f}%  CI [{d_all['lo']:+.2f}, {d_all['hi']:+.2f}]",
              f"S1 - LIVE  n={d_live['n_dates']}  mean={d_live['mean']:+.2f}%  CI [{d_live['lo']:+.2f}, {d_live['hi']:+.2f}]",
              ""]
    n = min(d_all["n_dates"], cluster_bootstrap_mean(arms["S1"])["n_dates"])
    s1 = cluster_bootstrap_mean(arms["S1"])
    if n < N_MIN_DATES:
        lines.append(f"VERDICT: CONTINUE — {n}/{N_MIN_DATES} resolved dates. Nothing here justifies a live change yet.")
    elif s1["lo"] > 0 and d_all["lo"] > 0:
        lines.append("VERDICT: S1 PASSES pre-registered criteria (net excess CI > 0 and beats ALL). "
                     "Eligible to PROPOSE a live change to the owner — not to apply it.")
    else:
        lines.append("VERDICT: S1 does NOT pass the pre-registered criteria. Do not change the live selector on this evidence.")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=0.5)
    a = ap.parse_args(argv)
    txt = report(_read(OUT_DIR / "shadow_picks.jsonl"), _read(OUT_DIR / "shadow_outcomes.jsonl"), a.cost)
    print(txt)
    (OUT_DIR / "shadow_report.txt").write_text(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
