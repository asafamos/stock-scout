"""ML Calibration Audit — what does the model's probability MEAN?

Reads pending_scans.jsonl + scan_outcomes.jsonl and produces:
  1. Reliability diagram: predicted prob bin → actual hit rate
  2. Brier score (calibration quality, lower is better)
  3. Isotonic regression curve (what the calibrated probs WOULD be)
  4. Recommendation on threshold (raise / lower / leave at 0.33)

Why this matters: the live ML range is 0.30–0.37 (suspicious narrow).
Without this audit you can't know if:
  (A) the model is well-calibrated and just has weak signal (AUC 0.631)
  (B) the model is undercalibrated and 0.36 is much better than 0.32

The thread between scan and outcome is the (scan_date, ticker) key.
Outcomes are resolved 20 trading days after scan.

Usage:
    python -m scripts.ml_calibration_audit                # console report
    python -m scripts.ml_calibration_audit --json out.json # for piping
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
PENDING = ROOT / "data" / "outcomes" / "pending_scans.jsonl"
RESOLVED = ROOT / "data" / "outcomes" / "scan_outcomes.jsonl"


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def join_outcomes() -> List[Tuple[float, bool, Dict]]:
    """Return list of (predicted_prob, hit_target, full_record).

    `hit_target` is True if the resolved outcome reached the target
    price within the 20d window. We use this as a proxy for "this
    pick was good" vs "this pick was bad."
    """
    pending = _load_jsonl(PENDING)
    resolved = _load_jsonl(RESOLVED)

    # Index resolved by (date, ticker)
    by_key = {}
    for r in resolved:
        key = (r.get("scan_date"), r.get("ticker"))
        by_key[key] = r

    out = []
    for p in pending:
        ml_prob = float(p.get("ml_prob", 0) or 0)
        if ml_prob <= 0:
            continue
        key = (p.get("scan_date"), p.get("ticker"))
        outcome = by_key.get(key)
        if outcome is None:
            continue  # not yet resolved
        hit = bool(outcome.get("hit_target", False))
        out.append((ml_prob, hit, {**p, **outcome}))
    return out


def reliability_diagram(samples: List[Tuple[float, bool, Dict]],
                        bins: int = 10) -> List[Dict]:
    """Bucket by predicted prob, compute observed hit-rate per bucket."""
    if not samples:
        return []
    # Determine bin edges over the actual prob range, not [0,1] —
    # because ML range is 0.30-0.37; uniform [0,1] bins would put
    # almost everything in one bin and lose all resolution.
    probs = [s[0] for s in samples]
    lo, hi = min(probs), max(probs)
    if hi - lo < 0.001:
        return [{"bin": 0, "lo": lo, "hi": hi, "count": len(samples),
                 "predicted_mean": lo, "observed_rate": sum(1 for s in samples if s[1]) / len(samples)}]
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    buckets = []
    for i in range(bins):
        lo_i, hi_i = edges[i], edges[i + 1]
        in_bucket = [
            s for s in samples
            if lo_i <= s[0] < hi_i
            or (i == bins - 1 and s[0] == hi_i)
        ]
        if not in_bucket:
            continue
        avg_pred = sum(s[0] for s in in_bucket) / len(in_bucket)
        hit_rate = sum(1 for s in in_bucket if s[1]) / len(in_bucket)
        buckets.append({
            "bin": i, "lo": round(lo_i, 4), "hi": round(hi_i, 4),
            "count": len(in_bucket),
            "predicted_mean": round(avg_pred, 4),
            "observed_rate": round(hit_rate, 4),
        })
    return buckets


def brier_score(samples: List[Tuple[float, bool, Dict]]) -> float:
    """Mean squared error between predicted and observed (0 or 1)."""
    if not samples:
        return float("nan")
    return sum((p - (1 if h else 0)) ** 2 for p, h, _ in samples) / len(samples)


def auc(samples: List[Tuple[float, bool, Dict]]) -> float:
    """Compute AUC-ROC manually (no sklearn dependency)."""
    if not samples:
        return float("nan")
    pos = sum(1 for _, h, _ in samples if h)
    neg = len(samples) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    sorted_s = sorted(samples, key=lambda x: x[0])
    rank_sum = 0.0
    rank = 1
    for prob, hit, _ in sorted_s:
        if hit:
            rank_sum += rank
        rank += 1
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def threshold_grid(samples: List[Tuple[float, bool, Dict]]) -> List[Dict]:
    """For each candidate threshold, report n picks + win rate above it."""
    if not samples:
        return []
    probs = sorted(set(round(s[0], 3) for s in samples))
    out = []
    for t in probs:
        above = [s for s in samples if s[0] >= t]
        if not above:
            continue
        wins = sum(1 for s in above if s[1])
        out.append({
            "threshold": round(t, 3),
            "n_picks": len(above),
            "n_wins": wins,
            "win_rate": round(wins / len(above), 4),
        })
    return out


def isotonic_calibrate(samples: List[Tuple[float, bool, Dict]]) -> List[Tuple[float, float]]:
    """Tiny isotonic regression — pool adjacent violators (PAV) algorithm.
    Returns sorted list of (predicted, calibrated) pairs.
    """
    if not samples:
        return []
    sorted_s = sorted(samples, key=lambda x: x[0])
    # Start with each point as its own block
    blocks = [(p, float(h)) for p, h, _ in sorted_s]
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks) - 1):
            if blocks[i][1] > blocks[i + 1][1]:
                # Merge adjacent decreasing pair
                w1, v1 = 1.0, blocks[i][1]
                w2, v2 = 1.0, blocks[i + 1][1]
                merged_v = (v1 * w1 + v2 * w2) / (w1 + w2)
                blocks[i] = (blocks[i][0], merged_v)
                blocks.pop(i + 1)
                changed = True
                break
    return [(p, round(v, 4)) for p, v in blocks]


def main():
    samples = join_outcomes()
    if not samples:
        print("No resolved outcomes yet — nothing to audit.")
        print(f"Pending: {len(_load_jsonl(PENDING))}")
        print(f"Resolved: {len(_load_jsonl(RESOLVED))}")
        return 0

    print(f"=== ML CALIBRATION AUDIT ===")
    print(f"Resolved samples: {len(samples)}")
    print(f"Hit rate (overall): {sum(1 for _,h,_ in samples if h)/len(samples):.3f}")
    print()

    print("--- ROC AUC ---")
    a = auc(samples)
    print(f"  AUC: {a:.4f}  ({'> 0.7 = decent' if a > 0.7 else '< 0.7 = weak signal'})")
    print()

    print("--- Brier Score ---")
    b = brier_score(samples)
    print(f"  Brier: {b:.4f}  (lower is better; calibrated baseline ≈ 0.25)")
    print()

    print("--- Reliability Diagram (predicted vs actual hit rate per bin) ---")
    diagram = reliability_diagram(samples, bins=10)
    print(f"  {'bin':>4} {'range':>20} {'count':>7} {'pred':>8} {'actual':>8} {'gap':>8}")
    for b_ in diagram:
        gap = b_["observed_rate"] - b_["predicted_mean"]
        print(f"  {b_['bin']:>4} [{b_['lo']:.3f},{b_['hi']:.3f}] {b_['count']:>7} "
              f"{b_['predicted_mean']:>8.4f} {b_['observed_rate']:>8.4f} {gap:>+8.4f}")
    print()

    print("--- Threshold Grid (would picking only above THIS prob improve win rate?) ---")
    grid = threshold_grid(samples)
    # Print top 8 thresholds by win rate (filtered to >=10 picks)
    candidates = [g for g in grid if g["n_picks"] >= 10]
    candidates.sort(key=lambda x: -x["win_rate"])
    print(f"  {'thresh':>8} {'picks':>6} {'wins':>6} {'win_rate':>10}")
    for g in candidates[:8]:
        print(f"  {g['threshold']:>8.3f} {g['n_picks']:>6} {g['n_wins']:>6} {g['win_rate']:>10.3f}")
    print()

    # Recommendation
    print("--- Recommendation ---")
    overall_rate = sum(1 for _, h, _ in samples if h) / len(samples)
    best = candidates[0] if candidates else None
    if best is None or best["win_rate"] <= overall_rate * 1.10:
        print(f"  No threshold above current 0.33 meaningfully beats baseline win rate")
        print(f"  ({overall_rate:.3f}). The model has real signal (AUC {a:.2f}) but its")
        print(f"  point predictions don't separate winners from losers well. Best ROI:")
        print(f"  feature engineering, NOT threshold tuning.")
    else:
        improve = (best["win_rate"] / overall_rate - 1) * 100
        print(f"  Raising threshold to {best['threshold']:.3f} would improve win rate")
        print(f"  from {overall_rate:.3f} → {best['win_rate']:.3f} (+{improve:.1f}%) at the")
        print(f"  cost of trading {best['n_picks']} picks instead of {len(samples)}.")
        print(f"  RECOMMENDED: edit core/trading/config.py min_ml_prob → {best['threshold']:.3f}")

    if "--json" in sys.argv:
        idx = sys.argv.index("--json") + 1
        if idx < len(sys.argv):
            outp = Path(sys.argv[idx])
            outp.write_text(json.dumps({
                "n_samples": len(samples),
                "auc": a,
                "brier": b,
                "reliability": diagram,
                "thresholds": grid,
                "isotonic": isotonic_calibrate(samples),
            }, default=str, indent=2))
            print(f"\nFull report written to {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
