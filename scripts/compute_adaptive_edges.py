"""Adaptive edges: nightly recompute of selection parameters from live data.

Reads data/outcomes/scan_outcomes.jsonl (rolling 90-day window),
computes optimal cohort/sector/window edges, writes results to
data/adaptive/current_edges.json. Downstream code (sector_champion.py,
order_manager.py) READS this JSON at scan time — no hardcoded lists.

Everything is DYNAMIC:
- Champion cohorts (sector × score) — any cohort with n>=MIN_N + WR>=CHAMPION_WR + p<0.05 vs universe
- Sector blocks — any sector with n>=MIN_N + mean<BLOCK_MEAN_THR + p<0.05 vs zero
- ML window — sub-buckets with mean>=0 stay; negative buckets get flagged
- RR cap — highest RR bucket with mean>=0 sets the cap

Safeguards:
- MIN_N per cohort/bucket — refuse to change if sample too small
- BASELINE preserved: falls back to hardcoded config if adaptive file missing
- HISTORY: keeps last 30 edge snapshots for audit/rollback
- APPROVAL_MODE: default is 'report_only' (writes recommendations but code still uses hardcoded).
  Set ADAPTIVE_EDGES_APPLY=1 in .env.trading to actually apply.

Runs nightly via systemd timer (see deploy/setup_vps.sh).
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import sqrt, erfc
from pathlib import Path
from typing import Optional

# Config
MIN_COHORT_N = 50        # min sample per cohort for champion status
CHAMPION_MIN_WR = 55.0   # min win rate for champion (%)
CHAMPION_MIN_MEAN = 2.0  # min mean return (%)
BLOCK_MEAN_THR = -1.0    # sector auto-block if mean below this
BLOCK_MIN_N = 100        # min sample for auto-block
WINDOW_DAYS = 90         # rolling window (calendar days) for input data
P_SIG = 0.05             # significance threshold

ROOT = Path("/home/stockscout/stock-scout-2") if Path("/home/stockscout").exists() \
       else Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "outcomes" / "scan_outcomes.jsonl"
IN_ALT = ROOT / "data" / "outcomes" / "scan_outcomes_recovered.jsonl"
OUT_DIR = ROOT / "data" / "adaptive"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CURRENT = OUT_DIR / "current_edges.json"
HISTORY_DIR = OUT_DIR / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _stats(rets):
    n = len(rets)
    if n == 0:
        return None
    m = sum(rets) / n
    if n > 1:
        var = sum((r - m) ** 2 for r in rets) / (n - 1)
        sd = sqrt(var)
    else:
        sd = 0.0
    sem = sd / sqrt(n) if n else 0.0
    wr = sum(1 for r in rets if r > 0) / n * 100
    return {"n": n, "mean": m, "sd": sd, "sem": sem, "wr": wr}


def _p_vs_zero(rets):
    s = _stats(rets)
    if not s or s["n"] < 5 or s["sem"] == 0:
        return 1.0
    t = s["mean"] / s["sem"]
    return erfc(abs(t) / sqrt(2))


def _load_records():
    path = IN_PATH if IN_PATH.exists() else IN_ALT
    if not path.exists():
        print(f"[compute_adaptive_edges] no scan_outcomes file at {path}", file=sys.stderr)
        return [], None

    cutoff = (datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)).isoformat()
    records = []
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("resolved"):
                continue
            rr = r.get("realized_return_pct")
            if rr is None:
                continue
            recorded = r.get("recorded_at") or r.get("scan_date") or ""
            if recorded < cutoff[:len(recorded)] if recorded else False:
                continue
            records.append(r)
    return records, str(path)


def _compute_champions(records):
    """Discover sector × score-band cohorts that outperform."""
    cohorts = defaultdict(list)
    for r in records:
        sec = (r.get("sector") or "").strip()
        sc = r.get("score") or r.get("final_score") or 0
        if not sec or sc <= 0:
            continue
        band = int(sc // 5) * 5
        cohorts[(sec, band, band + 5)].append(r["realized_return_pct"])

    champions = []
    for (sec, lo, hi), rets in cohorts.items():
        s = _stats(rets)
        if not s or s["n"] < MIN_COHORT_N:
            continue
        if s["wr"] < CHAMPION_MIN_WR or s["mean"] < CHAMPION_MIN_MEAN:
            continue
        p = _p_vs_zero(rets)
        if p > P_SIG:
            continue
        champions.append({
            "sector": sec,
            "score_min": lo,
            "score_max": hi,
            "n": s["n"],
            "mean": round(s["mean"], 3),
            "wr": round(s["wr"], 1),
            "p": round(p, 4),
        })
    champions.sort(key=lambda c: -c["mean"])
    return champions


def _compute_sector_blocks(records):
    """Auto-block sectors with consistently negative mean."""
    sec_rets = defaultdict(list)
    for r in records:
        s = (r.get("sector") or "").strip()
        if s:
            sec_rets[s].append(r["realized_return_pct"])

    blocks, keeps = [], []
    for sec, rets in sec_rets.items():
        s = _stats(rets)
        if not s or s["n"] < BLOCK_MIN_N:
            continue
        p = _p_vs_zero(rets)
        entry = {
            "sector": sec,
            "n": s["n"],
            "mean": round(s["mean"], 3),
            "wr": round(s["wr"], 1),
            "p": round(p, 4),
        }
        if s["mean"] < BLOCK_MEAN_THR and p < P_SIG:
            blocks.append(entry)
        else:
            keeps.append(entry)
    blocks.sort(key=lambda x: x["mean"])
    keeps.sort(key=lambda x: -x["mean"])
    return blocks, keeps


def _compute_ml_window(records):
    """Find contiguous ML sub-window with best mean."""
    edges = [i / 100 for i in range(20, 65, 5)]
    buckets = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        rets = [r["realized_return_pct"] for r in records if lo <= (r.get("ml_prob") or 0) < hi]
        s = _stats(rets)
        if s and s["n"] >= 200:
            buckets.append({
                "lo": lo, "hi": hi, "n": s["n"],
                "mean": round(s["mean"], 3), "wr": round(s["wr"], 1),
                "p": round(_p_vs_zero(rets), 4),
            })

    # Optimal contiguous window: longest run with mean > 0
    best = None
    for start in range(len(buckets)):
        for end in range(start, len(buckets)):
            window = buckets[start:end + 1]
            all_positive = all(b["mean"] > 0 for b in window)
            if not all_positive:
                continue
            total_n = sum(b["n"] for b in window)
            if total_n < 1000:
                continue
            wtd_mean = sum(b["mean"] * b["n"] for b in window) / total_n
            score = wtd_mean * (total_n ** 0.5)  # reward n
            if best is None or score > best["score"]:
                best = {
                    "lo": window[0]["lo"],
                    "hi": window[-1]["hi"],
                    "n": total_n,
                    "weighted_mean": round(wtd_mean, 3),
                    "score": score,
                }
    return {"buckets": buckets, "optimal": best}


def _compute_rr_cap(records):
    """Find highest RR bucket with mean>=0 to set the effective cap."""
    edges = [2.0, 2.5, 3.5, 5.0, 7.0, 10.0, 999]
    buckets = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        rets = [r["realized_return_pct"] for r in records if lo <= (r.get("rr") or 0) < hi]
        s = _stats(rets)
        if s and s["n"] >= 100:
            buckets.append({
                "lo": lo, "hi": hi, "n": s["n"],
                "mean": round(s["mean"], 3), "wr": round(s["wr"], 1),
            })
    # Highest RR bucket where mean >= 0
    max_cap = None
    for b in reversed(buckets):
        if b["mean"] >= 0.5:  # min +0.5% mean to be safe
            max_cap = b["hi"]
            break
    if max_cap == 999:
        max_cap = 10.0  # no upper cap ~effective
    return {"buckets": buckets, "recommended_cap": max_cap}


def _compute_fund_tier_signal(records):
    """Show fund tier stats — informs whether fund65+ deserves extra bonus."""
    tiers = [(30, 45), (45, 55), (55, 65), (65, 75), (75, 100)]
    out = []
    for lo, hi in tiers:
        rets = [r["realized_return_pct"] for r in records
                if lo <= (r.get("fundamental_score") or 0) < hi]
        s = _stats(rets)
        if s and s["n"] >= 100:
            out.append({
                "lo": lo, "hi": hi, "n": s["n"],
                "mean": round(s["mean"], 3), "wr": round(s["wr"], 1),
            })
    return out


def main():
    records, src = _load_records()
    if not records:
        print("[compute_adaptive_edges] no records in window — skipping", file=sys.stderr)
        return 1

    print(f"[compute_adaptive_edges] loaded {len(records)} records from {src}")

    edges = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "window_days": WINDOW_DAYS,
        "source": src,
        "n_records": len(records),
        "config": {
            "MIN_COHORT_N": MIN_COHORT_N,
            "CHAMPION_MIN_WR": CHAMPION_MIN_WR,
            "CHAMPION_MIN_MEAN": CHAMPION_MIN_MEAN,
            "BLOCK_MEAN_THR": BLOCK_MEAN_THR,
            "BLOCK_MIN_N": BLOCK_MIN_N,
            "P_SIG": P_SIG,
        },
        "champions": _compute_champions(records),
        "sector_blocks": _compute_sector_blocks(records)[0],
        "sector_keeps": _compute_sector_blocks(records)[1],
        "ml_window": _compute_ml_window(records),
        "rr_cap": _compute_rr_cap(records),
        "fund_tier_signal": _compute_fund_tier_signal(records),
    }

    # Write current
    with OUT_CURRENT.open("w") as f:
        json.dump(edges, f, indent=2)

    # Snapshot to history
    hist_path = HISTORY_DIR / f"edges_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    with hist_path.open("w") as f:
        json.dump(edges, f, indent=2)

    # Keep only last 30 snapshots
    snaps = sorted(HISTORY_DIR.glob("edges_*.json"), reverse=True)
    for old in snaps[30:]:
        old.unlink()

    print(f"[compute_adaptive_edges] wrote {OUT_CURRENT}")
    print(f"  champions: {len(edges['champions'])}")
    print(f"  sector_blocks: {len(edges['sector_blocks'])} recommended")
    print(f"  ml_window optimal: {edges['ml_window']['optimal']}")
    print(f"  rr_cap recommended: {edges['rr_cap']['recommended_cap']}")

    # Telegram digest if changes are significant
    apply = os.getenv("ADAPTIVE_EDGES_APPLY", "0") in ("1", "true", "yes")
    mode = "APPLY" if apply else "REPORT-ONLY"
    try:
        from core.trading.notifications import _send
        msg = f"🧠 <b>Adaptive Edges Recomputed — {mode}</b>\n\n"
        msg += f"Source: {src.split('/')[-1] if src else '?'} (n={len(records)})\n\n"
        msg += f"<b>Champions ({len(edges['champions'])}):</b>\n"
        for c in edges["champions"][:5]:
            msg += f"  • {c['sector']} score {c['score_min']}-{c['score_max']}: "
            msg += f"n={c['n']} mean={c['mean']:+.2f}% wr={c['wr']:.0f}%\n"
        if edges["sector_blocks"]:
            msg += f"\n<b>Sector auto-blocks ({len(edges['sector_blocks'])}):</b>\n"
            for b in edges["sector_blocks"][:5]:
                msg += f"  • {b['sector']}: n={b['n']} mean={b['mean']:+.2f}%\n"
        opt = edges["ml_window"].get("optimal")
        if opt:
            msg += f"\n<b>Optimal ML window:</b> {opt['lo']:.2f}-{opt['hi']:.2f} "
            msg += f"(mean={opt['weighted_mean']:+.2f}%)\n"
        rr = edges["rr_cap"]["recommended_cap"]
        msg += f"<b>Recommended RR cap:</b> {rr}\n\n"
        msg += f"Mode: <b>{mode}</b>\n"
        msg += ("Ranker/gates now read this file." if apply
                else "Ranker/gates still use hardcoded config.\n"
                     "Enable: ADAPTIVE_EDGES_APPLY=1 in .env.trading")
        _send(msg)
    except Exception as e:
        print(f"[compute_adaptive_edges] telegram send failed: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
