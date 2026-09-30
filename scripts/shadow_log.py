"""Shadow selector logger — pre-registered, additive, trades nothing.

Every scan day this logs the WHOLE scan (not just a top-50 by Score, which is what
scan_outcomes holds and why it can never answer "would a different rule have picked better?")
together with the pick flags of pre-registered selection rules. Outcomes are resolved later by
scripts/shadow_resolve.py and judged by scripts/shadow_report.py against the criteria written in
docs/shadow_selector_prereg.md BEFORE any result exists.

Nothing here feeds a gate, the ranker or an order. Removing it changes nothing live.

Rules (frozen — change = new RULE id, never edit in place):
  S1_v1  universe: Fundamental_Score finite and >= 45, sector not in CONFIG.blocked_sectors_list,
         Close > 0.  Rank: Fundamental_Score desc, ties by ticker asc.  Pick: top 3.
  S2_v1  (EXPLORATORY, added 2026-09-29 after the offline 7-year test showed the only OOS signal in
         the technical+fundamental set was a volatility/size tilt)  universe: Close > 0, ATR_Pct > 0,
         market_cap > 0, sector not blocked.  Score = rank_pct(ATR_Pct) + rank_pct(-market_cap) among
         the universe of that day.  Pick: top 3 (ties by ticker asc).  No other gates.
  S3_v1  = S2 + tradability filters (price >= 5, avg dollar volume >= $5M). This is EXACTLY the rule the
         live v2 sleeve trades (core/trading/v2_selector.py), so live and measured cannot drift apart.
  LIVE   `live_gate_pass`: policy.evaluate_static_gates(row) with the production CONFIG at log
         time (score/ML/RR/ATR/fund/confidence/sector/regime/reliability gates). Not the
         IB-dependent gates (cash, slots, quote drift) and not the ranker.

Usage: python -m scripts.shadow_log [--parquet PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("shadow_log")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "outcomes"
PICKS_PATH = OUT_DIR / "shadow_picks.jsonl"      # one row per scanned ticker per scan day
SCANS_PATH = OUT_DIR / "shadow_scans.jsonl"      # one meta row per scan day
DEFAULT_PARQUET = ROOT / "data" / "scans" / "latest_scan.parquet"

RULE_ID = "S1_v1"
S1_MIN_FUND = 45.0
S1_TOP_N = 3


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _read_keys(path: Path, key: str) -> set:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        try:
            keys.add(json.loads(line).get(key))
        except Exception:
            continue
    return keys


def select_s1(rows: List[Dict], blocked_sectors: set) -> Dict[str, int]:
    """Return {ticker: rank} (1-based) for the S1_v1 picks."""
    universe = []
    for r in rows:
        fund, close = r.get("fund"), r.get("close")
        if fund is None or fund < S1_MIN_FUND or not close or close <= 0:
            continue
        if (r.get("sector") or "") in blocked_sectors:
            continue
        universe.append(r)
    universe.sort(key=lambda r: (-r["fund"], r["ticker"]))
    return {r["ticker"]: i + 1 for i, r in enumerate(universe[:S1_TOP_N])}


def select_s2(rows: List[Dict], blocked_sectors: set) -> Dict[str, int]:
    """Return {ticker: rank} (1-based) for the S2_v1 picks (vol + small-size tilt)."""
    uni = [r for r in rows
           if r.get("close") and r["close"] > 0 and r.get("atr_pct") and r["atr_pct"] > 0
           and r.get("market_cap") and r["market_cap"] > 0
           and (r.get("sector") or "") not in blocked_sectors]
    if not uni:
        return {}
    def pct_rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        rk = [0.0] * len(vals)
        for pos, i in enumerate(order):
            rk[i] = (pos + 1) / len(vals)
        return rk
    a = pct_rank([r["atr_pct"] for r in uni])
    m = pct_rank([-r["market_cap"] for r in uni])
    scored = sorted(range(len(uni)), key=lambda i: (-(a[i] + m[i]), uni[i]["ticker"]))
    return {uni[i]["ticker"]: k + 1 for k, i in enumerate(scored[:S1_TOP_N])}


def log_scan(parquet: Path = DEFAULT_PARQUET, picks_path: Path = PICKS_PATH,
             scans_path: Path = SCANS_PATH, cfg=None) -> int:
    import pandas as pd
    if cfg is None:
        from core.trading.config import CONFIG as cfg
    from core.trading.policy import evaluate_static_gates

    df = pd.read_parquet(parquet)
    if df.empty or "As_Of_Date" not in df.columns:
        logger.warning("shadow_log: empty scan or no As_Of_Date — nothing logged")
        return 0
    scan_date = str(pd.to_datetime(df["As_Of_Date"]).max().date())

    picks_path.parent.mkdir(parents=True, exist_ok=True)
    if scan_date in _read_keys(scans_path, "scan_date"):
        logger.info("shadow_log: %s already logged — skipping (first scan of the day wins)", scan_date)
        return 0

    def g(row, *names):
        for n in names:
            if n in row.index and row[n] is not None:
                v = _f(row[n])
                if v is not None:
                    return v
        return None

    def s(row, *names):
        for n in names:
            if n in row.index and isinstance(row[n], str) and row[n]:
                return row[n]
        return ""

    rows: List[Dict] = []
    for _, r in df.iterrows():
        tkr = str(r.get("Ticker") or "").strip().upper()
        if not tkr:
            continue
        rec = {
            "scan_date": scan_date, "ticker": tkr,
            "close": g(r, "Close", "Price"), "entry_price": g(r, "Entry_Price"),
            "fund": g(r, "Fundamental_Score"), "tech": g(r, "TechScore_20d"),
            "score": g(r, "FinalScore_20d", "Score"), "ml": g(r, "ML_20d_Prob"),
            "rr": g(r, "RewardRisk", "RR_Ratio"), "atr_pct": g(r, "ATR_Pct"),
            "reliability": g(r, "Reliability_Score"),
            "fund_coverage_pct": g(r, "Fundamental_Coverage_Pct"),
            "fund_sources": g(r, "Fundamental_Sources_Count"),
            "market_cap": g(r, "market_cap", "Market_Cap"), "vol_avg": g(r, "vol_avg", "Vol_Avg"),
            "sector": s(r, "Sector", "sector"), "regime": s(r, "Market_Regime").upper(),
            "signal_quality": s(r, "SignalQuality"),
        }
        try:
            gr = evaluate_static_gates(r, cfg=cfg, state={}, held_tickers=set())
            rec["live_gate_pass"] = bool(gr.would_buy)
            rec["live_gate_fail"] = (gr.primary_reason if not gr.would_buy else "")
        except Exception as e:  # never let logging break on one odd row
            rec["live_gate_pass"] = None
            rec["live_gate_fail"] = f"gate_error:{e!r}"[:80]
        rows.append(rec)

    blocked = {x.strip() for x in getattr(cfg, "blocked_sectors_list", []) if x.strip()}
    ranks = select_s1(rows, blocked)
    ranks2 = select_s2(rows, blocked)
    from core.trading.v2_selector import select_s3
    ranks3 = select_s3(rows, blocked, top_n=S1_TOP_N)
    for rec in rows:
        rec["s1_rank"] = ranks.get(rec["ticker"])
        rec["s2_rank"] = ranks2.get(rec["ticker"])
        rec["s3_rank"] = ranks3.get(rec["ticker"])
        rec["rule_id"] = RULE_ID

    logged_at = datetime.now(timezone.utc).isoformat()
    with open(picks_path, "a") as f:
        for rec in rows:
            rec["logged_at"] = logged_at
            f.write(json.dumps(rec) + "\n")
    with open(scans_path, "a") as f:
        f.write(json.dumps({
            "scan_date": scan_date, "logged_at": logged_at, "rule_id": RULE_ID, "n_rows": len(rows),
            "s1_min_fund": S1_MIN_FUND, "s1_top_n": S1_TOP_N, "blocked_sectors": sorted(blocked),
            "s1_picks": list(ranks), "s2_picks": list(ranks2), "s3_picks": list(ranks3), "n_live_pass": sum(1 for r in rows if r.get("live_gate_pass")),
        }) + "\n")
    logger.info("shadow_log: %s — %d rows, S1 picks %s, %d pass live gates",
                scan_date, len(rows), list(ranks), sum(1 for r in rows if r.get("live_gate_pass")))
    return len(rows)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(DEFAULT_PARQUET))
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    log_scan(Path(a.parquet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
