"""Scan Outcome Tracker — feedback loop for ML improvement.

For every scan snapshot, records the tickers we recommended + their scan-time
prices. Then, 20 trading days later, it looks up the actual outcome
(max/min/close) and writes (scan_row, actual_return, hit_target?, hit_stop?)
to `data/outcomes/scan_outcomes.jsonl` — append-only.

This labeled dataset is the TRUTH the model should be learning from. Much
more valuable than synthetic backtest labels because:
  - It reflects ACTUAL market conditions during your operating period.
  - It captures scan→execution latency effects.
  - Bad signals we didn't trade are still labeled (full coverage).

Usage:
  # Record today's scan (run after every scan):
  python -m scripts.track_scan_outcomes --record

  # Resolve mature outcomes (run nightly — looks for ≥20-trading-day-old):
  python -m scripts.track_scan_outcomes --resolve

Schedule (VPS cron or systemd timer):
  # After scan: piggyback on the scan workflow
  # Resolve nightly: 02:00 UTC

The resulting JSONL is the feed for the next retraining cycle.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUTCOMES_DIR = ROOT / "data" / "outcomes"
PENDING_PATH = OUTCOMES_DIR / "pending_scans.jsonl"    # scan snapshot awaiting outcome
RESOLVED_PATH = OUTCOMES_DIR / "scan_outcomes.jsonl"    # resolved outcomes (training data)
SCAN_PARQUET = ROOT / "data" / "scans" / "latest_scan.parquet"
SCAN_JSON = ROOT / "data" / "scans" / "latest_scan.json"


def _ensure_dirs():
    OUTCOMES_DIR.mkdir(parents=True, exist_ok=True)
    for p in (PENDING_PATH, RESOLVED_PATH):
        if not p.exists():
            p.touch()


def _read_jsonl(path: Path) -> List[Dict]:
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


def _append_jsonl(path: Path, records: List[Dict]):
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


def _capture_market_context() -> Dict:
    """Snapshot market state at scan time — CRITICAL for regime-aware learning.

    Without this, we can't later segment outcomes by regime (which is the
    whole point of phase 3c multi-regime model). Capture once per scan and
    attach to every candidate from that scan.
    """
    ctx = {
        "spy_close": None,
        "spy_sma50_pct": None,   # SPY / SMA50 - 1, positive = above
        "spy_sma200_pct": None,
        "spy_ret_5d": None,
        "spy_ret_20d": None,
        "vix_proxy": None,       # SPY 20-day realized vol × √252
        "market_regime": None,
    }
    try:
        import yfinance as yf
        import numpy as np
        hist = yf.Ticker("SPY").history(period="250d", interval="1d")
        if hist is None or len(hist) < 200:
            return ctx
        closes = hist["Close"].dropna()
        if len(closes) < 200:
            return ctx
        spy = float(closes.iloc[-1])
        sma50 = float(closes.iloc[-50:].mean())
        sma200 = float(closes.iloc[-200:].mean())
        ret_5d = float(closes.iloc[-1] / closes.iloc[-6] - 1) if len(closes) >= 6 else 0
        ret_20d = float(closes.iloc[-1] / closes.iloc[-21] - 1) if len(closes) >= 21 else 0
        # 20-day realized vol annualized — VIX proxy
        returns = closes.pct_change().dropna()
        vix_proxy = float(returns.iloc[-20:].std() * (252 ** 0.5) * 100) if len(returns) >= 20 else None

        ctx["spy_close"] = round(spy, 2)
        ctx["spy_sma50_pct"] = round((spy / sma50 - 1) * 100, 2)
        ctx["spy_sma200_pct"] = round((spy / sma200 - 1) * 100, 2)
        ctx["spy_ret_5d"] = round(ret_5d * 100, 2)
        ctx["spy_ret_20d"] = round(ret_20d * 100, 2)
        ctx["vix_proxy"] = round(vix_proxy, 2) if vix_proxy else None

        # Regime classification (same logic used in scan pipeline)
        if spy > sma50 > sma200 and ret_20d > 0.02:
            ctx["market_regime"] = "STRONG_UPTREND"
        elif spy > sma200 and ret_20d > 0:
            ctx["market_regime"] = "UPTREND"
        elif abs(ret_20d) < 0.02 and abs(spy / sma50 - 1) < 0.03:
            ctx["market_regime"] = "RANGING"
        elif spy < sma200 and ret_20d < -0.03:
            ctx["market_regime"] = "DOWNTREND"
        elif ret_5d < -0.05:
            ctx["market_regime"] = "PANIC"
        else:
            ctx["market_regime"] = "NEUTRAL"
    except Exception as e:
        logger.warning("Market context capture failed: %s", e)
    return ctx


def _load_scan() -> List[Dict]:
    """Load the latest scan. ALWAYS prefer parquet when it exists — the
    JSON file is metadata-only after the GH Actions save_scan rewrite,
    and after `git checkout` both files share identical mtimes so the
    old `>` mtime comparison was an arbitrary tiebreak that frequently
    picked the wrong (metadata-only) file. (Audit finding #4.)
    """
    import pandas as pd
    if SCAN_PARQUET.exists():
        df = pd.read_parquet(SCAN_PARQUET)
    elif SCAN_JSON.exists():
        df = pd.read_json(SCAN_JSON)
    else:
        return []
    # Normalize — keep top-50 candidates by score (not just the top-3 bought)
    score_col = next(
        (c for c in ["FinalScore_20d", "Score", "final_score"] if c in df.columns),
        None,
    )
    if score_col is None:
        logger.warning("No score column found in scan")
        return []
    df = df.nlargest(50, score_col).reset_index(drop=True)
    return df.to_dict(orient="records")


def record_today():
    """Append today's top-50 scan candidates to the pending queue.

    Each record includes a snapshot of MARKET CONTEXT at scan time
    (SPY levels, regime, VIX proxy) — this is what makes the dataset
    useful for regime-aware learning later.
    """
    _ensure_dirs()
    rows = _load_scan()
    if not rows:
        logger.warning("No scan rows loaded — nothing to record")
        return 0
    today = date.today().isoformat()

    # Capture market context ONCE per record-run (cheap for the 50 tickers to share)
    ctx = _capture_market_context()
    logger.info(
        "Market context: regime=%s spy=%s sma50_gap=%s%% sma200_gap=%s%% vix=%s",
        ctx.get("market_regime"), ctx.get("spy_close"),
        ctx.get("spy_sma50_pct"), ctx.get("spy_sma200_pct"),
        ctx.get("vix_proxy"),
    )

    # Dedupe: don't record the same (date, ticker) twice
    existing = _read_jsonl(PENDING_PATH)
    existing_keys = {
        (r.get("scan_date"), r.get("ticker"))
        for r in existing if not r.get("resolved")
    }

    new_records = []
    for row in rows:
        ticker = row.get("Ticker") or row.get("ticker") or ""
        if not ticker or (today, ticker) in existing_keys:
            continue
        rec = {
            "scan_date": today,
            "ticker": ticker,
            "entry_price": float(row.get("Entry_Price", row.get("Close", 0)) or 0),
            "stop_loss": float(row.get("Stop_Loss", row.get("stop_loss", 0)) or 0),
            "target_price": float(row.get("Target_Price", row.get("target_price", 0)) or 0),
            "score": float(row.get("FinalScore_20d", row.get("Score", 0)) or 0),
            "rr": float(row.get("RewardRisk", row.get("RR", 0)) or 0),
            "ml_prob": float(row.get("ML_20d_Prob", row.get("ML_Probability", 0)) or 0),
            "sector": row.get("Sector") or row.get("sector") or "",
            "atr_pct": float(row.get("ATR_Pct", row.get("atr_pct", 0)) or 0),
            "holding_days": int(row.get("HoldingDays", row.get("holding_days", 20)) or 20),
            # Additional scan-time features (when present) — enables richer learning
            "technical_score": float(row.get("TechScore_20d", row.get("TechScore", 0)) or 0),
            "fundamental_score": float(row.get("Fundamental_Score", 0) or 0),
            "reliability": float(row.get("Reliability_Score", 0) or 0),
            "volume_surge": float(row.get("VolumeSurge", row.get("Volume_Surge", 0)) or 0),
            "rsi": float(row.get("RSI", 0) or 0),
            "recorded_at": datetime.utcnow().isoformat(),
            "resolved": False,
        }
        # Attach market context
        rec.update({f"mkt_{k}": v for k, v in ctx.items()})
        new_records.append(rec)

    _append_jsonl(PENDING_PATH, new_records)
    logger.info("Recorded %d new scan candidates for %s (regime: %s)",
                len(new_records), today, ctx.get("market_regime"))
    return len(new_records)


def _resolve_one(rec: Dict) -> Dict:
    """Look up actual outcome for a pending scan record using yfinance.

    Success criteria:
      hit_target = high >= target_price within holding_days
      hit_stop   = low  <= stop_loss within holding_days
      (if both, whichever triggered first wins — approximated via day-level OHLC)

    Adds fields: max_return_pct, min_return_pct, final_return_pct,
                 hit_target, hit_stop, resolved=True, resolved_at
    """
    import yfinance as yf
    ticker = rec["ticker"]
    scan_date = datetime.fromisoformat(rec["scan_date"]).date()
    holding_days = int(rec.get("holding_days", 20))
    entry = float(rec["entry_price"])
    target = float(rec["target_price"])
    stop = float(rec["stop_loss"])
    end_date = scan_date + timedelta(days=int(holding_days * 1.6) + 5)  # buffer for weekends

    try:
        # +1 day start to skip the scan-close entry bar
        hist = yf.Ticker(ticker).history(
            start=(scan_date + timedelta(days=1)).isoformat(),
            end=end_date.isoformat(),
            interval="1d",
        )
        if hist is None or len(hist) == 0:
            return {**rec, "resolved": False, "resolve_error": "no_history"}

        # Truncate to N trading days
        hist = hist.head(holding_days)
        highs = hist["High"].to_list()
        lows = hist["Low"].to_list()
        closes = hist["Close"].to_list()
        if not closes:
            return {**rec, "resolved": False, "resolve_error": "empty_ohlc"}

        max_h = max(highs)
        min_l = min(lows)
        final_close = closes[-1]

        hit_target = (target > 0 and max_h >= target)
        hit_stop = (stop > 0 and min_l <= stop)

        max_ret_pct = (max_h / entry - 1) * 100 if entry > 0 else 0
        min_ret_pct = (min_l / entry - 1) * 100 if entry > 0 else 0
        final_ret_pct = (final_close / entry - 1) * 100 if entry > 0 else 0

        # Realized outcome: hit target → target return; hit stop → stop return;
        # otherwise final close return
        if hit_target and not hit_stop:
            realized_ret = (target / entry - 1) * 100
            outcome = "target_hit"
        elif hit_stop and not hit_target:
            realized_ret = (stop / entry - 1) * 100
            outcome = "stop_hit"
        elif hit_target and hit_stop:
            # Both — assume stop (conservative; a mid-day look would be ideal)
            realized_ret = (stop / entry - 1) * 100
            outcome = "both_(stop_assumed)"
        else:
            realized_ret = final_ret_pct
            outcome = "time_expired"

        return {
            **rec,
            "resolved": True,
            "resolved_at": datetime.utcnow().isoformat(),
            "max_return_pct": round(max_ret_pct, 2),
            "min_return_pct": round(min_ret_pct, 2),
            "final_return_pct": round(final_ret_pct, 2),
            "realized_return_pct": round(realized_ret, 2),
            "outcome": outcome,
            "hit_target": hit_target,
            "hit_stop": hit_stop,
            "trading_days_data": len(closes),
        }
    except Exception as e:
        return {**rec, "resolved": False, "resolve_error": str(e)[:120]}


def resolve_matured(min_age_days: int = 20, max_per_run: int = 200) -> int:
    """Find unresolved scan records ≥ min_age_days old and try to resolve them.

    Writes resolved records to scan_outcomes.jsonl. Keeps unresolved ones
    in pending_scans.jsonl for retry.
    """
    _ensure_dirs()
    pending = _read_jsonl(PENDING_PATH)
    if not pending:
        logger.info("No pending records to resolve")
        return 0

    cutoff = date.today() - timedelta(days=min_age_days)
    matured = [
        r for r in pending
        if not r.get("resolved")
        and r.get("scan_date")
        and datetime.fromisoformat(r["scan_date"]).date() <= cutoff
    ][:max_per_run]

    if not matured:
        logger.info("No matured records yet (min_age=%dd)", min_age_days)
        return 0

    logger.info("Resolving %d matured records", len(matured))
    resolved = []
    still_pending = []
    for rec in matured:
        out = _resolve_one(rec)
        if out.get("resolved"):
            resolved.append(out)
        else:
            still_pending.append(rec)

    # Rewrite pending file: keep unresolved (either too young OR just-failed)
    matured_ids = {(r["scan_date"], r["ticker"]) for r in matured}
    keep_records = [
        r for r in pending
        if (r.get("scan_date"), r.get("ticker")) not in matured_ids
        or not r.get("resolved", False)  # double-safety
    ]
    # remove already-matured-and-just-resolved from keep_records
    resolved_ids = {(r["scan_date"], r["ticker"]) for r in resolved}
    keep_records = [r for r in keep_records
                    if (r.get("scan_date"), r.get("ticker")) not in resolved_ids]
    # ... and re-add still-pending (ones that failed to resolve but matured)
    keep_records.extend(still_pending)

    # Atomic rewrite
    tmp = PENDING_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for r in keep_records:
            f.write(json.dumps(r, default=str) + "\n")
    tmp.replace(PENDING_PATH)

    _append_jsonl(RESOLVED_PATH, resolved)
    logger.info("Resolved %d, still pending %d (total kept %d)",
                len(resolved), len(still_pending), len(keep_records))
    return len(resolved)


def summarize():
    """Print a quick summary of the resolved outcomes file."""
    _ensure_dirs()
    pending = _read_jsonl(PENDING_PATH)
    outcomes = _read_jsonl(RESOLVED_PATH)

    print("═" * 60)
    print("SCAN OUTCOMES TRACKER")
    print("═" * 60)
    pending_unresolved = [r for r in pending if not r.get("resolved")]
    distinct_dates = len({r.get("scan_date") for r in pending_unresolved})
    print(f"Pending (awaiting 20d maturity): {len(pending_unresolved)} "
          f"across {distinct_dates} scan dates")
    if pending_unresolved:
        earliest = min((r.get("scan_date") for r in pending_unresolved), default="")
        latest   = max((r.get("scan_date") for r in pending_unresolved), default="")
        print(f"  Date range: {earliest} → {latest}")
        # Regime distribution of pending
        regime_counts: Dict[str, int] = {}
        for r in pending_unresolved:
            rg = r.get("mkt_market_regime") or "UNKNOWN"
            regime_counts[rg] = regime_counts.get(rg, 0) + 1
        print("  Regime dist:")
        for rg, n in sorted(regime_counts.items(), key=lambda x: -x[1]):
            print(f"    {rg:16s} n={n}")

    print(f"\nResolved outcomes: {len(outcomes)}")
    if not outcomes:
        print("  <need ≥20 trading days from scan → outcome>")
        return

    n = len(outcomes)
    hits_t = sum(1 for r in outcomes if r.get("hit_target"))
    hits_s = sum(1 for r in outcomes if r.get("hit_stop"))
    avg_realized = sum(r.get("realized_return_pct", 0) for r in outcomes) / n
    wins = [r for r in outcomes if r.get("realized_return_pct", 0) > 0]
    losses = [r for r in outcomes if r.get("realized_return_pct", 0) < 0]
    win_rate = len(wins) / n * 100 if n else 0
    avg_win = sum(r["realized_return_pct"] for r in wins) / len(wins) if wins else 0
    avg_loss = sum(r["realized_return_pct"] for r in losses) / len(losses) if losses else 0
    pf = (sum(r["realized_return_pct"] for r in wins)
          / abs(sum(r["realized_return_pct"] for r in losses))) if losses else float("inf")

    print(f"  Win rate:        {win_rate:.1f}%")
    print(f"  Target hit rate: {hits_t/n*100:.1f}%")
    print(f"  Stop hit rate:   {hits_s/n*100:.1f}%")
    print(f"  Avg realized:    {avg_realized:+.2f}%")
    print(f"  Avg win / loss:  {avg_win:+.2f}% / {avg_loss:+.2f}%")
    print(f"  Profit factor:   {pf:.2f}×")

    # By score tier
    print("\nBy score tier:")
    tiers = [("≥82", 82, 200), ("78-82", 78, 82), ("75-78", 75, 78), ("<75", 0, 75)]
    for label, lo, hi in tiers:
        subset = [r for r in outcomes if lo <= r.get("score", 0) < hi]
        if subset:
            w = sum(1 for r in subset if r.get("realized_return_pct", 0) > 0) / len(subset) * 100
            a = sum(r.get("realized_return_pct", 0) for r in subset) / len(subset)
            print(f"  Score {label:7s}: n={len(subset):3d}  win={w:5.1f}%  avg={a:+6.2f}%")

    # By ML probability tier
    print("\nBy ML probability tier:")
    ml_tiers = [("≥0.40", 0.40, 1.01), ("0.33-0.40", 0.33, 0.40),
                ("0.30-0.33", 0.30, 0.33), ("<0.30", 0, 0.30)]
    for label, lo, hi in ml_tiers:
        subset = [r for r in outcomes if lo <= r.get("ml_prob", 0) < hi]
        if subset:
            w = sum(1 for r in subset if r.get("realized_return_pct", 0) > 0) / len(subset) * 100
            a = sum(r.get("realized_return_pct", 0) for r in subset) / len(subset)
            print(f"  ML {label:11s}: n={len(subset):3d}  win={w:5.1f}%  avg={a:+6.2f}%")

    # By regime (the whole point of regime tagging!)
    print("\nBy market regime at scan time:")
    regimes: Dict[str, List] = {}
    for r in outcomes:
        rg = r.get("mkt_market_regime") or "UNKNOWN"
        regimes.setdefault(rg, []).append(r)
    for rg, subset in sorted(regimes.items(), key=lambda x: -len(x[1])):
        w = sum(1 for r in subset if r.get("realized_return_pct", 0) > 0) / len(subset) * 100
        a = sum(r.get("realized_return_pct", 0) for r in subset) / len(subset)
        print(f"  {rg:16s}: n={len(subset):3d}  win={w:5.1f}%  avg={a:+6.2f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record", action="store_true", help="Record today's scan candidates")
    p.add_argument("--resolve", action="store_true", help="Resolve matured (≥20d) outcomes")
    p.add_argument("--summarize", action="store_true", help="Show outcome stats")
    p.add_argument("--min-age", type=int, default=20, help="Min days before resolving")
    args = p.parse_args()

    if args.record:
        record_today()
    if args.resolve:
        resolve_matured(min_age_days=args.min_age)
    if args.summarize or not (args.record or args.resolve):
        summarize()


if __name__ == "__main__":
    main()
