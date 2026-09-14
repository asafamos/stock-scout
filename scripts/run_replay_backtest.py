"""Fast replay backtest: consume `scan_outcomes.jsonl` records that the LIVE
pipeline already produced + resolved, apply the production gates, simulate a
portfolio, and write the same-shape report as `run_full_backtest.py`.

Rationale (see [[deep-investigation-sep14]] Phase 3 follow-up):
- `run_full_backtest.py` recomputes scores against a top-200-by-market-cap
  universe. Big-cap stocks rarely hit our 73-85 score band, so the LIVE-gates
  path returned n=0 trades in the Sep 2025-Sep 2026 window — universe mismatch,
  not a gate defect.
- The LIVE pipeline scans ~2000 tickers 3x/day and persists the resolved
  outcomes to `scan_outcomes.jsonl`. Replaying gates over that dataset uses
  the exact universe production sees, in ~1s, no API calls, no compute.
- realized_return_pct in scan_outcomes reflects hold-to-target/expiry
  outcome (per the resolver). It does NOT model trail stops, so this report
  is an UPPER bound on trail-aware performance — a follow-up will simulate
  trail 9%->5.5% via the max_return_pct / min_return_pct fields already
  captured in each record.

Usage::

    python scripts/run_replay_backtest.py --output reports/backtest_latest.json
    python scripts/run_replay_backtest.py --start 2025-09-15 --end 2026-09-14
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta
from math import sqrt
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_replay_backtest")


def _pass_gates(r: dict) -> bool:
    """Apply the same gates as core.backtest.engine._apply_production_gates.
    Kept as a local mirror to avoid the full engine's import cost + coupling."""
    from core.trading.config import CONFIG
    score = r.get("score", 0) or 0
    fund  = r.get("fundamental_score", 0) or 0
    ml    = r.get("ml_prob", 0) or 0
    rr    = r.get("rr", 0) or 0
    atr   = r.get("atr_pct", 0) or 0
    sect  = (r.get("sector") or "").strip()
    try:
        score, fund, ml, rr, atr = float(score), float(fund), float(ml), float(rr), float(atr)
    except Exception:
        return False
    if not (float(getattr(CONFIG, "min_score_to_trade", 73.0)) <= score <= float(getattr(CONFIG, "max_score_to_trade", 85.0))):
        return False
    if fund and fund < float(getattr(CONFIG, "min_fundamental_score", 45.0)):
        return False
    if ml:
        lo = float(getattr(CONFIG, "min_ml_prob", 0.40))
        hi = float(getattr(CONFIG, "max_ml_prob", 0.60))
        if not (lo <= ml <= hi):
            return False
    if rr:
        lo = float(getattr(CONFIG, "min_rr_to_trade", 2.5))
        hi = float(getattr(CONFIG, "max_rr_to_trade", 5.0) or 0.0)
        if rr < lo or (hi > 0 and rr > hi):
            return False
    if atr and atr < float(getattr(CONFIG, "min_atr_pct", 0.03)):
        return False
    blocked = getattr(CONFIG, "blocked_sectors_list", None) or []
    if sect in blocked:
        return False
    return True


def _load_records(path: Path, start_d: date, end_d: date) -> list:
    if not path.exists() or path.stat().st_size == 0:
        alt = path.parent / "scan_outcomes_recovered.jsonl"
        if alt.exists() and alt.stat().st_size > 0:
            path = alt
        else:
            logger.error("No usable scan_outcomes file at %s or %s", path, alt)
            sys.exit(1)
    logger.info("Reading %s", path)
    out = []
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("resolved"):
                continue
            if r.get("realized_return_pct") is None:
                continue
            d = (r.get("scan_date") or "")[:10]
            if not d:
                continue
            try:
                dt = date.fromisoformat(d)
            except Exception:
                continue
            if start_d <= dt <= end_d:
                out.append(r)
    logger.info("Loaded %d resolved records in window %s..%s", len(out), start_d, end_d)
    return out


def _fetch_spy_return(start_d: date, end_d: date) -> float:
    """Cheap benchmark: SPY close-to-close return over the window via yfinance."""
    try:
        import yfinance as yf
        h = yf.Ticker("SPY").history(start=start_d.isoformat(), end=(end_d + timedelta(days=1)).isoformat())
        if h is None or h.empty or len(h) < 2:
            return 0.0
        first, last = float(h["Close"].iloc[0]), float(h["Close"].iloc[-1])
        return (last / first - 1.0) * 100.0
    except Exception as e:
        logger.warning("SPY benchmark fetch failed: %s", e)
        return 0.0


def _trail_adjusted_return(max_ret: float, min_ret: float, realized: float, holding_days: int) -> tuple[float, str]:
    """Approximate the return an OUR-trail exit would have produced.

    Uses (max_return_pct, min_return_pct) from scan_outcomes to model the
    9pct -> 5.5pct(day 7) -> T0/T1/T2/T3 ratchet trail defined in CLAUDE.md.

    Limitation: scan_outcomes doesn't preserve time ordering of peak/trough,
    so we assume conservatively that if drawdown-from-peak >= trail width,
    the trail fired at (peak - trail). This may UNDERSTATE returns for
    positions where the trough came BEFORE the peak (real trail wouldn't
    have been armed yet). Best-available approximation without intraday data.

    Returns (adjusted_return_pct, exit_reason).
    """
    if max_ret is None or min_ret is None:
        return realized, "no_trail_data"

    # Effective trail width by ratchet + time-tighten (per CLAUDE.md Trail section).
    if   max_ret >= 30: trail, tier = 2.5, "T3"
    elif max_ret >= 22: trail, tier = 3.5, "T2"
    elif max_ret >= 14: trail, tier = 4.5, "T1"
    elif max_ret >= 10: trail, tier = 5.0, "T0"
    elif (holding_days or 0) >= 7: trail, tier = 5.5, "phase_B"
    else: trail, tier = 9.0, "phase_A"

    trail_exit = max_ret - trail
    # Trail fires if drawdown from peak >= trail width.
    if min_ret <= trail_exit:
        return trail_exit, f"trail_fired@{tier}"
    return realized, f"held_no_trail_{tier}"


def _simulate(records: list, max_positions: int, apply_trail: bool = True) -> dict:
    """Portfolio simulation. Per scan_date, close matured positions then open
    up to (max_positions - open) new ones ranked by score, one per ticker."""
    by_date: dict = {}
    for r in records:
        d = r["scan_date"][:10]
        by_date.setdefault(d, []).append(r)

    open_positions: list = []
    closed: list = []
    eligible_seen = 0
    n_scan_dates = len(by_date)

    for scan_d_str in sorted(by_date.keys()):
        scan_d = date.fromisoformat(scan_d_str)
        # Close positions whose exit is on/before this scan date.
        still_open = []
        for pos in open_positions:
            if pos["exit_date"] <= scan_d:
                closed.append(pos)
            else:
                still_open.append(pos)
        open_positions = still_open

        # Rank eligible candidates by score desc.
        cands = [r for r in by_date[scan_d_str] if _pass_gates(r)]
        cands.sort(key=lambda x: -(x.get("score") or 0))
        eligible_seen += len(cands)

        held = {p["ticker"] for p in open_positions}
        for c in cands:
            if len(open_positions) >= max_positions:
                break
            tk = c["ticker"]
            if tk in held:
                continue
            hd_days = c.get("holding_days") or 20
            # Trading days ≈ calendar * 7/5 (rough — good enough for buckets)
            exit_d = scan_d + timedelta(days=max(int(hd_days) * 7 // 5, 1))
            raw_ret = float(c["realized_return_pct"])
            if apply_trail:
                adj_ret, exit_reason = _trail_adjusted_return(
                    max_ret=float(c.get("max_return_pct") or raw_ret),
                    min_ret=float(c.get("min_return_pct") or raw_ret),
                    realized=raw_ret,
                    holding_days=int(hd_days),
                )
            else:
                adj_ret, exit_reason = raw_ret, "no_trail"
            open_positions.append({
                "ticker": tk,
                "entry_date": scan_d,
                "exit_date": exit_d,
                "return_pct": adj_ret,
                "raw_realized_pct": raw_ret,
                "exit_reason": exit_reason,
                "sector": c.get("sector", ""),
                "score": float(c.get("score", 0) or 0),
                "ml_prob": float(c.get("ml_prob", 0) or 0),
                "rr": float(c.get("rr", 0) or 0),
            })
            held.add(tk)

    # Force-close remaining positions at end of window.
    closed.extend(open_positions)

    return {"trades": closed, "n_scan_dates": n_scan_dates, "eligible_seen": eligible_seen}


def _metrics(trades: list, start_d: date, end_d: date, max_positions: int, initial: float, benchmark_ret: float) -> dict:
    n = len(trades)
    if n == 0:
        return {
            "n_trades": 0,
            "total_return": 0.0, "cagr": 0.0, "avg_trade_return": 0.0,
            "median_trade_return": 0.0, "sharpe": 0.0, "max_dd": 0.0,
            "volatility": 0.0, "win_rate_pct": 0.0, "profit_factor": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0,
            "benchmark_return": benchmark_ret, "excess_return": -benchmark_ret,
        }

    rets = [t["return_pct"] for t in trades]
    mean_ret = sum(rets) / n
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    wr = 100 * len(wins) / n

    # Per-trade portfolio effect: each trade uses 1/max_positions of capital.
    equity = initial
    for t in sorted(trades, key=lambda x: x["exit_date"]):
        equity *= (1.0 + (t["return_pct"] / 100.0) / max_positions)
    total_return = (equity / initial - 1.0) * 100.0

    # Sharpe from per-trade returns, annualized by trades/year ≈ 252/20 = 12.6.
    if n > 1:
        var = sum((r - mean_ret) ** 2 for r in rets) / (n - 1)
        sd = sqrt(var)
        sharpe = (mean_ret / sd * sqrt(252 / 20)) if sd > 0 else 0.0
    else:
        sd = 0.0
        sharpe = 0.0

    # Max DD from running equity across trade exits.
    eq_curve = [initial]
    eq = initial
    for t in sorted(trades, key=lambda x: x["exit_date"]):
        eq *= (1.0 + (t["return_pct"] / 100.0) / max_positions)
        eq_curve.append(eq)
    peak, max_dd = eq_curve[0], 0.0
    for e in eq_curve:
        if e > peak: peak = e
        dd = (e - peak) / peak * 100.0
        if dd < max_dd: max_dd = dd

    days = max((end_d - start_d).days, 1)
    years = days / 365.0
    cagr = ((equity / initial) ** (1.0 / max(years, 1e-6)) - 1.0) * 100.0

    tot_win = sum(wins) if wins else 0.0
    tot_loss = -sum(losses) if losses else 0.0
    pf = (tot_win / tot_loss) if tot_loss > 0 else (float("inf") if tot_win > 0 else 0.0)

    return {
        "n_trades": n,
        "total_return": total_return,
        "cagr": cagr,
        "avg_trade_return": mean_ret,
        "median_trade_return": sorted(rets)[n // 2],
        "sharpe": sharpe,
        "max_dd": max_dd,
        "volatility": sd,
        "win_rate_pct": wr,
        "profit_factor": pf if pf != float("inf") else 999.99,
        "avg_win": (sum(wins) / len(wins)) if wins else 0.0,
        "avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
        "benchmark_return": benchmark_ret,
        "excess_return": total_return - benchmark_ret,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast replay backtest from scan_outcomes.jsonl")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--outcomes", type=str, default="data/outcomes/scan_outcomes.jsonl")
    parser.add_argument("--output", type=str, default="reports/backtest_latest.json")
    parser.add_argument("--no-trail", action="store_true",
                        help="Use raw realized_return_pct instead of the 9pct->5.5pct->ratchet approximation (baseline check).")
    args = parser.parse_args()

    today = date.today()
    end_d = date.fromisoformat(args.end) if args.end else today
    start_d = date.fromisoformat(args.start) if args.start else (end_d - timedelta(days=365))

    try:
        from core.trading.config import CONFIG
    except Exception as e:
        logger.error("Cannot import CONFIG: %s", e)
        return 1
    max_positions = int(getattr(CONFIG, "max_open_positions", 3) or 3)

    records = _load_records(Path(args.outcomes), start_d, end_d)
    if not records:
        logger.error("No records in window — nothing to replay")
        return 1

    sim = _simulate(records, max_positions=max_positions, apply_trail=not args.no_trail)
    logger.info("Simulated %d trades across %d scan-dates (%d eligible cands seen)",
                len(sim["trades"]), sim["n_scan_dates"], sim["eligible_seen"])

    benchmark = _fetch_spy_return(start_d, end_d)
    logger.info("Benchmark SPY over window: %+.2f%%", benchmark)

    metrics = _metrics(sim["trades"], start_d, end_d, max_positions, args.capital, benchmark)

    report = {
        "start_date": str(start_d),
        "end_date": str(end_d),
        "n_periods": sim["n_scan_dates"],
        **metrics,
        "information_ratio": None,
        "p_value": None,
        "ci_lower": None,
        "ci_upper": None,
        "config": {
            "start_date": str(start_d),
            "end_date": str(end_d),
            "apply_prod_gates": True,
            "backtest_mode": "replay",
            "trail_simulation": not args.no_trail,  # 9pct->5.5pct->ratchet applied when true
            "max_positions": max_positions,
            "initial_capital": args.capital,
            "source": args.outcomes,
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Wrote %s", args.output)

    # Print summary
    print(json.dumps({k: metrics[k] for k in ("n_trades", "total_return", "cagr", "sharpe", "max_dd", "win_rate_pct", "profit_factor", "excess_return", "benchmark_return")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
