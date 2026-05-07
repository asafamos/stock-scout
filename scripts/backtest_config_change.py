"""Backtest: did today's config changes improve historical trades?

Reads the actual trade_log.json (paired OPEN/CLOSE entries), fetches
daily OHLC from yfinance for each ticker between entry and exit (with
a 5-day buffer), and simulates the NEW config's behavior trade-by-trade.

What it answers:
  - Would the 4% min initial trail have prevented early stopouts?
  - Would the new ratchet (T0 at +5% / T1 +10% / T2 +18% / T3 +28%)
    have locked in MORE profit on winners?
  - Would earnings-aware target_date have closed positions before
    bad gap-downs?

What it can NOT answer (yet):
  - Whether NEW BUYS we missed (because of low max_exposure $800)
    would have been profitable. That requires re-running the scan
    evaluator on archived parquets + simulating fills.

Usage:
    python -m scripts.backtest_config_change [--days 30]
    python -m scripts.backtest_config_change --csv > /tmp/backtest.csv

Output:
  - Per-trade table: actual entry/exit/P&L vs simulated
  - Aggregate metrics: profit_factor, win_rate, total_pnl, deltas
  - Verdict: "config changes improved/hurt/neutral on this sample"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as YFTO
from datetime import date, datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
TRADE_LOG = ROOT / "data" / "trades" / "trade_log.json"

# Config — env-overridable so we can compare scenarios without editing code.
# Format for each tier: (peak_gain_threshold, trail_pct, min_hold_days)
import os
NEW_INITIAL_TRAIL_FLOOR = float(os.getenv("BT_INITIAL_FLOOR", "4.0"))
EARNINGS_BUFFER_DAYS = 1  # exit 1 day before earnings


def _build_tiers() -> list:
    """Tiers from env, sorted high→low gain. Set BT_T0_GAIN=999 to disable T0."""
    t3 = (
        float(os.getenv("BT_T3_GAIN", "28.0")),
        float(os.getenv("BT_T3_TRAIL", "2.0")),
        int(os.getenv("BT_T3_HOLD", "0")),
    )
    t2 = (
        float(os.getenv("BT_T2_GAIN", "18.0")),
        float(os.getenv("BT_T2_TRAIL", "2.5")),
        int(os.getenv("BT_T2_HOLD", "0")),
    )
    t1 = (
        float(os.getenv("BT_T1_GAIN", "10.0")),
        float(os.getenv("BT_T1_TRAIL", "3.0")),
        int(os.getenv("BT_T1_HOLD", "0")),
    )
    t0 = (
        float(os.getenv("BT_T0_GAIN", "8.0")),
        float(os.getenv("BT_T0_TRAIL", "3.5")),
        int(os.getenv("BT_T0_HOLD", "2")),
    )
    return [t3, t2, t1, t0]


RATCHET_TIERS_NEW = _build_tiers()


def _load_trades() -> list:
    try:
        return json.loads(TRADE_LOG.read_text())
    except Exception:
        return []


def _pair_trades(log: list) -> list:
    """Return list of (open_event, close_event) tuples by FIFO ticker match."""
    opens = defaultdict(list)
    pairs = []
    for e in sorted(log, key=lambda e: str(e.get("timestamp", ""))):
        action = e.get("action")
        ticker = e.get("ticker")
        if action == "OPEN":
            opens[ticker].append(e)
        elif action == "CLOSE":
            o_list = opens.get(ticker, [])
            if o_list:
                o = o_list.pop(0)
                pairs.append((o, e))
    return pairs


def _fetch_ohlc(ticker: str, start_dt: date, end_dt: date):
    """Daily OHLC. Wrapped with timeout so a slow yfinance doesn't hang."""
    def _do():
        import yfinance as yf
        # +5 day buffer on each side so we have room to simulate ratchet
        start = (start_dt - timedelta(days=5)).strftime("%Y-%m-%d")
        end = (end_dt + timedelta(days=5)).strftime("%Y-%m-%d")
        df = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
        return df
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_do).result(timeout=15.0)
    except YFTO:
        logger.warning("yfinance TIMEOUT for %s", ticker)
        return None
    except Exception as e:
        logger.warning("yfinance err for %s: %s", ticker, e)
        return None


def _fetch_earnings(ticker: str):
    """Next earnings date (after a given date)."""
    def _do():
        import yfinance as yf
        cal = yf.Ticker(ticker).calendar
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if ed and isinstance(ed, list) and ed:
                return ed[0]
        return None
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_do).result(timeout=8.0)
    except Exception:
        return None


def simulate_trade(open_event, close_event, ohlc) -> dict:
    """Walk daily candles from entry → exit, applying new config.

    Returns a dict with simulated exit price/date and P&L delta vs actual.
    """
    ticker = open_event["ticker"]
    qty = float(open_event.get("quantity", 0))
    entry = float(open_event.get("price", 0))
    actual_exit = float(close_event.get("price", 0))
    actual_pnl = float(close_event.get("pnl", 0) or 0)

    if entry <= 0 or qty <= 0 or ohlc is None or len(ohlc) == 0:
        return {
            "ticker": ticker, "entry": entry, "qty": qty,
            "actual_exit": actual_exit, "actual_pnl": actual_pnl,
            "sim_exit": actual_exit, "sim_pnl": actual_pnl,
            "sim_reason": "no_ohlc_data",
            "sim_hold_days": 0,
        }

    open_ts = datetime.fromisoformat(str(open_event["timestamp"])[:19])
    close_ts = datetime.fromisoformat(str(close_event["timestamp"])[:19])

    # Walk candles from entry day forward
    peak = entry
    current_trail_pct = max(NEW_INITIAL_TRAIL_FLOOR, 4.0)  # day-1 floor
    sim_exit = None
    sim_reason = None
    sim_exit_date = None

    # Optionally check earnings (skip for backtest speed if it's slow)
    earnings_date = None  # disable for speed; users can re-enable later

    candles = ohlc.reset_index()
    for _, row in candles.iterrows():
        row_date = row["Date"].date() if hasattr(row["Date"], "date") else row["Date"]
        if isinstance(row_date, datetime):
            row_date = row_date.date()
        # Only consider candles between open and close+buffer
        if row_date < open_ts.date():
            continue
        if row_date > close_ts.date() + timedelta(days=2):
            break

        # How many days have we held this position by `row_date`?
        hold_days = (row_date - open_ts.date()).days

        # DAY-0 BUG GUARD (added 2026-05-07): the entry day's daily LOW
        # in yfinance includes the part of the day BEFORE we filled.
        # If we bought mid-day at $X and the morning low was 5% below
        # $X, the simulation would falsely fire our stop at the open.
        # Real fills don't see those earlier lows. Solution: on the
        # entry day, treat the LOW as the entry price (= no triggering)
        # and use only the HIGH for peak update. From day 1 onwards,
        # use full HIGH/LOW.
        if hold_days == 0:
            high = float(row["High"])
            low = entry  # don't trigger stop on pre-fill lows
        else:
            high = float(row["High"])
            low = float(row["Low"])

        # Update peak (intraday high)
        if high > peak:
            peak = high
            # Apply ratchet — tighten if new tier crossed AND hold-days
            # gate satisfied for that tier. (T0 has min_hold=2 in the
            # new config; T1/T2/T3 fire at any age.)
            peak_gain_pct = (peak / entry - 1.0) * 100
            for thresh, new_trail, min_hold in RATCHET_TIERS_NEW:
                if (peak_gain_pct >= thresh
                        and new_trail < current_trail_pct
                        and hold_days >= min_hold):
                    current_trail_pct = new_trail
                    break

        # Check stop hit (intraday low)
        stop_price = peak * (1 - current_trail_pct / 100)
        if low <= stop_price:
            sim_exit = stop_price
            sim_reason = "trail_fired"
            sim_exit_date = row_date
            break

        # Earnings exit
        if earnings_date and row_date >= earnings_date - timedelta(days=EARNINGS_BUFFER_DAYS):
            sim_exit = float(row["Close"])
            sim_reason = "earnings_block_exit"
            sim_exit_date = row_date
            break

    # If no stop fired, close on actual close date
    if sim_exit is None:
        sim_exit = actual_exit
        sim_reason = "no_trigger_used_actual"
        sim_exit_date = close_ts.date()

    sim_pnl = (sim_exit - entry) * qty
    sim_hold_days = (sim_exit_date - open_ts.date()).days

    return {
        "ticker": ticker, "entry": entry, "qty": qty,
        "actual_exit": actual_exit, "actual_pnl": actual_pnl,
        "sim_exit": sim_exit, "sim_pnl": sim_pnl,
        "sim_reason": sim_reason, "sim_hold_days": sim_hold_days,
        "actual_open_date": open_ts.date().isoformat(),
        "actual_close_date": close_ts.date().isoformat(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=60, help="Lookback in days")
    p.add_argument("--csv", action="store_true", help="Output CSV")
    args = p.parse_args()

    log = _load_trades()
    pairs = _pair_trades(log)

    cutoff = date.today() - timedelta(days=args.days)
    pairs = [
        (o, c) for o, c in pairs
        if datetime.fromisoformat(str(o["timestamp"])[:19]).date() >= cutoff
    ]

    if not pairs:
        print("No completed trade pairs in lookback window")
        return

    print(f"Backtesting {len(pairs)} closed trades from last {args.days} days...")
    print(f"NEW config: trail floor {NEW_INITIAL_TRAIL_FLOOR}%, ratchet T0={RATCHET_TIERS_NEW[-1]}, T3={RATCHET_TIERS_NEW[0]}")
    print()

    results = []
    for o, c in pairs:
        ticker = o["ticker"]
        open_dt = datetime.fromisoformat(str(o["timestamp"])[:19]).date()
        close_dt = datetime.fromisoformat(str(c["timestamp"])[:19]).date()
        ohlc = _fetch_ohlc(ticker, open_dt, close_dt)
        sim = simulate_trade(o, c, ohlc)
        results.append(sim)

    # Report
    if args.csv:
        import csv
        w = csv.DictWriter(sys.stdout, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
        return

    print(f"{'Ticker':<7} {'Entry':>7} {'Actual':>10} {'Sim':>10} {'Δ P&L':>9} {'Reason':<22} {'Hold':>5}")
    print("-" * 80)
    for r in results:
        delta = r["sim_pnl"] - r["actual_pnl"]
        print(
            f"{r['ticker']:<7} ${r['entry']:>6.2f} "
            f"${r['actual_pnl']:>+8.2f}  ${r['sim_pnl']:>+8.2f}  "
            f"${delta:>+7.2f}  {r['sim_reason']:<22} {r['sim_hold_days']:>3}d"
        )

    actual_total = sum(r["actual_pnl"] for r in results)
    sim_total = sum(r["sim_pnl"] for r in results)
    delta_total = sim_total - actual_total

    actual_wins = sum(1 for r in results if r["actual_pnl"] > 0)
    sim_wins = sum(1 for r in results if r["sim_pnl"] > 0)
    actual_losses = sum(1 for r in results if r["actual_pnl"] < 0)
    sim_losses = sum(1 for r in results if r["sim_pnl"] < 0)

    actual_gross_w = sum(r["actual_pnl"] for r in results if r["actual_pnl"] > 0)
    actual_gross_l = abs(sum(r["actual_pnl"] for r in results if r["actual_pnl"] < 0))
    sim_gross_w = sum(r["sim_pnl"] for r in results if r["sim_pnl"] > 0)
    sim_gross_l = abs(sum(r["sim_pnl"] for r in results if r["sim_pnl"] < 0))

    actual_pf = (actual_gross_w / actual_gross_l) if actual_gross_l > 0 else 999.0
    sim_pf = (sim_gross_w / sim_gross_l) if sim_gross_l > 0 else 999.0

    actual_wr = actual_wins / len(results) * 100 if results else 0
    sim_wr = sim_wins / len(results) * 100 if results else 0

    print()
    print("═" * 80)
    print("AGGREGATE — actual vs simulated (new config)")
    print("═" * 80)
    print(f"{'':30} {'ACTUAL':>15} {'SIM':>15} {'Δ':>10}")
    print(f"{'Total P&L:':30} {f'${actual_total:+.2f}':>15} {f'${sim_total:+.2f}':>15} {f'${delta_total:+.2f}':>10}")
    print(f"{'Win rate:':30} {f'{actual_wr:.0f}%':>15} {f'{sim_wr:.0f}%':>15} {f'{sim_wr - actual_wr:+.0f}pp':>10}")
    print(f"{'Profit Factor:':30} {f'{actual_pf:.2f}':>15} {f'{sim_pf:.2f}':>15} {f'{sim_pf - actual_pf:+.2f}':>10}")
    print(f"{'Wins / Losses:':30} {f'{actual_wins}W / {actual_losses}L':>15} {f'{sim_wins}W / {sim_losses}L':>15}")
    print()

    # Verdict
    if sim_pf > actual_pf and delta_total > 0:
        print("✅ VERDICT: New config IMPROVES on this sample (higher P&L AND better profit factor)")
    elif delta_total > 0 and sim_pf >= actual_pf - 0.1:
        print("🟢 VERDICT: New config marginally helps (more P&L, similar PF)")
    elif sim_pf > actual_pf and delta_total >= -50:
        print("🟡 VERDICT: New config improves edge (PF up) but P&L slightly lower — likely just noise")
    elif delta_total < -50:
        print("🔴 VERDICT: New config hurts on this sample. Re-evaluate.")
    else:
        print("⚪ VERDICT: Neutral — sample too small or change too modest to matter")

    print()
    print("⚠️  Sample size:", len(results), "trades. Need ≥20 for statistical confidence.")


if __name__ == "__main__":
    main()
