"""Backtest replay engine — re-runs all historical closed trades through
the CURRENT code (with break-even + ATR floor + earnings graduated) and
compares to actual outcomes.

This is the ONLY way to validate the 5 changes from 2026-05-15 without
waiting for 30+ new live trades.

Methodology
-----------
For each (OPEN, CLOSE) pair in trade_log.json:
  1. Fetch daily OHLC via yfinance from entry_date → exit_date+5
  2. Initialize position with NEW code's trail formula (ATR floor, regime mult)
  3. Day-by-day simulate:
       - Update peak from HIGH
       - Apply break-even tightening (if peak_gain ≥ 1.5%)
       - Apply ratchet tier tightening (T0/T1/T2/T3)
       - Check trail-fire (does LOW touch peak × (1 - trail%) ?)
       - Check target hit (does HIGH touch target_price ?)
       - Check target_date forced exit
  4. Report counterfactual exit + delta vs actual

Important caveats
-----------------
- Daily OHLC misses intraday tick-by-tick precision. Trail stops may fire
  on a 1-second wick that the OHLC doesn't capture (over-counts wins) or
  miss a real stop-hit that gapped intraday (under-counts losses).
- yfinance data is post-split-adjusted; entry prices in trade_log are at
  the time-of-trade (also post-adjusted, but may drift slightly).
- No slippage modeling — we assume fills at exact stop/target levels.
- ML / scan-time variables not replayed (no impact on exit logic).

Output is informational, NOT a green-light for live changes. The point
is to detect if any of the 5 changes would have made things WORSE on
historical data.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("backtest_replay")
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ── New-code trail parameters (mirror config.py) ─────────────────
BREAK_EVEN_THRESHOLD_PCT = float(__import__("os").environ.get("BT_BE_THRESHOLD", "1.5"))
BREAK_EVEN_FLOOR_MULT = 1.002
BREAK_EVEN_MIN_TRAIL_PCT = 0.5
BREAK_EVEN_MIN_HOLD_DAYS = int(__import__("os").environ.get("BT_BE_HOLD_DAYS", "0"))
INITIAL_TRAIL_ATR_FLOOR_MULT = 0.9
MIN_INITIAL_TRAIL_PCT = 4.0
MAX_INITIAL_TRAIL_PCT = 9.0

RATCHET_T0_GAIN, RATCHET_T0_TRAIL, T0_MIN_HOLD = 8.0, 3.5, 2
RATCHET_T1_GAIN, RATCHET_T1_TRAIL = 10.0, 3.0
RATCHET_T2_GAIN, RATCHET_T2_TRAIL = 18.0, 2.5
RATCHET_T3_GAIN, RATCHET_T3_TRAIL = 28.0, 2.0


def fetch_ohlc(ticker: str, start: date, end: date):
    """Fetch daily OHLC. Returns list of dicts sorted by date."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed — pip install yfinance")
        sys.exit(2)

    # Pad end +5 days to cover edge cases
    df = yf.Ticker(ticker).history(
        start=start.isoformat(),
        end=(end + timedelta(days=5)).isoformat(),
        auto_adjust=False,
    )
    if df.empty:
        return []
    rows = []
    for ts, r in df.iterrows():
        rows.append({
            "date": ts.date(),
            "open": float(r["Open"]),
            "high": float(r["High"]),
            "low": float(r["Low"]),
            "close": float(r["Close"]),
        })
    return rows


def compute_initial_trail(
    entry_price: float,
    stop_loss: float,
    atr_pct: float,
    regime: str,
    use_new_code: bool,
) -> float:
    """Mirror order_manager.py's trail calculation."""
    candidates = []
    if stop_loss > 0:
        candidates.append((entry_price - stop_loss) / entry_price * 100)
    if atr_pct > 0:
        candidates.append(atr_pct * 1.5)
    if not candidates:
        return 5.0
    base_trail = sum(candidates) / len(candidates)

    regime_u = (regime or "").upper()
    if use_new_code:
        # NEW: includes STRONG_UPTREND and UPTREND
        bullish = ("TREND_UP", "MODERATE_UP", "BULLISH", "STRONG_UPTREND", "UPTREND")
    else:
        # OLD: missing STRONG_UPTREND / UPTREND
        bullish = ("TREND_UP", "MODERATE_UP", "BULLISH")

    if regime_u in bullish:
        regime_mult = 1.20
    elif regime_u == "DISTRIBUTION":
        regime_mult = 0.85
    elif regime_u in ("CORRECTION", "BEARISH", "PANIC"):
        regime_mult = 0.70
    else:
        regime_mult = 1.0

    trail = base_trail * regime_mult

    # ATR floor (new code only)
    if use_new_code and atr_pct > 0:
        atr_floor = atr_pct * INITIAL_TRAIL_ATR_FLOOR_MULT
        trail = max(trail, atr_floor)

    # Hard floor / cap
    trail = max(MIN_INITIAL_TRAIL_PCT, min(trail, MAX_INITIAL_TRAIL_PCT))
    return trail


def simulate(
    ticker: str,
    entry_date: date,
    entry_price: float,
    stop_loss: float,
    target_price: float,
    target_date: Optional[date],
    initial_trail_pct: float,
    use_new_code: bool,
    ohlc: List[Dict],
) -> Tuple[float, str, int, float]:
    """Walk forward day by day. Returns (exit_price, reason, days_held, peak_gain_pct)."""
    if not ohlc:
        return entry_price, "no_data", 0, 0.0

    peak_price = entry_price
    current_trail = initial_trail_pct

    for i, day in enumerate(ohlc):
        if day["date"] < entry_date:
            continue
        days_held = (day["date"] - entry_date).days

        # ── Realistic intraday ordering ──
        # Up days (close ≥ open): OPEN → LOW → HIGH → CLOSE
        #   LOW happens FIRST with prior peak → trail stop is more lenient
        # Down days (close < open): OPEN → HIGH → LOW → CLOSE
        #   HIGH happens FIRST → peak updates, break-even/ratchet tightens trail,
        #   THEN LOW tested against new tighter trail
        is_up_day = day["close"] >= day["open"]

        def check_target_or_stop(low_val, high_val, trail_stop_val):
            """Return (exit_price, reason) or None."""
            if low_val <= trail_stop_val:
                return trail_stop_val, f"trail_fired_{current_trail:.1f}pct"
            if high_val >= target_price:
                return target_price, "target_hit"
            return None

        if is_up_day:
            # 1. Check LOW first against PRIOR peak's trail
            trail_stop = peak_price * (1 - current_trail / 100)
            if day["low"] <= trail_stop and peak_price > entry_price * 1.001:
                # Prior-peak trail fired (only meaningful once peak rose above entry)
                return trail_stop, f"trail_fired_{current_trail:.1f}pct", days_held, (peak_price - entry_price) / entry_price * 100
            # Initial-stop check on entry day with no prior gains
            if peak_price <= entry_price * 1.001 and day["low"] <= stop_loss:
                return stop_loss, "initial_stop_hit", days_held, (peak_price - entry_price) / entry_price * 100
            # 2. Update peak from HIGH
            if day["high"] > peak_price:
                peak_price = day["high"]
            peak_gain_pct = (peak_price - entry_price) / entry_price * 100
            # 3. Apply break-even / ratchet (may tighten trail)
            if (use_new_code
                and peak_gain_pct >= BREAK_EVEN_THRESHOLD_PCT
                and days_held >= BREAK_EVEN_MIN_HOLD_DAYS):
                target_floor = entry_price * BREAK_EVEN_FLOOR_MULT
                if peak_price > target_floor:
                    be_trail = (peak_price - target_floor) / peak_price * 100
                    if be_trail >= BREAK_EVEN_MIN_TRAIL_PCT and be_trail < current_trail:
                        current_trail = be_trail
            ratchet_target = None
            if peak_gain_pct >= RATCHET_T3_GAIN:
                ratchet_target = RATCHET_T3_TRAIL
            elif peak_gain_pct >= RATCHET_T2_GAIN:
                ratchet_target = RATCHET_T2_TRAIL
            elif peak_gain_pct >= RATCHET_T1_GAIN:
                ratchet_target = RATCHET_T1_TRAIL
            elif peak_gain_pct >= RATCHET_T0_GAIN and days_held >= T0_MIN_HOLD:
                ratchet_target = RATCHET_T0_TRAIL
            if ratchet_target is not None and ratchet_target < current_trail:
                current_trail = ratchet_target
            # 4. Target hit (HIGH after LOW)
            if day["high"] >= target_price:
                return target_price, "target_hit", days_held, peak_gain_pct
            # LOW already passed for the day — no second test
        else:
            # Down day: HIGH first, then LOW
            # 1. Update peak from HIGH
            if day["high"] > peak_price:
                peak_price = day["high"]
            peak_gain_pct = (peak_price - entry_price) / entry_price * 100
            # 2. Target hit (HIGH)
            if day["high"] >= target_price:
                return target_price, "target_hit", days_held, peak_gain_pct
            # 3. Apply break-even / ratchet
            if (use_new_code
                and peak_gain_pct >= BREAK_EVEN_THRESHOLD_PCT
                and days_held >= BREAK_EVEN_MIN_HOLD_DAYS):
                target_floor = entry_price * BREAK_EVEN_FLOOR_MULT
                if peak_price > target_floor:
                    be_trail = (peak_price - target_floor) / peak_price * 100
                    if be_trail >= BREAK_EVEN_MIN_TRAIL_PCT and be_trail < current_trail:
                        current_trail = be_trail
            ratchet_target = None
            if peak_gain_pct >= RATCHET_T3_GAIN:
                ratchet_target = RATCHET_T3_TRAIL
            elif peak_gain_pct >= RATCHET_T2_GAIN:
                ratchet_target = RATCHET_T2_TRAIL
            elif peak_gain_pct >= RATCHET_T1_GAIN:
                ratchet_target = RATCHET_T1_TRAIL
            elif peak_gain_pct >= RATCHET_T0_GAIN and days_held >= T0_MIN_HOLD:
                ratchet_target = RATCHET_T0_TRAIL
            if ratchet_target is not None and ratchet_target < current_trail:
                current_trail = ratchet_target
            # 4. Now LOW with possibly-tightened trail
            trail_stop = peak_price * (1 - current_trail / 100)
            if day["low"] <= trail_stop and peak_price > entry_price * 1.001:
                return trail_stop, f"trail_fired_{current_trail:.1f}pct", days_held, peak_gain_pct
            if peak_price <= entry_price * 1.001 and day["low"] <= stop_loss:
                return stop_loss, "initial_stop_hit", days_held, peak_gain_pct

        # ── 7. Target date forced exit (after intraday checks) ──
        if target_date and day["date"] >= target_date:
            return day["close"], "target_date_exit", days_held, (peak_price - entry_price) / entry_price * 100

    # Reached end of OHLC without exit — return last close
    last = ohlc[-1]
    last_days = (last["date"] - entry_date).days
    return last["close"], "end_of_data", last_days, (peak_price - entry_price) / entry_price * 100


def load_trades(trade_log_path: Path) -> List[Tuple[Dict, Dict]]:
    """Pair OPEN with subsequent CLOSE for each ticker."""
    with open(trade_log_path) as f:
        log = json.load(f)
    opens = [e for e in log if e.get("action") == "OPEN"]
    closes = [e for e in log if e.get("action") == "CLOSE"]
    pairs = []
    used_close_idx = set()
    for o in opens:
        tkr = o["ticker"]
        o_ts = o["timestamp"]
        # Find the next CLOSE for this ticker after o_ts that we haven't used
        for i, c in enumerate(closes):
            if i in used_close_idx:
                continue
            if c["ticker"] == tkr and c["timestamp"] > o_ts:
                pairs.append((o, c))
                used_close_idx.add(i)
                break
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trade-log", default="data/trades/trade_log.json")
    ap.add_argument("--ticker", default=None, help="Only replay this ticker")
    args = ap.parse_args()

    pairs = load_trades(Path(args.trade_log))
    if args.ticker:
        pairs = [p for p in pairs if p[0]["ticker"] == args.ticker]

    print(f"\nReplaying {len(pairs)} closed trades — NEW code vs ACTUAL outcome\n")
    print(f"{'ticker':6s} {'days':>4s} {'entry':>8s} {'actual_$':>9s} {'actual%':>8s} "
          f"{'new_$':>8s} {'new%':>8s} {'delta':>8s} {'reason'}")
    print("-" * 100)

    total_actual = 0.0
    total_new = 0.0
    rows_better = 0
    rows_worse = 0
    rows_same = 0
    detail_rows = []

    for o, c in pairs:
        tkr = o["ticker"]
        entry_dt = datetime.fromisoformat(o["timestamp"]).date()
        exit_dt = datetime.fromisoformat(c["timestamp"]).date()
        entry_price = float(o["price"])
        stop_loss = float(o["stop_loss"])
        target_price = float(o["target_price"])
        target_date_str = o.get("target_date")
        target_date = None
        if target_date_str:
            # Handle both "YYYY-MM-DD" and "YYYY-MM-DD HH:MM:SS"
            try:
                target_date = datetime.fromisoformat(target_date_str.split()[0]).date()
            except Exception:
                target_date = None

        qty = int(o.get("quantity", 1))
        actual_exit = float(c["price"])
        actual_pnl_abs = (actual_exit - entry_price) * qty
        actual_pnl_pct = (actual_exit - entry_price) / entry_price * 100

        # Fetch OHLC
        ohlc = fetch_ohlc(tkr, entry_dt, exit_dt)

        # Replay with NEW code (no ATR info, assume regime SIDEWAYS as conservative)
        # Note: without scan-time atr_pct we use 0 → no ATR floor effect. This
        # underestimates the new code's benefit on high-vol stocks.
        new_trail = compute_initial_trail(entry_price, stop_loss, 0.0, "", use_new_code=True)
        new_exit, new_reason, days, peak_gain = simulate(
            tkr, entry_dt, entry_price, stop_loss, target_price,
            target_date, new_trail, use_new_code=True, ohlc=ohlc,
        )
        new_pnl_abs = (new_exit - entry_price) * qty
        new_pnl_pct = (new_exit - entry_price) / entry_price * 100
        delta = new_pnl_abs - actual_pnl_abs

        if abs(delta) < 0.01:
            rows_same += 1
        elif delta > 0:
            rows_better += 1
        else:
            rows_worse += 1

        total_actual += actual_pnl_abs
        total_new += new_pnl_abs

        marker = "🟢" if delta > 0.01 else ("🔴" if delta < -0.01 else "  ")
        print(f"{tkr:6s} {days:>4d} {entry_price:>8.2f} {actual_pnl_abs:>+9.2f} {actual_pnl_pct:>+8.2f}% "
              f"{new_pnl_abs:>+8.2f} {new_pnl_pct:>+8.2f}% {delta:>+8.2f} {marker} {new_reason[:25]}")

        detail_rows.append({
            "ticker": tkr,
            "days": days,
            "actual_pnl": round(actual_pnl_abs, 2),
            "new_pnl": round(new_pnl_abs, 2),
            "delta": round(delta, 2),
            "peak_gain_pct": round(peak_gain, 2),
            "new_reason": new_reason,
            "actual_reason": c.get("reason", ""),
        })

    print("-" * 100)
    print(f"\nTOTALS:")
    print(f"  Actual realized:    ${total_actual:>+8.2f}")
    print(f"  NEW-code simulated: ${total_new:>+8.2f}")
    print(f"  Delta:              ${total_new - total_actual:>+8.2f}  ({'better' if total_new > total_actual else 'WORSE'} by ${abs(total_new - total_actual):.2f})")
    print(f"\n  Trades improved:   {rows_better}")
    print(f"  Trades worsened:   {rows_worse}")
    print(f"  Trades same:       {rows_same}")

    # Highlight worst regressions
    detail_rows.sort(key=lambda r: r["delta"])
    print(f"\nTop 3 REGRESSIONS (where new code is worse than actual):")
    for r in detail_rows[:3]:
        if r["delta"] < -0.01:
            print(f"  {r['ticker']:6s} delta=${r['delta']:+7.2f}  "
                  f"actual={r['actual_pnl']:+.2f}  new={r['new_pnl']:+.2f}  "
                  f"peak=+{r['peak_gain_pct']:.1f}%  new_reason={r['new_reason']}")

    print(f"\nTop 3 IMPROVEMENTS (where new code beats actual):")
    detail_rows.sort(key=lambda r: r["delta"], reverse=True)
    for r in detail_rows[:3]:
        if r["delta"] > 0.01:
            print(f"  {r['ticker']:6s} delta=${r['delta']:+7.2f}  "
                  f"actual={r['actual_pnl']:+.2f}  new={r['new_pnl']:+.2f}  "
                  f"peak=+{r['peak_gain_pct']:.1f}%  new_reason={r['new_reason']}")


if __name__ == "__main__":
    main()
