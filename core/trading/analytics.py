"""Trading performance analytics — win rate, expectancy, Sharpe, equity curve.

Computes stats from trade_log.json (VPS live trades) and optionally
compares to SPY benchmark.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math

logger = logging.getLogger(__name__)


def _parse_dt(ts: Any) -> Optional[datetime]:
    """Parse timestamp string to datetime."""
    if not ts:
        return None
    s = str(ts).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        try:
            return datetime.fromisoformat(s[:19])
        except Exception:
            return None


def build_trade_pairs(trade_log: List[Dict]) -> List[Dict]:
    """Pair OPEN+CLOSE events per ticker into complete round trips.

    Returns list of {ticker, entry_price, exit_price, qty, entry_date, exit_date,
                     pnl_abs, pnl_pct, hold_days, reason}
    """
    # Group by ticker, handle multiple opens/closes
    by_ticker: Dict[str, List[Dict]] = {}
    for t in trade_log:
        tkr = t.get("ticker", "")
        if tkr:
            by_ticker.setdefault(tkr, []).append(t)

    pairs: List[Dict] = []
    for tkr, events in by_ticker.items():
        events = sorted(events, key=lambda x: x.get("timestamp", ""))
        # FIFO: pair each CLOSE with the earliest open OPEN
        open_queue: List[Dict] = []
        for e in events:
            if e.get("action") == "OPEN":
                open_queue.append(e)
            elif e.get("action") == "CLOSE" and open_queue:
                opn = open_queue.pop(0)
                entry_p = float(opn.get("price") or 0)
                exit_p = float(e.get("price") or 0)
                qty = int(opn.get("quantity") or e.get("quantity") or 0)
                pnl_abs = (exit_p - entry_p) * qty
                pnl_pct = ((exit_p / entry_p) - 1) * 100 if entry_p > 0 else 0
                entry_dt = _parse_dt(opn.get("timestamp"))
                exit_dt = _parse_dt(e.get("timestamp"))
                hold_days = (exit_dt - entry_dt).days if (entry_dt and exit_dt) else 0
                pairs.append({
                    "ticker": tkr,
                    "entry_price": entry_p,
                    "exit_price": exit_p,
                    "quantity": qty,
                    "entry_date": entry_dt.date().isoformat() if entry_dt else "",
                    "exit_date": exit_dt.date().isoformat() if exit_dt else "",
                    "pnl_abs": round(pnl_abs, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "hold_days": hold_days,
                    "reason": e.get("reason", ""),
                    "score": opn.get("score"),
                })
    return pairs


def compute_metrics(pairs: List[Dict]) -> Dict[str, Any]:
    """Compute trading performance metrics from paired trades.

    Returns:
      total_trades, wins, losses, win_rate, avg_win, avg_loss,
      expectancy, profit_factor, total_pnl_abs, total_pnl_pct,
      avg_hold_days, best_trade, worst_trade
    """
    if not pairs:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "avg_win": 0, "avg_loss": 0, "expectancy": 0, "profit_factor": 0,
            "total_pnl_abs": 0, "total_pnl_pct": 0, "avg_hold_days": 0,
            "best_trade": None, "worst_trade": None,
        }

    wins = [p for p in pairs if p["pnl_abs"] > 0]
    losses = [p for p in pairs if p["pnl_abs"] < 0]
    even = [p for p in pairs if p["pnl_abs"] == 0]

    total = len(pairs)
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = (n_wins / total * 100) if total else 0

    avg_win_abs = sum(p["pnl_abs"] for p in wins) / len(wins) if wins else 0
    avg_loss_abs = sum(p["pnl_abs"] for p in losses) / len(losses) if losses else 0
    avg_win_pct = sum(p["pnl_pct"] for p in wins) / len(wins) if wins else 0
    avg_loss_pct = sum(p["pnl_pct"] for p in losses) / len(losses) if losses else 0

    # Expectancy: (win_rate * avg_win) - (loss_rate * |avg_loss|)
    # Use percentage form — more portable
    loss_rate = (n_losses / total * 100) if total else 0
    expectancy_pct = ((win_rate / 100) * avg_win_pct) + ((loss_rate / 100) * avg_loss_pct)

    # Profit factor = gross wins / gross losses (abs)
    gross_wins = sum(p["pnl_abs"] for p in wins)
    gross_losses = abs(sum(p["pnl_abs"] for p in losses))
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (float("inf") if gross_wins > 0 else 0)

    total_pnl_abs = sum(p["pnl_abs"] for p in pairs)
    total_pnl_pct = sum(p["pnl_pct"] for p in pairs)
    avg_hold = sum(p["hold_days"] for p in pairs) / total if total else 0

    best = max(pairs, key=lambda p: p["pnl_pct"]) if pairs else None
    worst = min(pairs, key=lambda p: p["pnl_pct"]) if pairs else None

    return {
        "total_trades": total,
        "wins": n_wins,
        "losses": n_losses,
        "even": len(even),
        "win_rate": round(win_rate, 1),
        "avg_win_abs": round(avg_win_abs, 2),
        "avg_loss_abs": round(avg_loss_abs, 2),
        "avg_win_pct": round(avg_win_pct, 2),
        "avg_loss_pct": round(avg_loss_pct, 2),
        "expectancy_pct": round(expectancy_pct, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else None,
        "total_pnl_abs": round(total_pnl_abs, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "avg_hold_days": round(avg_hold, 1),
        "best_trade": best,
        "worst_trade": worst,
    }


def build_equity_curve(pairs: List[Dict], starting_balance: float = 1000.0) -> List[Dict]:
    """Build equity curve from closed trades, sorted by exit_date.

    Returns list of {date, equity, pnl_abs, ticker, cumulative_pct}
    """
    if not pairs:
        return []
    sorted_pairs = sorted(pairs, key=lambda p: p.get("exit_date", ""))
    equity = starting_balance
    curve = [{
        "date": (sorted_pairs[0].get("entry_date") or sorted_pairs[0].get("exit_date")),
        "equity": starting_balance,
        "pnl_abs": 0,
        "ticker": "start",
        "cumulative_pct": 0,
    }]
    for p in sorted_pairs:
        equity += p["pnl_abs"]
        curve.append({
            "date": p["exit_date"],
            "equity": round(equity, 2),
            "pnl_abs": p["pnl_abs"],
            "ticker": p["ticker"],
            "cumulative_pct": round((equity / starting_balance - 1) * 100, 2),
        })
    return curve


def compute_drawdown(equity_curve: List[Dict]) -> Dict[str, Any]:
    """Compute max drawdown from equity curve."""
    if not equity_curve or len(equity_curve) < 2:
        return {"max_dd_pct": 0, "max_dd_abs": 0, "current_dd_pct": 0}

    equities = [e["equity"] for e in equity_curve]
    running_peak = equities[0]
    max_dd_abs = 0
    max_dd_pct = 0

    for eq in equities:
        if eq > running_peak:
            running_peak = eq
        dd = running_peak - eq
        dd_pct = (dd / running_peak * 100) if running_peak > 0 else 0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            max_dd_abs = dd

    current_eq = equities[-1]
    current_peak = max(equities)
    current_dd_pct = ((current_peak - current_eq) / current_peak * 100) if current_peak > 0 else 0

    return {
        "max_dd_pct": round(max_dd_pct, 2),
        "max_dd_abs": round(max_dd_abs, 2),
        "current_dd_pct": round(current_dd_pct, 2),
    }


def get_spy_return(start_date: str, end_date: str) -> Optional[float]:
    """Fetch SPY return between dates for benchmark comparison."""
    try:
        import yfinance as yf
        spy = yf.Ticker("SPY")
        hist = spy.history(start=start_date, end=end_date)
        if len(hist) < 2:
            return None
        start_price = float(hist["Close"].iloc[0])
        end_price = float(hist["Close"].iloc[-1])
        return round(((end_price / start_price) - 1) * 100, 2)
    except Exception as e:
        logger.debug("SPY fetch failed: %s", e)
        return None


def get_full_performance(trade_log_path: str = "data/trades/trade_log.json",
                         starting_balance: float = 1000.0) -> Dict[str, Any]:
    """Load trade log, compute all metrics, return full performance dict."""
    path = Path(trade_log_path)
    if not path.exists():
        return {"error": "no trade log found", "pairs": [], "metrics": {}}

    try:
        log = json.loads(path.read_text())
    except Exception as e:
        return {"error": f"parse failed: {e}", "pairs": [], "metrics": {}}

    pairs = build_trade_pairs(log)
    metrics = compute_metrics(pairs)
    curve = build_equity_curve(pairs, starting_balance)
    dd = compute_drawdown(curve)

    # SPY benchmark
    spy_return = None
    if pairs:
        earliest = min((p.get("entry_date", "") for p in pairs if p.get("entry_date")), default="")
        latest = max((p.get("exit_date", "") for p in pairs if p.get("exit_date")), default="")
        if earliest and latest:
            spy_return = get_spy_return(earliest, latest)

    alpha = None
    if spy_return is not None:
        alpha = round(metrics["total_pnl_pct"] - spy_return, 2)

    return {
        "pairs": pairs,
        "metrics": metrics,
        "equity_curve": curve,
        "drawdown": dd,
        "spy_return_pct": spy_return,
        "alpha_pct": alpha,
    }
