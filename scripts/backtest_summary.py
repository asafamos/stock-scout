"""Weekly backtest summary → Telegram + regression detection.

Runs AFTER `run_full_backtest.py` inside the weekly_backtest.yml workflow.

What it does:
    1. Loads current run (`reports/backtest_latest.json`).
    2. Loads prior run (last non-current row in `reports/backtest_history.jsonl`).
    3. Formats a compact HTML message with metrics + deltas from prior week.
    4. Fires regression alerts when key metrics move worse than a threshold.
    5. Appends the current run to `backtest_history.jsonl` (rolling record).
    6. Sends to Telegram via core.trading.notifications._send.

Exit codes:
    0 = success (message sent, no regression OR regression sent)
    1 = missing report / send failed
    2 = regression detected AND alert fired (workflow can decide to fail-loud)

Env:
    TRADE_TELEGRAM_TOKEN, TRADE_TELEGRAM_CHAT_ID   — from GH secrets, same as other bots
    BACKTEST_REPORT_PATH                            — default reports/backtest_latest.json
    BACKTEST_HISTORY_PATH                           — default reports/backtest_history.jsonl
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


# Regression thresholds — worse than this vs prior week fires an alert.
# Tuned for weekly cadence + small n (~99 trades). Tighter thresholds
# would false-alarm on Monte-Carlo noise; looser would miss real breakage.
REGRESSION = {
    "sharpe_drop":         0.20,   # abs decline
    "total_return_drop":   3.0,    # pp
    "excess_return_drop":  3.0,    # pp — vs benchmark
    "max_dd_widen":        5.0,    # pp (max_dd becomes MORE negative)
    "win_rate_drop":       5.0,    # pp
}


def _fmt_delta(cur: float, prev: float | None, suffix: str = "", precision: int = 2) -> str:
    """Format `+0.02` / `-1.30pp` — with an arrow icon or a neutral dash."""
    if prev is None:
        return "—"
    d = cur - prev
    arrow = "▲" if d > 0 else ("▼" if d < 0 else "•")
    sign = "+" if d >= 0 else ""
    return f"{arrow} {sign}{d:.{precision}f}{suffix}"


def _load_report(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: report not found at {path}", file=sys.stderr)
        sys.exit(1)
    with path.open() as f:
        return json.load(f)


def _load_prior(history: Path, current_key: tuple) -> dict | None:
    """Return the newest history row whose (start,end,n_trades) differs from current."""
    if not history.exists():
        return None
    prior = None
    with history.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            key = (row.get("start_date"), row.get("end_date"), row.get("n_trades"))
            if key == current_key:
                # Same run — skip. Keeps re-runs of the same week from
                # comparing against themselves.
                continue
            prior = row  # keep the newest non-matching row
    return prior


def _append_history(history: Path, current: dict) -> None:
    """Append current run to history — small subset, one line."""
    history.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "start_date":  current.get("start_date"),
        "end_date":    current.get("end_date"),
        "n_trades":    current.get("n_trades"),
        "sharpe":      current.get("sharpe"),
        "total_return":current.get("total_return"),
        "cagr":        current.get("cagr"),
        "max_dd":      current.get("max_dd"),
        "win_rate_pct":current.get("win_rate_pct"),
        "profit_factor":current.get("profit_factor"),
        "excess_return":current.get("excess_return"),
        "benchmark_return":current.get("benchmark_return"),
    }
    with history.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _regression_alerts(cur: dict, prev: dict | None) -> list[str]:
    """Return a list of regression alerts vs prior week. Empty if none."""
    if prev is None:
        return []
    alerts = []
    dsharpe = (cur.get("sharpe") or 0) - (prev.get("sharpe") or 0)
    if dsharpe < -REGRESSION["sharpe_drop"]:
        alerts.append(f"Sharpe dropped {dsharpe:+.2f} (threshold −{REGRESSION['sharpe_drop']:.2f})")
    dret = (cur.get("total_return") or 0) - (prev.get("total_return") or 0)
    if dret < -REGRESSION["total_return_drop"]:
        alerts.append(f"Total return dropped {dret:+.2f}pp (threshold −{REGRESSION['total_return_drop']:.1f}pp)")
    dexc = (cur.get("excess_return") or 0) - (prev.get("excess_return") or 0)
    if dexc < -REGRESSION["excess_return_drop"]:
        alerts.append(f"Excess-vs-benchmark dropped {dexc:+.2f}pp (threshold −{REGRESSION['excess_return_drop']:.1f}pp)")
    # max_dd is negative — "widening" means MORE negative → cur_dd - prev_dd < -threshold
    ddd = (cur.get("max_dd") or 0) - (prev.get("max_dd") or 0)
    if ddd < -REGRESSION["max_dd_widen"]:
        alerts.append(f"Max drawdown widened {ddd:+.2f}pp (threshold −{REGRESSION['max_dd_widen']:.1f}pp)")
    dwr = (cur.get("win_rate_pct") or 0) - (prev.get("win_rate_pct") or 0)
    if dwr < -REGRESSION["win_rate_drop"]:
        alerts.append(f"Win rate dropped {dwr:+.2f}pp (threshold −{REGRESSION['win_rate_drop']:.1f}pp)")
    return alerts


def _format_message(cur: dict, prev: dict | None, alerts: list[str]) -> str:
    start = cur.get("start_date", "?")
    end   = cur.get("end_date", "?")
    n     = cur.get("n_trades", 0)

    def _p(key, suffix="", prec=2):
        return _fmt_delta(cur.get(key) or 0, (prev or {}).get(key), suffix, prec)

    def _v(key, fmt=".2f"):
        v = cur.get(key)
        if v is None:
            return "—"
        return f"{v:{fmt}}"

    header = "🚨 <b>Weekly Backtest — REGRESSION</b>" if alerts else "📊 <b>Weekly Backtest</b>"
    lines = [
        header,
        f"<i>Period {start} → {end}, n={n} trades</i>",
        "",
        f"<b>Sharpe:</b>       {_v('sharpe')}        {_p('sharpe')}",
        f"<b>Total return:</b> {_v('total_return', '.2f')}%    {_p('total_return', 'pp')}",
        f"<b>CAGR:</b>         {_v('cagr', '.2f')}%    {_p('cagr', 'pp')}",
        f"<b>Max DD:</b>       {_v('max_dd', '.2f')}%   {_p('max_dd', 'pp')}",
        f"<b>Win rate:</b>     {_v('win_rate_pct', '.1f')}%   {_p('win_rate_pct', 'pp', 1)}",
        f"<b>Profit factor:</b> {_v('profit_factor')}   {_p('profit_factor')}",
        f"<b>vs benchmark:</b> {_v('excess_return', '+.2f')}pp  {_p('excess_return', 'pp')}",
    ]
    if alerts:
        lines.append("")
        lines.append("⚠️ <b>Alerts:</b>")
        for a in alerts:
            lines.append(f"  • {a}")
    if prev is None:
        lines.append("")
        lines.append("<i>(first run — no prior week to compare)</i>")
    return "\n".join(lines)


def main() -> int:
    report_path  = Path(os.getenv("BACKTEST_REPORT_PATH",  "reports/backtest_latest.json"))
    history_path = Path(os.getenv("BACKTEST_HISTORY_PATH", "reports/backtest_history.jsonl"))

    current = _load_report(report_path)
    key = (current.get("start_date"), current.get("end_date"), current.get("n_trades"))
    prior = _load_prior(history_path, key)
    alerts = _regression_alerts(current, prior)
    message = _format_message(current, prior, alerts)

    # Print to stdout so it also shows in the GH Actions log
    print(message.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))

    # Send to Telegram
    try:
        from core.trading.notifications import _send
        ok = _send(message)
        if not ok:
            print("WARN: Telegram send returned False (missing token/chat_id?)", file=sys.stderr)
            # Not fatal — history still gets appended below.
    except Exception as e:
        print(f"ERROR sending telegram: {e}", file=sys.stderr)

    # Append to history AFTER comparing so this run doesn't self-match.
    try:
        _append_history(history_path, current)
    except Exception as e:
        print(f"WARN: could not append history: {e}", file=sys.stderr)

    return 2 if alerts else 0


if __name__ == "__main__":
    sys.exit(main())
