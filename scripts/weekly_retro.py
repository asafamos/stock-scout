"""Weekly retrospective — Fires Friday 20:30 UTC = 23:30 IL.

Aggregates the week's trading and operational data into a single
Telegram message. Designed to answer the question that comes up
every Friday: "what happened this week, and what should change?"

Three sections:
  1. PERFORMANCE — P&L, win rate, profit factor (vs last week)
  2. EXECUTION QUALITY — close reasons, hold time, drift events
  3. STRATEGIC VERDICT — ready to scale capital? what to fix?

The verdict is rule-based, not vibes:
  - profit_factor >= 1.5 AND avg_hold >= 5d AND ≥5 trades → "ready to add capital"
  - profit_factor 1.0-1.5: "promising, need more data"
  - profit_factor < 1.0: "drawdown — investigate"
  - <5 trades: "insufficient sample"

Schedule: Fri 20:30 UTC (after EOD summary at 20:30 too — fire 1 min
later via OnCalendar to avoid race). On other weekdays this script
won't run.

Manual: python -m scripts.weekly_retro
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
TRADE_LOG = ROOT / "data" / "trades" / "trade_log.json"

TOKEN = os.getenv("TRADE_TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("TRADE_TELEGRAM_CHAT_ID", "")
if not TOKEN:
    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        for line in secrets_path.read_text().splitlines():
            if "TELEGRAM_TOKEN" in line and "=" in line:
                TOKEN = line.split("=", 1)[1].strip().strip('"').strip("'")
            if "TELEGRAM_CHAT_ID" in line and "=" in line:
                CHAT_ID = line.split("=", 1)[1].strip().strip('"').strip("'")


def send_telegram(text: str) -> bool:
    if not TOKEN or not CHAT_ID:
        logger.error("Telegram not configured")
        print(text)
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=15,
        )
        r.raise_for_status()
        return True
    except Exception as e:
        logger.error("Telegram send failed: %s", e)
        return False


def _load_log() -> list:
    try:
        return json.loads(TRADE_LOG.read_text())
    except Exception:
        return []


def _pair_trades(log: list) -> list:
    """FIFO-pair OPEN/CLOSE events per ticker. Returns (open, close, pnl, hold_days).

    CRITICAL (added 2026-05-09): also returns CLOSE entries that have NO
    matching OPEN — they could be for adopted positions (no OPEN ever
    written, see ORCL adopted from IB) or trades opened in a prior week
    not in our window. Without this fallback the weekly retro silently
    drops these trades and reports "INSUFFICIENT DATA" while a real
    +$55 winner sits unaccounted-for. (Real-world: weekly retro on
    2026-05-08 reported only ELVN -$19.56 and missed ORCL +$55.24
    + RDDT +$6.89, both adopted/cross-week.)
    """
    opens = defaultdict(list)
    pairs = []
    for e in sorted(log, key=lambda e: str(e.get("timestamp", ""))):
        a = e.get("action")
        t = e.get("ticker")
        if a == "OPEN":
            opens[t].append(e)
        elif a == "CLOSE":
            o_list = opens.get(t, [])
            pnl = float(e.get("pnl", 0) or 0)
            if o_list:
                o = o_list.pop(0)
                try:
                    ts_o = datetime.fromisoformat(str(o.get("timestamp", ""))[:19])
                    ts_c = datetime.fromisoformat(str(e.get("timestamp", ""))[:19])
                    hd = (ts_c - ts_o).days
                except Exception:
                    hd = 0
                pairs.append((o, e, pnl, hd))
            else:
                # Unpaired CLOSE — adopted position or pre-window OPEN.
                # Synthesize a placeholder open with hold_days=0 so it
                # still gets counted in P&L / win-rate / PF aggregates.
                # avg_hold loses precision but no other metric does.
                pairs.append(({"ticker": t, "timestamp": e.get("timestamp", "")}, e, pnl, 0))
    return pairs


def _filter_by_window(pairs: list, start: date, end: date) -> list:
    """Trades whose CLOSE is in [start, end)."""
    out = []
    for o, c, pnl, hd in pairs:
        try:
            close_dt = datetime.fromisoformat(str(c.get("timestamp", ""))[:19]).date()
            if start <= close_dt < end:
                out.append((o, c, pnl, hd))
        except Exception:
            pass
    return out


def _metrics(window_pairs: list) -> dict:
    if not window_pairs:
        return {"n": 0}
    pnls = [p for _, _, p, _ in window_pairs]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_w = sum(wins)
    gross_l = abs(sum(losses))
    pf = (gross_w / gross_l) if gross_l > 0 else (999.0 if gross_w > 0 else 0.0)
    holds = [hd for _, _, _, hd in window_pairs]
    avg_hold = sum(holds) / len(holds) if holds else 0.0

    # Close reason buckets
    reasons = defaultdict(int)
    for _, c, _, _ in window_pairs:
        r = str(c.get("reason", "unknown")).lower()
        if "trail" in r:
            reasons["trail"] += 1
        elif "target" in r or "limit" in r:
            reasons["target"] += 1
        elif "ladder" in r or "partial" in r:
            reasons["ladder"] += 1
        elif "earnings" in r or "target_date" in r:
            reasons["target_date"] += 1
        else:
            reasons["other"] += 1

    return {
        "n": len(window_pairs),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(window_pairs) * 100 if window_pairs else 0.0,
        "total_pnl": sum(pnls),
        "gross_w": gross_w,
        "gross_l": gross_l,
        "profit_factor": pf,
        "avg_hold": avg_hold,
        "biggest_win": max(pnls, default=0),
        "biggest_loss": min(pnls, default=0),
        "reasons": dict(reasons),
    }


def _operational_events(start: date, end: date) -> dict:
    """Count drift/abort/recovery events from systemd journal in window."""
    out = {"drift": 0, "aborts": 0, "auto_recovers": 0, "auto_blocks": 0}
    try:
        result = subprocess.run(
            ["journalctl", "--since", start.isoformat(),
             "--until", end.isoformat(), "--no-pager"],
            capture_output=True, text=True, timeout=30,
        )
        text = result.stdout or ""
        out["drift"] = len(re.findall(r"DRIFT DETECTED", text))
        out["aborts"] = len(re.findall(r"PIPELINE ABORTED", text))
        out["auto_recovers"] = len(re.findall(r"FALLBACK SUCCESS|recovered", text))
        out["auto_blocks"] = len(re.findall(r"AUTO-BLOCKED", text))
    except Exception as e:
        logger.warning("journalctl failed: %s", e)
    return out


def _verdict(this_week: dict, last_week: dict) -> str:
    """Rule-based capital-scaling verdict."""
    n = this_week.get("n", 0)
    if n == 0:
        return "⚪ <b>NO TRADES THIS WEEK</b> — system idle or capacity-bound."
    if n < 5:
        return (
            f"🟡 <b>INSUFFICIENT DATA</b> ({n} trades this week)\n"
            f"  Need ≥5/week + 4 weeks of data before considering capital scale-up."
        )
    pf = this_week.get("profit_factor", 0)
    avg_hold = this_week.get("avg_hold", 0)
    pnl = this_week.get("total_pnl", 0)
    if pf >= 1.5 and avg_hold >= 5 and pnl > 0:
        return (
            "🟢 <b>READY TO SCALE</b> — strong week:\n"
            f"  PF {pf:.2f} ≥ 1.5  |  hold {avg_hold:.1f}d ≥ 5  |  P&L ${pnl:+.2f}\n"
            f"  Suggest: add $1k → $2k account → unlock cash-tier features."
        )
    if pf >= 1.0:
        return (
            "🟡 <b>PROMISING</b> — keep current capital:\n"
            f"  PF {pf:.2f} (target ≥1.5)  |  hold {avg_hold:.1f}d (target ≥5)\n"
            f"  P&L ${pnl:+.2f}. Need 2-3 more weeks like this."
        )
    return (
        "🔴 <b>UNDERPERFORMING</b> — investigate before next week:\n"
        f"  PF {pf:.2f} below break-even (need ≥1.0)\n"
        f"  P&L ${pnl:+.2f}. Don't add capital. Review close reasons + ratchet behavior."
    )


def _cohort_breakdown(pairs: list) -> list:
    """Group lifetime pairs into cohorts by entry signal. Surfaces WHICH
    types of trades win (vs lose). Once we have ≥20 trades, cohort win-rates
    inform which gates to tighten or loosen.

    Cohorts:
      - Score bucket:  <75 / 75-80 / 80-85 / 85+
      - R:R bucket:    <2.5 / 2.5-4 / 4+
      - Hold bucket:   0-1d / 2-3d / 4-7d / 8+d
      - Close reason:  trail_fired / target_hit / other

    Skips adopted positions (hold=0 because no real OPEN, would skew Hold
    bucket).
    """
    lines = []

    def _score_bucket(s):
        if s < 75: return "<75"
        if s < 80: return "75-80"
        if s < 85: return "80-85"
        return "85+"

    def _rr_bucket(rr):
        if rr < 2.5: return "<2.5"
        if rr < 4.0: return "2.5-4"
        return "4+"

    def _hold_bucket(d):
        if d <= 1: return "0-1d"
        if d <= 3: return "2-3d"
        if d <= 7: return "4-7d"
        return "8+d"

    def _bucket_metric(bucket_key_fn, pairs_subset, label):
        from collections import defaultdict
        bins = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})
        for o, c, pnl, hd in pairs_subset:
            key = bucket_key_fn(o, c, pnl, hd)
            if key is None:
                continue
            bins[key]["pnl"] += pnl
            if pnl > 0:
                bins[key]["wins"] += 1
            else:
                bins[key]["losses"] += 1
        rows = []
        for k, v in sorted(bins.items()):
            n = v["wins"] + v["losses"]
            if n == 0:
                continue
            wr = v["wins"] / n * 100
            rows.append(f"    {k:<8} {v['wins']}W/{v['losses']}L (WR {wr:>3.0f}%)  P&L ${v['pnl']:+.2f}")
        if rows:
            lines.append(f"  <b>by {label}:</b>")
            lines.extend(rows)

    # Filter out adopted (open ticker is dict with no score field) — adopted
    # placeholders have only ticker+timestamp.
    real_pairs = [(o, c, pnl, hd) for o, c, pnl, hd in pairs
                  if "score" in o]

    if len(real_pairs) < 3:
        return []

    # Score cohort
    def _score_key(o, c, pnl, hd):
        s = o.get("score")
        return _score_bucket(s) if s else None
    _bucket_metric(_score_key, real_pairs, "Score")

    # R:R cohort (derived from stop_loss / target_price / entry)
    def _rr_key(o, c, pnl, hd):
        e = o.get("price", 0)
        s = o.get("stop_loss", 0)
        t = o.get("target_price", 0)
        if e <= 0 or s <= 0 or t <= 0 or e <= s:
            return None
        reward = t - e
        risk = e - s
        if risk <= 0:
            return None
        rr = reward / risk
        return _rr_bucket(rr)
    _bucket_metric(_rr_key, real_pairs, "R:R")

    # Hold cohort
    def _hold_key(o, c, pnl, hd):
        return _hold_bucket(hd)
    _bucket_metric(_hold_key, real_pairs, "Hold days")

    # Close reason cohort
    def _reason_key(o, c, pnl, hd):
        r = str(c.get("reason", "")).lower()
        if "trail" in r:
            return "trail"
        if "target" in r or "limit" in r:
            return "target"
        if "ladder" in r:
            return "ladder"
        return "other"
    _bucket_metric(_reason_key, real_pairs, "Close reason")

    return lines


def _suggestions(metrics: dict, ops: dict) -> list:
    """Specific, actionable suggestions based on observed data."""
    s = []
    if metrics.get("n", 0) >= 3:
        # Trail too tight?
        if metrics["avg_hold"] < 3:
            s.append(
                "⚠️ Avg hold &lt;3 days — trail likely too tight. "
                "Consider widening initial floor 4% → 5%."
            )
        # Mostly trail-fired?
        rs = metrics.get("reasons", {})
        if rs.get("trail", 0) >= max(1, metrics["n"] - 1):
            s.append(
                "⚠️ Almost all closes are trail_fired. Few targets hit. "
                "Either targets are too ambitious or strategy doesn't reach them."
            )
        # Win rate way off
        if metrics["win_rate"] < 30 and metrics["profit_factor"] < 1.5:
            s.append(
                "⚠️ Win rate &lt;30% AND PF &lt;1.5 — selection is letting through losers. "
                "Tighten min_score from 73 → 75 next week to test."
            )
    if ops.get("drift", 0) > 5:
        s.append(
            f"⚠️ {ops['drift']} drift events this week — IB↔tracker sync unstable. "
            "Investigate ratchet modify path."
        )
    if ops.get("aborts", 0) > 0:
        s.append(
            f"🚨 {ops['aborts']} pipeline aborts. Check pre-flight IB / scan staleness."
        )
    if ops.get("auto_blocks", 0) > 0:
        s.append(
            f"ℹ️ {ops['auto_blocks']} ticker(s) auto-blocked (IB Error 201). "
            f"Use /unblock TICKER if you completed verification."
        )
    return s


def build_retro(today: date = None) -> str:
    today = today or date.today()
    log = _load_log()
    pairs = _pair_trades(log)

    # This week: Mon..today (if today is Fri, that's M-T-W-T-F closes)
    monday = today - timedelta(days=today.weekday())
    next_monday = monday + timedelta(days=7)
    last_monday = monday - timedelta(days=7)

    this_week = _filter_by_window(pairs, monday, next_monday)
    last_week = _filter_by_window(pairs, last_monday, monday)
    lifetime = pairs

    m_now = _metrics(this_week)
    m_prev = _metrics(last_week)
    m_life = _metrics(lifetime)
    ops = _operational_events(monday, next_monday)

    lines = [
        f"<b>📅 WEEKLY RETRO — week of {monday.strftime('%b %d')}</b>",
        f"<i>{today.strftime('%A %b %d, %Y')}</i>",
        "",
    ]

    # 1. Performance
    lines.append("<b>1️⃣ PERFORMANCE</b>")
    if m_now["n"] == 0:
        lines.append("  No closed trades this week.")
    else:
        emoji = "🟢" if m_now["total_pnl"] >= 0 else "🔴"
        lines.append(
            f"  {emoji} P&L: <b>${m_now['total_pnl']:+.2f}</b>  "
            f"({m_now['n']} trades, {m_now['wins']}W/{m_now['losses']}L)"
        )
        lines.append(
            f"  Win rate: <b>{m_now['win_rate']:.0f}%</b>  "
            f"|  Profit Factor: <b>{m_now['profit_factor']:.2f}</b>"
        )
        lines.append(
            f"  Biggest win: ${m_now['biggest_win']:+.2f}  "
            f"|  Biggest loss: ${m_now['biggest_loss']:+.2f}"
        )
        # vs last week
        if m_prev["n"] > 0:
            d_pnl = m_now["total_pnl"] - m_prev["total_pnl"]
            d_pf = m_now["profit_factor"] - m_prev["profit_factor"]
            lines.append(
                f"  <i>vs last week: P&L Δ ${d_pnl:+.2f}, PF Δ {d_pf:+.2f}</i>"
            )

    # 2. Execution quality
    lines.append("")
    lines.append("<b>2️⃣ EXECUTION QUALITY</b>")
    if m_now["n"] > 0:
        lines.append(f"  Avg hold: <b>{m_now['avg_hold']:.1f}d</b> (target 5-15d)")
        rs = m_now.get("reasons", {})
        if rs:
            reason_str = " | ".join(f"{k}:{v}" for k, v in sorted(rs.items(), key=lambda x: -x[1]))
            lines.append(f"  Close reasons: <code>{reason_str}</code>")
    else:
        lines.append("  N/A (no closes)")
    lines.append(
        f"  Ops events: drift {ops['drift']} | aborts {ops['aborts']} | "
        f"auto-recovers {ops['auto_recovers']} | auto-blocks {ops['auto_blocks']}"
    )

    # 3. Lifetime context
    lines.append("")
    lines.append("<b>3️⃣ LIFETIME (since start)</b>")
    if m_life["n"] > 0:
        emoji = "🟢" if m_life["total_pnl"] >= 0 else "🔴"
        lines.append(
            f"  {emoji} Total: ${m_life['total_pnl']:+.2f} over {m_life['n']} trades  "
            f"(WR {m_life['win_rate']:.0f}%, PF {m_life['profit_factor']:.2f})"
        )

    # 4. Cohort breakdown (lifetime — needs sample to be useful)
    cohort_lines = _cohort_breakdown(lifetime)
    if cohort_lines:
        lines.append("")
        lines.append("<b>4️⃣ COHORTS (lifetime — which setups win?)</b>")
        lines.extend(cohort_lines)

    # 5. Verdict
    lines.append("")
    lines.append("<b>5️⃣ VERDICT</b>")
    lines.append("  " + _verdict(m_now, m_prev))

    # 6. Suggestions
    suggs = _suggestions(m_now, ops)
    if suggs:
        lines.append("")
        lines.append("<b>6️⃣ SUGGESTIONS</b>")
        for s in suggs:
            lines.append(f"  {s}")

    return "\n".join(lines)


if __name__ == "__main__":
    msg = build_retro()
    if send_telegram(msg):
        logger.info("Weekly retro sent (%d chars)", len(msg))
    else:
        print(msg)
