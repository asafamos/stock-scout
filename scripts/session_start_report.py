"""Session-start report — give a fresh Claude / human operator the full
current-state picture in one command.

Runs LOCALLY (from the operator's machine or the VPS). Reads state from
whichever is available; prefers live VPS state when SSH is reachable.

Output sections (all read-only, side-effect-free):
  1. Portfolio (open positions + NetLiq + realized today)
  2. Systemd timer health (last-run + drift-detector last verdict)
  3. Followups past due (from memory/project_open_followups.md)
  4. Recent pipeline outcomes (last 3 cycles)
  5. Recent Telegram-relevant events (BUY/SELL/errors in last 24h)
  6. Config guardrails snapshot (frozen state vars from .env.trading)

The point: next session says 'read this', gets the ground-truth state
in 30 seconds, doesn't waste time grep'ing across memories/logs/VPS.
"""
from __future__ import annotations
import json, os, sys, subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
MEMORY_DIR = Path.home() / ".claude" / "projects" / "-Users-asafamos-StockScout-stock-scout-2" / "memory"


def _run_vps(cmd: str, timeout: int = 30) -> tuple[bool, str]:
    """Run a command on VPS via SSH. Returns (ok, output)."""
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "root@87.99.142.12", cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return (r.returncode == 0, (r.stdout or r.stderr).strip())
    except Exception as e:
        return (False, str(e))


def section(title: str):
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


# ─────────────────────────────────────────────────────────────
# 1. Portfolio
# ─────────────────────────────────────────────────────────────
section("1. PORTFOLIO — open positions + cash")

ok, out = _run_vps("cat /home/stockscout/stock-scout-2/data/trades/open_positions.json 2>/dev/null")
if ok and out and out != "[]":
    try:
        pos = json.loads(out)
        print(f"  {len(pos)} open position(s):")
        for p in pos:
            tk = p.get("ticker","?"); qty = p.get("quantity","?")
            ep = p.get("entry_price",0) or 0; tp = p.get("trailing_stop_pct",0) or 0
            opened = (p.get("opened_at","") or "")[:10]
            print(f"    {tk:6s}  {qty:>4}sh @ ${ep:.2f}  trail {tp}%  opened {opened}")
    except Exception as e:
        print(f"  parse error: {e}")
else:
    print("  (no open positions or VPS unreachable)")

# Latest snapshot for NetLiq
ok2, out2 = _run_vps("cat /home/stockscout/stock-scout-2/data/state/portfolio_snapshot.json 2>/dev/null")
if ok2 and out2:
    try:
        snap = json.loads(out2)
        print(f"  NetLiq: ${snap.get('net_liquidation','?')} | "
              f"Cash: ${snap.get('cash_balance','?')} | "
              f"snapshot_at: {snap.get('snapshot_at','?')}")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────
# 2. Timer health
# ─────────────────────────────────────────────────────────────
section("2. TIMERS — systemd timer health + last-run")

ok, out = _run_vps("systemctl list-timers --all 'stockscout-*' --no-pager 2>/dev/null | head -20")
if ok and out:
    print(out)
else:
    print("  (could not query timers)")

# ─────────────────────────────────────────────────────────────
# 3. Followups past due
# ─────────────────────────────────────────────────────────────
section("3. FOLLOWUPS past due")

fu_path = MEMORY_DIR / "project_open_followups.md"
if not fu_path.exists():
    print("  (followups file not found)")
else:
    today = datetime.now().date()
    text = fu_path.read_text()
    # Grep for **OPEN — ... — due YYYY-MM-DD** patterns
    past_due = []
    for m in re.finditer(r"\*\*(OPEN|CHECK_NOW|🔴)[^*]*due\s+(\d{4}-\d{2}-\d{2})", text):
        try:
            d = datetime.strptime(m.group(2), "%Y-%m-%d").date()
            if d <= today:
                # grab the line
                start = m.start()
                line_end = text.find("\n", start)
                if line_end < 0: line_end = len(text)
                past_due.append((d, text[start:line_end][:200]))
        except Exception:
            pass
    if past_due:
        past_due.sort()
        print(f"  {len(past_due)} past-due item(s):")
        for d, line in past_due[:10]:
            days_over = (today - d).days
            print(f"    [{days_over:>3}d overdue] {d}  {line}")
    else:
        print("  ✅ none past due")

# ─────────────────────────────────────────────────────────────
# 4. Recent pipeline outcomes (last 3 cycles)
# ─────────────────────────────────────────────────────────────
section("4. PIPELINES — last 3 outcomes")

ok, out = _run_vps("journalctl -u stockscout-pipeline --since '48 hours ago' --no-pager 2>/dev/null "
                   "| grep -E 'Recorded [0-9]+ new scan|Filters:|No stocks pass|BUY |SELL |SCAN_ONLY' | tail -25")
if ok and out:
    for line in out.split("\n"):
        # Strip journalctl prefix for readability
        if "]:" in line:
            print(f"    {line.split(']:', 1)[-1].strip()[:150]}")
        else:
            print(f"    {line[:150]}")
else:
    print("  (no pipeline events in last 48h)")

# ─────────────────────────────────────────────────────────────
# 5. Recent monitor events
# ─────────────────────────────────────────────────────────────
section("5. MONITOR — Telegram-relevant events last 24h")

ok, out = _run_vps("journalctl -u stockscout-monitor --since '24 hours ago' --no-pager 2>/dev/null "
                   "| grep -iE 'notify|BUY |SELL |SLD|closed_externally|alerted|trail_fired|Auto-recover|proceeding with close|ERROR|missing from IB' "
                   "| grep -v 'Heartbeat\\|Disconnect\\|Sync\\|updatePortfolio\\|execDetails' | tail -20")
if ok and out:
    for line in out.split("\n"):
        if "]:" in line:
            print(f"    {line.split(']:', 1)[-1].strip()[:150]}")
        else:
            print(f"    {line[:150]}")
else:
    print("  (no monitor events in last 24h)")

# ─────────────────────────────────────────────────────────────
# 6. Config guardrails snapshot
# ─────────────────────────────────────────────────────────────
section("6. CONFIG guardrails — frozen state (verify vs CLAUDE.md)")

ok, out = _run_vps("grep -E 'TRADE_(MIN|MAX)_(SCORE|ML|RR)|TRADE_MIN_FUND|TRADE_BLOCKED_SECTORS|TRADE_MAX_OPEN|TRADE_ADAPTIVE|TRADE_DAY_N|TRADE_RATCHET' "
                   "/home/stockscout/stock-scout-2/.env.trading 2>/dev/null | grep -v '^#'")
if ok and out:
    for line in sorted(out.split("\n")):
        print(f"    {line}")
else:
    print("  (env unreachable)")

# ─────────────────────────────────────────────────────────────
# 7. Latest drift-detector verdict
# ─────────────────────────────────────────────────────────────
section("7. DRIFT DETECTOR — last verdict")

ok, out = _run_vps("journalctl -u stockscout-drift-check --since '48 hours ago' --no-pager 2>/dev/null "
                   "| grep -iE 'DRIFT|matches|no drift' | tail -5")
if ok and out:
    for line in out.split("\n"):
        if "]:" in line:
            print(f"    {line.split(']:', 1)[-1].strip()[:150]}")
        else:
            print(f"    {line[:150]}")
else:
    print("  (no drift-check runs in last 48h)")

print()
print("=" * 80)
print(f"  Report generated: {datetime.now(timezone.utc).isoformat()}")
print("=" * 80)
