"""Weekly followup audit — runs Friday 05:00 UTC (08:00 IL).

Reads data/followups.json, runs a live verifier for each open item, and
auto-closes anything whose success criterion is already met. Sends a
single Telegram summary: what auto-closed, what's still open, what's
overdue.

Purpose: prevent the 'deployed and forgot' pattern. Daily health surfaces
the LIST of followups; this weekly script actually CHECKS THEM against
live state and closes what's done, so the list stays honest.

Design rules:
- Verifiers are dumb, fast, and independent — one per item id.
- A verifier returns (done: bool, note: str). done=True → auto-close.
- Unknown IDs are reported as 'no verifier' — human handles.
- Fail-safe on any exception — never crash, always try to send summary.
- Atomic write for followups.json (tmp file + rename) to avoid corrupt
  state if the process dies mid-write.
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path("/home/stockscout/stock-scout-2") if Path("/home/stockscout").exists() \
       else Path(__file__).resolve().parents[1]

FOLLOWUPS = ROOT / "data" / "followups.json"
MODEL_META = ROOT / "models" / "model_20d_v3.metadata.json"
OPEN_POSITIONS = ROOT / "data" / "trades" / "open_positions.json"
SCAN_OUTCOMES = ROOT / "data" / "outcomes" / "scan_outcomes.jsonl"
LEDGER = ROOT / "data" / "trades" / "executions.jsonl"

VerifierResult = Tuple[bool, str]


# ── Verifiers ──────────────────────────────────────────────────────────
# Each returns (done: bool, note: str). If done, item auto-closes.
# Keep verifiers cheap (< 1s) — this runs weekly, but should stay snappy.


def _v_ml_training(item: dict) -> VerifierResult:
    """ML training v3.9-reduced: feature_list <= 6 after 2026-08-14 03:00 UTC."""
    if not MODEL_META.exists():
        return False, "metadata.json missing"
    try:
        m = json.loads(MODEL_META.read_text())
        feats = m.get("feature_list", [])
        ts_str = m.get("training_timestamp_utc", "")
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else None
        n = len(feats)
        # Must be a training AFTER our commit (2026-08-14 12:07 UTC trigger)
        # AND have dropped features to <= 6
        if ts and ts.date() >= date(2026, 8, 14) and n <= 6:
            auc = m.get("metrics", {}).get("oos_auc", 0)
            return True, f"feature_list={n} (from 10), AUC={auc:.4f}, trained {ts_str[:16]}"
        return False, f"feature_list={n}, trained {ts_str[:16]} (need <=6 after 2026-08-14)"
    except Exception as e:
        return False, f"parse error: {e}"


def _v_positions_close(item: dict) -> VerifierResult:
    """3 open positions target_date: AAL/FRSH/FTNT all gone from tracker."""
    tickers_to_close = {"AAL", "FRSH", "FTNT"}
    if not OPEN_POSITIONS.exists():
        return False, "open_positions.json missing"
    try:
        pos = json.loads(OPEN_POSITIONS.read_text())
        if not isinstance(pos, list):
            return False, "not a list"
        held = {p.get("ticker") for p in pos}
        still_open = held & tickers_to_close
        if not still_open:
            return True, "AAL/FRSH/FTNT all closed"
        return False, f"still open: {sorted(still_open)}"
    except Exception as e:
        return False, f"parse error: {e}"


def _v_analyst_pt_resolved(item: dict) -> VerifierResult:
    """Analyst PT: scan_outcomes.jsonl PT count > 145 (baseline)."""
    if not SCAN_OUTCOMES.exists():
        return False, "scan_outcomes.jsonl missing"
    try:
        n = 0
        for line in SCAN_OUTCOMES.open():
            if '"analyst_mean_pt"' in line:
                n += 1
        if n > 145:
            return True, f"{n} resolved records with analyst PT (was 145 baseline)"
        return False, f"{n} resolved records (need > 145 baseline)"
    except Exception as e:
        return False, f"read error: {e}"


def _v_post_freeze_review(item: dict) -> VerifierResult:
    """Post-freeze 10-close review: SELL count post-2026-07-09 >= 10."""
    if not LEDGER.exists():
        return False, "executions.jsonl missing"
    try:
        n = 0
        for line in LEDGER.open():
            try:
                r = json.loads(line)
                if r.get("side") == "SELL" and r.get("time", "")[:10] >= "2026-07-09":
                    n += 1
            except Exception:
                continue
        if n >= 10:
            return True, f"{n} auto SELL closes post-freeze (target 10) — review ready"
        return False, f"{n}/10 auto SELL closes post-freeze"
    except Exception as e:
        return False, f"read error: {e}"


def _v_ladder_suppress(item: dict) -> VerifierResult:
    """LADDER suppression: no 'LADDER T% SELL failed' errors in last 7 days."""
    try:
        r = subprocess.run(
            ["journalctl", "-u", "stockscout-monitor", "--since", "7 days ago", "--no-pager"],
            capture_output=True, text=True, timeout=30,
        )
        err_count = sum(1 for line in r.stdout.splitlines() if "LADDER" in line and "SELL failed" in line)
        suppress_count = sum(1 for line in r.stdout.splitlines() if "LADDER/partial suppressed" in line)
        if suppress_count > 0 and err_count == 0:
            return True, f"7d: {suppress_count} suppression logs, 0 errors — suppression working"
        if err_count > 0:
            return False, f"7d: {err_count} LADDER errors still occurring"
        return False, f"7d: {suppress_count} suppression logs, no ratchet events yet"
    except Exception as e:
        return False, f"journalctl error: {e}"


VERIFIERS: dict[str, Callable[[dict], VerifierResult]] = {
    "ml-training-aug14": _v_ml_training,
    "positions-close-aug24": _v_positions_close,
    "analyst-pt-resolution-aug24": _v_analyst_pt_resolved,
    "post-freeze-10close-review": _v_post_freeze_review,
    "ladder-suppress-verify": _v_ladder_suppress,
}


# ── Main audit + report ────────────────────────────────────────────────


def _write_json_atomic(path: Path, data: dict):
    """Write JSON atomically to avoid corrupt state on crash."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def run_audit() -> str:
    """Run all verifiers, auto-close items, return Telegram-ready summary."""
    if not FOLLOWUPS.exists():
        return "🟠 <b>Weekly Followup Audit</b>\n\nNo data/followups.json found."

    try:
        data = json.loads(FOLLOWUPS.read_text())
    except Exception as e:
        return f"🔴 <b>Weekly Followup Audit — ERROR</b>\n\nCould not parse followups.json: {e}"

    items = data.get("items", []) or []
    today = date.today().isoformat()
    closed_this_run = []
    still_open = []
    no_verifier = []
    overdue = []

    for it in items:
        if it.get("status") == "closed":
            continue
        item_id = it.get("id", "?")
        desc = it.get("description", "?")
        due_str = it.get("due_date", "")
        try:
            due = date.fromisoformat(due_str)
            delta = (due - date.today()).days
        except (ValueError, TypeError):
            delta = 999

        verifier = VERIFIERS.get(item_id)
        if verifier is None:
            no_verifier.append((item_id, desc, delta))
            continue

        try:
            done, note = verifier(it)
        except Exception as e:
            done, note = False, f"verifier crashed: {e}"

        if done:
            it["status"] = "closed"
            it["closed_date"] = today
            it["closed_note"] = f"auto-closed by weekly audit: {note}"
            closed_this_run.append((desc, note))
        else:
            if delta < 0:
                overdue.append((desc, delta, note))
            else:
                still_open.append((desc, delta, note))

    # Persist any auto-closes
    if closed_this_run:
        _write_json_atomic(FOLLOWUPS, data)

    # Build the Telegram summary
    lines = [f"📋 <b>Weekly Followup Audit — {today}</b>\n"]

    if closed_this_run:
        lines.append(f"✅ <b>Auto-closed ({len(closed_this_run)}):</b>")
        for desc, note in closed_this_run:
            lines.append(f"  ✓ {desc}")
            lines.append(f"      <i>{note}</i>")
        lines.append("")

    if overdue:
        overdue.sort(key=lambda x: x[1])  # most overdue first
        lines.append(f"🔴 <b>Overdue ({len(overdue)}):</b>")
        for desc, delta, note in overdue:
            lines.append(f"  🔴 {desc} — {-delta}d overdue")
            lines.append(f"      <i>{note}</i>")
        lines.append("")

    if still_open:
        still_open.sort(key=lambda x: x[1])  # soonest first
        lines.append(f"🟢 <b>Still open ({len(still_open)}):</b>")
        for desc, delta, note in still_open:
            lines.append(f"  🟢 {desc} — {delta}d away")
            lines.append(f"      <i>{note}</i>")
        lines.append("")

    if no_verifier:
        lines.append(f"⚠️ <b>No verifier ({len(no_verifier)}):</b> manual check")
        for item_id, desc, delta in no_verifier:
            when = f"{-delta}d overdue" if delta < 0 else f"{delta}d away"
            lines.append(f"  • [{item_id}] {desc} — {when}")

    if not (closed_this_run or overdue or still_open or no_verifier):
        lines.append("✅ All followups closed. No open items.")

    return "\n".join(lines)


def main():
    summary = run_audit()
    try:
        from core.trading.notifications import _send
        _send(summary)
        print("Sent weekly followup audit to Telegram")
        return 0
    except Exception as e:
        logger.error("Telegram send failed: %s", e)
        print(summary)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
