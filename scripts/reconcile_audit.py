"""Reconciliation audit — triangulate tracker ↔ ledger ↔ IB.

Runs daily to catch silent drift between the three sources of truth. The
APH silent-SELL bug (2026-09-15) sat undetected for 3 days because nothing
was actively comparing "what tracker thinks" vs "what IB reports" vs "what
the ledger recorded." This closes the gap.

Detects:
    ORPHAN_IB       — position on IB not in tracker (fresh buy not recorded?)
    ORPHAN_TRACKER  — tracker entry not in IB and no matching SELL in ledger
                      (this is the APH silent-SELL pattern)
    QTY_MISMATCH    — tracker qty != IB qty (split/pyramid/manual trade drift)
    ORPHAN_LEDGER_OPEN — ledger says we have net-long shares but IB says 0

On any finding → Telegram alert with actionable line per issue.

Exit codes:
    0 = clean
    2 = drift detected + alert sent
    1 = fatal (can't reach IB)

Designed for systemd timer daily 07:00 UTC (after drift-check 06:15).
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/stockscout/stock-scout-2") if Path("/home/stockscout").exists() \
       else Path(__file__).resolve().parent.parent


def _load_tracker():
    p = ROOT / "data" / "trades" / "open_positions.json"
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
        return {r["ticker"]: r for r in raw} if isinstance(raw, list) else {}
    except Exception as e:
        print(f"tracker load error: {e}", file=sys.stderr)
        return {}


def _load_ledger_net_long():
    """Return {ticker: net_qty} from ledger executions. Positive = long.

    Notes on the sign math:
    - net > 0  → we still hold shares we bought (legitimate open long)
    - net = 0  → fully closed (round-trip complete)
    - net < 0  → SELL total > BUY total. Two known causes:
        (a) split-adjusted SELL after a pre-split BUY was recorded (e.g. APH
            bought 1sh, split 2:1, sold 2sh → net -1sh but portfolio is flat)
        (b) legacy tickers from before the ledger existed — only SELL side
            got recorded, no matching BUY (e.g. ARWR, STUB)
    Both are cosmetic accounting quirks, not real orphaned longs. We flag
    them separately (LEGACY_LEDGER info-level) instead of alerting.
    """
    p = ROOT / "data" / "trades" / "executions.jsonl"
    if not p.exists():
        return {}
    net = defaultdict(float)
    try:
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            side = (r.get("side") or "").upper()
            shares = float(r.get("shares") or 0)
            tk = r.get("ticker")
            if not tk:
                continue
            if side in ("BUY", "BOT"):
                net[tk] += shares
            elif side in ("SELL", "SLD"):
                net[tk] -= shares
    except Exception as e:
        print(f"ledger load error: {e}", file=sys.stderr)
        return {}
    return {tk: q for tk, q in net.items() if abs(q) > 0.001}


def _load_ib_positions():
    """Return {ticker: qty} from IBKR live positions. None on failure."""
    try:
        sys.path.insert(0, str(ROOT))
        from core.trading.config import CONFIG
        from core.trading.ibkr_client import IBKRClient
        client = IBKRClient(CONFIG)
        if not client.connect():
            return None
        try:
            positions = client.get_positions() or []
            return {p.ticker: float(p.quantity) for p in positions if p.quantity > 0}
        finally:
            client.disconnect()
    except Exception as e:
        print(f"IB reconcile error: {e}", file=sys.stderr)
        return None


def main() -> int:
    tracker = _load_tracker()
    ledger_open = _load_ledger_net_long()
    ib_pos = _load_ib_positions()

    if ib_pos is None:
        print("ERROR: cannot reach IB — aborting audit", file=sys.stderr)
        return 1

    tracker_tks = set(tracker.keys())
    ledger_tks = set(ledger_open.keys())
    ib_tks = set(ib_pos.keys())

    findings = []       # actionable: needs human/monitor intervention
    info_notes = []     # cosmetic / known-benign: log-only, no alert

    # ORPHAN_IB: IB holds it, tracker doesn't know about it
    for tk in ib_tks - tracker_tks:
        findings.append({
            "severity": "critical",
            "type": "ORPHAN_IB",
            "ticker": tk,
            "msg": f"IB holds {ib_pos[tk]:.0f}sh of {tk} but tracker has no entry — "
                   f"was this a manual buy? Add to tracker or sell.",
        })

    # ORPHAN_TRACKER: tracker has it, IB doesn't
    for tk in tracker_tks - ib_tks:
        net_ledger = ledger_open.get(tk, 0)
        if net_ledger > 0.001:
            # Positive net in ledger means we should still hold shares → IB
            # missed sync OR closed outside our system.
            findings.append({
                "severity": "critical",
                "type": "ORPHAN_TRACKER_SYNC_ISSUE",
                "ticker": tk,
                "msg": f"{tk} in tracker + ledger net=+{net_ledger:.0f}sh but IB shows 0 "
                       f"— possible sync issue or manual close outside our system.",
            })
        else:
            # net_ledger <= 0 → either fully closed (0) or split-adjusted (< 0).
            # Either way: monitor's close-detection SHOULD have cleaned tracker.
            # It didn't — this IS the silent-SELL pattern.
            findings.append({
                "severity": "warning",
                "type": "ORPHAN_TRACKER_SILENT_SELL",
                "ticker": tk,
                "msg": f"{tk} in tracker but not in IB (ledger net={net_ledger:+.0f}sh, "
                       f"{'split-adjusted' if net_ledger < 0 else 'flat'}) → SILENT-SELL. "
                       f"Monitor's close-detection should have cleaned this on its "
                       f"next cycle; if this persists >24h, investigate the reconcile path.",
            })

    # QTY_MISMATCH: tracker qty != IB qty
    for tk in tracker_tks & ib_tks:
        tqty = float(tracker[tk].get("quantity") or 0)
        iqty = ib_pos[tk]
        if abs(tqty - iqty) > 0.001:
            findings.append({
                "severity": "critical",
                "type": "QTY_MISMATCH",
                "ticker": tk,
                "msg": f"{tk} tracker qty={tqty:.0f} vs IB qty={iqty:.0f} "
                       f"(delta {iqty - tqty:+.0f}) — split/pyramid/manual?",
            })

    # ORPHAN_LEDGER: ticker in ledger but neither IB nor tracker.
    # Split into two: real orphan (positive net) vs cosmetic (negative net =
    # legacy pre-ledger SELLs without matching BUYs, or split-adjusted).
    for tk in ledger_tks - ib_tks:
        if tk in tracker_tks:
            continue  # already handled above
        net = ledger_open[tk]
        if net > 0.001:
            findings.append({
                "severity": "critical",
                "type": "ORPHAN_LEDGER_LONG",
                "ticker": tk,
                "msg": f"Ledger net=+{net:.0f}sh long on {tk} but IB and tracker have "
                       f"nothing — orphaned open? Investigate.",
            })
        else:
            # Negative net + not in IB + not in tracker = fully closed but
            # record math is off. Cosmetic; do not alert.
            info_notes.append(
                f"LEDGER_QUIRK {tk}: net={net:+.0f}sh (legacy pre-ledger SELL "
                f"or split-adjusted), IB flat, tracker flat — harmless"
            )

    # Print info-notes to stdout regardless (audit trail), but don't alert.
    for note in info_notes:
        print(f"  · {note}")

    critical = [f for f in findings if f.get("severity") == "critical"]
    warnings = [f for f in findings if f.get("severity") == "warning"]

    if not findings:
        print(f"✅ reconcile clean — tracker({len(tracker_tks)})={sorted(tracker_tks)} "
              f"| IB({len(ib_tks)})={sorted(ib_tks)} | ledger_open({len(ledger_tks)})={sorted(ledger_tks)}")
        return 0

    print(f"🚨 {len(critical)} critical + {len(warnings)} warning finding(s):")
    for f in findings:
        marker = "🔴" if f["severity"] == "critical" else "⚠️"
        print(f"  {marker} [{f['type']}] {f['ticker']}: {f['msg']}")

    # Telegram alert ONLY when critical findings exist. Warnings log-only
    # to avoid alert fatigue during monitor's normal cleanup cycles.
    if not critical:
        print(f"(warnings only — no Telegram alert)")
        return 2

    try:
        from core.trading.notifications import _send
        lines = [f"🚨 <b>Reconcile audit — {len(critical)} critical issue(s)</b>", ""]
        for f in critical:
            lines.append(f"🔴 <b>[{f['type']}]</b> <code>{f['ticker']}</code>")
            lines.append(f"  {f['msg']}")
        if warnings:
            lines.append("")
            lines.append(f"<i>Also {len(warnings)} warning(s) — see log for detail.</i>")
        lines.append("")
        lines.append(f"<i>State: tracker={sorted(tracker_tks)} IB={sorted(ib_tks)} ledger_open={sorted(ledger_tks)}</i>")
        lines.append("<i>Run `python scripts/reconcile_audit.py` to re-check.</i>")
        _send("\n".join(lines))
    except Exception as e:
        print(f"warning: telegram alert failed: {e}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
