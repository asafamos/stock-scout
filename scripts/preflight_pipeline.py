"""Pre-flight capacity check for the scan-trade pipeline.

Purpose: skip the ~45-min GH Actions scan + IB API + trade evaluation when
there is no capacity to actually take a trade. Called from
`deploy/scan_and_trade.sh` BEFORE triggering the GH Actions dispatch.

Exit codes:
    0 = PROCEED (capacity available or DRY_RUN)
    1 = SKIP (no capacity)
    2 = IB_UNAVAILABLE (can't determine — proceed conservatively)

Skip criteria:
    * n_open_positions >= MAX_OPEN_POSITIONS  → no slot for new position
    * cash < min_viable_position_usd          → not enough for smallest buy

Also prints a status line to stdout for the shell caller to relay to Telegram.

Kill switch: TRADE_SKIP_WHEN_FULL=0 disables the whole thing (always proceed).
"""
from __future__ import annotations

import os
import sys
import logging

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    if os.getenv("TRADE_SKIP_WHEN_FULL", "1").strip() in ("0", "false", "no"):
        print("PROCEED:disabled_via_env")
        return 0

    # In DRY_RUN we still want to verify the system runs end-to-end
    if os.getenv("TRADE_DRY_RUN", "0").strip() in ("1", "true", "yes"):
        print("PROCEED:dry_run_mode")
        return 0

    try:
        from core.trading.config import CONFIG
        from core.trading.ibkr_client import IBKRClient
    except Exception as e:
        print(f"IB_UNAVAILABLE:import_failed:{e}")
        return 2

    max_pos = int(getattr(CONFIG, "max_open_positions", 3) or 3)
    min_viable = float(getattr(CONFIG, "min_viable_position_usd", 30.0) or 30.0)

    # READ-ONLY DETECTION (added 2026-08-24). IB Gateway sometimes drops
    # to read-only mode after 24h continuous session (or authenticated
    # with the read-only checkbox). Reads still work — the shell caller
    # would see PROCEED — but BUY orders silently fail with Error 321.
    # We saw 9 days of 0 buys before diagnosing. Now: capture errors via
    # ib_insync errorEvent during preflight; if 321 fires, ABORT +
    # Telegram-alert so the shell can surface it.
    read_only_detected = {"flag": False, "msg": ""}

    def _err_handler(reqId, errorCode, errorString, contract):
        if errorCode == 321 or "Read-Only" in (errorString or ""):
            read_only_detected["flag"] = True
            read_only_detected["msg"] = errorString

    client = IBKRClient(CONFIG)
    try:
        if not client.connect():
            print("IB_UNAVAILABLE:connect_failed")
            return 2
        # Hook error handler BEFORE any IB calls so we catch read-only errors
        try:
            client._ib.errorEvent += _err_handler
        except Exception:
            pass  # errorEvent not available on all builds — non-fatal
        positions = client.get_positions() or []
        n_pos = len(positions)
        try:
            cash = float(client.get_cash_balance() or 0.0)
        except Exception as _e:
            print(f"IB_UNAVAILABLE:cash_balance_failed:{_e}")
            return 2
    finally:
        try:
            client.disconnect()
        except Exception:
            pass

    # Read-only wins over capacity: even if slots are free, BUYS will fail.
    if read_only_detected["flag"]:
        msg = read_only_detected["msg"][:120] if read_only_detected["msg"] else "detected"
        print(f"IB_READ_ONLY:{msg}:positions={n_pos}:cash=${cash:.0f}")
        # Telegram alert — piggyback on core notifications so operator sees it
        try:
            from core.trading.notifications import _send
            _send(
                f"🚨 <b>IB READ-ONLY MODE DETECTED</b>\n\n"
                f"Preflight caught Error 321 — BUYS will silently fail.\n"
                f"Positions: {n_pos}, cash ${cash:.0f}\n\n"
                f"Fix: <code>ssh root@87.99.142.12 'docker restart ibgateway'</code>\n"
                f"Then approve 2FA on IBKR Mobile."
            )
        except Exception:
            pass
        return 2  # IB_UNAVAILABLE — shell should abort dispatch

    # Decide
    reasons = []
    if n_pos >= max_pos:
        reasons.append(f"positions_full({n_pos}/{max_pos})")
    if cash < min_viable:
        reasons.append(f"cash_insufficient(${cash:.0f}<${min_viable:.0f})")

    if reasons:
        print(f"SKIP:{','.join(reasons)}:positions={n_pos}:cash=${cash:.0f}")
        return 1
    print(f"PROCEED:capacity_ok:positions={n_pos}/{max_pos}:cash=${cash:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
