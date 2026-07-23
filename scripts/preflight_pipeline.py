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

    client = IBKRClient(CONFIG)
    try:
        if not client.connect():
            print("IB_UNAVAILABLE:connect_failed")
            return 2
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
