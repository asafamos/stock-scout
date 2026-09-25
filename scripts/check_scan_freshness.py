"""Scan-freshness watchdog — alert if latest_scan.parquet's As_Of_Date is stale.

Root cause it prevents: 2026-09-25 investigation found VPS's
`data/scans/latest_scan.parquet` had been frozen at commit 7862775d
(As_Of_Date 09-17) for 8 DAYS while pipelines silently reported success.
Root: `git checkout origin/main -- ...` listed a non-existent
`latest_scan.meta.json`; the atomic checkout failed silently and no
files were updated. See [[project_scan_freshness_bug_sep25]].

The pipeline itself now has a freshness assertion (aborts trade if
As_Of_Date > 5d old). This watchdog is defense-in-depth: fires DAILY
regardless of whether the pipeline ran, alerting on any silent staleness.

Exit codes:
    0 = fresh (As_Of_Date within tolerance)
    2 = stale + Telegram alert sent
    1 = fatal (can't read parquet)

Designed for systemd timer daily 06:30 UTC.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Tolerance: how many days old is "too stale" for daily watchdog.
# The in-pipeline assertion uses 5d (permissive during weekends). The
# daily watchdog uses a wider 7d to accommodate long weekends + holidays
# without noise, while still catching the 8-day drift pattern from 09-25.
STALENESS_THRESHOLD_DAYS = int(os.getenv("SCAN_FRESHNESS_MAX_DAYS", "7"))


def main() -> int:
    dry_run = "--dry-run" in sys.argv or os.getenv("DRIFT_CHECK_DRY_RUN", "0") == "1"

    root = Path("/home/stockscout/stock-scout-2") if Path("/home/stockscout").exists() \
           else Path(__file__).resolve().parent.parent
    scan_path = root / "data" / "scans" / "latest_scan.parquet"

    if not scan_path.exists():
        print(f"ERROR: {scan_path} missing", file=sys.stderr)
        return 1

    try:
        import pandas as pd
        df = pd.read_parquet(scan_path)
    except Exception as e:
        print(f"ERROR: cannot read parquet: {e}", file=sys.stderr)
        return 1

    if "As_Of_Date" not in df.columns or len(df) == 0:
        print(f"ERROR: parquet has no As_Of_Date column or is empty", file=sys.stderr)
        return 1

    as_of_raw = df["As_Of_Date"].iloc[0]
    try:
        import pandas as pd
        as_of = pd.Timestamp(as_of_raw)
        if as_of.tz is None:
            as_of = as_of.tz_localize("UTC")
    except Exception as e:
        print(f"ERROR: cannot parse As_Of_Date {as_of_raw!r}: {e}", file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc)
    days_old = (now - as_of.to_pydatetime()).days

    if days_old <= STALENESS_THRESHOLD_DAYS:
        print(f"✅ scan fresh: As_Of_Date={as_of.date()} ({days_old}d old, threshold {STALENESS_THRESHOLD_DAYS}d)")
        return 0

    print(f"🚨 SCAN STALE: As_Of_Date={as_of.date()} ({days_old}d old, threshold {STALENESS_THRESHOLD_DAYS}d)")

    if dry_run:
        print("(--dry-run: skipping Telegram alert)")
        return 2

    try:
        sys.path.insert(0, str(root))
        from core.trading.notifications import _send
        msg = (
            f"🚨 <b>SCAN FRESHNESS ALERT</b>\n"
            f"Host: <code>{scan_path}</code>\n\n"
            f"As_Of_Date: <b>{as_of.date()}</b> ({days_old} days old)\n"
            f"Threshold: {STALENESS_THRESHOLD_DAYS} days\n\n"
            f"The scan file has not been refreshed. Possible causes:\n"
            f"• Pipeline not running (check systemd timers)\n"
            f"• git checkout silently failing (see 2026-09-25 fix)\n"
            f"• GH Actions scan workflow broken\n"
            f"• VPS/GH network issue\n\n"
            f"See [[project_scan_freshness_bug_sep25]]."
        )
        _send(msg)
    except Exception as e:
        print(f"warning: telegram alert failed: {e}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
