#!/usr/bin/env bash
# v2 sleeve — runs at 09:31 ET (systemd timer). Trades the top S3_v1 name from the PRIOR close's scan.
# The scan is read straight from origin/main into data/state/ (the working tree's tracked parquet is
# NOT touched — see the 2026-09-25 freshness incident). Does nothing unless TRADE_V2_SLEEVE=1.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
mkdir -p data/state
if [ "${TRADE_V2_SLEEVE:-0}" != "1" ]; then
    echo "v2 sleeve disabled (TRADE_V2_SLEEVE != 1) — nothing to do"; exit 0
fi
git fetch -q origin main || { echo "git fetch failed"; exit 1; }
TMP="data/state/v2_scan.parquet.tmp"
if git show origin/main:data/scans/latest_scan.parquet > "$TMP" 2>/dev/null && [ -s "$TMP" ]; then
    mv -f "$TMP" data/state/v2_scan.parquet
else
    rm -f "$TMP"; echo "could not read latest_scan.parquet from origin/main"; exit 1
fi
TRADE_LIVE_CONFIRMED=1 $PY -m scripts.run_v2_sleeve --parquet data/state/v2_scan.parquet
