#!/usr/bin/env bash
# Daily shadow-selector job (additive; trades nothing). See scripts/shadow_log.py and
# docs/shadow_selector_prereg.md.
#   1. read the newest scan parquet straight from origin/main (working tree is NOT touched —
#      the 2026-09-25 freshness incident was caused by checkout games in the working tree)
#   2. log it (whole scan + rule flags), idempotent per scan date
#   3. resolve matured days (20 sessions) and refresh the report
# Any failing step makes the unit fail -> OnFailure Telegram alert.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
RC=0

git fetch -q origin main || { echo "git fetch failed"; RC=1; }
TMP="$(mktemp /tmp/shadow-scan-XXXXXX.parquet)"
if git show origin/main:data/scans/latest_scan.parquet > "$TMP" 2>/dev/null && [ -s "$TMP" ]; then
    $PY -m scripts.shadow_log --parquet "$TMP" || { echo "shadow_log failed"; RC=1; }
else
    echo "could not read data/scans/latest_scan.parquet from origin/main"; RC=1
fi
rm -f "$TMP"

$PY -m scripts.shadow_resolve || { echo "shadow_resolve failed"; RC=1; }
$PY -m scripts.shadow_report  || { echo "shadow_report failed";  RC=1; }
exit $RC
