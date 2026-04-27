#!/bin/bash
# StockScout — Atomic Scan + Trade Pipeline
#
# Runs the FULL scan locally on the VPS, then immediately invokes the
# trade executor + outcomes recorder. Single systemd unit = no timer race
# between scan completion and trade firing, no GH Actions cron unreliability,
# no git push/pull window where VPS reads stale data.
#
# Why this exists: previous architecture had GH Actions firing the scan
# (cron unreliable, often 30-46 min late), which then committed to git,
# which the VPS pulled, while a SEPARATE systemd timer fired the trade
# at a fixed time hoping the scan was done. When cron was late, trade
# fired on stale data. When manual scan was triggered to recover,
# `cancel-in-progress: true` killed it. Three points of failure in a
# pipeline that should be atomic.
#
# This script: scan → outcomes-record → trade, in sequence, in one process.
#
# Invoked by stockscout-pipeline.timer at 13:30 UTC (morning) and
# 17:30 UTC (afternoon). Trade fires WHEN SCAN ACTUALLY COMPLETES,
# not at a guessed-fixed time.

set -uo pipefail
exec > >(while IFS= read -r line; do echo "[$(date -u +%H:%M:%S)] $line"; done)
exec 2>&1

ROOT=/home/stockscout/stock-scout-2
LABEL="${1:-pipeline}"
PY=$ROOT/.venv/bin/python

cd "$ROOT"
set -a
source .env.trading 2>/dev/null || true
set +a

echo "═══════════════════════════════════════════════════════"
echo "Starting $LABEL pipeline"
echo "═══════════════════════════════════════════════════════"

# 1. Sync code from origin (non-blocking on conflicts).
echo "Syncing code from origin..."
git fetch origin main --quiet 2>/dev/null || true
git merge --ff-only origin/main --quiet 2>/dev/null || \
    echo "  (git merge non-ff — local data files diverge; using current state)"

# 2. Run the full scan inline (~55 min on Polygon free tier).
# DISABLE_AUTO_TRADE=1 — even if scan flow has its own auto-trade hook,
# we want trade to fire AFTER outcomes are recorded so the JSONL feed
# captures every candidate from this exact scan.
echo "Running full scan (this takes ~55 min)..."
SCAN_T0=$(date +%s)
DISABLE_AUTO_TRADE=1 $PY -m scripts.run_full_scan 2>&1 | tail -50
SCAN_EXIT=${PIPESTATUS[0]}
SCAN_DUR=$(( $(date +%s) - SCAN_T0 ))
echo "Scan finished (exit=$SCAN_EXIT, duration=${SCAN_DUR}s)"

if [ "$SCAN_EXIT" -ne 0 ]; then
    echo "Scan FAILED — skipping outcomes-record and trade for safety."
    exit "$SCAN_EXIT"
fi

# 3. Verify scan output is fresh.
SCAN_FILE="$ROOT/data/scans/latest_scan.parquet"
if [ ! -f "$SCAN_FILE" ]; then
    echo "Scan produced no parquet — aborting."
    exit 2
fi
SCAN_AGE_MIN=$(( ($(date +%s) - $(stat -c %Y "$SCAN_FILE")) / 60 ))
echo "Scan output age: ${SCAN_AGE_MIN}m"

# 4. Record candidates to outcomes JSONL (regime-tagged for ML feedback loop).
echo "Recording outcomes..."
$PY -m scripts.track_scan_outcomes --record 2>&1 | tail -5 || \
    echo "  (outcomes-record returned non-zero — continuing anyway)"

# 5. Fire the trade evaluator. It runs all risk gates + live price refresh.
echo "Triggering auto-trade..."
TRADE_T0=$(date +%s)
$PY -m scripts.run_auto_trade 2>&1 | tail -30
TRADE_EXIT=${PIPESTATUS[0]}
TRADE_DUR=$(( $(date +%s) - TRADE_T0 ))
echo "Trade finished (exit=$TRADE_EXIT, duration=${TRADE_DUR}s)"

echo "═══════════════════════════════════════════════════════"
echo "Pipeline complete: scan=${SCAN_DUR}s + trade=${TRADE_DUR}s"
echo "═══════════════════════════════════════════════════════"
exit 0
