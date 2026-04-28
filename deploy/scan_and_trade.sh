#!/bin/bash
# StockScout — Event-Driven Scan→Trade Pipeline
#
# Architecture: scan runs in GH Actions (where the data-provider API keys
# live), VPS watches git for new scan parquet, fires trade IMMEDIATELY
# when one lands. No fixed-time trade timer = no race against the cron
# variability that has cost us a week of trading days.
#
# Flow per invocation:
#   1. Note current scan parquet's git hash.
#   2. Trigger GH Actions auto_scan via repository_dispatch (best-effort;
#      if no GITHUB_TOKEN available, rely on the workflow's own cron).
#   3. Poll origin every 30s for up to 90 min.
#   4. When the parquet hash changes → pull → record outcomes → trade.
#   5. Exit. Next invocation handled by stockscout-pipeline.timer.
#
# Manual: bash deploy/scan_and_trade.sh [label]
# Systemd: stockscout-pipeline.service / .timer

set -uo pipefail
exec > >(while IFS= read -r line; do echo "[$(date -u +%H:%M:%S)] $line"; done)
exec 2>&1

ROOT=/home/stockscout/stock-scout-2
LABEL="${1:-pipeline}"
PY=$ROOT/.venv/bin/python
MAX_WAIT_SEC=$((150 * 60))  # 150 min hard cap — generous to absorb GH Actions
                            # cron delays (often 30–60 min late) PLUS the scan
                            # duration (~55 min). 90 min hit the cap on
                            # 2026-04-28 (cron 47 min late + 49 min scan).
POLL_SEC=30

cd "$ROOT"
set -a
source .env.trading 2>/dev/null || true
set +a

echo "═══════════════════════════════════════════════════════"
echo "Starting $LABEL pipeline (event-driven scan→trade)"
echo "═══════════════════════════════════════════════════════"

# Snapshot the current scan parquet hash so we can detect a NEW one.
git fetch origin main --quiet 2>/dev/null || true
START_HASH=$(git log -1 --format=%H origin/main -- data/scans/latest_scan.parquet 2>/dev/null || echo "none")
echo "Current scan parquet hash: ${START_HASH:0:12}"

# Best-effort: trigger a fresh GH Actions scan via REST.
# Requires GITHUB_TOKEN (PAT or fine-grained) with `actions:write` scope.
# If unavailable, skip — relies on the workflow's own cron schedule.
if [ -n "${GITHUB_TOKEN:-}" ]; then
    echo "Triggering GH Actions auto_scan workflow..."
    curl -fsS -X POST \
        -H "Authorization: token ${GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github+json" \
        https://api.github.com/repos/asafamos/stock-scout/actions/workflows/auto_scan.yml/dispatches \
        -d '{"ref":"main"}' && echo "  ✓ Workflow dispatched" || echo "  ✗ Dispatch failed (continuing with poll-only)"
else
    echo "No GITHUB_TOKEN — relying on workflow's own cron schedule."
fi

# Poll for new scan parquet on origin/main.
echo "Polling for new scan (every ${POLL_SEC}s, max $((MAX_WAIT_SEC / 60))min)..."
START_TIME=$(date +%s)
NEW_HASH="$START_HASH"
while [ "$NEW_HASH" = "$START_HASH" ]; do
    if [ $(($(date +%s) - START_TIME)) -gt "$MAX_WAIT_SEC" ]; then
        echo "TIMEOUT after $((MAX_WAIT_SEC / 60))min — no new scan committed. Aborting."
        exit 3
    fi
    sleep "$POLL_SEC"
    git fetch origin main --quiet 2>/dev/null || true
    NEW_HASH=$(git log -1 --format=%H origin/main -- data/scans/latest_scan.parquet 2>/dev/null || echo "none")
done

WAIT_DUR=$(( $(date +%s) - START_TIME ))
echo "✓ New scan detected after ${WAIT_DUR}s — hash ${NEW_HASH:0:12}"

# Pull origin's data/scans/* AUTHORITATIVELY (it's the source of truth for
# scan results). Don't stash-then-pop — that creates conflict markers in
# JSON files when the VPS has any local modification (it always does:
# data/trades/*.json reflects live IB state). The trade/positions JSONs
# are independent of git so we ONLY pull the scan files.
echo "Pulling latest scan files to VPS..."
git fetch origin main --quiet
git checkout origin/main -- data/scans/latest_scan.parquet data/scans/latest_scan.json data/scans/latest_scan.meta.json 2>/dev/null || true
# Also pull any code changes (excluding data/trades/* which we keep local)
git checkout origin/main -- core/ scripts/ deploy/ ml/ models/ 2>/dev/null || true

# Verify.
SCAN_FILE="$ROOT/data/scans/latest_scan.parquet"
if [ ! -f "$SCAN_FILE" ]; then
    echo "FATAL: scan parquet missing after pull. Aborting."
    exit 4
fi
SCAN_AGE_MIN=$(( ($(date +%s) - $(stat -c %Y "$SCAN_FILE")) / 60 ))
echo "Scan file age: ${SCAN_AGE_MIN}m"

# Record outcomes (regime-tagged JSONL for ML feedback loop).
echo "Recording outcomes..."
$PY -m scripts.track_scan_outcomes --record 2>&1 | tail -3 || \
    echo "  (outcomes-record returned non-zero — continuing)"

# Fire the trade evaluator. Live price refresh + 8 risk gates inside.
# TRADE_LIVE_CONFIRMED=1 is the systemd-pipeline authorization to run live;
# manual ssh runs (without this env var and without --live flag) default to
# DRY mode for safety. See run_auto_trade.py for the full policy.
echo "Triggering auto-trade..."
TRADE_T0=$(date +%s)
TRADE_LIVE_CONFIRMED=1 $PY -m scripts.run_auto_trade 2>&1 | tail -25
TRADE_EXIT=${PIPESTATUS[0]}
TRADE_DUR=$(( $(date +%s) - TRADE_T0 ))
echo "Trade finished (exit=$TRADE_EXIT, duration=${TRADE_DUR}s)"

echo "═══════════════════════════════════════════════════════"
echo "Pipeline complete: wait=${WAIT_DUR}s + trade=${TRADE_DUR}s"
echo "═══════════════════════════════════════════════════════"
exit 0
