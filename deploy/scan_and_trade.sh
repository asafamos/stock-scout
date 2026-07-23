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

set -euo pipefail
# `-e` added 2026-04-30: previous setup let `git fetch` failures fall
# through silently (returning "none" hash) which triggered false-positive
# "new scan" detections on the very next successful fetch — leading to
# trades against stale parquets. (Audit finding #10.)
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

# ─── Single-instance lock ─────────────────────────────────────────────
# Prevents the 17:30 timer from firing while the 13:30 instance is still
# polling/trading (a slow scan can run >4h). Without this, two pipelines
# could send duplicate buy orders. (Audit finding #9.)
#
# Use /tmp/ rather than /run/ — /run/ is writable only by root on most
# distros, and the pipeline runs as the stockscout user. /tmp/ is
# user-writable and tmpfs-mounted (auto-cleared on reboot, same as /run).
# Initial bug 2026-04-30: lock at /run/ caused `set -e` to exit
# immediately at startup, missing the entire 13:30 trading window.
LOCK_FILE=/tmp/stockscout-pipeline.lock
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Another pipeline instance is running (lock $LOCK_FILE held). Exiting."
    exit 7
fi

cd "$ROOT"
set -a
source .env.trading 2>/dev/null || true
set +a

# Telegram helper — defined early so the pre-flight IB check (and any
# other early failure path) can use it. Body is plain text; HTML allowed.
TG_SEND() {
    local emoji="$1"; local title="$2"; local body="$3"
    [ -z "${TRADE_TELEGRAM_TOKEN:-}" ] && return 0
    [ -z "${TRADE_TELEGRAM_CHAT_ID:-}" ] && return 0
    curl -fsS -X POST \
        "https://api.telegram.org/bot${TRADE_TELEGRAM_TOKEN}/sendMessage" \
        -d "chat_id=${TRADE_TELEGRAM_CHAT_ID}" \
        -d "parse_mode=HTML" \
        --data-urlencode "$(printf 'text=%s <b>%s</b>\n%s' "$emoji" "$title" "$body")" \
        >/dev/null 2>&1 || true
}

echo "═══════════════════════════════════════════════════════"
echo "Starting $LABEL pipeline (event-driven scan→trade)"
echo "═══════════════════════════════════════════════════════"

# Notify Telegram pipeline START — gives the user visibility into what's
# happening before the 50min scan completes, instead of silence followed by
# either a buy or "no candidates passed" with no context.
if [ -n "${TRADE_TELEGRAM_TOKEN:-}" ] && [ -n "${TRADE_TELEGRAM_CHAT_ID:-}" ]; then
    curl -fsS -X POST \
        "https://api.telegram.org/bot${TRADE_TELEGRAM_TOKEN}/sendMessage" \
        -d "chat_id=${TRADE_TELEGRAM_CHAT_ID}" \
        -d "parse_mode=HTML" \
        --data-urlencode "$(printf 'text=🔍 <b>Pipeline started</b> (%s)\nWaiting for fresh scan from GH Actions...' "$LABEL")" \
        >/dev/null 2>&1 || true
fi

# ─── Pre-flight capacity check (2026-07-23, task #143) ─────────────────
# Skip triggering GH Actions if we have no free slots or no cash. This
# saves ~45min of scan compute + IB API calls per empty pipeline. If IB
# is unreachable we PROCEED conservatively (better a wasted scan than a
# missed buy).
# Kill switch: TRADE_SKIP_WHEN_FULL=0 disables the entire preflight.
PREFLIGHT_OUT=$($PY -m scripts.preflight_pipeline 2>&1 || true)
PREFLIGHT_RC=$?
echo "Preflight: $PREFLIGHT_OUT"
case "$PREFLIGHT_OUT" in
    SKIP:*)
        REASON="${PREFLIGHT_OUT#SKIP:}"
        echo "Pipeline SKIPPED — $REASON"
        TG_SEND "⏭️" "Pipeline SKIPPED (no capacity)" "Reason: <code>${REASON}</code>
No GH Actions scan triggered, no compute wasted.
Kill: <code>TRADE_SKIP_WHEN_FULL=0</code>"
        exit 0
        ;;
    IB_UNAVAILABLE:*)
        echo "Preflight inconclusive — IB unavailable. Proceeding conservatively."
        # Fall through — better a wasted scan than a missed opportunity
        ;;
    PROCEED:*)
        echo "Preflight OK — capacity available."
        ;;
esac

# Snapshot the current scan parquet hash so we can detect a NEW one.
# Fail-fast if the initial fetch fails — we must have a true baseline,
# not "none", or the next successful fetch will trip the change detector
# and trade on the stale parquet still on disk. (Audit finding #10.)
if ! git fetch origin main --quiet; then
    echo "FATAL: initial git fetch failed — cannot establish baseline. Aborting."
    exit 5
fi
START_HASH=$(git log -1 --format=%H origin/main -- data/scans/latest_scan.parquet 2>/dev/null || echo "none")
if [ "$START_HASH" = "none" ]; then
    echo "FATAL: no parquet hash found on origin/main. Aborting."
    exit 5
fi
echo "Current scan parquet hash: ${START_HASH:0:12}"

# Best-effort: trigger a fresh GH Actions scan via REST.
# Requires GITHUB_TOKEN (PAT or fine-grained) with `actions:write` scope.
# If unavailable, skip — relies on the workflow's own cron schedule.
if [ -n "${GITHUB_TOKEN:-}" ]; then
    echo "Triggering GH Actions auto_scan workflow..."
    DISPATCH_HTTP=$(curl -fsS -o /tmp/dispatch_resp -w "%{http_code}" -X POST \
        -H "Authorization: token ${GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github+json" \
        https://api.github.com/repos/asafamos/stock-scout/actions/workflows/auto_scan.yml/dispatches \
        -d '{"ref":"main"}' 2>&1) || DISPATCH_HTTP="curl_failed"
    if [ "$DISPATCH_HTTP" = "204" ]; then
        echo "  ✓ Workflow dispatched (HTTP 204)"
    else
        echo "  ✗ Dispatch failed (HTTP=$DISPATCH_HTTP)"
        # Telegram alert — silent dispatch failure (token expired, rate
        # limited) used to mean a 2.5h timeout before we noticed.
        if [ -n "${TRADE_TELEGRAM_TOKEN:-}" ] && [ -n "${TRADE_TELEGRAM_CHAT_ID:-}" ]; then
            DISP_BODY=$(head -c 200 /tmp/dispatch_resp 2>/dev/null || true)
            curl -fsS -X POST \
                "https://api.telegram.org/bot${TRADE_TELEGRAM_TOKEN}/sendMessage" \
                -d "chat_id=${TRADE_TELEGRAM_CHAT_ID}" \
                -d "parse_mode=HTML" \
                --data-urlencode "$(printf 'text=⚠️ <b>PIPELINE DISPATCH FAIL</b>\nHTTP=%s\n%s\nContinuing with cron-only fallback.' "$DISPATCH_HTTP" "$DISP_BODY")" \
                >/dev/null 2>&1 || true
        fi
        echo "  → Continuing with poll-only (cron fallback)"
    fi
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

# CRITICAL FIX (2026-05-05): refresh the scan file mtime to NOW.
# `git checkout` preserves the BLOB's original mtime when the file content
# is identical to the working-tree version (and sometimes even when it
# differs — see git's smudge filter behavior). On 2026-05-04 this caused
# both the 13:30 UTC and 17:30 UTC trade pipelines to abort with
# "Scan file is 66.8h old" even though the parquet hash had just changed.
# `_load_scan_results` reads file mtime to decide staleness, so the
# trade fired with a 4h staleness gate, then refused to execute. Touch
# the files after checkout so mtime reflects "this VPS just received it".
for f in data/scans/latest_scan.parquet data/scans/latest_scan.json data/scans/latest_scan.meta.json; do
    [ -f "$f" ] && touch "$f"
done

# Verify.
SCAN_FILE="$ROOT/data/scans/latest_scan.parquet"
if [ ! -f "$SCAN_FILE" ]; then
    echo "FATAL: scan parquet missing after pull. Aborting."
    exit 4
fi
SCAN_AGE_MIN=$(( ($(date +%s) - $(stat -c %Y "$SCAN_FILE")) / 60 ))
echo "Scan file age: ${SCAN_AGE_MIN}m"

# Record outcomes (regime-tagged JSONL for ML feedback loop).
# Capture full output to a file (for diagnostics) AND tail to stdout.
# Failure here doesn't block the trade, but it WILL Telegram-alert so we
# don't silently lose the ML feedback loop again like 2026-04-23..27.
echo "Recording outcomes..."
OUT_LOG=/tmp/outcomes-record-$$.log
$PY -m scripts.track_scan_outcomes --record >"$OUT_LOG" 2>&1
OUT_EXIT=$?
tail -3 "$OUT_LOG"
if [ "$OUT_EXIT" -ne 0 ]; then
    echo "  ✗ outcomes-record FAILED (exit=$OUT_EXIT)"
    if [ -n "${TRADE_TELEGRAM_TOKEN:-}" ] && [ -n "${TRADE_TELEGRAM_CHAT_ID:-}" ]; then
        ERR_TAIL=$(tail -c 500 "$OUT_LOG" 2>/dev/null || true)
        curl -fsS -X POST \
            "https://api.telegram.org/bot${TRADE_TELEGRAM_TOKEN}/sendMessage" \
            -d "chat_id=${TRADE_TELEGRAM_CHAT_ID}" \
            -d "parse_mode=HTML" \
            --data-urlencode "$(printf 'text=⚠️ <b>OUTCOMES-RECORD FAILED</b>\nexit=%s\n<pre>%s</pre>\nML feedback loop at risk. Trade will still run.' "$OUT_EXIT" "$ERR_TAIL")" \
            >/dev/null 2>&1 || true
    fi
fi
rm -f "$OUT_LOG"

# Notify Telegram that scan landed + how many candidates we'll evaluate
# (gives the user context before the trade evaluator runs).
SCAN_REGIME=$($PY -c "
import pandas as pd
try:
    df = pd.read_parquet('$SCAN_FILE')
    print(df['Market_Regime'].mode().iloc[0] if 'Market_Regime' in df.columns and len(df) else 'unknown')
except Exception:
    print('unknown')
" 2>/dev/null || echo "unknown")
SCAN_TOP=$($PY -c "
import pandas as pd
try:
    df = pd.read_parquet('$SCAN_FILE')
    print(f\"{len(df)} rows, top score {df['FinalScore_20d'].max():.1f}\" if len(df) else '0 rows')
except Exception:
    print('?')
" 2>/dev/null || echo "?")
if [ -n "${TRADE_TELEGRAM_TOKEN:-}" ] && [ -n "${TRADE_TELEGRAM_CHAT_ID:-}" ]; then
    curl -fsS -X POST \
        "https://api.telegram.org/bot${TRADE_TELEGRAM_TOKEN}/sendMessage" \
        -d "chat_id=${TRADE_TELEGRAM_CHAT_ID}" \
        -d "parse_mode=HTML" \
        --data-urlencode "$(printf 'text=📊 <b>Scan landed</b>\nRegime: %s\n%s\nEvaluating candidates...' "$SCAN_REGIME" "$SCAN_TOP")" \
        >/dev/null 2>&1 || true
fi

# ── PRE-FLIGHT IB CONNECTIVITY CHECK (added 2026-05-05) ──
# healthcheck.timer runs every 15 min; if the IB session dies between its
# last check and this pipeline fire, the trade evaluator wastes ~30s
# setting up before failing with cryptic IB errors that only sometimes
# match the FATAL grep below. Catch it here, attempt one auto-restart,
# and abort cleanly with a Telegram alert if recovery fails. This closes
# the worst-case 15-min window between healthcheck cycles.
echo "Pre-flight IB connectivity check..."
_ib_check_handshake() {
    $PY -c "
from ib_insync import IB
import random
ib = IB()
try:
    ib.connect('127.0.0.1', 7496, clientId=200+random.randint(1,500), timeout=10)
    print('OK' if ib.isConnected() else 'NOCONN')
    ib.disconnect()
except Exception as e:
    print(f'ERR:{type(e).__name__}')
" 2>/dev/null
}
# Layer 1 — TCP port. Fast (sub-second).
if ! timeout 5 nc -z 127.0.0.1 7496 2>/dev/null; then
    echo "  ⚠ TCP port 7496 not responding — restarting ibgateway container..."
    TG_SEND "⚠️" "Pipeline pre-flight: IB port DOWN" "Auto-restarting ibgateway container..."
    docker restart ibgateway >/dev/null 2>&1 || true
    sleep 45
    if ! timeout 5 nc -z 127.0.0.1 7496 2>/dev/null; then
        TG_SEND "🚨" "PIPELINE ABORTED — IB unreachable" "Port 7496 still down after auto-restart. Manual: <a href=\"http://87.99.142.12:5800/vnc.html\">VNC</a>"
        echo "FATAL: IB port still down after restart — aborting"
        exit 6
    fi
    TG_SEND "✅" "IB Gateway recovered (port)" "Auto-restart succeeded — pipeline continuing"
    echo "  ✓ TCP recovered"
fi
# Layer 2 — actual ib_insync handshake. TCP open ≠ session authenticated.
PRE_HANDSHAKE=$(_ib_check_handshake)
if [ "$PRE_HANDSHAKE" != "OK" ]; then
    echo "  ⚠ IB handshake failed ($PRE_HANDSHAKE) — restarting ibgateway container..."
    TG_SEND "⚠️" "Pipeline pre-flight: IB session DOWN" "Handshake: ${PRE_HANDSHAKE}. Auto-restarting..."
    docker restart ibgateway >/dev/null 2>&1 || true
    sleep 45
    PRE_HANDSHAKE2=$(_ib_check_handshake)
    if [ "$PRE_HANDSHAKE2" != "OK" ]; then
        TG_SEND "🚨" "PIPELINE ABORTED — IB session dead" "Handshake: ${PRE_HANDSHAKE2}. Likely needs IB Key 2FA approval. <a href=\"http://87.99.142.12:5800/vnc.html\">VNC</a>"
        echo "FATAL: handshake still failing — aborting"
        exit 6
    fi
    TG_SEND "✅" "IB Gateway recovered (session)" "Auto-restart succeeded — pipeline continuing"
    echo "  ✓ Handshake recovered"
else
    echo "  ✓ IB connectivity OK"
fi

# CAPACITY PRE-CHECK (added 2026-05-07).
# If we're already at max_open_positions, the trade evaluator will run
# 30-60s of price refresh + risk gates only to reject every candidate
# with "Max open positions reached". Skip the eval entirely in that
# case — saves API calls (yfinance rate-limit budget) and ~5 min per
# pipeline. We still need a position-close to free a slot; that path
# is handled by the monitor's _try_opportunistic_buy on the LATEST
# scan, not by this pipeline's eval.
#
# Stays conservative: only skips when current open count strictly
# matches the cap. If something changed mid-pipeline (e.g., a stop
# fired during the scan poll), we still run the eval — better safe.
CAPACITY_SKIP=0
if [ -f "$ROOT/data/trades/open_positions.json" ]; then
    OPEN_COUNT=$($PY -c "
import json, os
try:
    with open('$ROOT/data/trades/open_positions.json') as f:
        print(len(json.load(f)))
except Exception:
    print(-1)
" 2>/dev/null || echo "-1")
    MAX_OPEN=${TRADE_MAX_OPEN_POSITIONS:-2}
    if [ "$OPEN_COUNT" -ge "$MAX_OPEN" ] 2>/dev/null; then
        echo "Pre-eval: already at $OPEN_COUNT/$MAX_OPEN open — skipping trade evaluator (would reject all)"
        TG_SEND "⏭" "Scan: capacity full ($OPEN_COUNT/$MAX_OPEN)" "Trade eval skipped — no slot for new positions until one closes."
        CAPACITY_SKIP=1
    fi
fi

if [ "$CAPACITY_SKIP" -eq 0 ]; then
# Fire the trade evaluator. Live price refresh + 8 risk gates inside.
# TRADE_LIVE_CONFIRMED=1 is the systemd-pipeline authorization to run live;
# manual ssh runs (without this env var and without --live flag) default to
# DRY mode for safety. See run_auto_trade.py for the full policy.
echo "Triggering auto-trade..."
TRADE_T0=$(date +%s)
TRADE_OUT=/tmp/trade-output-$$.log
TRADE_LIVE_CONFIRMED=1 $PY -m scripts.run_auto_trade > "$TRADE_OUT" 2>&1 || true
TRADE_EXIT=$?
tail -25 "$TRADE_OUT"

# ── Telegram diagnostic alerts (rewritten 2026-05-05) ──
# Yesterday's "0 buys" went silent because (a) the trade aborted on
# stale-scan BEFORE entering filters, so "No candidates passed filters"
# never appeared in the log, and (b) the SKIP grep pattern `^\s*SKIP `
# only matched lines starting with "SKIP", but the real log format is
# `2026-05-04 14:13:57,560 [INFO] core.trading.order_manager: SKIP ...`
# (timestamp prefix). The user got NO Telegram alert about the failure.
#
# New logic — three exclusive paths in priority order:
#   1. Stale-scan / FATAL / TIMEOUT abort  →  loud 🚨 alert
#   2. Filter chain returned 0 candidates  →  ⏭ alert with REAL skip reasons
#   3. Trade ran but bought 0 (all gates rejected at exec time)  →  ⏭ alert
# (TG_SEND is defined near the top of the file so the pre-flight IB check
# can also use it.)

# Path 1: hard abort (stale scan, FATAL, no scan results, IBKR fail)
if grep -qE "stale data|No scan results available|FATAL|Failed to connect to IBKR" "$TRADE_OUT"; then

    # ── RETRY-ON-STALE (added 2026-05-05) ──
    # If the abort was scan-staleness, the most likely cause is a race:
    # GH Actions just committed a new parquet but the local touch+pull
    # didn't propagate fast enough. Wait 60s, refresh the local files,
    # and retry the trade ONCE before alerting. Catches the common case
    # where the next scheduled fire (3-4h away) would otherwise be the
    # earliest retry.
    if grep -qE "stale data|No scan results available" "$TRADE_OUT"; then
        echo "STALE-SCAN abort detected — refreshing scan + retrying once after 60s..."
        sleep 60
        git fetch origin main --quiet 2>/dev/null || true
        git checkout origin/main -- data/scans/latest_scan.parquet data/scans/latest_scan.json data/scans/latest_scan.meta.json 2>/dev/null || true
        for f in data/scans/latest_scan.parquet data/scans/latest_scan.json data/scans/latest_scan.meta.json; do
            [ -f "$f" ] && touch "$f"
        done
        echo "Retrying auto-trade..."
        TRADE_RETRY_OUT=/tmp/trade-retry-$$.log
        TRADE_LIVE_CONFIRMED=1 $PY -m scripts.run_auto_trade > "$TRADE_RETRY_OUT" 2>&1 || true
        tail -25 "$TRADE_RETRY_OUT"
        if ! grep -qE "stale data|No scan results available|FATAL" "$TRADE_RETRY_OUT"; then
            echo "✓ Retry succeeded — replacing original output"
            mv "$TRADE_RETRY_OUT" "$TRADE_OUT"
            # Re-run the path detection on the retry's output
            if ! grep -qE "stale data|No scan results available|FATAL|Failed to connect to IBKR" "$TRADE_OUT"; then
                # Retry was clean — fall through to the elif paths below
                if grep -q "No candidates passed filters" "$TRADE_OUT"; then
                    FILTER_SUMMARY=$(grep -E "filter dropped|No stocks pass|Smart filter|MARKET REGIME BLOCK" "$TRADE_OUT" | tail -8 | sed 's/^.*\] //g' || true)
                    TG_SEND "⏭" "Scan: 0 candidates (after retry)" "<pre>${FILTER_SUMMARY:-Filter chain produced 0}</pre>"
                fi
                rm -f "$TRADE_OUT"
                TRADE_DUR=$(( $(date +%s) - TRADE_T0 ))
                echo "Trade finished after retry (duration=${TRADE_DUR}s)"
                echo "═══════════════════════════════════════════════════════"
                echo "Pipeline complete WITH retry"
                echo "═══════════════════════════════════════════════════════"
                exit 0
            fi
        else
            rm -f "$TRADE_RETRY_OUT"
        fi
    fi

    ABORT_REASON=$(grep -E "stale data|No scan results available|FATAL|Failed to connect" "$TRADE_OUT" | head -3 | sed 's/^.*\] //g' || true)
    TG_SEND "🚨" "PIPELINE ABORTED" "<pre>${ABORT_REASON:-(no detail)}</pre>"

# Path 2: filter chain dropped everything (Score/RR/ML/Confidence/etc)
elif grep -q "No candidates passed filters" "$TRADE_OUT"; then
    # Capture the filter-stage breakdown that order_manager logs.
    # Pull the LAST line of each "filter dropped N stocks" + the regime line.
    FILTER_SUMMARY=$(grep -E "filter dropped|No stocks pass|Smart filter|MARKET REGIME BLOCK" "$TRADE_OUT" | tail -8 | sed 's/^.*\] //g' || true)
    TG_SEND "⏭" "Scan: 0 candidates after filters" "<pre>${FILTER_SUMMARY:-Filter chain produced 0}</pre>"

# Path 3: candidates entered execution but all rejected by risk gates
elif grep -qE "core\.trading\.order_manager: SKIP " "$TRADE_OUT" && \
     ! grep -qE "BUY filled|notify_buy" "$TRADE_OUT"; then
    # Real SKIP log lines (from order_manager._execute_single)
    SKIP_REASONS=$(grep -E "core\.trading\.order_manager: SKIP " "$TRADE_OUT" \
        | head -5 \
        | sed -E 's/^.*SKIP /SKIP /' \
        | sed 's/SKIP \([A-Z]*\): /\1: /' || true)
    TG_SEND "⏭" "Scan: 0 buys (all rejected by risk gates)" "<pre>${SKIP_REASONS:-(no detail)}</pre>"
fi
rm -f "$TRADE_OUT"
TRADE_DUR=$(( $(date +%s) - TRADE_T0 ))
echo "Trade finished (exit=$TRADE_EXIT, duration=${TRADE_DUR}s)"
fi  # end CAPACITY_SKIP guard

echo "═══════════════════════════════════════════════════════"
echo "Pipeline complete: wait=${WAIT_DUR}s${CAPACITY_SKIP:+ (capacity-skip)}"
echo "═══════════════════════════════════════════════════════"
exit 0
