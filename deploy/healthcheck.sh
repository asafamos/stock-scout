#!/bin/bash
# ============================================================
# StockScout Healthcheck + Weekly IB Key Reminder
#
# Runs every 15 min via systemd timer during market hours.
# Checks Docker Gateway, API port, monitor daemon.
# Sends weekly reminder to approve IB Key.
# ============================================================

TELEGRAM_TOKEN="${TRADE_TELEGRAM_TOKEN}"
CHAT_ID="${TRADE_TELEGRAM_CHAT_ID}"
IB_PORT=7496

# State dir for alert de-duplication. Files here act as "we already warned you
# about this recently" markers so we don't spam Telegram every 15 min while the
# same issue persists.
STATE_DIR="/var/tmp/stockscout-healthcheck"
mkdir -p "${STATE_DIR}" 2>/dev/null

send_alert() {
    local msg="$1"
    if [ -n "${TELEGRAM_TOKEN}" ] && [ -n "${CHAT_ID}" ]; then
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
            -d chat_id="${CHAT_ID}" \
            -d text="${msg}" \
            -d parse_mode="HTML" \
            > /dev/null 2>&1
    fi
    echo "[ALERT] ${msg}"
}

# Send an alert only if we haven't already sent one of this kind in the last
# ${2:-7200} seconds (default 2h). Key must be a short filename-safe tag.
send_alert_dedup() {
    local key="$1"
    local msg="$2"
    local cooldown="${3:-7200}"
    local marker="${STATE_DIR}/alert_${key}"
    if [ -f "${marker}" ]; then
        local age=$(( $(date +%s) - $(stat -c %Y "${marker}" 2>/dev/null || echo 0) ))
        if [ "${age}" -lt "${cooldown}" ]; then
            echo "[SKIP-DEDUP] ${key} (last sent ${age}s ago < ${cooldown}s)"
            return 0
        fi
    fi
    send_alert "${msg}"
    touch "${marker}"
}

# Clear a dedup marker once the underlying issue recovers, so the next failure
# triggers a fresh alert instead of being silently suppressed.
clear_alert_dedup() {
    rm -f "${STATE_DIR}/alert_$1" 2>/dev/null
}

# Actual ib_insync handshake. Returns "OK" or "ERR:<name>" / "NOCONN".
# Use a RANDOM clientId per run (200-899): if a previous healthcheck left a
# zombie session on a fixed clientId, a fresh fixed id would get rejected with
# "client id already in use" → false handshake-failure alert. Randomizing
# sidesteps the collision since IB allows many concurrent client IDs.
# Range avoids known ids: 1 (order_manager), 10 (reapply), 93-99 (old
# healthcheck + statusbot + daily_summary).
run_handshake_check() {
    local cid=$(( (RANDOM % 700) + 200 ))
    HC_CID="${cid}" runuser -u stockscout --preserve-environment -- bash -c '/home/stockscout/stock-scout-2/.venv/bin/python -c "
import os
from ib_insync import IB
ib = IB()
try:
    ib.connect(\"127.0.0.1\", 7496, clientId=int(os.environ[\"HC_CID\"]), timeout=10)
    if ib.isConnected():
        print(\"OK\")
    else:
        print(\"NOCONN\")
    ib.disconnect()
except Exception as e:
    print(f\"ERR: {type(e).__name__}\")
"' 2>/dev/null
}

# Skip outside market hours (Mon-Fri 13:30-20:30 UTC / 9:30 AM - 4:30 PM ET)
DOW=$(date -u +%u)  # 1=Mon, 7=Sun
HOUR=$(date -u +%H)
if [ "$DOW" -gt 5 ] || [ "$HOUR" -lt 13 ] || [ "$HOUR" -gt 20 ]; then
    # Weekly IB Key reminder — Sunday 15:00 UTC (before market opens Monday)
    if [ "$DOW" -eq 7 ] && [ "$HOUR" -eq 15 ]; then
        send_alert "$(echo -e '\xF0\x9F\x94\x91') <b>Weekly IB Key Reminder</b>

IB Gateway needs re-authentication before market opens Monday.

1. Open: http://87.99.142.12:5800/vnc.html
2. Click Connect
3. Select IB Key in 2FA dialog
4. Approve on IBKR Mobile app

Takes 10 seconds!"
    fi
    echo "Outside market hours — skipping healthcheck"
    exit 0
fi

ISSUES=0

# 1. Check Docker IB Gateway container
if ! docker ps --format '{{.Names}}' | grep -q ibgateway; then
    send_alert_dedup "container_down" "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') IB Gateway Docker container is DOWN — attempting restart"
    docker start ibgateway 2>/dev/null || docker restart ibgateway 2>/dev/null
    sleep 30
    if docker ps --format '{{.Names}}' | grep -q ibgateway; then
        send_alert "$(echo -e '\xe2\x9c\x85') IB Gateway container restarted"
        clear_alert_dedup "container_down"
    else
        send_alert_dedup "container_restart_failed" "$(echo -e '\xf0\x9f\x9a\xa8') CRITICAL: IB Gateway restart FAILED"
        ISSUES=$((ISSUES + 1))
    fi
fi

# 2. Check API port connectivity (TCP-level only)
if ! nc -z 127.0.0.1 ${IB_PORT} 2>/dev/null; then
    send_alert_dedup "port_down" "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') IB Gateway port ${IB_PORT} not responding — may need 2FA approval

Open: http://87.99.142.12:5800/vnc.html"
    ISSUES=$((ISSUES + 1))
else
    clear_alert_dedup "port_down"
    # 2b. DEEP API CHECK — TCP open doesn't prove IB is authenticated.
    # The session can die while the container stays up (happens daily).
    # Try an actual ib_insync handshake; timeout fast (10s).
    API_CHECK=$(run_handshake_check)
    if [ "$API_CHECK" != "OK" ]; then
        # SELF-HEAL: session died but container is up. Most of the time a
        # `docker restart ibgateway` brings it back — the Gateway's autologin
        # then triggers a fresh 2FA push to IBKR Mobile, which can be approved
        # from the phone. Only alert if the restart fails to recover.
        echo "[AUTO-HEAL] handshake failed (${API_CHECK}) — restarting ibgateway container"
        docker restart ibgateway >/dev/null 2>&1
        # IB Gateway is slow to boot + autologin; give it enough time.
        sleep 45
        API_CHECK2=$(run_handshake_check)
        if [ "$API_CHECK2" = "OK" ]; then
            send_alert "$(echo -e '\xe2\x9c\x85') IB Gateway auto-recovered after handshake failure (was: ${API_CHECK})"
            clear_alert_dedup "handshake_failed"
        else
            send_alert_dedup "handshake_failed" "$(echo -e '\xf0\x9f\x9a\xa8') IB API handshake FAILED (${API_CHECK2}) — auto-restart did not recover.

Session needs IB Key re-approval. From phone: open IBKR Mobile and approve the pending push (container was just restarted, push should be waiting).

If no push arrived: ssh root@87.99.142.12 'docker restart ibgateway' to re-trigger autologin."
            ISSUES=$((ISSUES + 1))
        fi
    else
        clear_alert_dedup "handshake_failed"
    fi
fi

# 3. Check monitor daemon
if ! systemctl is-active --quiet stockscout-monitor; then
    send_alert_dedup "monitor_down" "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') Monitor daemon is DOWN — restarting"
    sudo systemctl restart stockscout-monitor
    sleep 5
    if systemctl is-active --quiet stockscout-monitor; then
        echo "Monitor daemon restarted"
        clear_alert_dedup "monitor_down"
    else
        send_alert_dedup "monitor_restart_failed" "$(echo -e '\xf0\x9f\x9a\xa8') Monitor daemon restart FAILED"
        ISSUES=$((ISSUES + 1))
    fi
else
    clear_alert_dedup "monitor_down"
    clear_alert_dedup "monitor_restart_failed"
fi

# 4. Check status bot daemon (handles /panic, /pnl, /status, /history)
if ! systemctl is-active --quiet stockscout-statusbot; then
    send_alert_dedup "statusbot_down" "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') Status bot is DOWN — restarting"
    sudo systemctl restart stockscout-statusbot
    sleep 3
    if systemctl is-active --quiet stockscout-statusbot; then
        echo "Status bot restarted"
        clear_alert_dedup "statusbot_down"
    else
        send_alert_dedup "statusbot_restart_failed" "$(echo -e '\xf0\x9f\x9a\xa8') Status bot restart FAILED — /panic unavailable"
        ISSUES=$((ISSUES + 1))
    fi
else
    clear_alert_dedup "statusbot_down"
    clear_alert_dedup "statusbot_restart_failed"
fi

# 5. Detect monitor stuck in restart loop (>3 restarts in last 5 min).
# Trim whitespace so the integer comparison doesn't choke on journalctl output.
MONITOR_RESTARTS=$(journalctl -u stockscout-monitor --since '5 min ago' --no-pager 2>/dev/null \
    | grep -c 'Started StockScout Position Monitor')
MONITOR_RESTARTS=${MONITOR_RESTARTS//[^0-9]/}
MONITOR_RESTARTS=${MONITOR_RESTARTS:-0}
if [ "$MONITOR_RESTARTS" -gt 3 ]; then
    send_alert_dedup "monitor_restart_loop" "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') Monitor restarted ${MONITOR_RESTARTS}x in 5 min — check logs"
    ISSUES=$((ISSUES + 1))
fi

# 6. Orphan IB position drift — only report if we can prove the state.
# Runs as stockscout (no sudo needed since healthcheck runs as root, we use
# runuser to drop privileges). If the IB query fails or returns no answer,
# we SKIP the drift check entirely instead of false-alarming on every
# tracker ticker (which happened when `sudo -u` failed for not-in-sudoers).
TRACKER="/home/stockscout/stock-scout-2/data/trades/open_positions.json"
if [ -f "${TRACKER}" ]; then
    # Random clientId (see run_handshake_check for rationale).
    DRIFT_CID=$(( (RANDOM % 700) + 200 ))
    export DRIFT_CID
    IB_QUERY=$(runuser -u stockscout --preserve-environment -- bash -c 'cd /home/stockscout/stock-scout-2 && set -a && source .env.trading 2>/dev/null && set +a && /home/stockscout/stock-scout-2/.venv/bin/python -c "
import os
from ib_insync import IB
try:
    ib = IB(); ib.connect(\"127.0.0.1\", 7496, clientId=int(os.environ[\"DRIFT_CID\"]), timeout=10)
    syms = sorted({p.contract.symbol for p in ib.positions() if p.position != 0})
    ib.disconnect()
    print(\"OK\", \" \".join(syms))
except Exception as e:
    print(\"ERR\", e)
"' 2>/dev/null)
    STATUS=$(echo "${IB_QUERY}" | awk '{print $1}')
    if [ "$STATUS" = "OK" ]; then
        IB_TICKERS=$(echo "${IB_QUERY}" | cut -d' ' -f2-)
        TRACKER_TICKERS=$(python3 -c "
import json
try:
    with open('${TRACKER}') as f:
        syms = sorted({p['ticker'] for p in json.load(f)})
    print(' '.join(syms))
except Exception:
    print('')
" 2>/dev/null)
        for sym in ${IB_TICKERS}; do
            if [ -n "${sym}" ] && ! echo " ${TRACKER_TICKERS} " | grep -q " ${sym} "; then
                send_alert "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') DRIFT: IB holds <b>${sym}</b> but tracker doesn't know"
                ISSUES=$((ISSUES + 1))
            fi
        done
        for sym in ${TRACKER_TICKERS}; do
            if [ -n "${sym}" ] && ! echo " ${IB_TICKERS} " | grep -q " ${sym} "; then
                send_alert "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') DRIFT: tracker holds <b>${sym}</b> but IB doesn't"
                ISSUES=$((ISSUES + 1))
            fi
        done
    else
        # Could not query IB (connection refused, no env vars, etc.)
        # Silent skip — this is NOT a drift, just an unreachable gateway.
        echo "Drift check skipped: IB query failed (${IB_QUERY})"
    fi
fi

# 7. Scan outcomes tracker health — alert if recording has stalled.
# The pending_scans.jsonl file should grow by ~50 rows per trading day.
# If no new entries in 2+ weekdays, something is wrong with the record cron.
PENDING="/home/stockscout/stock-scout-2/data/outcomes/pending_scans.jsonl"
if [ -f "${PENDING}" ]; then
    # Most-recent scan_date in the file
    LAST_SCAN=$(tail -100 "${PENDING}" 2>/dev/null | grep -oE '"scan_date": "[0-9-]+"' | tail -1 \
        | sed 's/.*"\([0-9-]*\)".*/\1/')
    if [ -n "${LAST_SCAN}" ]; then
        AGE_DAYS=$(( ($(date +%s) - $(date -d "${LAST_SCAN}" +%s 2>/dev/null || echo 0)) / 86400 ))
        if [ "${AGE_DAYS}" -gt 3 ]; then
            send_alert "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') Scan outcomes tracker STALE — last record ${LAST_SCAN} (${AGE_DAYS} days ago). ML feedback loop broken."
            ISSUES=$((ISSUES + 1))
        fi
    fi
fi

if [ "$ISSUES" -eq 0 ]; then
    echo "Healthcheck OK — all services running"
fi

exit 0
