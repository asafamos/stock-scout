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
    send_alert "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') IB Gateway Docker container is DOWN — attempting restart"
    docker start ibgateway 2>/dev/null || docker restart ibgateway 2>/dev/null
    sleep 30
    if docker ps --format '{{.Names}}' | grep -q ibgateway; then
        send_alert "$(echo -e '\xe2\x9c\x85') IB Gateway container restarted"
    else
        send_alert "$(echo -e '\xf0\x9f\x9a\xa8') CRITICAL: IB Gateway restart FAILED"
        ISSUES=$((ISSUES + 1))
    fi
fi

# 2. Check API port connectivity
if ! nc -z 127.0.0.1 ${IB_PORT} 2>/dev/null; then
    send_alert "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') IB Gateway port ${IB_PORT} not responding — may need 2FA approval

Open: http://87.99.142.12:5800/vnc.html"
    ISSUES=$((ISSUES + 1))
fi

# 3. Check monitor daemon
if ! systemctl is-active --quiet stockscout-monitor; then
    send_alert "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') Monitor daemon is DOWN — restarting"
    sudo systemctl restart stockscout-monitor
    sleep 5
    if systemctl is-active --quiet stockscout-monitor; then
        echo "Monitor daemon restarted"
    else
        send_alert "$(echo -e '\xf0\x9f\x9a\xa8') Monitor daemon restart FAILED"
        ISSUES=$((ISSUES + 1))
    fi
fi

if [ "$ISSUES" -eq 0 ]; then
    echo "Healthcheck OK — all services running"
fi

exit 0
