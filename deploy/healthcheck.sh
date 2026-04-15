#!/bin/bash
# ============================================================
# StockScout IB Gateway Healthcheck
#
# Runs every 15 min via systemd timer during market hours.
# Checks IB Gateway, API port, and monitor daemon.
# Sends Telegram alert + auto-restart on failure.
#
# Pure bash — works even if Python environment is broken.
# ============================================================

TELEGRAM_TOKEN="${TRADE_TELEGRAM_TOKEN}"
CHAT_ID="${TRADE_TELEGRAM_CHAT_ID}"
IB_PORT=7496

send_alert() {
    local msg="$1"
    if [ -n "${TELEGRAM_TOKEN}" ] && [ -n "${CHAT_ID}" ]; then
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
            -d chat_id="${CHAT_ID}" \
            -d text="$(echo -e "\xe2\x9a\xa0\xef\xb8\x8f") StockScout VPS: ${msg}" \
            > /dev/null 2>&1
    fi
    echo "[ALERT] ${msg}"
}

# Skip outside market hours (Mon-Fri 13:30-20:00 UTC / 9:30 AM - 4:00 PM ET)
DOW=$(date -u +%u)  # 1=Mon, 7=Sun
HOUR=$(date -u +%H)
if [ "$DOW" -gt 5 ] || [ "$HOUR" -lt 13 ] || [ "$HOUR" -gt 20 ]; then
    echo "Outside market hours — skipping healthcheck"
    exit 0
fi

ISSUES=0

# 1. Check IB Gateway service
if ! systemctl is-active --quiet ibgateway; then
    send_alert "IB Gateway is DOWN — attempting restart"
    sudo systemctl restart ibgateway
    sleep 30
    if systemctl is-active --quiet ibgateway; then
        send_alert "IB Gateway restarted successfully"
    else
        send_alert "CRITICAL: IB Gateway restart FAILED — manual intervention needed"
        ISSUES=$((ISSUES + 1))
    fi
fi

# 2. Check API port connectivity
if ! nc -z 127.0.0.1 ${IB_PORT} 2>/dev/null; then
    send_alert "IB Gateway port ${IB_PORT} not responding (service may be starting)"
    ISSUES=$((ISSUES + 1))
fi

# 3. Check monitor daemon
if ! systemctl is-active --quiet stockscout-monitor; then
    send_alert "Monitor daemon is DOWN — restarting"
    sudo systemctl restart stockscout-monitor
    sleep 5
    if systemctl is-active --quiet stockscout-monitor; then
        echo "Monitor daemon restarted successfully"
    else
        send_alert "Monitor daemon restart FAILED"
        ISSUES=$((ISSUES + 1))
    fi
fi

if [ "$ISSUES" -eq 0 ]; then
    echo "Healthcheck OK — all services running"
fi

exit 0
