#!/bin/bash
# ============================================================
# install_telegram_bot.sh
#
# One-shot installer for the merged Telegram status bot
# (telegram_status_bot.py now includes the IB Key 2FA watchdog).
#
# Run this ONCE on the VPS (87.99.142.12, as the stockscout user
# with sudo) to:
#   1. Stop any old telegram bot processes (the two-poller bug)
#   2. Disable a stale ibkey systemd unit if one exists
#   3. Install/refresh the stockscout-telegram-bot.service unit
#   4. Pull the latest code on whatever branch is checked out
#   5. Start + enable the new service
#
# Usage (from iPhone Termius/Blink, after the new code is on main
# or the deploy branch):
#
#     ssh stockscout@87.99.142.12
#     cd ~/stock-scout-2
#     git pull
#     sudo bash deploy/install_telegram_bot.sh
#
# Idempotent — safe to re-run.
# ============================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/stockscout/stock-scout-2}"
VENV_PY="${REPO_DIR}/.venv/bin/python"
ENV_FILE="${REPO_DIR}/.env.trading"
LOG_DIR="${REPO_DIR}/logs"
SERVICE_NAME="stockscout-telegram-bot"
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

echo "==> 1/5  Stopping any running telegram bot processes..."
# pkill returns 1 if nothing matched — don't let it abort the script.
pkill -f "scripts.telegram_status_bot"  || true
pkill -f "scripts.ibkey_telegram_bot"   || true
# Disable a stale ibkey systemd unit if one was ever installed.
if systemctl list-unit-files | grep -q "stockscout-ibkey"; then
    echo "    found stale stockscout-ibkey unit — disabling"
    sudo systemctl disable --now stockscout-ibkey-bot.service 2>/dev/null || true
    sudo systemctl disable --now stockscout-ibkey.service     2>/dev/null || true
fi

echo "==> 2/5  Verifying prerequisites..."
[ -x "$VENV_PY" ]   || { echo "ERROR: venv missing at $VENV_PY"; exit 1; }
[ -f "$ENV_FILE" ]  || { echo "ERROR: env file missing at $ENV_FILE"; exit 1; }
mkdir -p "$LOG_DIR"

echo "==> 3/5  Writing systemd unit to $UNIT_PATH ..."
sudo tee "$UNIT_PATH" > /dev/null << SVCEOF
[Unit]
Description=StockScout Telegram status bot (+ IB Key 2FA watchdog)
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
User=stockscout
Group=stockscout
WorkingDirectory=${REPO_DIR}
EnvironmentFile=${ENV_FILE}
# Enable the IB Gateway 2FA watchdog (the bot auto-skips it if the
# docker container is not present, but the env var makes intent explicit).
Environment="TRADE_IBKEY_WATCHDOG=1"
ExecStart=${VENV_PY} -m scripts.telegram_status_bot
Restart=always
RestartSec=10
KillMode=process
StandardOutput=append:${LOG_DIR}/telegram_bot.log
StandardError=append:${LOG_DIR}/telegram_bot.log

[Install]
WantedBy=multi-user.target
SVCEOF

echo "==> 4/5  Reloading systemd + enabling unit..."
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}.service"
sudo systemctl restart "${SERVICE_NAME}.service"

sleep 3

echo "==> 5/5  Status:"
sudo systemctl status "${SERVICE_NAME}.service" --no-pager --lines=12 || true
echo
echo "Tail logs with:"
echo "    journalctl -u ${SERVICE_NAME} -f"
echo "    tail -f ${LOG_DIR}/telegram_bot.log"
echo
echo "Verify the bot is alive by sending 'status' to the Telegram chat."
