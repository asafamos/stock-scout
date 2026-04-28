#!/bin/bash
# ============================================================
# StockScout VPS Setup — IB Gateway + Auto-Trading
#
# Tested on Ubuntu 22.04 (Hetzner CX22, $4.35/month)
#
# ============================================================
# HETZNER SETUP GUIDE (from zero to first live trade)
# ============================================================
#
# 1. CREATE SERVER
#    - Go to console.hetzner.cloud → New Project → "StockScout"
#    - Add Server → Location: Ashburn (US East, low latency to IB)
#    - Image: Ubuntu 22.04 → Type: CX22 ($4.35/mo, 2 vCPU, 4GB RAM)
#    - SSH Key: paste your public key (~/.ssh/id_ed25519.pub)
#    - Create & Buy
#
# 2. INITIAL SSH
#    ssh root@<IP>
#    apt update && apt upgrade -y && reboot
#    ssh root@<IP>
#
# 3. UPLOAD AND RUN THIS SCRIPT
#    scp deploy/setup_vps.sh root@<IP>:~/
#    ssh root@<IP> 'chmod +x setup_vps.sh && ./setup_vps.sh'
#
# 4. CONFIGURE CREDENTIALS (as stockscout user)
#    su - stockscout
#    nano ~/stock-scout-2/.env.trading
#    → Fill in IBKR_USERNAME, IBKR_PASSWORD, TELEGRAM_TOKEN, CHAT_ID
#    sudo nano /opt/ibc/config.ini
#    → Set IbLoginId and IbPassword
#
# 5. INSTALL IB GATEWAY (interactive)
#    sudo bash /tmp/ibgateway-install.sh
#    → Accept defaults, install to /home/stockscout/Jts
#
# 6. START SERVICES
#    sudo systemctl enable --now xvfb ibgateway
#    sleep 30
#    journalctl -u ibgateway -f   # verify "connected"
#
# 7. TEST DRY RUN
#    cd ~/stock-scout-2 && source .venv/bin/activate
#    TRADE_DRY_RUN=1 python -m scripts.run_auto_trade
#
# 8. TEST LIVE (manual, single run)
#    TRADE_AUTO_CONFIRM=1 TRADE_DRY_RUN=0 TRADE_PAPER_MODE=0 \
#      python -m scripts.run_auto_trade
#
# 9. ENABLE AUTOMATION
#    sudo systemctl enable --now stockscout-pipeline.timer
#    sudo systemctl enable --now stockscout-monitor
#    sudo systemctl enable --now stockscout-healthcheck.timer
#    systemctl list-timers
#
# 10. MONITORING
#    journalctl -u stockscout-pipeline --since today
#    journalctl -u stockscout-monitor -f
#    journalctl -u ibgateway -f
#    systemctl list-timers
# ============================================================

set -e

SCOUT_USER="stockscout"
SCOUT_HOME="/home/${SCOUT_USER}"
PROJECT_DIR="${SCOUT_HOME}/stock-scout-2"

echo "=========================================="
echo " StockScout VPS Setup"
echo "=========================================="

# ── System packages ──────────────────────────────────────────
echo ""
echo "[1/7] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3.11 python3.11-venv python3-pip \
    git xvfb unzip wget curl netcat-openbsd \
    openjdk-17-jre-headless

# ── Dedicated user ───────────────────────────────────────────
echo ""
echo "[2/7] Creating stockscout user..."
if ! id "${SCOUT_USER}" &>/dev/null; then
    sudo useradd -m -s /bin/bash "${SCOUT_USER}"
    echo "User ${SCOUT_USER} created"
else
    echo "User ${SCOUT_USER} already exists"
fi

# Copy SSH keys so you can ssh directly as stockscout
if [ -f ~/.ssh/authorized_keys ]; then
    sudo mkdir -p "${SCOUT_HOME}/.ssh"
    sudo cp ~/.ssh/authorized_keys "${SCOUT_HOME}/.ssh/"
    sudo chown -R "${SCOUT_USER}:${SCOUT_USER}" "${SCOUT_HOME}/.ssh"
    sudo chmod 700 "${SCOUT_HOME}/.ssh"
    sudo chmod 600 "${SCOUT_HOME}/.ssh/authorized_keys"
fi

# ── IB Gateway download ─────────────────────────────────────
echo ""
echo "[3/7] Downloading IB Gateway..."
IB_GATEWAY_URL="https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh"
wget -q -O /tmp/ibgateway-install.sh "$IB_GATEWAY_URL" 2>/dev/null || {
    echo "WARNING: Could not download IB Gateway automatically."
    echo "Download manually from: https://www.interactivebrokers.com/en/trading/ibgateway-stable.php"
}

if [ -f /tmp/ibgateway-install.sh ]; then
    chmod +x /tmp/ibgateway-install.sh
    echo "IB Gateway installer ready at /tmp/ibgateway-install.sh"
    echo "Run: sudo bash /tmp/ibgateway-install.sh"
fi

# ── IBC (auto-login controller) ─────────────────────────────
echo ""
echo "[4/7] Installing IBC..."
IBC_VERSION="3.18.0"
wget -q -O /tmp/ibc.zip \
    "https://github.com/IbcAlpha/IBC/releases/download/${IBC_VERSION}/IBCLinux-${IBC_VERSION}.zip"
sudo mkdir -p /opt/ibc
sudo unzip -qo /tmp/ibc.zip -d /opt/ibc
sudo chmod +x /opt/ibc/*.sh

# IBC config template
sudo tee /opt/ibc/config.ini > /dev/null << 'IBCEOF'
# IBC Configuration for StockScout
# Fill in IbLoginId and IbPassword before starting

LogToConsole=yes
FIX=no

IbLoginId=
IbPassword=

TradingMode=live
IbDir=/home/stockscout/Jts

AcceptIncomingConnectionAction=accept
AcceptNonBrokerageAccountWarning=yes
ExistingSessionDetectedAction=primaryoverride

# Must match TRADE_PAPER_MODE=0 → port 7496 (live)
OverrideTwsApiPort=7496
ReadOnlyApi=no

# Auto-accept warnings
DismissPasswordExpiryWarning=yes
DismissNSEComplianceNotice=yes
IBCEOF

echo "IBC config created at /opt/ibc/config.ini"
echo "→ Edit IbLoginId and IbPassword before starting!"

# ── Project setup ────────────────────────────────────────────
echo ""
echo "[5/7] Setting up StockScout project..."
sudo -u "${SCOUT_USER}" bash << PROJEOF
cd ~
if [ ! -d "stock-scout-2" ]; then
    git clone https://github.com/asafamos/stock-scout.git stock-scout-2
fi
cd stock-scout-2

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

# ib_insync is NOT in requirements.txt (breaks Streamlit Cloud)
# Install separately for VPS trading
pip install ib_insync -q

echo "Python environment ready"
PROJEOF

# ── Environment file ─────────────────────────────────────────
echo ""
echo "[6/7] Creating .env.trading..."
ENV_FILE="${PROJECT_DIR}/.env.trading"
if [ ! -f "${ENV_FILE}" ]; then
    sudo -u "${SCOUT_USER}" tee "${ENV_FILE}" > /dev/null << 'ENVEOF'
# ── StockScout Trading Configuration ──
# This file is read by systemd services via EnvironmentFile=

# IBKR credentials (for IBC auto-login reference)
IBKR_USERNAME=your_ibkr_username
IBKR_PASSWORD=your_ibkr_password

# Trading mode
TRADE_DRY_RUN=0
TRADE_PAPER_MODE=0
TRADE_AUTO_CONFIRM=1

# Position sizing (tuned for ~$1000 account)
TRADE_MAX_POSITION_SIZE=300
TRADE_MAX_OPEN_POSITIONS=3
TRADE_MAX_DAILY_BUYS=3
TRADE_MAX_PORTFOLIO_EXPOSURE=900

# Signal filters
TRADE_MIN_SCORE=73.0
TRADE_MAX_SCORE=95.0
TRADE_MIN_ML_PROB=0.33
TRADE_MIN_RR=2.0
TRADE_BLOCKED_SECTORS=Consumer Defensive

# Risk management
TRADE_TRAILING_STOP_PCT=5.0

# Telegram notifications
TRADE_TELEGRAM_TOKEN=your_bot_token_here
TRADE_TELEGRAM_CHAT_ID=your_chat_id_here
ENVEOF
    echo "Created ${ENV_FILE} — edit with your credentials!"
else
    echo "${ENV_FILE} already exists — skipping"
fi

# ── Systemd services ─────────────────────────────────────────
echo ""
echo "[7/7] Creating systemd services..."

# --- Xvfb (virtual display for IB Gateway) ---
sudo tee /etc/systemd/system/xvfb.service > /dev/null << 'SVCEOF'
[Unit]
Description=Xvfb virtual display
After=network.target

[Service]
Type=simple
User=stockscout
ExecStart=/usr/bin/Xvfb :1 -screen 0 1024x768x24 -ac
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SVCEOF

# --- IB Gateway (via IBC for auto-login) ---
sudo tee /etc/systemd/system/ibgateway.service > /dev/null << 'SVCEOF'
[Unit]
Description=IB Gateway via IBC
After=xvfb.service
Requires=xvfb.service

[Service]
Type=simple
User=stockscout
Environment=DISPLAY=:1
ExecStart=/opt/ibc/gatewaystart.sh -inline \
    --tws-settings-path /home/stockscout/Jts
Restart=on-failure
RestartSec=60
StartLimitIntervalSec=600
StartLimitBurst=5

[Install]
WantedBy=multi-user.target
SVCEOF

# --- Event-driven scan→trade pipeline (replaces the old time-based ---
#     stockscout-pipeline.timer that fired at fixed times. The fixed-time
#     design lost ~1 trading day per week to GH Actions cron variability:
#     scan finished too late → trade ran on stale data, or scan still in
#     progress → trade ran with no fresh recommendations.
#
#     The pipeline (deploy/scan_and_trade.sh) is event-driven:
#       1. Snapshot current scan parquet hash on origin/main
#       2. Best-effort dispatch GH Actions (needs GITHUB_TOKEN in env)
#       3. Poll origin every 30s up to 150 min for a NEW hash
#       4. When new scan lands → pull → record outcomes → run_auto_trade
#       5. Exit. Next invocation handled by the timer below.
sudo tee /etc/systemd/system/stockscout-pipeline.service > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout atomic scan+trade pipeline
After=ibgateway.service
Requires=ibgateway.service

[Service]
Type=oneshot
User=stockscout
WorkingDirectory=/home/stockscout/stock-scout-2
EnvironmentFile=/home/stockscout/stock-scout-2/.env.trading
ExecStart=/bin/bash /home/stockscout/stock-scout-2/deploy/scan_and_trade.sh
TimeoutStartSec=10800
SVCEOF

# Timer: fire BEFORE each GH Actions scheduled scan window.
# The pipeline then triggers/awaits the scan and trades immediately on
# arrival — no race against cron variability.
sudo tee /etc/systemd/system/stockscout-pipeline.timer > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout pipeline timer (event-driven scan→trade)

[Timer]
# Pre-market scan (cron 13:30 UTC) — pipeline fires at 13:30 to dispatch
# the workflow and poll for results.
OnCalendar=Mon..Fri 13:30:00 UTC
# Afternoon scan (cron 17:30 UTC)
OnCalendar=Mon..Fri 17:30:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
SVCEOF

# --- Position monitor daemon ---
sudo tee /etc/systemd/system/stockscout-monitor.service > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout Position Monitor
After=ibgateway.service
Requires=ibgateway.service

[Service]
Type=simple
User=stockscout
WorkingDirectory=/home/stockscout/stock-scout-2
EnvironmentFile=/home/stockscout/stock-scout-2/.env.trading
ExecStart=/home/stockscout/stock-scout-2/.venv/bin/python -m scripts.monitor_positions --daemon
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
SVCEOF

# --- Healthcheck (oneshot, triggered by timer) ---
sudo tee /etc/systemd/system/stockscout-healthcheck.service > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout healthcheck
After=ibgateway.service

[Service]
Type=oneshot
User=stockscout
EnvironmentFile=/home/stockscout/stock-scout-2/.env.trading
ExecStart=/home/stockscout/stock-scout-2/deploy/healthcheck.sh
SVCEOF

sudo tee /etc/systemd/system/stockscout-healthcheck.timer > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout healthcheck timer

[Timer]
OnCalendar=Mon..Fri *:00/15 UTC
Persistent=true

[Install]
WantedBy=timers.target
SVCEOF

# --- Daily morning health summary (07:00 UTC = 10:00 IL) ---
# Surface bugs that took days to detect in the past:
# outcomes-record stale, services down, positions without protective
# orders. One Telegram message per weekday morning — quick to scan.
sudo tee /etc/systemd/system/stockscout-daily-summary.service > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout daily morning health summary

[Service]
Type=oneshot
User=stockscout
WorkingDirectory=/home/stockscout/stock-scout-2
EnvironmentFile=/home/stockscout/stock-scout-2/.env.trading
ExecStart=/home/stockscout/stock-scout-2/.venv/bin/python -m scripts.daily_morning_summary
SVCEOF

sudo tee /etc/systemd/system/stockscout-daily-summary.timer > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout daily health summary timer

[Timer]
OnCalendar=Mon..Fri 07:00:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
SVCEOF

sudo systemctl daemon-reload

echo ""
echo "=========================================="
echo " Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Install IB Gateway:  sudo bash /tmp/ibgateway-install.sh"
echo "  2. Edit credentials:    su - stockscout && nano ~/stock-scout-2/.env.trading"
echo "  3. Configure IBC:       sudo nano /opt/ibc/config.ini"
echo "     → Set IbLoginId and IbPassword"
echo "  4. Start services:      sudo systemctl enable --now xvfb ibgateway"
echo "  5. Test dry run:        cd ~/stock-scout-2 && source .venv/bin/activate"
echo "                          TRADE_DRY_RUN=1 python -m scripts.run_auto_trade"
echo "  6. Enable automation:   sudo systemctl enable --now stockscout-pipeline.timer"
echo "                          sudo systemctl enable --now stockscout-monitor"
echo "                          sudo systemctl enable --now stockscout-healthcheck.timer"
echo ""
echo "Monitoring:"
echo "  journalctl -u stockscout-pipeline -f"
echo "  journalctl -u stockscout-monitor -f"
echo "  journalctl -u ibgateway -f"
echo "  systemctl list-timers"
