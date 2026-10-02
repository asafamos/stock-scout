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

# --- OnFailure notification template (2026-09-28) ---
# Every stockscout-* service unit below has OnFailure= pointing to this
# template. When any unit fails, this fires and Telegram-alerts the operator.
# Prevents silent crashes — previously a crashed oneshot left no alert until
# downstream file-age checks eventually noticed (hours later).
sudo tee /etc/systemd/system/stockscout-notify-failure@.service > /dev/null << 'SVCEOF'
[Unit]
Description=Telegram alert when stockscout-%i failed

[Service]
Type=oneshot
EnvironmentFile=/home/stockscout/stock-scout-2/.env.trading
ExecStart=/bin/bash -c '\
  UNIT="%i"; \
  MSG="🚨 <b>stockscout-$UNIT FAILED</b> (systemd OnFailure trigger) — run: <code>journalctl -u stockscout-$UNIT -n 30 --no-pager</code>"; \
  curl -sf -o /dev/null -X POST "https://api.telegram.org/bot${TRADE_TELEGRAM_TOKEN}/sendMessage" \
    -d "chat_id=${TRADE_TELEGRAM_CHAT_ID}" \
    --data-urlencode "text=$MSG" \
    -d "parse_mode=HTML" || echo "telegram delivery failed" >&2'

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
OnFailure=stockscout-notify-failure@pipeline.service

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
# Schedule is defined in NEW YORK time so it follows DST (2026-09-29 audit: the
# old fixed-UTC times drift an hour vs the market from 2026-11-01 — the 13:30 UTC
# run would land at 08:30 ET, pre-market, and its buys would be refused).
# #1 MARKET OPEN 09:30 ET (16:30 IL), #2 OPENING-RANGE BREAKOUT 11:00 ET,
# #3 POWER HOUR 15:15 ET.
OnCalendar=Mon..Fri 09:30:00 America/New_York
OnCalendar=Mon..Fri 11:00:00 America/New_York
OnCalendar=Mon..Fri 15:15:00 America/New_York
Persistent=true

[Install]
WantedBy=timers.target
SVCEOF

# --- Position monitor daemon ---
# Uses clientId=2 to avoid colliding with the pipeline (clientId=1) and any
# manual ssh runs (which fall back to random 100-999 via auto-retry).
sudo tee /etc/systemd/system/stockscout-monitor.service > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout Position Monitor
After=ibgateway.service
Requires=ibgateway.service
OnFailure=stockscout-notify-failure@monitor.service

[Service]
Type=simple
User=stockscout
WorkingDirectory=/home/stockscout/stock-scout-2
EnvironmentFile=/home/stockscout/stock-scout-2/.env.trading
Environment="TRADE_IBKR_CLIENT_ID=2"
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
OnFailure=stockscout-notify-failure@healthcheck.service

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
# 2026-05-18: was Mon..Fri only — left the weekend uncovered, so the
# Saturday IB Gateway maintenance window's session-expiry sat un-noticed
# until 16:00 IL Monday (10 minutes before market open). Now runs every
# day, every 15 min. The script itself differentiates market-hours
# (deep) vs off-hours (deep too but with longer alert-dedup so it
# doesn't spam at 3am).
OnCalendar=*-*-* *:00/15 UTC
Persistent=true

[Install]
WantedBy=timers.target
SVCEOF

# --- State Broadcaster (every 30s) ---
# Builds system_state.json + force-pushes to GitHub state-feed branch.
# Streamlit reads via raw URL → near-real-time dashboard without lagging
# on git checkouts. Lightweight (~3s per run, mostly file IO).
sudo tee /etc/systemd/system/stockscout-state-broadcaster.service > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout state broadcaster (VPS → state-feed branch)
OnFailure=stockscout-notify-failure@state-broadcaster.service

[Service]
Type=oneshot
User=stockscout
WorkingDirectory=/home/stockscout/stock-scout-2
EnvironmentFile=/home/stockscout/stock-scout-2/.env.trading
ExecStart=/home/stockscout/stock-scout-2/.venv/bin/python -m scripts.state_broadcaster
TimeoutStartSec=60
SVCEOF

sudo tee /etc/systemd/system/stockscout-state-broadcaster.timer > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout state broadcaster timer (every 30s)

[Timer]
OnBootSec=60
OnUnitActiveSec=30
AccuracySec=5
Persistent=false

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
OnFailure=stockscout-notify-failure@daily-summary.service

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

# --- Weekly followup auto-verify audit (Fri 05:00 UTC = 08:00 IL) ---
# Runs verifiers over data/followups.json and auto-closes items whose
# success criterion is met on live state. Prevents the 'deployed and
# forgot' pattern (see project_bugfixes_aug14 memory for context).
sudo tee /etc/systemd/system/stockscout-followup-audit.service > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout weekly followup auto-verify audit
OnFailure=stockscout-notify-failure@followup-audit.service

[Service]
Type=oneshot
User=stockscout
WorkingDirectory=/home/stockscout/stock-scout-2
EnvironmentFile=/home/stockscout/stock-scout-2/.env.trading
ExecStart=/home/stockscout/stock-scout-2/.venv/bin/python -m scripts.weekly_followup_audit
SVCEOF

sudo tee /etc/systemd/system/stockscout-followup-audit.timer > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout weekly followup audit timer (Fri 05:00 UTC = 08:00 IL)

[Timer]
OnCalendar=Fri *-*-* 05:00:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
SVCEOF

# --- Adaptive edges nightly recompute (04:00 UTC = 07:00 IL) ---
# Runs compute_adaptive_edges.py which analyzes scan_outcomes.jsonl
# over rolling 90-day window and writes data/adaptive/current_edges.json
# with dynamic champion cohorts, sector blocks, ML window, RR cap.
# REPORT-ONLY by default — set ADAPTIVE_EDGES_APPLY=1 to activate.
sudo tee /etc/systemd/system/stockscout-adaptive-edges.service > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout adaptive edges — nightly recompute of selection parameters
OnFailure=stockscout-notify-failure@adaptive-edges.service

[Service]
Type=oneshot
User=stockscout
WorkingDirectory=/home/stockscout/stock-scout-2
EnvironmentFile=/home/stockscout/stock-scout-2/.env.trading
ExecStart=/home/stockscout/stock-scout-2/.venv/bin/python -m scripts.compute_adaptive_edges
SVCEOF

sudo tee /etc/systemd/system/stockscout-adaptive-edges.timer > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout adaptive edges timer (nightly 04:00 UTC = 07:00 IL)

[Timer]
OnCalendar=*-*-* 04:00:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
SVCEOF

# --- Command Poller (long-running, polls every 15s) ---
# Watches the `commands` branch on GitHub for entries appended by the
# command_dispatch.yml workflow (which fires on Streamlit dispatch
# button clicks via repository_dispatch). Picks up pending commands,
# runs them through core.control.command_bus, posts results back to
# Telegram. Without this, Streamlit's "Trigger VPS Scan" / "Pause" /
# "Sell All" buttons silently queue up and never execute.
#
# Type=simple (long-running, not oneshot) — the script has its own
# 15s poll loop. Restart=always so it survives transient git/network
# errors. KillMode=process ensures clean shutdown via SIGTERM.
sudo tee /etc/systemd/system/stockscout-command-poller.service > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout command poller (commands branch -> command_bus)
After=network-online.target
Wants=network-online.target
OnFailure=stockscout-notify-failure@command-poller.service

[Service]
Type=simple
User=stockscout
WorkingDirectory=/home/stockscout/stock-scout-2
EnvironmentFile=/home/stockscout/stock-scout-2/.env.trading
ExecStart=/home/stockscout/stock-scout-2/.venv/bin/python -m scripts.command_poller
Restart=always
RestartSec=10
KillMode=process

[Install]
WantedBy=multi-user.target
SVCEOF

# --- Telegram status bot (+ merged IB Key 2FA watchdog) ---
# Single Telegram getUpdates poller. Running this alongside the old
# stockscout-ibkey-bot.service causes 409 Conflicts and silent message
# drops — disable that unit if it exists. Restart=always covers
# crashes; the in-process auto-restart wrapper covers exceptions.
sudo tee /etc/systemd/system/stockscout-telegram-bot.service > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout Telegram status bot (+ IB Key 2FA watchdog)
After=network-online.target docker.service
Wants=network-online.target
OnFailure=stockscout-notify-failure@telegram-bot.service

[Service]
Type=simple
User=stockscout
Group=stockscout
WorkingDirectory=/home/stockscout/stock-scout-2
EnvironmentFile=/home/stockscout/stock-scout-2/.env.trading
Environment="TRADE_IBKEY_WATCHDOG=1"
ExecStart=/home/stockscout/stock-scout-2/.venv/bin/python -m scripts.telegram_status_bot
Restart=always
RestartSec=10
KillMode=process
StandardOutput=append:/home/stockscout/stock-scout-2/logs/telegram_bot.log
StandardError=append:/home/stockscout/stock-scout-2/logs/telegram_bot.log

[Install]
WantedBy=multi-user.target
SVCEOF

# ── Watchdog units (2026-09) — added after freshness+drift incidents ──
# These were created ad-hoc on the VPS during the Sep 14-28 investigation
# and are documented here so re-provisioning restores them.

# --- Drift check (2026-09-14): compares .env.trading vs CLAUDE.md EXPECTED
sudo tee /etc/systemd/system/stockscout-drift-check.service > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout env-vs-docs drift check
OnFailure=stockscout-notify-failure@drift-check.service

[Service]
Type=oneshot
User=stockscout
WorkingDirectory=/home/stockscout/stock-scout-2
EnvironmentFile=/home/stockscout/stock-scout-2/.env.trading
ExecStart=/home/stockscout/stock-scout-2/.venv/bin/python scripts/check_env_vs_docs.py
SVCEOF

sudo tee /etc/systemd/system/stockscout-drift-check.timer > /dev/null << 'SVCEOF'
[Unit]
Description=Daily env drift check (06:15 UTC)

[Timer]
OnCalendar=*-*-* 06:15:00
Persistent=true

[Install]
WantedBy=timers.target
SVCEOF

# --- Reconcile audit (2026-09-18): triangulates tracker ↔ ledger ↔ IB
sudo tee /etc/systemd/system/stockscout-reconcile-audit.service > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout tracker/ledger/IB reconciliation audit
OnFailure=stockscout-notify-failure@reconcile-audit.service

[Service]
Type=oneshot
User=stockscout
WorkingDirectory=/home/stockscout/stock-scout-2
EnvironmentFile=/home/stockscout/stock-scout-2/.env.trading
ExecStart=/home/stockscout/stock-scout-2/.venv/bin/python scripts/reconcile_audit.py
SVCEOF

sudo tee /etc/systemd/system/stockscout-reconcile-audit.timer > /dev/null << 'SVCEOF'
[Unit]
Description=Daily tracker/ledger/IB reconciliation (07:00 UTC)

[Timer]
OnCalendar=*-*-* 07:00:00
Persistent=true

[Install]
WantedBy=timers.target
SVCEOF

# --- Scan freshness watchdog (2026-09-25): alerts if parquet As_Of_Date > 7d old
sudo tee /etc/systemd/system/stockscout-scan-freshness.service > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout scan-freshness watchdog
OnFailure=stockscout-notify-failure@scan-freshness.service

[Service]
Type=oneshot
User=stockscout
WorkingDirectory=/home/stockscout/stock-scout-2
EnvironmentFile=/home/stockscout/stock-scout-2/.env.trading
ExecStart=/home/stockscout/stock-scout-2/.venv/bin/python scripts/check_scan_freshness.py
SVCEOF

sudo tee /etc/systemd/system/stockscout-scan-freshness.timer > /dev/null << 'SVCEOF'
[Unit]
Description=Daily scan-freshness watchdog (fires ~06:30 UTC)

[Timer]
OnCalendar=*-*-* 06:30:00
Persistent=true
RandomizedDelaySec=90

[Install]
WantedBy=timers.target
SVCEOF

# --- Outcomes resolver (nightly). 2026-09-29: also re-resolves legacy short-window (<20 bars) rows in batches.
sudo tee /etc/systemd/system/stockscout-outcomes-resolve.service > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout scan outcomes resolver
OnFailure=stockscout-notify-failure@outcomes-resolve.service

[Service]
Type=oneshot
User=stockscout
WorkingDirectory=/home/stockscout/stock-scout-2
EnvironmentFile=/home/stockscout/stock-scout-2/.env.trading
ExecStart=/home/stockscout/stock-scout-2/.venv/bin/python -m scripts.track_scan_outcomes --resolve
ExecStartPost=/home/stockscout/stock-scout-2/.venv/bin/python -m scripts.track_scan_outcomes --reresolve-short --reresolve-limit 300
TimeoutStartSec=3000
SVCEOF

# --- v2 sleeve (2026-09-29): volatility+size tilt, trades the prior close's scan at the open. No-op
# unless TRADE_V2_SLEEVE=1 in .env.trading.
sudo tee /etc/systemd/system/stockscout-v2-sleeve.service > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout v2 sleeve (next-open entry from the prior close scan)
OnFailure=stockscout-notify-failure@v2-sleeve.service

[Service]
Type=oneshot
User=stockscout
WorkingDirectory=/home/stockscout/stock-scout-2
EnvironmentFile=/home/stockscout/stock-scout-2/.env.trading
ExecStart=/bin/bash /home/stockscout/stock-scout-2/deploy/v2_sleeve.sh
TimeoutStartSec=900
SVCEOF

sudo tee /etc/systemd/system/stockscout-v2-sleeve.timer > /dev/null << 'SVCEOF'
[Unit]
Description=v2 sleeve 1 minute after the US open (DST-aware)

[Timer]
OnCalendar=Mon..Fri 09:31:00 America/New_York
Persistent=false

[Install]
WantedBy=timers.target
SVCEOF

# --- Security hardening (2026-09-29 incident: IB API + VNC were reachable from the internet) ------------------
# 1. keep IB Gateway API/VNC ports off the public interface (idempotent, re-applied at boot)
sudo install -m 0755 "$(dirname "$0")/stockscout-firewall.sh" /usr/local/sbin/stockscout-firewall.sh
sudo tee /etc/systemd/system/stockscout-firewall.service > /dev/null << 'SVCEOF'
[Unit]
Description=Keep IB Gateway API/VNC off the public internet
After=docker.service network-online.target
Wants=docker.service
[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/stockscout-firewall.sh
[Install]
WantedBy=multi-user.target
SVCEOF
# 2. fail2ban for sshd
sudo apt-get install -y fail2ban >/dev/null 2>&1 || true
sudo mkdir -p /etc/fail2ban/jail.d
sudo tee /etc/fail2ban/jail.d/stockscout-sshd.local > /dev/null << 'SVCEOF'
[sshd]
enabled = true
backend = systemd
maxretry = 4
findtime = 10m
bantime = 12h
SVCEOF
# 3. key-only SSH — ONLY if a key is already installed (never lock yourself out of a fresh box)
if [ -s /root/.ssh/authorized_keys ]; then
  sudo mkdir -p /etc/ssh/sshd_config.d
  sudo tee /etc/ssh/sshd_config.d/00-stockscout-hardening.conf > /dev/null << 'SVCEOF'
PasswordAuthentication no
KbdInteractiveAuthentication no
MaxAuthTries 3
PermitRootLogin prohibit-password
SVCEOF
  sudo sshd -t && sudo systemctl reload ssh || echo "WARNING: sshd config test failed — hardening not applied"
else
  echo "WARNING: /root/.ssh/authorized_keys is empty — SSH password auth left ON. Add a key, then re-run."
fi
sudo systemctl daemon-reload
sudo systemctl enable --now stockscout-firewall.service fail2ban >/dev/null 2>&1 || true

# --- CoreTrend (2026-10-02): QQQ/IEF 10-month trend rule, 2-3 trades/year. No-op unless TRADE_CORETREND=1.
sudo tee /etc/systemd/system/stockscout-coretrend.service > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout CoreTrend executor (idempotent daily run)
OnFailure=stockscout-notify-failure@coretrend.service

[Service]
Type=oneshot
User=stockscout
WorkingDirectory=/home/stockscout/stock-scout-2
EnvironmentFile=/home/stockscout/stock-scout-2/.env.trading
Environment=TRADE_LIVE_CONFIRMED=1
ExecStart=/home/stockscout/stock-scout-2/.venv/bin/python -m scripts.run_coretrend
TimeoutStartSec=600
SVCEOF

sudo tee /etc/systemd/system/stockscout-coretrend.timer > /dev/null << 'SVCEOF'
[Unit]
Description=CoreTrend 09:45 New York time, Mon-Fri (DST-aware)

[Timer]
OnCalendar=Mon..Fri 09:45:00 America/New_York
Persistent=false

[Install]
WantedBy=timers.target
SVCEOF

# --- CoreTrend month-end decision alert (Telegram only on the last trading day of a month)
sudo tee /etc/systemd/system/stockscout-coretrend-alert.service > /dev/null << 'SVCEOF'
[Unit]
Description=CoreTrend month-end decision alert
OnFailure=stockscout-notify-failure@coretrend-alert.service

[Service]
Type=oneshot
User=stockscout
WorkingDirectory=/home/stockscout/stock-scout-2
EnvironmentFile=/home/stockscout/stock-scout-2/.env.trading
ExecStart=/home/stockscout/stock-scout-2/.venv/bin/python -m scripts.coretrend_paper --alert-if-last-day
SVCEOF

sudo tee /etc/systemd/system/stockscout-coretrend-alert.timer > /dev/null << 'SVCEOF'
[Unit]
Description=CoreTrend month-end alert 17:50 New York time, Mon-Fri

[Timer]
OnCalendar=Mon..Fri 17:50:00 America/New_York
Persistent=false

[Install]
WantedBy=timers.target
SVCEOF

# --- Weekly scorecard (2026-09-30): account vs SPY, sleeve stats, perf-guard level -> Telegram (Fri after close)
sudo tee /etc/systemd/system/stockscout-weekly-vs-spy.service > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout weekly scorecard (account vs SPY)
OnFailure=stockscout-notify-failure@weekly-vs-spy.service

[Service]
Type=oneshot
User=stockscout
WorkingDirectory=/home/stockscout/stock-scout-2
EnvironmentFile=/home/stockscout/stock-scout-2/.env.trading
ExecStart=/home/stockscout/stock-scout-2/.venv/bin/python -m scripts.weekly_vs_spy
SVCEOF

sudo tee /etc/systemd/system/stockscout-weekly-vs-spy.timer > /dev/null << 'SVCEOF'
[Unit]
Description=Weekly scorecard, Friday after the US close (DST-aware)

[Timer]
OnCalendar=Fri 17:45:00 America/New_York
Persistent=true

[Install]
WantedBy=timers.target
SVCEOF

# --- Shadow selector (2026-09-29): logs the whole scan + pre-registered rule flags, resolves 20-session
# outcomes, writes data/outcomes/shadow_report.txt. Additive — trades nothing.
sudo tee /etc/systemd/system/stockscout-shadow.service > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout shadow selector (log + resolve + report)
OnFailure=stockscout-notify-failure@shadow.service

[Service]
Type=oneshot
User=stockscout
WorkingDirectory=/home/stockscout/stock-scout-2
EnvironmentFile=/home/stockscout/stock-scout-2/.env.trading
ExecStart=/bin/bash /home/stockscout/stock-scout-2/deploy/shadow_daily.sh
TimeoutStartSec=1500
SVCEOF

sudo tee /etc/systemd/system/stockscout-shadow.timer > /dev/null << 'SVCEOF'
[Unit]
Description=Daily shadow selector (after the US close, DST-aware)

[Timer]
OnCalendar=Mon..Fri 17:30:00 America/New_York
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
echo "                          sudo systemctl enable --now stockscout-telegram-bot"
echo "                          sudo systemctl enable --now stockscout-daily-summary.timer"
echo "                          sudo systemctl enable --now stockscout-followup-audit.timer"
echo ""
echo "Monitoring:"
echo "  journalctl -u stockscout-pipeline -f"
echo "  journalctl -u stockscout-monitor -f"
echo "  journalctl -u ibgateway -f"
echo "  systemctl list-timers"
