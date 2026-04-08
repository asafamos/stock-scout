#!/bin/bash
# ============================================================
# StockScout VPS Setup — IB Gateway + Auto-Trading
#
# Run on a fresh Ubuntu 22.04 VPS (DigitalOcean, Hetzner, etc.)
# Estimated cost: $5-10/month for a basic droplet
#
# Usage:
#   scp deploy/setup_vps.sh user@your-vps:~/
#   ssh user@your-vps
#   chmod +x setup_vps.sh && ./setup_vps.sh
# ============================================================

set -e

echo "=========================================="
echo " StockScout VPS Setup"
echo "=========================================="

# ── System packages ──────────────────────────────────────────
echo "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3.11 python3.11-venv python3-pip \
    git xvfb unzip wget curl \
    openjdk-17-jre-headless  # Required for IB Gateway

# ── IB Gateway (headless) ───────────────────────────────────
echo ""
echo "Downloading IB Gateway..."
# IB Gateway stable release
IB_GATEWAY_URL="https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh"
wget -q -O /tmp/ibgateway-install.sh "$IB_GATEWAY_URL" 2>/dev/null || {
    echo "WARNING: Could not download IB Gateway automatically."
    echo "Download manually from: https://www.interactivebrokers.com/en/trading/ibgateway-stable.php"
    echo "Then run: bash ibgateway-install.sh"
}

if [ -f /tmp/ibgateway-install.sh ]; then
    chmod +x /tmp/ibgateway-install.sh
    echo "Run: sudo bash /tmp/ibgateway-install.sh"
    echo "(Interactive installer — follow the prompts)"
fi

# ── IBC (IB Controller — auto-login for IB Gateway) ─────────
echo ""
echo "Installing IBC (auto-login controller)..."
IBC_VERSION="3.18.0"
wget -q -O /tmp/ibc.zip \
    "https://github.com/IbcAlpha/IBC/releases/download/${IBC_VERSION}/IBCLinux-${IBC_VERSION}.zip"
sudo mkdir -p /opt/ibc
sudo unzip -qo /tmp/ibc.zip -d /opt/ibc
sudo chmod +x /opt/ibc/*.sh

# ── Project setup ────────────────────────────────────────────
echo ""
echo "Setting up StockScout..."
cd ~
if [ ! -d "stock-scout-2" ]; then
    git clone https://github.com/asafamos/stock-scout.git stock-scout-2
fi
cd stock-scout-2

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

# ── Environment file ─────────────────────────────────────────
echo ""
echo "Creating .env template..."
cat > .env.trading << 'ENVEOF'
# IBKR credentials (for IBC auto-login)
IBKR_USERNAME=your_ibkr_username
IBKR_PASSWORD=your_ibkr_password

# Trading config
TRADE_DRY_RUN=1
TRADE_PAPER_MODE=1
TRADE_MAX_POSITION_SIZE=1000
TRADE_MAX_OPEN_POSITIONS=10
TRADE_TRAILING_STOP_PCT=5.0

# Telegram notifications
TRADE_TELEGRAM_TOKEN=your_bot_token
TRADE_TELEGRAM_CHAT_ID=your_chat_id

# Auto-trade trigger (set to 1 to enable)
AUTO_TRADE_ENABLED=0
ENVEOF

echo "Edit .env.trading with your credentials!"

# ── Systemd services ─────────────────────────────────────────
echo ""
echo "Creating systemd services..."

# IB Gateway service (with Xvfb for headless display)
sudo tee /etc/systemd/system/ibgateway.service > /dev/null << 'SVCEOF'
[Unit]
Description=IB Gateway (headless)
After=network.target

[Service]
Type=simple
User=root
Environment=DISPLAY=:1
ExecStartPre=/usr/bin/Xvfb :1 -screen 0 1024x768x24 &
ExecStart=/opt/ibc/gatewaystart.sh -inline
Restart=on-failure
RestartSec=60

[Install]
WantedBy=multi-user.target
SVCEOF

# StockScout auto-trade service
sudo tee /etc/systemd/system/stockscout-trade.service > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout Auto-Trade (pre-market scan + execute)
After=ibgateway.service

[Service]
Type=oneshot
User=root
WorkingDirectory=/root/stock-scout-2
EnvironmentFile=/root/stock-scout-2/.env.trading
ExecStart=/root/stock-scout-2/.venv/bin/python -m scripts.run_auto_trade
SVCEOF

# Timer: run at 13:15 UTC (9:15 AM ET — 15 min before market open)
sudo tee /etc/systemd/system/stockscout-trade.timer > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout daily pre-market trade timer

[Timer]
OnCalendar=Mon..Fri 13:15:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
SVCEOF

# Position monitor service
sudo tee /etc/systemd/system/stockscout-monitor.service > /dev/null << 'SVCEOF'
[Unit]
Description=StockScout Position Monitor Daemon
After=ibgateway.service

[Service]
Type=simple
User=root
WorkingDirectory=/root/stock-scout-2
EnvironmentFile=/root/stock-scout-2/.env.trading
ExecStart=/root/stock-scout-2/.venv/bin/python -m scripts.monitor_positions --daemon
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
SVCEOF

sudo systemctl daemon-reload

echo ""
echo "=========================================="
echo " Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Install IB Gateway: sudo bash /tmp/ibgateway-install.sh"
echo "  2. Edit credentials: nano .env.trading"
echo "  3. Configure IBC: nano /opt/ibc/config.ini"
echo "     - Set IbLoginId and IbPassword"
echo "     - Set TradingMode=paper (or live)"
echo "  4. Start IB Gateway: sudo systemctl start ibgateway"
echo "  5. Test dry run: source .venv/bin/activate && python -m scripts.run_auto_trade"
echo "  6. Enable timer: sudo systemctl enable --now stockscout-trade.timer"
echo "  7. Enable monitor: sudo systemctl enable --now stockscout-monitor"
echo ""
echo "Monitoring commands:"
echo "  journalctl -u stockscout-trade -f    # Trade logs"
echo "  journalctl -u stockscout-monitor -f  # Monitor logs"
echo "  journalctl -u ibgateway -f           # IB Gateway logs"
echo "  systemctl list-timers                 # Check timer"
