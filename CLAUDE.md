# StockScout - Project Memory

## Owner
Asaf Amos (asafamos)

## What This Project Does
AI-powered stock recommendation system that scans 3,000+ US stocks using technical analysis (20+ indicators), fundamental scoring, and ML (XGBoost ensemble). Deployed on Streamlit Cloud.

## Live URLs
- **Streamlit App**: https://stock-scout-fm4iuuknxjwjbcg95inbcj.streamlit.app
- **GitHub**: https://github.com/asafamos/stock-scout

## Auto-Trading (IBKR) - Setup Complete as of April 12, 2026
- **Account**: U25201431 (Interactive Brokers)
- **Balance**: ~$977.50 (initial deposit)
- **IB Gateway 10.37**: Installed on Mac, configured with IB API, port 7496 (live)
- **ib_insync**: Installed locally (not in requirements.txt to avoid breaking Streamlit Cloud)
- **Connection tested**: Successfully connects from Mac to IB Gateway

### Trading Config (adjusted for $977 balance)
- Max position size: $300
- Max open positions: 3
- Max daily buys: 2
- Max portfolio exposure: $900
- Score filter: 75-95 (Q3-Q4 sweet spot)
- ML probability min: 0.40
- Trailing stop: 5%
- Blocked sectors: Consumer Defensive

### Trading Architecture
- `core/trading/ibkr_client.py` - IB connection + OCA bracket orders (buy + trailing stop + limit sell)
- `core/trading/order_manager.py` - Scan → filter → risk check → execute → track
- `core/trading/position_tracker.py` - JSON-backed position tracking
- `core/trading/risk_manager.py` - Pre-trade validation
- `core/trading/notifications.py` - Telegram alerts
- `core/trading/config.py` - All settings (override via TRADE_* env vars)
- `scripts/run_auto_trade.py` - CLI entry point
- `scripts/monitor_positions.py` - Position monitoring daemon

### Safety: DRY_RUN=True by default, requires explicit override for live trading

## Telegram Bot
- Bot name: StockScout Alerts (@stockscout_asaf_bot)
- Bot Token: 8502546938:AAE1l92Ucy54j22W8EFFm-wMc6aijYH0u7k
- Chat ID: 5600927421 (Asaf Amos)
- Connected and working since April 8, 2026
- Sends: buy alerts, sell alerts, errors, scan summaries, daily summaries
- Config via: TRADE_TELEGRAM_TOKEN and TRADE_TELEGRAM_CHAT_ID env vars
- Stored in: GitHub Secrets + .streamlit/secrets.toml (local)

## Automation
- **GitHub Actions**: 4x daily scans (pre-market, 10AM, 3PM, EOD), nightly outcome tracking, weekly ML retraining + backtest
- **VPS deploy script**: `deploy/setup_vps.sh` for Ubuntu VPS (~$5/month) - NOT YET DEPLOYED
- **Next step**: Move from running on Mac to VPS for 24/7 operation

## Known Issues (Fixed)
- Streamlit scan freezing at 30-80%: Fixed April 12 by adding timeouts to as_completed() loops
- ib_insync removed from requirements.txt to not break Streamlit Cloud deployment

## Tech Stack
- Python 3.11, Streamlit, XGBoost, scikit-learn, pandas
- 10 data providers with fallback (Yahoo, Alpha Vantage, Finnhub, Polygon, Tiingo, FMP, etc.)
- DuckDB for analytics, JSON for position tracking
