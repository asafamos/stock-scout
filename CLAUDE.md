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
- Max position size: $300 (allows 1 share up to $600 for expensive stocks)
- Max open positions: 3
- Max daily buys: 3
- Max portfolio exposure: $900
- Score filter: 73-95 (lowered from 75 to catch near-miss stocks like BURL)
- ML probability min: 0.33 (lowered from 0.40 — real ML output range is 0.30-0.37, 0.40 blocked 100%)
- Trailing stop: adaptive per stock from scan's stop loss (floor 3%, cap 8%)
- Blocked sectors: Consumer Defensive
- Selection: sorted by 60% score + 40% R:R ratio (not just score)

### Trading Architecture
- `core/trading/ibkr_client.py` - IB connection + OCA bracket orders (buy + trailing stop + limit sell)
- `core/trading/order_manager.py` - Scan → filter → risk check → execute → track (auto-picks newest scan: GH Actions or Streamlit)
- `core/trading/position_tracker.py` - JSON-backed position tracking
- `core/trading/risk_manager.py` - Pre-trade validation
- `core/trading/notifications.py` - Telegram alerts
- `core/trading/config.py` - All settings (override via TRADE_* env vars)
- `scripts/run_auto_trade.py` - CLI entry point
- `scripts/monitor_positions.py` - Position monitoring daemon

### Safety: DRY_RUN=True by default, requires explicit override for live trading

## Telegram Bot
- Bot name: StockScout Alerts (@stockscout_asaf_bot)
- Token & Chat ID: stored in GitHub Secrets + .streamlit/secrets.toml (NEVER in code)
- Connected and working since April 8, 2026
- Sends: buy alerts, sell alerts, errors, scan summaries, daily summaries
- Config via: TRADE_TELEGRAM_TOKEN and TRADE_TELEGRAM_CHAT_ID env vars

## Automation
- **GitHub Actions**: 4x daily scans (pre-market 8:30AM, 10AM, 3PM, 4:30PM ET), nightly outcome tracking, weekly ML retraining + backtest
- **10AM scan** triggers auto-trade DRY RUN → sends Telegram alerts showing what it WOULD buy
- **Both scans (GH Actions + Streamlit)** save to Supabase with user_id=stockscout_owner
- **VPS deploy script**: `deploy/setup_vps.sh` for Ubuntu VPS (~$5/month) - NOT YET DEPLOYED
- **Next step**: Run first LIVE trade from Mac → then move to VPS for 24/7 operation
- **Scan loading**: order_manager auto-picks the most recent scan file (GH Actions parquet or Streamlit JSON) by modification time

## How to Run Auto-Trade (Manual from Mac)
```bash
# 1. Open IB Gateway 10.37 → Log in (Live Trading, IB API)
# 2. DRY RUN first:
cd ~/StockScout/stock-scout-2
.venv/bin/python -m scripts.run_auto_trade
# 3. LIVE (requires typed "CONFIRM LIVE"):
TRADE_DRY_RUN=0 .venv/bin/python -m scripts.run_auto_trade
# 4. Monitor positions:
.venv/bin/python -m scripts.monitor_positions --daemon
```

## Important Notes
- OCA bracket orders (trailing stop + limit sell) live on IB servers - no Mac dependency
- Target date exits require monitor_positions daemon (Mac or VPS must be running)
- System correctly abstains from buying when market regime is NEUTRAL (lower scores)
- Scans and auto-trade are identical between GH Actions and Streamlit (same pipeline, same save)
- Streamlit Cloud scans can get stuck on long runs (Streamlit reruns kill them) - GH Actions is more reliable

## Known Issues (Fixed on April 12, 2026)
- Streamlit scan freezing at 30-80%: added timeouts to as_completed() loops
- Streamlit scan completes but results not saved: moved Supabase save to immediately after pipeline completion (was 400 lines later, Streamlit rerun killed it before reaching save)
- Streamlit scan crash on Python 3.13: added .python-version=3.11 (KeyError: 'core.indicators')
- GitHub Actions scans cancelled at 45min: increased timeout to 90min
- GitHub Actions missing 5 API keys: added EODHD, SIMFIN, NASDAQ, MARKETSTACK, Telegram
- GitHub Actions overlapping scans: added concurrency group
- Telegram token exposed in CLAUDE.md: removed, token revoked and replaced
- ib_insync removed from requirements.txt to not break Streamlit Cloud deployment
- Git pushes during Streamlit scan kill the scan: avoid pushing while scan runs
- GitHub Actions scan not saving to Supabase: switched to core.scan_io.save_scan (same as Streamlit)
- GitHub Actions final save crash on Fundamental_Breakdown column: added object-to-string conversion
- GitHub Actions scanned 3000 instead of 2000: fixed default in core/config.py

## Tech Stack
- Python 3.11, Streamlit, XGBoost, scikit-learn, pandas
- 10 data providers with fallback (Yahoo, Alpha Vantage, Finnhub, Polygon, Tiingo, FMP, etc.)
- DuckDB for analytics, JSON for position tracking
