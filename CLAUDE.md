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

### Trading Config (RECALIBRATED 2026-06-06 on 18,709 Supabase production trades)
**Position sizing (adjusted for ~$977 balance):**
- Max position size: $400 (was $300; allows 1 share up to $600 for expensive stocks)
- Max open positions: 3 | Max daily buys: 3 | Max portfolio exposure: $1000
- min_viable_position_usd: $30 (early-exit candidate loop when cash exhausted)

**Trade GATES (WINDOW-based, validated on real fwd returns):**
- Score: 73-95 (max=95 because Q5 underperforms in-sample; 95+ OOS = PLTR/NFLX curve fit)
- ML probability: **0.40-0.55** ← SWEET SPOT (live ML mean 0.42, max 0.76. 0.45-0.50 = +5.07% returns)
- R:R: **3.0-5.0** ← SWEET SPOT (RR 2-2.5 returns -0.62%, RR 4-5 returns +3.81%, RR 7+ drops)
- ATR_Pct: **≥ 0.04** (NEW gate — ATR 5-7% returns +6.13%, low-vol returns negative)
- Min confidence: High | Min reliability: 50

**Blocked sectors (8, from 1):** Consumer Defensive, Utilities, Communication, Materials, Basic Materials, Consumer Cyclical, Financial, Financial Services
**Top sectors kept:** Technology +10.42%, Healthcare +2.79%, Industrials, Energy, Communication Services

**Blocked regimes: PANIC only** (CORRECTION removed — data showed +5.48%/55% WR, mean-reversion edge)
**Reduce_regimes: DISTRIBUTION** (half-size)

**Trail (RECALIBRATED 2026-06-10 on real-OHLC simulation of 512 trades):**
- **Initial trail: 9.0%** (was 5.5%) — absorbs intraday noise during days 1-7
- **At day 7: tightens automatically to 5.5%** via `_time_tighten_stops` in monitor
- 3.0% defensive (CORRECTION/BEARISH/PANIC/DISTRIBUTION)
- ATR floor 0.9×, regime mult (UP=1.20, SIDEWAYS=1.00, DISTRIBUTION=0.85, defensive=0.70)
- Then standard RATCHET tightens further as profit accumulates (T0 +10% → 5%, T1 +14% → 4.5%, T2 +22% → 3.5%, T3 +30% → 2.5%)
- Reasoning: postmortem on 12 trail-fired losers showed 9/12 (75%) recovered above ENTRY within 14 days. The 5.5% trail was firing on natural intraday noise. Real-OHLC backtest: 5.5%+0d=-1.73%/trade OOS; 9%+0d=+0.42%; 9%→5.5% at day 7=+1.26% (chosen).
- Env: TRADE_MIN_INITIAL_TRAIL_PCT, TRADE_TIME_TIGHTEN_ENABLED, TRADE_TIME_TIGHTEN_DAYS, TRADE_TIME_TIGHTEN_TARGET_PCT

**BREAK_EVEN: DISABLED** (backtest showed net -$74.81/$1k/trade — was a bad anecdote-based feature)

**Selection ranking weights (REBALANCED, sum=1.00):**
- **RR: 40%** (was 25% — strongest stable signal, corr +0.088)
- **ML: 25%** (was 20%)
- **ATR: 15%** (NEW signal — corr +0.140)
- **Score: 5%** (was 45% — anti-predictive as ranker, corr -0.13)
- Sector momentum: 5% | Insider boost: 5% | SPY momentum: 5% when available
- INSIGHT: ranking by score actually PICKED LOSERS — picking by RR gave +6.34%/trade vs +2.55% by score

### Trading Architecture
- `core/trading/policy.py` - **Single source of truth for buy gates + execution preview.**
  Both `risk_manager.can_open_position` (production) and `streamlit_components.evaluate_scan_row_for_buy` (dashboard preview) call into `evaluate_static_gates()`. Also `compute_execution_preview()` mirrors `_execute_single`'s price/stop/target/trail derivation.
- `core/trading/ibkr_client.py` - IB connection + OCA bracket orders. All Stock contracts routed through `_ib_symbol()` → `policy.normalize_ticker_for_ib()` (BRK.B → BRK B).
- `core/trading/order_manager.py` - Scan → filter → risk check → execute → track. **Tracker write BEFORE notify_buy** (Telegram BUY always backed by audit row). Pre-filter parity with SSOT (added 2026-06-06: ML window, RR window, ATR floor).
- `core/trading/position_tracker.py` - JSON-backed position tracking. `drop_metadata()` is the ledger-mode close path (no P&L fabrication).
- `core/trading/risk_manager.py` - Pre-trade validation. Calls policy.evaluate_static_gates as defense-in-depth.
- `core/trading/notifications.py` - Telegram alerts
- `core/trading/config.py` - All settings (override via TRADE_* env vars). All gate thresholds read FROM here — never hardcoded.
- **`core/trading/ledger.py` (NEW 2026-06-04)** - Event-sourced ledger. `executions.jsonl` idempotent by IB execId. Realized P&L from `commissionReport.realizedPNL` (net of fees, reconciles to NetLiq by construction). **THE source of truth** for closed-trade accounting — replaced the buggy trade_log-based path.
- `scripts/run_auto_trade.py` - CLI entry point
- `scripts/monitor_positions.py` - Position monitoring daemon (ingests ledger fills each cycle in ledger mode)
- `scripts/migrate_ledger.py` - ONE-TIME baseline anchor. Run after deploying ledger code; sets `pre_ledger_realized` so `/pnl` reconciliation reads Δ≈0.

### CRITICAL ARCHITECTURE LAYERS (do NOT confuse)
- **IB is the source of truth** for: positions, NetLiq, executions, realized P&L (via ledger)
- **The ledger (executions.jsonl)** is the source of truth for: per-execution realized P&L, win/loss record
- **The tracker (open_positions.json)** is the metadata annotation layer — NO P&L authority
- **The legacy trade_log.json** has pre-cutover history with phantom data. Do NOT compute lifetime P&L from it.
- **Lifetime realized = NetLiq − starting_capital − open_unrealized** (account-truth identity)
- See `docs/architecture_ledger.md` for full spec.

### Safety: DRY_RUN=True by default, requires explicit override for live trading
**Reversibility of recent changes:** all gates env-overridable (TRADE_MIN_SCORE, TRADE_MIN_ML_PROB, TRADE_MAX_ML_PROB, TRADE_MIN_RR, TRADE_MAX_RR, TRADE_MIN_ATR_PCT, TRADE_BLOCKED_SECTORS, TRADE_BLOCKED_REGIMES, TRADE_BREAK_EVEN_ENABLED, TRADE_LEDGER_ENABLED). Selection weights live in `order_manager.py` (need code change to revert).

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
