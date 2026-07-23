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
**Position sizing (adjusted for ~$867 balance, revised 2026-07-17):**
- Max position size: **$450** (was $400; env `TRADE_MAX_POSITION_SIZE=450` — accommodates $600+ stocks in 1-share bites)
- Max open positions: **3** | Max daily buys: **3** | Max portfolio exposure: **$1350**
  - 2026-07-17 drift fix: `.env.trading` on VPS had `MAX_OPEN_POSITIONS=2` / `MAX_DAILY_BUYS=2` / `MAX_PORTFOLIO_EXPOSURE=900` (silent drift from setup_vps.sh defaults). Restored to 3/3/1350 after 2 days of 0-buy cycles left cash idle. See memory `throughput-drift-jul17.md`.
- min_viable_position_usd: $30 (early-exit candidate loop when cash exhausted)

**⚠️ 2026-07-09 FREEZE: gates below are FROZEN until we have 10-20 new trade closes under this config. See memory `feedback_no_flipflop.md` for framework. Do NOT retune based on more simulated data — that's the anti-pattern that caused MAX_SCORE to flip 4× in a week.**

**Trade GATES (RECALIBRATED 2026-07-09 evening — final freeze state):**
- Score: **73-85** in CONFIG. **2026-07-03 (a9c6645):** runtime uses `max(REGIME_MIN_SCORE+5, CONFIG.min_score)` so CONFIG.min_score_to_trade is the HARD floor. **MAX_SCORE=85** confirmed on 2026-07-09 (commit 75d987b) after brief flip to 97 based on simulated (n=2588). REAL portfolio (n=402) says 85-89=-0.38%, 90-94=-0.48%. Framework: when sim vs real disagree, TRUST REAL. Only revisit when real n≥100 in 85+ range.
- Fundamental_Score: **≥ 45** (2026-07-09 commit 3f92573, was 40). Fund 40-45=-0.04% (dead zone), 45-50=+5.67% (BEST). Both sim + real agree on monotonic direction.
- ML probability: **0.40-0.60** (was 0.55; raised because bucket 0.55-0.60 = +4.93% mean vs in-window +3.0%)
- R:R: **2.5-5.0** (2026-07-09 commit 9ea1174 restored max=5.0 after fresh backtest: RR 5-7 = -3.49% mean, WR 0% (n=18) — DISASTER ZONE). Floor 2.5 stable.
- ATR_Pct: **≥ 0.03** (lowered from 0.04 on 2026-06-12; bucket 0.03-0.04 = +2.0% mean, was being excluded. ATR=0 treated as missing data pass-through.)
- Min confidence: High. 2026-07-03 FIX (commit 281b74c) — CONFIG is now HARD floor. Regime relax (High→Medium in bullish) is opt-in via `TRADE_CONFIDENCE_REGIME_RELAX=1`. Default disabled. Note: gate reads `SignalQuality` field, NOT `ML_Confidence_Status` — separate metrics, latter shown in status but not enforced.
- Min reliability: 50

**Blocked sectors (6, REVISED 2026-07-17 — real data trumps sim):**
- BLOCKED: Consumer Defensive (-3.13% p=0.006, n=22), Utilities (+0.88% borderline, kept), Communication (no real data, kept as precaution), Materials (-1.23% n=8, weak), Basic Materials (-0.80% n=28, weak), Real Estate (+1.08% n=15, borderline)
- **UNBLOCKED 2026-07-17: Energy (+3.07% p<0.001 n=70)** — was blocked from stale sim (-1.59% n=304). Real data flips: 402 REAL Supabase positions Mar-Jun 2026 show Energy = 65.7% win rate. Per no-flipflop framework: TRUST REAL when sim disagrees.
- **UNBLOCKED 2026-07-17: Financial (+3.05% p=0.062 n=6)** — small n but positive; previously blocked as ambiguous.
- Previous unblocks kept: Consumer Cyclical, Financial Services.
- Top real performers (n=402 Supabase): **Energy +3.07% (NEW)**, Industrials +2.53% (n=53), Technology +1.95% (n=46), Communication Services +2.25% (n=26).

**Blocked regimes: PANIC only** (CORRECTION removed — data showed +5.48%/55% WR, mean-reversion edge)
**Reduce_regimes: DISTRIBUTION** (half-size)

**Trail (RECALIBRATED 2026-06-10 on real-OHLC simulation of 512 trades — FROZEN per memory `calibration-freeze` + `data-driven-gates-jun24`):**
- **Initial trail: 9.0%** (was 5.5%) — absorbs intraday noise during days 1-7
- **At day 7: tightens automatically to 5.5%** via `_time_tighten_stops` in monitor
- 3.0% defensive (CORRECTION/BEARISH/PANIC/DISTRIBUTION)
- ATR floor 0.9×, regime mult (UP=1.20, SIDEWAYS=1.00, DISTRIBUTION=0.85, defensive=0.70)
- Then standard RATCHET tightens further as profit accumulates (T0 +10% → 5%, T1 +14% → 4.5%, T2 +22% → 3.5%, T3 +30% → 2.5%)
- Ratchet is idempotent (only tightens, never loosens — enforced by N4 guard in `modify_trailing_pct`)
- **DEFERRED (do NOT deploy without freeze exit)**: T-early tier at +5% peak, and T0 threshold drop to +7% (pipeline-deep-dive-jun26 suggestion). Both require 15-20 new closes as validation before deployment.
- Reasoning: postmortem on 12 trail-fired losers showed 9/12 (75%) recovered above ENTRY within 14 days. The 5.5% trail was firing on natural intraday noise. Real-OHLC backtest: 5.5%+0d=-1.73%/trade OOS; 9%+0d=+0.42%; 9%→5.5% at day 7=+1.26% (chosen).
- Env: TRADE_MIN_INITIAL_TRAIL_PCT, TRADE_TIME_TIGHTEN_ENABLED, TRADE_TIME_TIGHTEN_DAYS, TRADE_TIME_TIGHTEN_TARGET_PCT

**BREAK_EVEN: DISABLED** (backtest showed net -$74.81/$1k/trade — was a bad anecdote-based feature)

**DAY-N MOMENTUM KILL: ENABLED** (NEW 2026-07-17, `TRADE_DAY_N_KILL_ENABLED=1`)
- Rule: if position age ≥ 2 days AND peak_gain_pct < 5% → force sell (market, with force_exit_via_trail fallback for sub-$2k Error 201)
- Simulated on 16 real closed trades with Polygon 1-min bars: baseline -$50 → with kill +$30 (delta +$80). Real ledger was -$95 → sim projects +$30 (delta +$125).
- Fires ONCE per position (marks `day_n_kill_fired_at`)
- Rationale: losers mean hold 5.2d vs winners 8.9d; 7/10 real losers exited by day 4 with peaks +0-4%. No trail can catch these — momentum-selection layer needed.
- Small sample (n=16) → env kill-switch: `TRADE_DAY_N_KILL_ENABLED=0` to disable, `TRADE_DAY_N_KILL_CUTOFF_DAYS=X`, `TRADE_DAY_N_KILL_MIN_GAIN_PCT=X` to tune.

**ANALYST PT VETO (HARD, current default):**
- Rule: if analyst_mean_PT < current_price AND n_analysts ≥ 3 → SKIP the trade (in `order_manager._cap_target_with_analysts`)
- Env: `TRADE_ANALYST_VETO_OVERVALUED=1` (default). Set to 0 for SOFT mode (still buys, caps target at max(analyst_high, current × 1.06))
- **Data gap**: analyst PT NOT saved to scan_outcomes — impossible to backtest currently. yfinance query at scan time is too slow for full backtest (200+ tickers → 2min+ timeout).
- Code author's own note (order_manager.py:141-147): "systematically rejects the strongest MOMENTUM names — they often run ABOVE slow-to-update analyst targets, and we VALIDATED (out-of-time corr +0.22) that our score predicts returns regardless of analyst PT."
- **Real observed impact 2026-07-17**: 3 straight zero-buy scan cycles due to this + volume_surge gates. Not a bug — the system rejecting overvalued names is by design. If we want to unblock, flip env flag (reversible in seconds). NOT flipped as of 2026-07-17 (no historical evidence either way).
- **TODO to fix data gap**: modify `scripts/track_scan_outcomes.py` to fetch + save analyst PT with each recorded candidate. Then after 1-2 months of collection, backtest whether veto helps or hurts. NOT yet done.

**Selection ranking weights (REBALANCED 2026-06-26 + ELITE + SPEC + CHAMPION bonus):**
- **Fundamental: 25%** (NEW 2026-06-26 — corr +0.117 SIG, top-3 by fund alone = +8.98%)
- **RR: 30%** (was 40%, then 30% after 26/6)
- **ML: 20%** (was 25%)
- **ATR: 10%** (was 15%)
- **Technical: 5%** (NEW 2026-06-26 — top-3 by tech alone = +6.72%)
- **ELITE bonus: 5%** (NEW 2026-07-03 — mask ×1 if fund≥45 AND tech≥60 AND vs<1.0. ELITE cohort in backtest = +10.69% mean, WR 85%)
- **SPEC bonus: 5%** (NEW 2026-07-17 — mask ×1 if Risk_Level=speculative. Real Supabase n=80: +3.98% p<0.001 vs core +0.33% p=0.426. Delta +$3.65/trade. Env kill: TRADE_SPEC_BONUS_WEIGHT=0)
- **SECTOR CHAMPION bonus: 5%** (NEW 2026-07-21 — mask ×1 if (sector, score) matches a strong cohort. Real Supabase n=402: Energy 70-85 (n=49, WR 74-79%, mean +2.9 to +5.2%), Technology 70-85 (n=25, WR 70-75%, mean +3.9 to +6.0%), Healthcare 70-75 (n=13, WR 84.6%, mean +2.69%). Env kill: TRADE_SECTOR_CHAMPION_WEIGHT=0. See core/trading/sector_champion.py.)
- Score: 0% (was 5%, anti-predictive at ranker level even though it's the gate floor)
- Volume_surge INVERTED: 5% (2026-07-03 — corr -0.117 SIG within window)
- Sector momentum: 5% | Insider boost: 5% | SPY momentum: 5% when available

**BUY ALERT ENHANCEMENT (NEW 2026-07-21):**
- Header shows 🏆 CHAMPION / 💎 ELITE / 💎🏆 ELITE+CHAMPION when cohorts match
- New `Cohort:` line displays historical n/WR/mean from real Supabase closes
- ML deadzone warning downgraded to `(mid)` when in Champion cohort
- Motivation: prior alerts over-emphasized weaknesses (e.g. "ML deadzone") without showing empirical STRENGTH from historical cohort match. VG buy (2026-07-21) triggered this — 91% WR cohort was invisible while ML flag was prominent.

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
**Reversibility of recent changes:** all gates env-overridable (TRADE_MIN_SCORE, TRADE_MIN_ML_PROB, TRADE_MAX_ML_PROB, TRADE_MIN_RR, TRADE_MAX_RR, TRADE_MIN_ATR_PCT, TRADE_BLOCKED_SECTORS, TRADE_BLOCKED_REGIMES, TRADE_BREAK_EVEN_ENABLED, TRADE_LEDGER_ENABLED, TRADE_DAY_N_KILL_ENABLED, TRADE_SPEC_BONUS_WEIGHT, TRADE_ANALYST_VETO_OVERVALUED). Selection weights live in `order_manager.py` (need code change to revert).

**DRY_RUN safety (fixed 2026-07-17):**
- `_execute_single` returns early on DRY_RUN → no tracker write, no notify_buy (prevents phantom positions blocking real slots)
- `notify_buy` prepends `[DRY] ` tag when TRADE_DRY_RUN=1 (prevents Telegram confusion)
- Incident that triggered these fixes: verify-only DRY_RUN added MRX to tracker with order_ids=0, blocked a real trading slot until manual reset.

**Healthcheck auto-heal (fixed 2026-07-17):**
- `deploy/healthcheck.sh` detects stuck monitor (snapshot > 660s old) and Telegram-alerts "Auto-recovered: pkill -KILL + systemctl restart".
- **PRIOR BUG**: alert said "Auto-recovered" but sudo commands SILENTLY FAILED — `stockscout` user was not in sudoers. Real behavior: monitor stayed stuck; only 0-position luck prevented incident.
- Fixed by adding `/etc/sudoers.d/stockscout` with NOPASSWD for exactly: `pkill -KILL -u stockscout -f monitor_positions`, `systemctl restart stockscout-monitor{,.service}`, `systemctl restart stockscout-statusbot{,.service}`. Verified live: systemctl restart worked without password.
- Incident that revealed this: 2026-07-17 17:30 STALE alert during quiet 0-position window (safe, but auto-heal fake).

### Data Collection Status (as of 2026-07-17)
**Collected (backtestable):**
- `data/trades/executions.jsonl` — IB fills, ledger source of truth (34 fills through 2026-07-17)
- `data/trades/open_positions.json` — tracker (currently cleaned to match IB; may drift if monitor's reconcile misses a sale — historical drift observed with TEO 7/16)
- `data/outcomes/scan_outcomes.jsonl` — 2990 scan candidates with market context, scores, ML prob, fund/tech, RSI, ATR, volume_surge, sector. Resolved with return_5/10/20d.
- Supabase `portfolio_positions` — 402 closed positions with realized return, peak, exit_reason, sector, final_score, risk_class (but recommendation_id + scan_id fields are EMPTY — can't join to features)
- Supabase `scan_recommendations` — has full feature set but historical coverage very sparse (only 2 of our 16 recent trades matched)
- Local `data/scans/*.parquet` — 111 scan snapshots (each ~130 columns of features)
- Local `data/stockscout.duckdb` — 334 recs+outcomes rows but mostly early test data

**Data GAPS (would improve future backtests):**
- Analyst PT snapshot at scan/trade time — currently fetched live from yfinance, never persisted. Blocks `TRADE_ANALYST_VETO_OVERVALUED` backtest.
- Intraday peak per closed position — currently reconstructed on-demand via Polygon 1-min bars (rate-limited 5/min = ~4min per 20 tickers). Blocks fast peak-giveback tuning.
- Peak timing (hours from entry to peak) — has to be pulled fresh each analysis.
- Supabase `portfolio_positions.recommendation_id` + `.scan_id` fields are empty for all rows — breaks joins to feature-rich `scan_recommendations`.

### API Keys (for reference)
Local `.env` has all 10 provider keys (FMP, POLYGON, FINNHUB, TIINGO, ALPHA_VANTAGE, EODHD, SIMFIN, NASDAQ, MARKETSTACK, OPENAI). Same keys in GH secrets for CI scans. VPS `.env.trading` intentionally does NOT have them (VPS runs auto_trade + monitor, does NOT scan). FMP legacy `/api/v3/historical-chart/1min` deprecated 2026-07 → use `/stable/historical-chart/1min?symbol=X`. Polygon 1-min via `/v2/aggs/ticker/X/range/1/minute/from/to`.

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

### VPS Pipeline schedule (3x weekday, documented rationale in stockscout-pipeline.timer)
- **13:30 UTC / 09:30 ET / 16:30 IL** — Pipeline #1 (MARKET OPEN, overnight news reactions)
- **15:00 UTC / 11:00 ET / 18:00 IL** — Pipeline #2 (OPENING RANGE BREAKOUT, highest-edge swing window)
- **19:15 UTC / 15:15 ET / 22:15 IL** — Pipeline #3 (POWER HOUR, institutional flow + overnight gap potential)
- Each pipeline: (1) preflight capacity check → (2) trigger GH Actions scan via repository_dispatch → (3) poll for new parquet (30s cadence, 150min max) → (4) run auto-trade → (5) record outcomes.

### Pipeline efficiency + smart-retry (NEW 2026-07-23)
- **Preflight skip (task #143)**: `scripts/preflight_pipeline.py` runs BEFORE GH Actions dispatch. If `n_positions >= MAX_OPEN_POSITIONS` OR `cash < min_viable_position_usd` → skip entire scan (saves ~45min compute × wasted-empty run). Kill switch: `TRADE_SKIP_WHEN_FULL=0`. If IB unreachable → PROCEED conservatively.
- **Immediate re-eval after adaptive activation (task #144)**: when `_record_adaptive_outcome` newly-activates a relax flag (confidence or analyst_pt), the current run recursively retries `execute_recommendations(_adaptive_retry=True)` on the SAME scan_df. This catches candidates the just-relaxed gate would allow WITHOUT waiting 1-4h for the next pipeline. Max 1 retry per invocation (loop protection). See `core/trading/order_manager.py:execute_recommendations`.

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
