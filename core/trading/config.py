"""Trading configuration — all user-adjustable parameters.

Override via environment variables (prefixed TRADE_) or by editing this file.
Safety-critical defaults: DRY_RUN=True, PAPER_MODE=True.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str) -> str:
    return os.getenv(f"TRADE_{key}", default)


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(f"TRADE_{key}")
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes")


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(f"TRADE_{key}", str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(f"TRADE_{key}", str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class TradingConfig:
    """Central trading configuration."""

    # ── Safety ──────────────────────────────────────────────────
    dry_run: bool = field(default_factory=lambda: _env_bool("DRY_RUN", True))
    paper_mode: bool = field(default_factory=lambda: _env_bool("PAPER_MODE", True))

    # ── IBKR Connection ────────────────────────────────────────
    ibkr_host: str = field(default_factory=lambda: _env("IBKR_HOST", "127.0.0.1"))
    ibkr_port_paper: int = 7497
    ibkr_port_live: int = 7496
    ibkr_client_id: int = field(default_factory=lambda: _env_int("IBKR_CLIENT_ID", 1))
    ibkr_timeout: int = 30  # seconds

    # ── Position Sizing ────────────────────────────────────────
    # Audit H1 (2026-05-01): defaults rebalanced for the ~$1k account.
    # Old defaults (3 × $300 = $900 exposure) committed 90% of capital
    # and paid $1 commission per trade = 0.67% structural drag at $300.
    # New defaults (2 × $400 = $800 exposure) keep the same total
    # exposure ratio (80% vs 90%), commit to fewer concurrent bets so
    # entries can stagger by a day (decorrelating timing risk on
    # volatile open-prints), and reduce commission drag to 0.50% at
    # $400. Production VPS uses .env.trading which overrides these
    # defaults — to apply the new sizing on production, update
    # /home/stockscout/stock-scout-2/.env.trading explicitly.
    max_position_size: float = field(
        default_factory=lambda: _env_float("MAX_POSITION_SIZE", 400.0)
    )
    max_open_positions: int = field(
        default_factory=lambda: _env_int("MAX_OPEN_POSITIONS", 2)
    )
    max_daily_buys: int = field(
        default_factory=lambda: _env_int("MAX_DAILY_BUYS", 2)
    )
    max_portfolio_exposure: float = field(
        default_factory=lambda: _env_float("MAX_PORTFOLIO_EXPOSURE", 800.0)
    )
    cash_reserve: float = field(
        default_factory=lambda: _env_float("CASH_RESERVE", 20.0)
    )  # Keep this much cash aside (buffer for fees, slippage)
    # 2026-05-29: minimum deployable cash (after the sub-$2k buffer) below
    # which the candidate-evaluation loop stops early. Prevents spending
    # minutes evaluating 100+ candidates that can't be afforded after cash
    # is exhausted — that delay was blocking the monitor snapshot write and
    # tripping the stale auto-recover (which SIGKILLed a healthy monitor).
    min_viable_position_usd: float = field(
        default_factory=lambda: _env_float("MIN_VIABLE_POSITION_USD", 30.0)
    )
    # 2026-05-29: grace window (sec) before drift_check flags a position for
    # "NO active stop". A freshly-bought position's OCA bracket may not be in
    # the cycle's order snapshot yet (race with the opportunistic buy that
    # placed it), causing a false "DRIFT DETECTED" the moment after a buy.
    # Skip drift checks for positions younger than this; self-resolves next
    # cycle. Env: TRADE_DRIFT_FRESH_GRACE_SEC.
    drift_fresh_grace_sec: float = field(
        default_factory=lambda: _env_float("DRIFT_FRESH_GRACE_SEC", 300.0)
    )

    # ── Circuit Breakers (stop new buys when losses pile up) ──
    max_daily_loss_pct: float = field(
        default_factory=lambda: _env_float("MAX_DAILY_LOSS_PCT", 2.0)
    )  # Stop new buys if today's realized+unrealized P&L < -X% of portfolio
    max_drawdown_pct: float = field(
        default_factory=lambda: _env_float("MAX_DRAWDOWN_PCT", 10.0)
    )  # Pause trading if portfolio is down >X% from peak

    # ── Sector Concentration ─────────────────────────────────────
    # 2026-05-19: tightened from 2 → 1 after Monday's run produced
    # ARWR + ACHC (both Healthcare = 67% of portfolio). With max=2 the
    # cap allowed concentration; max=1 forces every new buy into a
    # different sector. Override via MAX_SECTOR_POSITIONS env var if
    # you want to relax during specific regimes.
    max_sector_positions: int = field(
        default_factory=lambda: _env_int("MAX_SECTOR_POSITIONS", 1)
    )  # Max # of positions in same sector (1 = strict diversification)

    # ── Portfolio Correlation ─────────────────────────────────────
    # Block a new entry if adding it would push the portfolio's mean
    # pairwise correlation above the threshold. Keeps the book diversified
    # even when sector labels hide related exposures (e.g. TDW + SLB are
    # both labeled "Energy" but CVX + SLB are not — MAX_PORTFOLIO_CORR
    # catches the deeper relationship).
    max_portfolio_correlation: float = field(
        default_factory=lambda: _env_float("MAX_PORTFOLIO_CORR", 0.65)
    )  # Mean pairwise 60-day correlation — above this, skip new buys.
    correlation_check_enabled: bool = field(
        default_factory=lambda: _env_bool("CORRELATION_CHECK_ENABLED", True)
    )

    # ── Trade Filters ──────────────────────────────────────────
    # 2026-06-05 EVENING REVERT: I tightened these earlier today based on a
    # backtest of the 16k training dataset, but a later investigation found
    # the analysis rested on a wrong assumption — the 16k dataset's
    # AdjustedScore_20d / ML_20d_Prob are DIFFERENT metrics from the LIVE
    # FinalScore_20d / ML_20d_Prob the production system uses. Specifically:
    #   • ML distribution in training: mean 0.10, in live: mean 0.42 (4x shift)
    #   • Training had AdjustedScore_20d (range 0-200+); live uses
    #     FinalScore_20d (range 0-100) — a 25/30/25/20 weighted sum of
    #     fundamental/momentum/RR/reliability, computed differently.
    # When I backtested on actual live scan history (55 scans, 5172 ticker-
    # date pairs, Feb-May 2026), FinalScore_20d was found to be NEGATIVELY
    # correlated with forward 20d returns (corr = -0.13) — high-score
    # stocks underperformed. ML showed a non-monotonic curve (sweet spot
    # 0.40-0.50, terrible >0.55). Only RR had positive predictive power.
    # ⇒ Reverting all 3 changes to the prior defaults until a more robust
    # investigation can be done (need ≥6 months of scan history, decomposed
    # by component, OOS validated). Don't trust today's "tighter" numbers
    # without that work. See conv 2026-06-05 + /tmp/live_backtest.py.
    min_score_to_trade: float = field(
        default_factory=lambda: _env_float("MIN_SCORE", 73.0)
    )
    max_score_to_trade: float = field(
        default_factory=lambda: _env_float("MAX_SCORE", 85.0)
    )  # 2026-07-03 EVENING REVISION: was 95, briefly 200 (based on
       # scan_outcomes.jsonl), now 85 (based on REAL portfolio_positions).
       # REAL trades data (n=402 closed) reveals sweet spot below 85:
       #   Score 73-79: n=104  mean=+3.05%  WR=67.3%
       #   Score 80-83: n=42   mean=+4.72%  WR=69.0% ← BEST
       #   Score 84-85: n=9    mean=-0.60%  WR=44.4% ← rolls over
       #   Score 85-89: n=20   mean=-0.38%  WR=40.0% ← BAD
       #   Score 90-94: n=10   mean=-0.48%  WR=50.0% ← BAD
       #   Score >=94:  n=2    mean=-3.79%  WR=0.0%  ← WORST
       # Simulated scan_outcomes had said high-score = best. Real trades
       # with real trail-stop execution disagree. Trust real data.
       # Env: TRADE_MAX_SCORE=200 to disable cap.
    # 2026-06-05 LATE NIGHT: re-calibrated on REAL production scan data
    # (Supabase: 18,709 scan recommendations + actual yfinance forward returns,
    # OOS validated on n=170 held-out trades).
    #
    #     STRATEGY                                  n     mean    WR    OOS
    #   Baseline (random pick)                  18709   +0.97%  47.6%
    #   OLD PROD (s≥73, ML≥0.33, RR≥2.0)        1957   +2.62%  53.0%
    #   FULL CALIBRATION ↓                       243  +10.26%  67.1%   +9.78% / 61% (n=170 OOS)
    #
    # Where FULL CALIBRATION = sweet-spot windows for ML, RR, ATR + expanded
    # sector block list + unblock CORRECTION regime (data showed +5.48%/55% WR).
    min_rr_to_trade: float = field(
        default_factory=lambda: _env_float("MIN_RR", 2.5)
    )  # 2026-07-03 EVENING (ef9d859+): lowered 3.0 → 2.5 based on backtest
       # under NEW gates (score>=73, ml 0.40-0.60, fund>=40, 2169 rows):
       #   RR >= 2.0: n=670  mean=+2.56%  WR=47.3%
       #   RR >= 2.5: n=364  mean=+2.92%  WR=44.2% ← BEST
       #   RR >= 3.0: n=292  mean=+2.76%  WR=42.5%  (was cutting 25% of best)
       #   RR >= 4.0: n=144  mean=+2.02%  WR=36.8%
       # Old comment claimed RR 4-5 best (+3.81%) but that was under FUND>=30
       # gates. Under FUND>=40, higher RR correlates with WORSE outcomes.
       # Env: TRADE_MIN_RR=3.0 to revert.
    max_rr_to_trade: float = field(
        default_factory=lambda: _env_float("MAX_RR", 5.0)
    )  # 2026-07-09 RE-INSTATE: fresh backtest on 2588 resolved trades
       # (was 2169 last calibration → +419 new samples) revealed the
       # RR 5-7 zone is CATASTROPHIC:
       #   RR 2.5-3.0: n=20   mean=+5.47%  WR=60.0%  ← BEST
       #   RR 3.0-4.0: n=43   mean=+2.28%  WR=44.2%
       #   RR 4.0-5.0: n=29   mean=+2.53%  WR=41.4%
       #   RR 5.0-7.0: n=18   mean=-3.49%  WR=0.0%   ← DISASTER
       #   RR 7.0+:    n=16   mean=+2.89%  WR=25.0%  (mostly noise, small n)
       # Cap at 5.0 blocks the disaster zone. Previously assumed RR 7+
       # was fine (jun24 data) — new data disproves.
       # TRADE_MAX_RR=0 to disable cap.
    min_confidence: str = field(
        default_factory=lambda: _env("MIN_CONFIDENCE", "High")
    )
    confidence_regime_relax: bool = field(
        default_factory=lambda: _env_bool("CONFIDENCE_REGIME_RELAX", False)
    )  # NEW 2026-07-03 — opt-in flag to preserve old bullish-regime relaxation
       # (High → Medium in TREND_UP/MODERATE_UP). Default DISABLED — CONFIG
       # is the hard floor. Set TRADE_CONFIDENCE_REGIME_RELAX=1 to re-enable
       # the old permissive behavior for testing.
    min_ml_prob: float = field(
        default_factory=lambda: _env_float("MIN_ML_PROB", 0.40)
    )  # Was 0.33; ML 0.20-0.30 = -1.16% (n=1256), ML 0.40-0.45 = +2.58%, ML 0.45-0.50 = +5.07% (BEST)
    max_ml_prob: float = field(
        default_factory=lambda: _env_float("MAX_ML_PROB", 0.55)
    )  # NEW gate; ML > 0.55 underperforms (likely model over-confidence on extended stocks)
    min_fundamental_score: float = field(
        default_factory=lambda: _env_float("MIN_FUNDAMENTAL_SCORE", 40.0)
    )  # 2026-07-03 RAISED 30 → 40 based on scan_outcomes.jsonl (2169 trades):
       #   Fund >= 30: n=1187  mean=+1.74%  WR=41.4%
       #   Fund >= 40: n= 686  mean=+2.54%  WR=47.2% ← +0.80pp mean, +5.8pp WR
       #   Fund 40-60: n= 559  mean=+3.16%  WR=50.4% ← sweet spot
       # We keep <60 uncapped for now (n>60 sample small); tighten after data.
       # Env: TRADE_MIN_FUNDAMENTAL_SCORE=30 to revert.
    max_volume_surge: float = field(
        default_factory=lambda: _env_float("MAX_VOLUME_SURGE", 1.5)
    )  # NEW 2026-06-26 — counter-intuitive but data is clear (n=303,
       # p=0.04 SIG): high volume_surge predicts LOWER returns. Buckets:
       # vs<0.5: +11.15%; 0.5-1: +6.48%; 1-1.5: +4.97%; 1.5-2: +0.70%;
       # >=2: -6.12%. Blocking vs>=1.5 removes ~5% of candidates that
       # collectively lose money. ENV: TRADE_MAX_VOLUME_SURGE (0 = disable).
    max_slippage_pct: float = field(
        default_factory=lambda: _env_float("MAX_SLIPPAGE_PCT", 3.0)
    )  # NEW 2026-06-26 — was hardcoded 5.0 in order_manager. Tightened
       # to 3.0 (data: 34% of 29 OPENs had >2% slippage, mean 1.79%).
       # ENV: TRADE_MAX_SLIPPAGE_PCT for flexibility.
    min_atr_pct: float = field(
        default_factory=lambda: _env_float("MIN_ATR_PCT", 0.03)
    )  # 2026-06-12 LOOSENED 0.04 → 0.03. At 0.04 we blocked 58% of universe
       # (live verified: 127/304 in latest scan pass; 5/8 days no buys at all).
       # Original data: ATR 3-4% = +0.68% (FIRST positive band) — that's
       # where the gate should sit, not at 4% (which removed half the universe).
       # policy.evaluate_static_gates ALSO treats atr==0 as missing data
       # (not low-vol). Upstream provider failures shouldn't kill picks.
    min_reliability: float = field(
        default_factory=lambda: _env_float("MIN_RELIABILITY", 50.0)
    )  # Filter stocks with incomplete data (Reliability_Score < 50)

    # ── Market Regime Gate ────────────────────────────────────
    # 2026-06-05: UNBLOCKED "CORRECTION" — Supabase data shows
    # CORRECTION regime returns +5.48% (n=527, WR 55.8%). Likely the model
    # picks oversold winners during corrections (mean-reversion). PANIC
    # stays blocked (-2.06%, WR 29%, n=55). TREND_UP shows -0.83% but
    # n=5492 is too large to ignore — investigate but don't block yet.
    blocked_regimes: str = field(
        default_factory=lambda: _env("BLOCKED_REGIMES", "PANIC")
    )  # Comma-separated list of regime names to block entirely
    reduce_regimes: str = field(
        default_factory=lambda: _env("REDUCE_REGIMES", "DISTRIBUTION")
    )  # Regimes where we still trade but smaller (half size)

    # ── Sector Blocklist ─────────────────────────────────────
    # 2026-06-05: EXPANDED based on 18,709 Supabase scan recs + actual
    # 20-day forward returns. Sectors with statistically meaningful (n≥100)
    # negative mean returns:
    #   Consumer Defensive  -2.75% (n=978)   ← already blocked
    #   Utilities           -3.70% (n=931)   ← NEW
    #   Communication       -5.69% (n=199)   ← NEW (NOT Communication Services)
    #   Materials           -2.01% (n=521)   ← NEW
    #   Basic Materials     -1.86% (n=1387)  ← NEW
    #   Consumer Cyclical   -2.33% (n=1827)  ← NEW
    #   Financial           -1.17% (n=303)   ← NEW
    #   Financial Services  -0.92% (n=1801)  ← NEW
    # Top performers KEPT: Technology +10.42%, Healthcare +2.79%,
    # Industrials +2.57%, Communication Services +1.97%, Energy +1.65%.
    blocked_sectors: str = field(
        default_factory=lambda: _env("BLOCKED_SECTORS",
            "Consumer Defensive,Utilities,Communication,Materials,"
            "Basic Materials,Financial,Energy,Real Estate")
    )  # 2026-07-08 SYNC: default was stale (blocked Consumer Cyclical and
       # Financial Services — both POSITIVE per real trade data). VPS env
       # override was correct; brought code default in sync with CLAUDE.md
       # + scan_outcomes data:
       #   BLOCKED: Cons Defensive (-2.36%), Utilities (-0.42%),
       #            Materials (-7.36%), Basic Materials (-2.51%),
       #            Financial (-3.82%), Energy (-1.59% SIG),
       #            Real Estate (-3.25%), Communication (blocked)
       #   UNBLOCKED (wrongly blocked before): Consumer Cyclical (+3.06%),
       #             Financial Services (+4.54%)


    # ── Earnings calendar gate (binary-risk reduction) ─────────
    # Block buys when an earnings announcement is within the window.
    # Earnings gaps are the #1 source of catastrophic losses for swing
    # strategies — a 5% TRAIL won't save you from a -22% overnight gap.
    earnings_gate_enabled: bool = field(
        default_factory=lambda: _env_bool("EARNINGS_GATE_ENABLED", True)
    )
    earnings_block_days: int = field(
        default_factory=lambda: _env_int("EARNINGS_BLOCK_DAYS", 5)
    )  # Audit H3 (2026-05-01): widened from 3 to 5 days. Options-implied
       # gap risk prices in at T-5; intraday volatility lifts at T-1; the
       # report itself can produce 15-25% gap-down on miss. 3-day window
       # only covered the report itself, not the lead-in elevated-vol period.

    # Graduated earnings-proximity tightening (added 2026-05-15).
    # The block-days gate prevents NEW buys near earnings, but positions
    # opened weeks earlier need defensive tightening as the event nears.
    # Tighter trail as the binary-risk window approaches:
    #   T-4..-5 days: trail floor 3.0% (defensive warning)
    #   T-2..-3 days: trail floor 2.5% (mid-zone)
    #   T-0..-1 days: trail floor 2.0% (final-zone)
    # Each level is a MAX trail (we'll only tighten, never loosen) — so if
    # ratchet already set tighter (e.g., 2.0% at +28% peak), no change.
    earnings_proximity_enabled: bool = field(
        default_factory=lambda: _env_bool("EARNINGS_PROXIMITY_ENABLED", True)
    )
    earnings_proximity_warn_days: int = field(
        default_factory=lambda: _env_int("EARNINGS_PROXIMITY_WARN_DAYS", 5)
    )   # Warn band — trail floor 3.0%
    earnings_proximity_warn_trail_pct: float = field(
        default_factory=lambda: _env_float("EARNINGS_PROXIMITY_WARN_TRAIL_PCT", 3.0)
    )
    earnings_proximity_mid_days: int = field(
        default_factory=lambda: _env_int("EARNINGS_PROXIMITY_MID_DAYS", 3)
    )   # Mid band — trail floor 2.5%
    earnings_proximity_mid_trail_pct: float = field(
        default_factory=lambda: _env_float("EARNINGS_PROXIMITY_MID_TRAIL_PCT", 2.5)
    )
    earnings_proximity_final_days: int = field(
        default_factory=lambda: _env_int("EARNINGS_PROXIMITY_FINAL_DAYS", 1)
    )   # Final band — trail floor 2.0%
    earnings_proximity_final_trail_pct: float = field(
        default_factory=lambda: _env_float("EARNINGS_PROXIMITY_FINAL_TRAIL_PCT", 2.0)
    )

    # ── Performance throttle (safety brake on losing streaks) ─────
    # Tracks the rolling win rate of the last N closed trades. When the
    # rate drops below thresholds, the system reduces exposure or halts
    # entirely. This protects against regime change / model drift
    # that the system itself can't detect — if recent live performance
    # is materially worse than expected, something is wrong; better to
    # cut size than to compound losses.
    throttle_enabled: bool = field(
        default_factory=lambda: _env_bool("THROTTLE_ENABLED", True)
    )
    throttle_window_trades: int = field(
        default_factory=lambda: _env_int("THROTTLE_WINDOW", 10)
    )   # Rolling window of last N closed trades for win-rate calc
    throttle_warn_winrate: float = field(
        default_factory=lambda: _env_float("THROTTLE_WARN_WIN_RATE", 0.30)
    )   # Below this → halve position sizes
    throttle_halt_winrate: float = field(
        default_factory=lambda: _env_float("THROTTLE_HALT_WIN_RATE", 0.20)
    )   # Below this → block all new buys
    throttle_min_trades: int = field(
        default_factory=lambda: _env_int("THROTTLE_MIN_TRADES", 5)
    )   # Need at least N trades before throttle kicks in (avoid early-noise)

    # ── EXPECTANCY-BASED THROTTLE (audit H2, 2026-05-01) ──
    # Win-rate alone misjudges a 2.0+ R:R strategy. A 30% WR strategy
    # with avg_win = $30 and avg_loss = $10 has expectancy
    #   = 0.3 × $30 − 0.7 × $10 = $9 − $7 = +$2/trade
    # which is profitable. The WR-based throttle (default 30% WARN)
    # punishes this strategy for behaving normally.
    #
    # When THROTTLE_MODE=expectancy, the throttle uses average per-trade
    # P&L instead of WR:
    #   - avg_pnl_pct >= warn → no throttle
    #   - 0 < avg_pnl_pct < warn (default 0%) → halve sizes
    #   - avg_pnl_pct <= halt (default -1.5%) → halt all new buys
    # When THROTTLE_MODE=winrate (default — backward-compat), uses the
    # legacy WR thresholds above.
    throttle_mode: str = field(
        default_factory=lambda: _env("THROTTLE_MODE", "expectancy")
    )   # 2026-07-08: switched default winrate → expectancy.
       # RR 2.5+ strategy is profitable at 35% WR (below the 30% warn
       # threshold). The WR throttle was firing on normal behavior;
       # expectancy tracks what actually matters: avg $ per trade.
       # Legacy: THROTTLE_MODE=winrate to revert.
    throttle_warn_expectancy_pct: float = field(
        default_factory=lambda: _env_float("THROTTLE_WARN_EXPECTANCY_PCT", 0.0)
    )   # avg pnl% under this → halve sizes (0 = unprofitable on average)
    throttle_halt_expectancy_pct: float = field(
        default_factory=lambda: _env_float("THROTTLE_HALT_EXPECTANCY_PCT", -1.5)
    )   # avg pnl% under this → halt buys (losing >1.5% per trade is severe)

    # ── Dynamic position sizing (by ML probability) ──────────────
    # Scale base position size by ML conviction. Higher-prob picks get
    # larger size, lower-prob picks smaller. Without this, we burn edge
    # by allocating equally across high and low conviction trades.
    # Formula: size = base × (1 + ml_sizing_slope × (ml_prob - ml_sizing_anchor))
    # Defaults: anchor 0.40, slope 2.0 → ML 0.50 = 1.20× base, ML 0.30 = 0.80×.
    # Capped at min_position_size on the low end and 1.3× max on the high end.
    ml_sizing_enabled: bool = field(
        default_factory=lambda: _env_bool("ML_SIZING_ENABLED", True)
    )
    ml_sizing_anchor: float = field(
        default_factory=lambda: _env_float("ML_SIZING_ANCHOR", 0.40)
    )
    ml_sizing_slope: float = field(
        default_factory=lambda: _env_float("ML_SIZING_SLOPE", 2.0)
    )
    ml_sizing_max_mult: float = field(
        default_factory=lambda: _env_float("ML_SIZING_MAX_MULT", 1.30)
    )   # Hard cap on multiplier — even ML 1.0 won't exceed this
    ml_sizing_min_mult: float = field(
        default_factory=lambda: _env_float("ML_SIZING_MIN_MULT", 0.70)
    )   # Hard floor — even ML 0.0 won't shrink below this

    # ── Opportunistic Trading (audit followup, 2026-05-05) ──
    # When a position closes intraday (TRAIL/LIMIT fires), the freed
    # cash sits idle until the next scheduled trade pipeline run
    # (13:30 or 17:30 UTC). On a 4h gap, that's potentially missing
    # the best opportunity of the day. Opportunistic trading watches
    # the monitor's close events and immediately re-evaluates the
    # most recent scan against the new account state — if there's
    # a candidate that NOW fits (cash freed, slot freed), buy it.
    #
    # Safety preserved:
    #   - Uses the same `evaluate_static_gates` + `risk_manager` that
    #     the scheduled pipeline uses (same gates, same throttle).
    #   - Only fires during regular trading hours (9:30-16:00 ET).
    #   - Only if the latest scan is < scan_max_age_hours old.
    #   - Respects max_open_positions and max_daily_buys (those count
    #     opportunistic buys too).
    #   - Cooldown: at most one opportunistic check per N seconds
    #     after a close, to prevent thrash on choppy intraday closes.
    opportunistic_buy_enabled: bool = field(
        default_factory=lambda: _env_bool("OPPORTUNISTIC_BUY_ENABLED", True)
    )
    opportunistic_buy_cooldown_sec: int = field(
        default_factory=lambda: _env_int("OPPORTUNISTIC_BUY_COOLDOWN_SEC", 300)
    )   # 5 min minimum between opportunistic triggers

    # ── Insider buying signal (SEC EDGAR Form 4) ─────────────────
    # Boost ranking for stocks where insiders (CEO/CFO/Director) bought
    # ≥$50K worth of stock on the open market in the last 30 days. One
    # of the most robust alpha signals — 4-7% annualized outperformance.
    # Disabled when offline or rate-limited; fail-OPEN.
    insider_signal_enabled: bool = field(
        default_factory=lambda: _env_bool("INSIDER_SIGNAL_ENABLED", True)
    )

    # 2026-05-29: analyst "overvalued" veto toggle. Default True = current
    # behavior (skip stocks trading above analyst mean PT with ≥3 analysts).
    # Set TRADE_ANALYST_VETO_OVERVALUED=0 to instead trade those momentum
    # names with a capped target — an A/B lever we can't backtest (no
    # historical analyst data) so it must be measured forward.
    analyst_veto_overvalued: bool = field(
        default_factory=lambda: _env_bool("ANALYST_VETO_OVERVALUED", True)
    )

    # Liquidity floor — minimum average daily dollar volume (vol_avg × price)
    # for a candidate to be tradeable. Defense-in-depth for the marketable
    # LIMIT entry: thin names have wide spreads where the limit won't fill at
    # ref. Default $10M is a low bar that mid/large-caps clear easily and
    # only excludes genuinely thin micro-caps. Env: TRADE_MIN_ADDV_USD.
    min_addv_usd: float = field(
        default_factory=lambda: _env_float("MIN_ADDV_USD", 10_000_000.0)
    )

    # ── Entry execution (2026-05-29: kill slippage) ──────────────
    # Live trade analysis over 11 buys (2026-04 → 2026-05) showed the
    # MARKET buy paid +1.93% avg slippage = $74.32 total = 106% of the
    # entire net realized profit ($69.84). Worst offenders were illiquid
    # names: ANDG +5.64% ($24.59), SOLS +4.88% ($16.65). Fix: place a
    # MARKETABLE LIMIT instead of a MARKET order. The limit sits a small
    # buffer above the live ask so it fills immediately on a normal
    # spread, but CAPS the worst-case fill price. If the price has run
    # away (e.g. live-price fetch failed and we fell back to a stale
    # scan price, or the stock gapped), the limit simply doesn't fill and
    # we skip the trade — far better than chasing it 5% higher with a MKT.
    entry_use_limit: bool = field(
        default_factory=lambda: _env_bool("ENTRY_USE_LIMIT", True)
    )
    # Buffer above the reference (live) price for the marketable limit.
    # 0.3% is enough to clear a normal NBBO spread on a liquid name while
    # capping slippage well below the 1.93% we were paying. Illiquid names
    # with wider spreads simply won't fill — which the liquidity filter
    # (Stage B) will keep out of the candidate set anyway.
    entry_limit_buffer_pct: float = field(
        default_factory=lambda: _env_float("ENTRY_LIMIT_BUFFER_PCT", 0.3)
    )
    # How long to wait for the limit to fill before cancelling + skipping.
    # Swing entries are not time-critical to the second; if it doesn't
    # fill in this window the price moved against us and we'd rather pass.
    entry_limit_fill_wait_sec: int = field(
        default_factory=lambda: _env_int("ENTRY_LIMIT_FILL_WAIT_SEC", 20)
    )

    # ── Stop / Target ─────────────────────────────────────────
    trailing_stop_pct: float = field(
        default_factory=lambda: _env_float("TRAILING_STOP_PCT", 5.0)
    )
    use_pipeline_stop: bool = field(
        default_factory=lambda: _env_bool("USE_PIPELINE_STOP", False)
    )  # If True, use StopLoss from scan instead of trailing %

    # Minimum INITIAL trail % at buy time. Prevents trades from being
    # stopped out by intraday noise on day 1. Trade history (2026-04-15
    # → 2026-05-05) showed 3/6 trades closed within 0-2 days at -0.7%
    # to -3.0% — pure noise on a 20-day-swing thesis. The ratchet still
    # tightens BELOW this floor later (tier 1 = 4%, tier 2 = 3%, tier 3
    # = 2%) once peak gains earn that protection.
    # 2026-05-29 RAISED 4.0 → 5.5 after a 21-trade backtest on real OHLC.
    # Fixed-width trail simulation across all closed trades:
    #     3% → net -$8 | 4% → -$35 | 5% → -$24 | 5.5% → +$15 (best realized)
    #     6% → +$218 (incl. 4 monster runners that never stopped out)
    # The actual effective trail (~3-3.5%) was NET NEGATIVE — it shook us
    # out of the exact trends the momentum strategy exists to capture
    # (ORCL clipped at +4% then ran to +16%; ILMN clipped at +1.3% then
    # ran +60%). 5.5% is the conservative choice: the best fully-realized
    # width, without over-fitting to the 4 lucky runners that drive the 6%
    # number. The ratchet (below) still tightens to lock gains once a
    # position is a real winner, and regime_mult ×0.70 auto-tightens this
    # in CORRECTION/PANIC.
    #
    # 2026-06-10: WIDENED 5.5% → 9.0% based on real-OHLC trail simulation
    # on 512 production trades (Mar-Jun 2026). Findings:
    #   • At 5.5% trail, OOS returns -1.73%/trade (the system was bleeding)
    #   • At 9.0% trail, OOS returns +0.42%/trade
    #   • At 5.5% trail + 7d min-hold, OOS +1.26% (best — Phase B)
    # Of 12 recent losers, 9 (75%) recovered above ENTRY within 14 days —
    # the 5.5% trail was firing on natural intraday noise BEFORE the
    # 20-day swing strategy could play out.
    # Risk: max single-trade loss now -9% × $400 = -$36 (was -$22).
    # But sim showed 0/512 trades lost >10%, so risk is bounded.
    # Env: TRADE_MIN_INITIAL_TRAIL_PCT to revert.
    min_initial_trail_pct: float = field(
        default_factory=lambda: _env_float("MIN_INITIAL_TRAIL_PCT", 9.0)
    )
    # 2026-05-29: lower floor for DEFENSIVE regimes (CORRECTION/BEARISH/
    # PANIC/DISTRIBUTION). The wide 5.5% floor is for normal/bull markets
    # (let winners run). In a falling market the regime_mult ×0.70 tightens
    # the stop to preserve capital — a flat 5.5% floor would override that.
    # This defensive floor (3.0%) lets the tightening work, so the trail is
    # "wide in good markets, tight in bad". Env: TRADE_MIN_INITIAL_TRAIL_PCT_DEFENSIVE.
    min_initial_trail_pct_defensive: float = field(
        default_factory=lambda: _env_float("MIN_INITIAL_TRAIL_PCT_DEFENSIVE", 3.0)
    )

    # ── Accounting / reconciliation ───────────────────────────
    # Initial deposit. Used to reconcile the trade-log realized P&L against
    # IB's net liquidation: starting_capital + Σ(CLOSE pnl) + Σ(open
    # unrealized) should ≈ IB net-liq. A divergence beyond
    # reconcile_tolerance_usd means the ledger lost trades (reconcile_drop),
    # double-counted, or is gross-of-commission — surfaced in /pnl + daily
    # summary instead of silently overstating performance.
    # Env: TRADE_STARTING_CAPITAL, TRADE_RECONCILE_TOLERANCE_USD.
    starting_capital: float = field(
        default_factory=lambda: _env_float("STARTING_CAPITAL", 977.50)
    )
    reconcile_tolerance_usd: float = field(
        default_factory=lambda: _env_float("RECONCILE_TOLERANCE_USD", 5.0)
    )

    # ── Trail activation (P2 — OFF by default, see docs/trail_tuning.md) ──
    # When enabled, the monitor only begins *tightening* the trail (ratchet,
    # break-even, partial) once a position has earned a minimum gain — so a
    # fresh position is not whipsawed out by day-1 noise before the swing
    # thesis (5-15d) plays out. Default 0.0 = current behavior (no-op).
    # This NEVER widens past the initial protective bracket and never leaves
    # a position unprotected; it only gates the *tightening* passes.
    # Env: TRADE_TRAIL_ACTIVATION_ENABLED, TRADE_TRAIL_ACTIVATION_MIN_GAIN_PCT.
    trail_activation_enabled: bool = field(
        default_factory=lambda: _env_bool("TRAIL_ACTIVATION_ENABLED", False)
    )
    trail_activation_min_gain_pct: float = field(
        default_factory=lambda: _env_float("TRAIL_ACTIVATION_MIN_GAIN_PCT", 0.0)
    )

    # ── Partial Profit-Taking (sell half at intermediate target) ─
    partial_profit_enabled: bool = field(
        default_factory=lambda: _env_bool("PARTIAL_PROFIT_ENABLED", True)
    )
    partial_profit_trigger_pct: float = field(
        default_factory=lambda: _env_float("PARTIAL_PROFIT_TRIGGER_PCT", 12.0)
    )  # When unrealized PnL crosses this %, sell a portion.
    # Raised from 6% → 12% so it fires AFTER ratchet tier 1 (at +10%),
    # not before. Lets winners run through normal ATR noise.
    partial_profit_fraction: float = field(
        default_factory=lambda: _env_float("PARTIAL_PROFIT_FRACTION", 0.33)
    )  # Fraction of position to sell (0.33 = one-third).
    # Reduced from 0.5 so two-thirds ride the target after partial.

    # ── Tiered profit-taking ladder (preferred over single partial) ─
    # Sells fractional pieces at multiple peak-gain thresholds instead
    # of one big sell at +12%. Locks in cumulative gains while letting
    # the remaining quarter ride the TRAIL toward the full target.
    # Set ladder_enabled=True to use this; falls back to single-trigger
    # partial above if disabled. (Recommendation #3 from 2026-04-30 audit.)
    profit_ladder_enabled: bool = field(
        default_factory=lambda: _env_bool("PROFIT_LADDER_ENABLED", True)
    )
    # 2026-05-29: minimum ORIGINAL position qty for the ladder to engage.
    # Below this the 25% fraction can't be expressed cleanly (rounds to 0 or
    # is forced to >33% of the lot), and selling half of a 2-share winner at
    # +10% fights the wide-trail "let winners run" design. Small lots ride
    # the trail whole; ladder resumes when capital grows positions to ≥4 sh.
    ladder_min_qty: int = field(
        default_factory=lambda: _env_int("LADDER_MIN_QTY", 4)
    )
    # Each tier: (peak_gain_threshold_pct, fraction_of_ORIGINAL_position_to_sell).
    # Defaults sell 25% at +10%, +18%, +28% → 75% locked, 25% rides.
    profit_ladder_tier1_gain: float = field(
        default_factory=lambda: _env_float("LADDER_T1_GAIN", 10.0)
    )
    profit_ladder_tier1_fraction: float = field(
        default_factory=lambda: _env_float("LADDER_T1_FRAC", 0.25)
    )
    profit_ladder_tier2_gain: float = field(
        default_factory=lambda: _env_float("LADDER_T2_GAIN", 18.0)
    )
    profit_ladder_tier2_fraction: float = field(
        default_factory=lambda: _env_float("LADDER_T2_FRAC", 0.25)
    )
    profit_ladder_tier3_gain: float = field(
        default_factory=lambda: _env_float("LADDER_T3_GAIN", 28.0)
    )
    profit_ladder_tier3_fraction: float = field(
        default_factory=lambda: _env_float("LADDER_T3_FRAC", 0.25)
    )

    # ── Dynamic Stop Ratcheting (tighten the trailing stop as it runs up) ─
    # When peak gain crosses thresholds, MODIFY the existing TRAIL order's
    # trailingPercent on IB's server (it auto-updates from there — no need
    # for our monitor to push every tick).
    #
    # Why TRAIL-tighten instead of replacing with STP (old design until
    # 2026-04-28): the static STP froze the floor at the threshold. A
    # stock that peaked at +17.9% would only have +3% locked because we
    # were waiting for tier 2 (+18%) to bump the floor. With dynamic
    # trail, IB tracks the peak continuously, so every penny above the
    # threshold raises the stop too.
    # ── Time-based trail tightening (Phase B, 2026-06-10) ─────
    # After Phase A widened initial trail to 9% (capture intraday noise),
    # we want to TIGHTEN to a sane 5.5% once the position has had time to
    # establish (no longer in chop zone). Backtest on 512 trades:
    #   trail 9% alone:      +0.42% / trade OOS
    #   trail 5.5% + 7d hold: +1.26% / trade OOS  (+0.84pp from tightening)
    # The tightening only runs AFTER ratchet (so a ratchet-tightened
    # position e.g. at 2.5% never loosens to 5.5%).
    time_tighten_enabled: bool = field(
        default_factory=lambda: _env_bool("TIME_TIGHTEN_ENABLED", True)
    )
    time_tighten_days: int = field(
        default_factory=lambda: _env_int("TIME_TIGHTEN_DAYS", 7)
    )  # After position is this old, tighten trail to time_tighten_target_pct
    time_tighten_target_pct: float = field(
        default_factory=lambda: _env_float("TIME_TIGHTEN_TARGET_PCT", 5.5)
    )  # The "normal" trail width once chop window passes

    ratchet_enabled: bool = field(
        default_factory=lambda: _env_bool("RATCHET_ENABLED", True)
    )
    # Threshold (peak gain %) → trail % to set.
    # Lower tier % = tighter stop (closer to current price).
    # The starting trail is 3–8% (scan-derived). Ratchet only TIGHTENS,
    # never loosens, so a stock that started with TRAIL 3% never widens.
    # Ratchet thresholds STAGGERED 2pp ABOVE ladder thresholds.
    # Audit H8 (2026-05-01): when ladder and ratchet shared the same
    # gain trigger (e.g. both at +10%), a stock that peaked at +10%
    # then retraced 4% with normal noise would BOTH partial-exit AND
    # get stopped out of the rest. Now ladder fires first (lock in a
    # quarter), then ratchet fires later (tighten what's left).
    # Order on a +12% peak that retraces 4%:
    #     +10% → ladder partial 25%  (locked)
    #     +12% → ratchet to 4%       (tightens trail on remaining 75%)
    #     -4%  from peak → trail fires on remaining 75%
    # Tier 0 — EARLY LOCK (added 2026-05-05, REVISED 2026-05-07 after
    # backtest revealed it was killing winners).
    #
    # Backtest 2026-05-07 on 6 closed trades showed the original T0
    # (+5% → 3.5%) cut profit factor from 2.22 → 1.00 because it triggered
    # on first-day FOMO spikes, then exited on normal pullback. CF (+22.24
    # actual, 14-day hold) would have been stopped out at -2.24 on day 0.
    #
    # Two changes:
    #   1. Threshold raised +5% → +8% — places it above typical day-1
    #      volatility band (~3-5% intraday range for $400-priced names).
    #   2. Hold-days gate: T0 ONLY fires after the position has been held
    #      for `min_hold_days_for_t0` days. Lets early momentum prove
    #      itself before the ratchet engages. T1/T2/T3 still fire
    #      immediately at any age — those tiers are at peaks (10/18/28%)
    #      that already imply a meaningful run.
    # 2026-05-29: tier trails softened in lockstep with the 5.5% initial
    # floor. The backtest showed partial-tightening into the 4-5% "valley
    # of death" is the worst place to be — wide enough to give back the
    # quick profit, too tight to survive the pullback before the real run.
    # So tier trails now sit only modestly below the 5.5% initial and
    # only deep winners (+18%/+28%) get pulled meaningfully tighter to
    # lock genuine gains. T0 threshold raised 8 → 10 so it can't fire on
    # day-2 FOMO. All env-overridable.
    ratchet_tier0_gain: float = field(
        default_factory=lambda: _env_float("RATCHET_T0_GAIN", 10.0)
    )
    ratchet_tier0_trail_pct: float = field(
        default_factory=lambda: _env_float("RATCHET_T0_TRAIL_PCT", 5.0)
    )   # Peak +10% → trail 5.0% (gentle first lock, just below 5.5% initial)
    min_hold_days_for_t0: int = field(
        default_factory=lambda: _env_int("RATCHET_T0_MIN_HOLD_DAYS", 1)
    )   # 2026-07-08: lowered 2 → 1. Data from pipeline-deep-dive: 30% of
       # stop-fired reached +10% peak. Day-1 monsters (news, breakout)
       # peak fast then reverse — 1-day min still filters day-of buy
       # noise but catches next-day peaks. Env: RATCHET_T0_MIN_HOLD_DAYS.

    ratchet_tier1_gain: float = field(
        default_factory=lambda: _env_float("RATCHET_T1_GAIN", 14.0)
    )
    ratchet_tier1_trail_pct: float = field(
        default_factory=lambda: _env_float("RATCHET_T1_TRAIL_PCT", 4.5)
    )   # 2026-05-29: +10%→3.0% softened to +14%→4.5% — was choking
        # developing winners in the 4-5% valley of death (see backtest)
    ratchet_tier2_gain: float = field(
        default_factory=lambda: _env_float("RATCHET_T2_GAIN", 22.0)
    )
    ratchet_tier2_trail_pct: float = field(
        default_factory=lambda: _env_float("RATCHET_T2_TRAIL_PCT", 3.5)
    )   # 2026-05-29: +18%→2.5% softened to +22%→3.5% (lock real gains, gentler)
    ratchet_tier3_gain: float = field(
        default_factory=lambda: _env_float("RATCHET_T3_GAIN", 30.0)
    )
    ratchet_tier3_trail_pct: float = field(
        default_factory=lambda: _env_float("RATCHET_T3_TRAIL_PCT", 2.5)
    )   # 2026-05-29: +28%→2.0% softened to +30%→2.5% — only monster
        # winners get pulled this tight, to lock a genuine +27%+ run

    # ── Break-Even Protection ──────────────────────────────────
    # Closes the gap between entry and the ratchet's lowest tier (T0 = +10%)
    # where a base trail of ~5.5% would still fire at a loss.
    #
    # Mechanism: when peak_gain ≥ break_even_threshold, dynamically
    # tighten the TRAIL % so the resulting stop floors at
    #   entry × break_even_floor_mult   (default 1.002 = +0.2% to cover commissions)
    #
    # ⚠️ DEFAULT FLIPPED TO FALSE ON 2026-06-05. ⚠️
    # The original (2026-05-15) justification rested on 2 anecdotes (ELVN, RSI)
    # — that's not validation, it's curve-fit storytelling. Proper backtest on
    # the 16k-row training dataset (score ≥ 80 cohort, n=1,450) showed:
    #   • 48.1% of trades end ABOVE +0.2% — BE cuts their profit at +0.2%.
    #   • 51.9% end below +0.2% — BE *might* save them, IF they peaked ≥ 1.5%.
    #   • Upper-bound net (BE's BEST possible case): -$68.79 per $1k per trade.
    #   • Realistic estimate (peak-doubling rule):    -$74.81 per $1k per trade.
    # i.e., even in the most BE-favorable accounting, it costs more than it
    # saves. PRM on 2026-06-05 was the canonical failure mode: BE armed at
    # +1.94% peak, locked exit at +0.2%, then the price kept climbing to
    # $30.63 (vs our $30.08 exit). One "save" anecdote (today's PRM "saved
    # $11.51 vs naive trail-fire-at-peak") was visible in the cash account;
    # the much larger aggregate cost was hidden in opportunity cost.
    #
    # Keep the mechanism + env switch intact so it's trivially re-enabled if
    # later data justifies it — but it ships OFF by default. Re-enabling needs
    # a fresh backtest, not another anecdote.
    break_even_enabled: bool = field(
        default_factory=lambda: _env_bool("BREAK_EVEN_ENABLED", False)
    )
    break_even_threshold_pct: float = field(
        default_factory=lambda: _env_float("BREAK_EVEN_THRESHOLD_PCT", 1.5)
    )   # Activate dynamic break-even trail when peak_gain ≥ this %.
        # 2026-05-15: set to 1.5% (≈ 0.5R for our typical 3% stop). Backtest
        # on 5 trail_fired trades showed +2.0% missed both ELVN (+0.41% peak)
        # and RSI (+1.17% peak) without saving any of the wider-peak trades
        # (ORCL/LION/MEOH base trail already locked above break-even). +1.5%
        # is the forward-looking default — saves any future trade that
        # peaks above +1.5% and then retraces past the base-trail stop.
    break_even_floor_mult: float = field(
        default_factory=lambda: _env_float("BREAK_EVEN_FLOOR_MULT", 1.002)
    )   # Floor stop at entry × this multiplier (1.002 = +0.2% above entry)
    break_even_min_trail_pct: float = field(
        default_factory=lambda: _env_float("BREAK_EVEN_MIN_TRAIL_PCT", 0.5)
    )   # Refuse to set trail tighter than this (noise floor — avoid whipsaw)

    # ── ATR floor on initial trail ─────────────────────────────
    # The trail formula avg(scan_stop, 1.5×ATR) can produce a trail TIGHTER
    # than the stock's normal daily volatility, leading to whipsaw exits.
    # Example: ILMN had ATR 4.49% but trail 4.0% — one standard-deviation
    # day = guaranteed stopout.
    #
    # When > 0, the initial trail is also floored at: this_mult × ATR%.
    # Default 0.9 means trail ≥ 90% of ATR. For ILMN: trail floored at
    # 0.9 × 4.49 = 4.04% (barely different from 4.0). For a low-ATR
    # stock with ATR 2%, this is moot (trail of 4% already exceeds 1.8%).
    # Conservatively starts at 0.9 — easy to nudge up to 1.0 or 1.2 later.
    initial_trail_atr_floor_mult: float = field(
        default_factory=lambda: _env_float("INITIAL_TRAIL_ATR_FLOOR_MULT", 0.9)
    )

    # ── Event-sourced ledger (deep fix for tracker↔IB drift) ──────
    # When enabled: IB is the single source of truth for position existence
    # and realized P&L; the durable ledger (executions.jsonl) is built by
    # idempotently ingesting IB executions (keyed by execId), and all
    # reporting reads from IB-live + ledger instead of the JSON tracker.
    # This eliminates the dual-source-of-truth that caused months of drift.
    # Setting it False fully restores the legacy trade_log behavior.
    # See docs/architecture_ledger.md. Env: TRADE_LEDGER_ENABLED.
    ledger_enabled: bool = field(
        default_factory=lambda: _env_bool("LEDGER_ENABLED", True)
    )

    # ── Paths ──────────────────────────────────────────────────
    scan_results_path: str = "data/scans/latest_scan_live.json"
    open_positions_path: str = "data/trades/open_positions.json"
    trade_log_path: str = "data/trades/trade_log.json"
    ledger_path: str = "data/trades/executions.jsonl"

    @property
    def ibkr_port(self) -> int:
        return self.ibkr_port_paper if self.paper_mode else self.ibkr_port_live

    @property
    def blocked_sectors_list(self) -> list:
        return [s.strip() for s in self.blocked_sectors.split(",") if s.strip()]

    @property
    def blocked_regimes_list(self) -> list:
        return [s.strip().upper() for s in self.blocked_regimes.split(",") if s.strip()]

    @property
    def reduce_regimes_list(self) -> list:
        return [s.strip().upper() for s in self.reduce_regimes.split(",") if s.strip()]

    def summary(self) -> str:
        mode = "DRY RUN" if self.dry_run else ("PAPER" if self.paper_mode else "LIVE")
        return (
            f"Trading Config [{mode}]\n"
            f"  Position: ${self.max_position_size:,.0f} | "
            f"Max open: {self.max_open_positions} | "
            f"Daily limit: {self.max_daily_buys}\n"
            f"  Filters: Score {self.min_score_to_trade}-{self.max_score_to_trade} | "
            f"ML>={self.min_ml_prob} | "
            f"RR>={self.min_rr_to_trade} | "
            f"Confidence>={self.min_confidence} | "
            f"Reliability>={self.min_reliability:.0f}\n"
            f"  Regimes: Block {self.blocked_regimes_list} | "
            f"Reduce {self.reduce_regimes_list}\n"
            f"  Stop: Trailing {self.trailing_stop_pct}%\n"
            f"  IBKR: {self.ibkr_host}:{self.ibkr_port} (client {self.ibkr_client_id})"
        )


# Singleton — import and use directly
CONFIG = TradingConfig()
