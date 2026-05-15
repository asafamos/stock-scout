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

    # ── Circuit Breakers (stop new buys when losses pile up) ──
    max_daily_loss_pct: float = field(
        default_factory=lambda: _env_float("MAX_DAILY_LOSS_PCT", 2.0)
    )  # Stop new buys if today's realized+unrealized P&L < -X% of portfolio
    max_drawdown_pct: float = field(
        default_factory=lambda: _env_float("MAX_DRAWDOWN_PCT", 10.0)
    )  # Pause trading if portfolio is down >X% from peak

    # ── Sector Concentration ─────────────────────────────────────
    max_sector_positions: int = field(
        default_factory=lambda: _env_int("MAX_SECTOR_POSITIONS", 2)
    )  # Max # of positions in same sector (avoid concentration)

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
    min_score_to_trade: float = field(
        default_factory=lambda: _env_float("MIN_SCORE", 73.0)
    )
    max_score_to_trade: float = field(
        default_factory=lambda: _env_float("MAX_SCORE", 95.0)
    )  # Q5 (highest scores) underperform — cap at 95
    min_rr_to_trade: float = field(
        default_factory=lambda: _env_float("MIN_RR", 2.0)
    )
    min_confidence: str = field(
        default_factory=lambda: _env("MIN_CONFIDENCE", "High")
    )
    min_ml_prob: float = field(
        default_factory=lambda: _env_float("MIN_ML_PROB", 0.33)
    )  # Calibrated to real ML output range (0.30-0.37); filters bottom half
    min_reliability: float = field(
        default_factory=lambda: _env_float("MIN_RELIABILITY", 50.0)
    )  # Filter stocks with incomplete data (Reliability_Score < 50)

    # ── Market Regime Gate ────────────────────────────────────
    # Skip buying when market is unfavorable
    blocked_regimes: str = field(
        default_factory=lambda: _env("BLOCKED_REGIMES", "PANIC,CORRECTION")
    )  # Comma-separated list of regime names to block entirely
    reduce_regimes: str = field(
        default_factory=lambda: _env("REDUCE_REGIMES", "DISTRIBUTION")
    )  # Regimes where we still trade but smaller (half size)

    # ── Sector Blocklist (portfolio analysis: Consumer Defensive = -4.47%, 20% win) ──
    blocked_sectors: str = field(
        default_factory=lambda: _env("BLOCKED_SECTORS", "Consumer Defensive")
    )  # Comma-separated list

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
        default_factory=lambda: _env("THROTTLE_MODE", "winrate")
    )   # "winrate" (default) or "expectancy"
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
    min_initial_trail_pct: float = field(
        default_factory=lambda: _env_float("MIN_INITIAL_TRAIL_PCT", 4.0)
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
    ratchet_tier0_gain: float = field(
        default_factory=lambda: _env_float("RATCHET_T0_GAIN", 8.0)
    )
    ratchet_tier0_trail_pct: float = field(
        default_factory=lambda: _env_float("RATCHET_T0_TRAIL_PCT", 3.5)
    )   # Peak +8% → trail 3.5% (early lock above noise band)
    min_hold_days_for_t0: int = field(
        default_factory=lambda: _env_int("RATCHET_T0_MIN_HOLD_DAYS", 2)
    )   # T0 ratchet only engages after position has aged ≥N days

    ratchet_tier1_gain: float = field(
        default_factory=lambda: _env_float("RATCHET_T1_GAIN", 10.0)
    )
    ratchet_tier1_trail_pct: float = field(
        default_factory=lambda: _env_float("RATCHET_T1_TRAIL_PCT", 3.0)
    )   # Peak +10% → trail 3.0% (was +12% → 4% — tightened 2026-05-05)
    ratchet_tier2_gain: float = field(
        default_factory=lambda: _env_float("RATCHET_T2_GAIN", 18.0)
    )
    ratchet_tier2_trail_pct: float = field(
        default_factory=lambda: _env_float("RATCHET_T2_TRAIL_PCT", 2.5)
    )   # Peak +18% → trail 2.5% (was +20% → 3% — tightened 2026-05-05)
    ratchet_tier3_gain: float = field(
        default_factory=lambda: _env_float("RATCHET_T3_GAIN", 28.0)
    )
    ratchet_tier3_trail_pct: float = field(
        default_factory=lambda: _env_float("RATCHET_T3_TRAIL_PCT", 2.0)
    )   # Peak +28% → trail 2.0% (was +30% — tightened 2026-05-05)

    # ── Break-Even Protection ──────────────────────────────────
    # Closes the gap between entry and the ratchet's lowest tier (T0 = +8%)
    # where a base trail of ~4% would still fire at a loss.
    #
    # Mechanism: when peak_gain ≥ break_even_threshold, dynamically
    # tighten the TRAIL % so the resulting stop floors at
    #   entry × break_even_floor_mult   (default 1.002 = +0.2% to cover commissions)
    # Once base trail naturally floors above break-even (roughly when
    # peak_gain ≥ base_trail_pct + 0.2pp), this becomes a no-op because
    # the "only-tighten" guard rejects the wider candidate.
    #
    # Forensic justification (2026-05-15): in last 7 trades, ELVN peaked
    # at +0.41% then closed at -3.60%; RSI peaked at +1.17% then closed
    # at -2.88%. Both were "winners turned losers". Break-even at +2%
    # threshold would have saved RSI (~$3.95 saved). ELVN's peak was
    # below threshold and can't be saved by any trail mechanism — that's
    # a pick-quality issue, not a trail issue.
    break_even_enabled: bool = field(
        default_factory=lambda: _env_bool("BREAK_EVEN_ENABLED", True)
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

    # ── Paths ──────────────────────────────────────────────────
    scan_results_path: str = "data/scans/latest_scan_live.json"
    open_positions_path: str = "data/trades/open_positions.json"
    trade_log_path: str = "data/trades/trade_log.json"

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
