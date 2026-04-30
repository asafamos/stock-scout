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
    max_position_size: float = field(
        default_factory=lambda: _env_float("MAX_POSITION_SIZE", 300.0)
    )
    max_open_positions: int = field(
        default_factory=lambda: _env_int("MAX_OPEN_POSITIONS", 3)
    )
    max_daily_buys: int = field(
        default_factory=lambda: _env_int("MAX_DAILY_BUYS", 3)
    )
    max_portfolio_exposure: float = field(
        default_factory=lambda: _env_float("MAX_PORTFOLIO_EXPOSURE", 900.0)
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
        default_factory=lambda: _env_int("EARNINGS_BLOCK_DAYS", 3)
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
    ratchet_tier1_gain: float = field(
        default_factory=lambda: _env_float("RATCHET_T1_GAIN", 10.0)
    )
    ratchet_tier1_trail_pct: float = field(
        default_factory=lambda: _env_float("RATCHET_T1_TRAIL_PCT", 4.0)
    )   # Peak +10% → trail tightens to 4%
    ratchet_tier2_gain: float = field(
        default_factory=lambda: _env_float("RATCHET_T2_GAIN", 18.0)
    )
    ratchet_tier2_trail_pct: float = field(
        default_factory=lambda: _env_float("RATCHET_T2_TRAIL_PCT", 3.0)
    )   # Peak +18% → trail tightens to 3%
    ratchet_tier3_gain: float = field(
        default_factory=lambda: _env_float("RATCHET_T3_GAIN", 28.0)
    )
    ratchet_tier3_trail_pct: float = field(
        default_factory=lambda: _env_float("RATCHET_T3_TRAIL_PCT", 2.0)
    )   # Peak +28% → trail tightens to 2%

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
