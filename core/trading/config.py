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
        default_factory=lambda: _env_float("MIN_ML_PROB", 0.40)
    )  # ML<0.4 has negative expected return

    # ── Sector Blocklist (portfolio analysis: Consumer Defensive = -4.47%, 20% win) ──
    blocked_sectors: str = field(
        default_factory=lambda: _env("BLOCKED_SECTORS", "Consumer Defensive")
    )  # Comma-separated list

    # ── Stop / Target ─────────────────────────────────────────
    trailing_stop_pct: float = field(
        default_factory=lambda: _env_float("TRAILING_STOP_PCT", 5.0)
    )
    use_pipeline_stop: bool = field(
        default_factory=lambda: _env_bool("USE_PIPELINE_STOP", False)
    )  # If True, use StopLoss from scan instead of trailing %

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
            f"Confidence>={self.min_confidence}\n"
            f"  Stop: Trailing {self.trailing_stop_pct}%\n"
            f"  IBKR: {self.ibkr_host}:{self.ibkr_port} (client {self.ibkr_client_id})"
        )


# Singleton — import and use directly
CONFIG = TradingConfig()
