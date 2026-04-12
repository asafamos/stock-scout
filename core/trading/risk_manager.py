"""Pre-trade risk checks — gate every order through here."""

from __future__ import annotations

import logging
import math
from typing import Tuple

from core.trading.config import CONFIG
from core.trading.ibkr_client import IBKRClient
from core.trading.position_tracker import PositionTracker

logger = logging.getLogger(__name__)


class RiskManager:
    """Validates every trade against position limits and account state."""

    def __init__(self, client: IBKRClient, tracker: PositionTracker,
                 config=None):
        self.client = client
        self.tracker = tracker
        self.cfg = config or CONFIG

    def can_open_position(
        self,
        ticker: str,
        price: float,
        score: float = 0.0,
        rr: float = 0.0,
    ) -> Tuple[bool, str]:
        """Return (allowed, reason). Reason is empty string if allowed."""

        # 1. Already holding
        if self.tracker.is_holding(ticker):
            return False, f"Already holding {ticker}"

        # 2. Max open positions
        if self.tracker.open_count >= self.cfg.max_open_positions:
            return False, (
                f"Max open positions reached "
                f"({self.tracker.open_count}/{self.cfg.max_open_positions})"
            )

        # 3. Daily buy limit
        daily = self.tracker.daily_buy_count()
        if daily >= self.cfg.max_daily_buys:
            return False, (
                f"Daily buy limit reached ({daily}/{self.cfg.max_daily_buys})"
            )

        # 4. Portfolio exposure
        new_exposure = self.tracker.total_exposure + self.cfg.max_position_size
        if new_exposure > self.cfg.max_portfolio_exposure:
            return False, (
                f"Would exceed max exposure "
                f"(${new_exposure:,.0f} > ${self.cfg.max_portfolio_exposure:,.0f})"
            )

        # 5. Cash balance
        cash = self.client.get_cash_balance()
        if cash < self.cfg.max_position_size:
            return False, (
                f"Insufficient cash (${cash:,.0f} < "
                f"${self.cfg.max_position_size:,.0f})"
            )

        # 6. Score filter
        if score < self.cfg.min_score_to_trade:
            return False, (
                f"Score too low ({score:.1f} < {self.cfg.min_score_to_trade})"
            )

        # 7. R:R filter
        if rr < self.cfg.min_rr_to_trade:
            return False, f"R:R too low ({rr:.2f} < {self.cfg.min_rr_to_trade})"

        # 8. Market hours
        if not self.client.is_market_open() and not self.cfg.dry_run:
            return False, "Market is closed"

        return True, ""

    def calculate_qty(self, price: float) -> int:
        """Calculate number of shares to buy within position size limit.

        Allows buying 1 share of expensive stocks (up to 2x max_position_size)
        so high-scoring stocks like UTHR ($564) aren't skipped entirely.
        """
        if price <= 0:
            return 0
        qty = math.floor(self.cfg.max_position_size / price)
        # Allow 1 share if price is within 2x max position size
        if qty == 0 and price <= self.cfg.max_position_size * 2:
            qty = 1
        return max(qty, 0)

    def get_portfolio_summary(self) -> dict:
        return {
            "cash": self.client.get_cash_balance(),
            "net_liquidation": self.client.get_net_liquidation(),
            "open_positions": self.tracker.open_count,
            "total_exposure": self.tracker.total_exposure,
            "daily_buys_today": self.tracker.daily_buy_count(),
            "remaining_capacity": (
                self.cfg.max_open_positions - self.tracker.open_count
            ),
        }
