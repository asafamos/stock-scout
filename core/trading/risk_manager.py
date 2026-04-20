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

        # 4. Calculate qty using DYNAMIC cash-aware sizing
        cash = self.client.get_cash_balance()
        available_cash = max(0, cash - self.cfg.cash_reserve)
        qty_est = self.calculate_qty(price, cash_available=available_cash)

        if qty_est == 0:
            return False, (
                f"Can't afford any shares "
                f"(cash=${cash:,.0f}, price=${price:.2f}, reserve=${self.cfg.cash_reserve:.0f})"
            )

        actual_cost = qty_est * price

        # 5. Portfolio exposure
        new_exposure = self.tracker.total_exposure + actual_cost
        if new_exposure > self.cfg.max_portfolio_exposure:
            return False, (
                f"Would exceed max exposure "
                f"(${new_exposure:,.0f} > ${self.cfg.max_portfolio_exposure:,.0f})"
            )

        # 6. Cash sanity check
        if cash < actual_cost:
            return False, (
                f"Insufficient cash (${cash:,.0f} < ${actual_cost:,.0f})"
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

    def calculate_qty(self, price: float, cash_available: float = None) -> int:
        """Calculate number of shares to buy — DYNAMIC cash-aware sizing.

        Logic:
        - target_spend = min(max_position_size, cash_available)
        - qty = floor(target_spend / price)
        - Fallback: if qty=0, allow 1 share if we can afford it
          (respects cash if passed, else up to 2x max_position_size)

        Examples:
        - cash=$500, max=$300, price=$50 → 6 shares ($300)
        - cash=$165, max=$300, price=$120 → 1 share ($120) ✓ (was 0 before)
        - cash=$165, max=$300, price=$85 → 1 share ($85) ✓ (was 3 before = exceeded cash!)
        - cash=$500, max=$300, price=$575 → 0 or 1? (expensive)
        """
        if price <= 0:
            return 0

        target_spend = self.cfg.max_position_size
        if cash_available is not None:
            target_spend = min(target_spend, cash_available)

        qty = math.floor(target_spend / price)

        # Fallback: allow 1 share of expensive stock if we can afford it
        if qty == 0:
            max_affordable = (
                cash_available if cash_available is not None
                else self.cfg.max_position_size * 2
            )
            if price <= max_affordable:
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
