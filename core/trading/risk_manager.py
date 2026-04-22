"""Pre-trade risk checks — gate every order through here."""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Optional, Tuple

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

    def check_daily_loss_breaker(self) -> Tuple[bool, str]:
        """Return (allowed, reason). Blocks new buys if today's P&L < -max_daily_loss_pct."""
        try:
            net = self.client.get_net_liquidation()
            if net <= 0:
                return True, ""
            # Today's realized P&L from trade log
            today = date.today().isoformat()
            realized = 0.0
            for t in self.tracker.get_trade_log():
                if t.get("action") == "CLOSE" and str(t.get("timestamp", "")).startswith(today):
                    realized += float(t.get("pnl", 0) or 0)
            # Unrealized P&L from live IB positions
            unrealized = 0.0
            try:
                for p in self.client._ib.portfolio():
                    if p.position != 0:
                        unrealized += float(p.unrealizedPNL or 0)
            except Exception:
                pass
            total_today = realized + unrealized
            pct = (total_today / net) * 100
            if pct <= -self.cfg.max_daily_loss_pct:
                return False, (
                    f"Daily loss breaker: P&L {pct:+.2f}% <= -{self.cfg.max_daily_loss_pct}% "
                    f"(realized ${realized:.0f} + unrealized ${unrealized:.0f})"
                )
            return True, ""
        except Exception as e:
            logger.warning("Daily loss breaker check failed: %s", e)
            return True, ""  # Fail open (don't block trades on error)

    def check_sector_concentration(self, new_sector: str) -> Tuple[bool, str]:
        """Block if we'd exceed max_sector_positions in same sector."""
        if not new_sector:
            return True, ""
        # Count existing positions in same sector
        same_sector = 0
        for p in self.tracker.get_open_positions():
            if str(p.get("sector", "")).strip().lower() == new_sector.strip().lower():
                same_sector += 1
        if same_sector >= self.cfg.max_sector_positions:
            return False, (
                f"Sector concentration: already {same_sector} positions in {new_sector} "
                f"(max {self.cfg.max_sector_positions})"
            )
        return True, ""

    def can_open_position(
        self,
        ticker: str,
        price: float,
        score: float = 0.0,
        rr: float = 0.0,
        sector: str = "",
        atr_pct: float = 0.0,
    ) -> Tuple[bool, str]:
        """Return (allowed, reason). Reason is empty string if allowed."""

        # 0. Daily loss circuit breaker
        allowed, reason = self.check_daily_loss_breaker()
        if not allowed:
            return False, reason

        # 0a. Day-trade prevention (cash account cannot re-buy same-day sell)
        try:
            today = date.today().isoformat()
            for t in self.tracker.get_trade_log():
                if (t.get("ticker") == ticker
                        and t.get("action") in ("CLOSE", "PARTIAL")
                        and str(t.get("timestamp", "")).startswith(today)):
                    return False, (
                        f"Day-trade block: {ticker} was sold today "
                        f"(cash account can't re-buy same day)"
                    )
        except Exception:
            pass

        # 0b. Sector concentration check
        if sector:
            allowed, reason = self.check_sector_concentration(sector)
            if not allowed:
                return False, reason

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

    def calculate_qty(self, price: float, cash_available: float = None,
                      atr_pct: float = 0.0) -> int:
        """Calculate number of shares to buy — cash-aware + volatility-aware sizing.

        Sizing logic:
        1. Base target = min(max_position_size, cash_available)
        2. Volatility scaling (if atr_pct given):
           - 2% ATR is "normal" baseline → 1.0x size
           - High volatility (>4% ATR) → reduce size (downweight)
           - Low volatility (<2% ATR) → keep full size (can't exceed 1.0x)
           - Formula: vol_factor = clamp(2.0 / max(atr_pct, 1.0), 0.5, 1.0)
        3. qty = floor(target_spend * vol_factor / price)
        4. Fallback: 1 share if we can afford it

        Examples:
        - price=$50, ATR 2% → 1.0x sizing (normal)
        - price=$50, ATR 4% → 0.5x sizing (high vol, reduce risk)
        - price=$100, ATR 3% → 0.67x sizing
        - price=$100, ATR 1% → 1.0x sizing (capped)
        """
        if price <= 0:
            return 0

        target_spend = self.cfg.max_position_size
        if cash_available is not None:
            target_spend = min(target_spend, cash_available)

        # Volatility factor: high-vol stocks get smaller positions
        vol_factor = 1.0
        if atr_pct and atr_pct > 0:
            vol_factor = max(0.5, min(2.0 / max(atr_pct, 1.0), 1.0))

        adjusted_spend = target_spend * vol_factor
        qty = math.floor(adjusted_spend / price)

        # Fallback: allow 1 share if we can afford it
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
