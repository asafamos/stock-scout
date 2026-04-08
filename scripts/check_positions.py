"""Monitor open positions and check for exits.

Usage:
    python -m scripts.check_positions
"""

import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    from core.trading.config import CONFIG
    from core.trading.order_manager import OrderManager

    print("=" * 60)
    print("  StockScout Position Monitor")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)
    print()

    manager = OrderManager()

    # Position summary
    print(manager.tracker.summary())
    print()

    # Portfolio summary
    summary = manager.risk.get_portfolio_summary()
    print(f"Cash Balance:      ${summary['cash']:,.2f}")
    print(f"Net Liquidation:   ${summary['net_liquidation']:,.2f}")
    print(f"Open Positions:    {summary['open_positions']}/{CONFIG.max_open_positions}")
    print(f"Total Exposure:    ${summary['total_exposure']:,.2f}/{CONFIG.max_portfolio_exposure:,.0f}")
    print(f"Daily Buys Today:  {summary['daily_buys_today']}/{CONFIG.max_daily_buys}")
    print(f"Remaining Slots:   {summary['remaining_capacity']}")
    print()

    # Check target-date exits
    expired = manager.tracker.check_target_date_exits()
    if expired:
        print(f"TARGET DATE EXITS ({len(expired)}):")
        for t in expired:
            pos = manager.tracker.get_position(t)
            print(f"  {t}: target date {pos.get('target_date')} reached "
                  f"(entry ${pos.get('entry_price', 0):.2f})")
    else:
        print("No positions at target date.")

    # Trade log summary
    log = manager.tracker.get_trade_log()
    if log:
        print(f"\nRecent trades ({len(log)} total):")
        for t in log[-5:]:
            print(f"  {t['timestamp'][:16]} | {t['action']} {t['ticker']} "
                  f"x{t['quantity']} @ ${t['price']:.2f}")


if __name__ == "__main__":
    main()
