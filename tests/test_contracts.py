from datetime import datetime

from core.interfaces import (
    TickerFeatures,
    TradeDecision,
    Action,
    DataQuality,
)


def main():
    # Instantiate a TickerFeatures with dummy data
    tf = TickerFeatures(
        ticker="AAPL",
        as_of_date=datetime(2026, 1, 17),
        data_timestamp=datetime(2026, 1, 17, 12, 0, 0),
        source_map={"price": "yfinance", "fundamentals": "finnhub"},
        quality=DataQuality.HIGH,
        point_in_time_ok=True,
        model_features={"feature1": 0.123, "feature2": -0.5},
        risk_metadata={
            "market_cap": 2_500_000_000_000,
            "sector": "Technology",
            "avg_volume": 50_000_000,
        },
    )

    # Instantiate a TradeDecision
    td = TradeDecision(
        ticker="AAPL",
        action=Action.BUY,
        quantity=100,
        limit_price=190.00,
        stop_loss_price=175.00,
        target_price=220.00,
        conviction=0.85,
        estimated_commission=1.50,
        primary_reason="Strong momentum and fundamentals",
        active_filters=["beta_cap", "earnings_blackout"],
        risk_penalties=["high_volatility"],
        explain_id="EXP-12345",
    )

    print("Contracts Validated Successfully")


if __name__ == "__main__":
    main()
