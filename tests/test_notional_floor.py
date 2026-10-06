"""Cost-aware notional floor (plan 2026-10-06): disabled by default; blocks dust positions when enabled."""
from core.trading.policy import notional_floor_reason as nfr


def test_disabled_by_default_changes_nothing():
    assert nfr(1, 13.03, 0.0) is None            # floor 0 = off
    assert nfr(1, 13.03, -5) is None


def test_blocks_dust_position_and_names_the_cost():
    r = nfr(1, 13.03, 150.0)                      # PATH-sized buy
    assert r and "below the cost-aware floor $150" in r
    assert "2.0%" in r                            # 1% cap each way -> ~2% round trip on a $13 position


def test_allows_position_at_or_above_floor():
    assert nfr(2, 100.0, 150.0) is None           # $200
    assert nfr(1, 150.0, 150.0) is None           # exactly at the floor


def test_round_trip_cost_pct_for_mid_sized_position():
    r = nfr(1, 98.0, 150.0)                       # HNGE-sized: $1 min capped at 1% -> $0.98/leg ~2.0% round trip
    assert r and "2.0%" in r


def test_effective_minimum_is_the_larger_of_old_threshold_and_floor(monkeypatch):
    from core.trading.config import TradingConfig
    monkeypatch.delenv("TRADE_MIN_POSITION_NOTIONAL_USD", raising=False)
    assert TradingConfig().effective_min_position_usd == 30.0           # default: unchanged behaviour
    monkeypatch.setenv("TRADE_MIN_POSITION_NOTIONAL_USD", "150")
    c = TradingConfig()
    assert c.min_position_notional_usd == 150.0 and c.effective_min_position_usd == 150.0
    monkeypatch.setenv("TRADE_MIN_POSITION_NOTIONAL_USD", "10")          # a floor below the old threshold never lowers it
    assert TradingConfig().effective_min_position_usd == 30.0
