import numpy as np
import pandas as pd
import pytest

from ml.targets import compute_smart_targets


def make_close_from_log_returns(log_returns: np.ndarray, start_price: float = 100.0) -> pd.DataFrame:
    # Construct price series: Close_t = Close_0 * exp(sum_{i<=t} r_i)
    cum_lr = np.cumsum(log_returns)
    prices = start_price * np.exp(cum_lr)
    return pd.DataFrame({"Close": prices})


def test_gold_case_low_vol_rising():
    horizon = 20
    # Constant positive log return -> zero volatility in window
    n = 60
    log_returns = np.full(n, 0.002)  # ~0.2% per period
    df = make_close_from_log_returns(log_returns)

    out = compute_smart_targets(df, horizon=horizon)

    # At index 0: forward Sharpe should be very large -> class 2
    assert out.loc[0, "target_class"] == 2
    assert out.loc[0, "forward_sharpe"] > 1.5


def test_silver_case_positive_return_high_vol():
    horizon = 20
    n = 60
    # Craft high volatility but overall positive over first horizon window
    # 19 small negatives and one large positive so sum ~ 0.035
    window = np.array([-0.005] * 19 + [0.13])
    # Extend to n with small zeros afterwards
    log_returns = np.concatenate([window, np.zeros(n - len(window))])
    df = make_close_from_log_returns(log_returns)

    out = compute_smart_targets(df, horizon=horizon)

    # Raw forward return should exceed 3% but Sharpe < 1.5 => class 1
    assert out.loc[0, "raw_forward_return"] > 0.03
    assert out.loc[0, "forward_sharpe"] < 1.5
    assert out.loc[0, "target_class"] == 1


def test_noise_case_flat_or_down():
    horizon = 20
    n = 60
    # Slightly negative returns -> overall decrease
    log_returns = np.full(n, -0.001)
    df = make_close_from_log_returns(log_returns)

    out = compute_smart_targets(df, horizon=horizon)

    assert out.loc[0, "raw_forward_return"] < 0.03
    assert out.loc[0, "target_class"] == 0


def test_horizon_dependency_alignment():
    horizon = 5
    n = 20
    # Simple deterministic returns
    log_returns = np.full(n, 0.01)  # 1% log return per step
    df = make_close_from_log_returns(log_returns, start_price=100.0)

    out = compute_smart_targets(df, horizon=horizon)

    # target_log_return at index 0 should be ln(C_{h} / C_0)
    C0 = df.loc[0, "Close"]
    Ch = df.loc[horizon, "Close"]
    expected = np.log(Ch / C0)
    assert np.isclose(out.loc[0, "target_log_return"], expected, rtol=1e-6, atol=1e-8)
