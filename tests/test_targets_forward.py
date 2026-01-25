import numpy as np
import pandas as pd

from ml.targets import compute_smart_targets


def test_forward_vol_constant_growth_h2():
    # Prices grow by 10% each step: constant multiplicative growth
    prices = [100.0, 110.0, 121.0, 133.1, 146.41]
    df = pd.DataFrame({"Close": prices})
    horizon = 2

    out = compute_smart_targets(df, horizon=horizon)

    # Forward window exists for indices 0..len-1-horizon
    for i in range(0, len(prices) - horizon):
        assert np.isfinite(out.loc[i, "future_volatility"]) and out.loc[i, "future_volatility"] == 0.0
    # Last H rows must be NaN
    assert np.isnan(out.loc[len(prices) - 2, "future_volatility"])  # index 3
    assert np.isnan(out.loc[len(prices) - 1, "future_volatility"])  # index 4


def test_forward_vol_alternating_returns_h2():
    # Construct prices to yield alternating log returns [a, b, a, b, ...]
    a = 0.02
    b = -0.01
    # Start at 100 and apply multiplicative steps exp(a), exp(b), ...
    steps = [a, b, a, b, a]
    prices = [100.0]
    for lr in steps:
        prices.append(prices[-1] * np.exp(lr))
    df = pd.DataFrame({"Close": prices})
    horizon = 2

    out = compute_smart_targets(df, horizon=horizon)

    # At index 0, forward window returns are [b, a]; std should equal population std of [a, b]
    m = (a + b) / 2.0
    expected_std = float(np.sqrt(((a - m) ** 2 + (b - m) ** 2) / 2.0))
    assert np.isclose(out.loc[0, "future_volatility"], expected_std, rtol=1e-12, atol=1e-12)
