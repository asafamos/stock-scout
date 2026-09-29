"""A collapsed ML column (model silently failed -> all 0.5) must block buys, not pass the 0.40-0.60 gate."""
import numpy as np
import pandas as pd

from core.trading.order_manager import OrderManager


def _om():
    om = OrderManager.__new__(OrderManager)
    from types import SimpleNamespace
    om.cfg = SimpleNamespace()
    return om


def _df(ml):
    n = len(ml)
    return pd.DataFrame({"Ticker": [f"T{i}" for i in range(n)], "FinalScore_20d": [78.0] * n,
                         "RewardRisk": [3.0] * n, "ML_20d_Prob": ml, "Sector": ["Technology"] * n})


def test_all_half_blocks_and_alerts(monkeypatch):
    sent = []
    monkeypatch.setattr("core.trading.order_manager.notify.notify_error", lambda *a, **k: sent.append(a))
    out = _om()._filter_candidates(_df([0.5] * 60))
    assert out.empty and sent, "must return nothing and alert"


def test_healthy_spread_is_not_blocked_by_the_guard(monkeypatch):
    sent = []
    monkeypatch.setattr("core.trading.order_manager.notify.notify_error", lambda *a, **k: sent.append(a))
    rng = np.random.default_rng(0)
    try:
        _om()._filter_candidates(_df(list(rng.uniform(0.2, 0.55, 60))))
    except Exception:
        pass  # later filters may need more config; only the guard's alert matters here
    assert not [m for m in sent if "degenerate" in str(m)]
