"""Performance guard statistics (replaces the noisy, permanently-halting throttle)."""
import numpy as np

from core.trading import performance_guard as pg


def test_insufficient_history_does_nothing():
    assert pg.assess([-5.0] * 10)["level"] == "insufficient"


def test_unlucky_streak_of_a_good_strategy_is_not_degraded():
    rng = np.random.default_rng(0)
    # true mean +1%, sd 8%: how often does a 30-trade window look 'degraded'? must be rare
    flags = [pg.assess(list(rng.normal(1.0, 8.0, 30)))["level"] == "degraded" for _ in range(2000)]
    assert np.mean(flags) < 0.03


def test_confident_losing_edge_is_degraded():
    rng = np.random.default_rng(1)
    r = list(rng.normal(-4.0, 6.0, 30))
    assert pg.assess(r)["level"] == "degraded"


def test_slightly_negative_but_noisy_is_only_watch():
    r = [-1.0, 3.0, -4.0, 5.0, -6.0, 4.0, -3.0, 2.0, -2.5, 1.0] * 3
    res = pg.assess(r)
    assert res["mean"] < 0 and res["level"] == "watch"


def test_never_halts_and_size_only_in_size_mode():
    assert pg.size_multiplier("degraded", "alert") == 1.0
    assert pg.size_multiplier("degraded", "off") == 1.0
    assert pg.size_multiplier("degraded", "size") == 0.5
    assert pg.size_multiplier("ok", "size") == 1.0


def test_alert_only_on_level_change(tmp_path):
    sent = []

    class N:
        @staticmethod
        def notify_error(title, msg): sent.append((title, msg))
    st = tmp_path / "s.json"
    res = {"level": "degraded", "n": 30, "mean": -3.0, "se": 1.0, "upper95": -1.3}
    assert pg.maybe_alert(res, st, N) is True
    assert pg.maybe_alert(res, st, N) is False, "same level again -> silent"
    assert pg.maybe_alert(dict(res, level="ok", mean=1.0, upper95=2.0), st, N) is True   # recovery announced
    assert len(sent) == 2
