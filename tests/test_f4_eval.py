import numpy as np
import pandas as pd

import scripts.forward.f4_eval as ev


def synth(seed, n=1000, weeks=30, slope=0.0, week_sd=2.0, noise=10.0):
    rng = np.random.default_rng(seed)
    wk = rng.integers(0, weeks, n)
    dates = pd.to_datetime("2026-11-02") + pd.to_timedelta(wk * 7 + rng.integers(0, 5, n), unit="D")
    score = rng.integers(-5, 6, n)
    wfac = rng.normal(0, week_sd, weeks)[wk]            # market-wide weekly shock shared by all events of the week
    excess = wfac + slope * score + rng.normal(0, noise, n)
    return pd.DataFrame({"entry_date": dates, "score": score, "excess": excess, "mcap": rng.choice([5e8, 2e9, 7e9], n)})


def run(df):
    return ev.evaluate(df, n_perm=300, n_boot=300)


def test_h0_false_positive_rate_is_low():
    wins = sum(run(synth(s))["verdict"] == "SUCCESS" for s in range(40))
    assert wins <= 2, f"{wins}/40 false successes under H0"


def test_h0_p_value_not_anticonservative():
    ps = [run(synth(100 + s))["p_ic"] for s in range(40)]
    assert sum(p < 0.05 for p in ps) <= 6          # nominal 5% of 40 = 2; generous bound for a small simulation


def test_real_effect_is_detected():
    res = run(synth(7, slope=0.9))                  # IC around 0.3, top bucket well above costs
    assert res["ic"] > 0.15 and res["p_ic"] < 0.05
    assert res["verdict"] == "SUCCESS", res["criteria"]


def test_effect_that_exists_but_is_eaten_by_costs_is_dead():
    res = run(synth(7, slope=0.05))                 # tiny edge: IC significant at most, top bucket nowhere near +1pp
    assert res["verdict"] == "DEAD"


def test_cost_model():
    assert ev.spread_bp(5e8) == 25 and ev.spread_bp(2e9) == 10 and ev.spread_bp(7e9) == 5
    assert abs(ev.cost_pct(2e9, 1.0) - (0.10 + 2 * 1.0 / 270 * 100 + 0.15)) < 1e-9
    assert ev.cost_pct(2e9, 0.35) < ev.cost_pct(2e9, 1.0)


def test_main_is_blind_before_stopping_rule(monkeypatch, capsys):
    import scripts.forward.f4_resolve as r
    monkeypatch.setattr(r, "main", lambda today=None: {"stopping_rule_met": False})
    called = []
    monkeypatch.setattr(ev, "build_frame", lambda: called.append(1))
    ev.main()
    assert "BLIND" in capsys.readouterr().out and not called
