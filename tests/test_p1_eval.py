"""P1 evaluator (prereg docs/research_prereg_P1_new_score_eval.md): must find a planted signal, must NOT find one in noise, and must
stay blind (coverage only) until the prereg sample criteria are met."""
import numpy as np
import pandas as pd

from scripts.research import p1_eval_new_score as p1


def _panel(n_dates=40, n_names=60, beta=0.0, seed=1):
    """excess return = beta * z(score) + noise (pct points); one record per (ticker, date)."""
    rng = np.random.default_rng(seed)
    rows = []
    for di in range(n_dates):
        d = (pd.Timestamp("2026-10-08") + pd.offsets.BDay(di)).strftime("%Y-%m-%d")
        s = rng.normal(0, 1, n_names)
        y = beta * s + rng.normal(0, 4, n_names)
        for i in range(n_names):
            rows.append({"scan_date": d, "ticker": f"T{i}", "score": 80 + 5 * s[i], "y": y[i], "atr_pct": 0.03 + 0.01 * rng.random(),
                         "fundamental_score": 50 + 5 * rng.normal(), "technical_score": 60 + 5 * rng.normal()})
    df = pd.DataFrame(rows)
    df["excess"] = df["y"] - df.groupby("scan_date")["y"].transform("mean")
    return df


def _elig_top(df, frac=0.1):
    out = set()
    for d, g in df.groupby("scan_date"):
        k = max(1, int(len(g) * frac))
        out |= {(t, d) for t in g.nlargest(k, "score")["ticker"]}
    return out


def test_planted_signal_passes_both_primary_tests():
    df = _panel(beta=3.0)
    el = _elig_top(df)
    r = p1.run(df, el)
    assert r["T1_score_IC"]["passed"] and r["T2_eligible_excess"]["passed"], r
    assert r["verdict"].startswith("DEMONSTRATED EDGE")


def test_pure_noise_does_not_pass_and_is_not_called_an_edge():
    df = _panel(beta=0.0, seed=7)
    el = _elig_top(df)
    r = p1.run(df, el)
    assert not r["T1_score_IC"]["passed"] and not r["T2_eligible_excess"]["passed"], r
    assert not r["verdict"].startswith(("DEMONSTRATED", "PARTIAL"))


def test_blind_until_the_sample_criteria_are_met():
    df = _panel(n_dates=10, beta=3.0)                  # too few dates: must show coverage only, no statistics
    r = p1.run(df, _elig_top(df))
    assert r["verdict"].startswith("NO VERDICT")
    assert set(r) == {"coverage", "verdict"}


def test_final_look_uses_the_halved_alpha():
    df = _panel(n_dates=10, beta=0.0)
    r = p1.run(df, _elig_top(df), final=True)
    assert r["alpha"] == p1.ALPHA_FINAL


def test_load_outcomes_keeps_resolved_post_cutoff_and_dedupes(tmp_path):
    rows = [
        {"scan_date": "2026-10-07", "ticker": "AAA", "resolved": True, "final_return_pct": 5, "score": 80, "recorded_at": "a"},    # before SINCE
        {"scan_date": "2026-10-08", "ticker": "AAA", "resolved": True, "final_return_pct": 2, "score": 80, "recorded_at": "a"},
        {"scan_date": "2026-10-08", "ticker": "AAA", "resolved": True, "final_return_pct": 9, "score": 80, "recorded_at": "b"},    # duplicate: keep earliest
        {"scan_date": "2026-10-08", "ticker": "BBB", "resolved": False, "final_return_pct": 1, "score": 70, "recorded_at": "a"},   # unresolved
        {"scan_date": "2026-10-08", "ticker": "CCC", "resolved": True, "final_return_pct": 4, "score": 75, "recorded_at": "a"},
    ]
    p = tmp_path / "o.jsonl"
    p.write_text("\n".join(__import__("json").dumps(r) for r in rows))
    d = p1.load_outcomes(p)
    assert sorted(d["ticker"]) == ["AAA", "CCC"]
    assert float(d.loc[d.ticker == "AAA", "y"].iloc[0]) == 2.0
    assert abs(float(d["excess"].sum())) < 1e-9
