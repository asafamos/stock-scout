"""Guards against silent contradictions between code defaults, thresholds and the frozen docs.

2026-09-29: three such contradictions cost real trading time —
  * core/scoring_config.ML_PROB_THRESHOLD = 0.70 while the trade gate max_ml_prob = 0.60, so the
    scan's ML bypass and the SigHigh ML bonus were unreachable for any tradeable stock;
  * core/trading/config max_ml_prob default 0.55 while CLAUDE.md / check_env_vs_docs said 0.60;
  * nothing in CI compared them.
Sizing/portfolio defaults (max_open, daily buys, position size, exposure, sector cap) are supplied by
.env.trading on purpose and guarded by scripts/check_env_vs_docs.py on the VPS; only GATE thresholds
are pinned here.
"""
import os

import pytest

from core import scoring_config as sc
from scripts import check_env_vs_docs as docs
from core.trading.config import TradingConfig


@pytest.fixture
def cfg(monkeypatch):
    for k in list(os.environ):
        if k.startswith("TRADE_"):
            monkeypatch.delenv(k, raising=False)
    return TradingConfig()


# (EXPECTED env key in the drift checker, TradingConfig field)
GATES = [
    ("TRADE_MIN_SCORE", "min_score_to_trade"),
    ("TRADE_MAX_SCORE", "max_score_to_trade"),
    ("TRADE_MIN_FUNDAMENTAL_SCORE", "min_fundamental_score"),
    ("TRADE_MIN_ML_PROB", "min_ml_prob"),
    ("TRADE_MAX_ML_PROB", "max_ml_prob"),
    ("TRADE_MIN_RR", "min_rr_to_trade"),
    ("TRADE_MAX_RR", "max_rr_to_trade"),
    ("TRADE_MIN_ATR_PCT", "min_atr_pct"),
    ("TRADE_MAX_SLIPPAGE_PCT", "max_slippage_pct"),
]


@pytest.mark.parametrize("env_key,field", GATES)
def test_code_default_matches_frozen_expected(cfg, env_key, field):
    assert float(getattr(cfg, field)) == pytest.approx(float(docs.EXPECTED[env_key])), (
        f"{field} default drifted from the frozen value in scripts/check_env_vs_docs.py ({env_key})"
    )


def test_windows_are_not_empty(cfg):
    assert cfg.min_score_to_trade < cfg.max_score_to_trade
    assert cfg.min_ml_prob < cfg.max_ml_prob
    assert cfg.min_rr_to_trade < cfg.max_rr_to_trade


def test_ml_bypass_threshold_is_reachable_within_the_trade_window(cfg):
    """A scan-side ML threshold above the trade cap is dead code (the 0.70 vs 0.60 bug)."""
    assert cfg.min_ml_prob <= sc.ML_PROB_THRESHOLD <= cfg.max_ml_prob


def test_adaptive_relaxation_only_relaxes(cfg):
    assert cfg.adaptive_ml_relaxed_floor <= cfg.min_ml_prob
    assert cfg.adaptive_rr_relaxed_floor <= cfg.min_rr_to_trade
