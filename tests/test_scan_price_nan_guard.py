"""2026-10-10: overnight scans came out with price columns partly/fully empty. Both the monitor (RED > 20%) and the
integrity check that blocks the commit (error > 30%) must catch it, and must not trip on healthy scans (max 18% NaN)."""
import numpy as np
import pandas as pd
import pytest

from scripts import scan_integrity_check as sic
from scripts.scan_quality_report import evaluate


def _scan(n=120, nan_share=0.0, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Ticker": [f"T{i}" for i in range(n)], "FinalScore_20d": rng.uniform(60, 95, n), "RewardRisk": rng.uniform(2, 5, n),
        "ML_20d_Prob": rng.uniform(0.2, 0.5, n), "Sector": "Technology", "Market_Regime": "MODERATE_UP",
        "Entry_Price": 10.0, "Stop_Loss": 9.0, "Target_Price": 12.0, "Close": 10.0, "RS_63d": rng.normal(0, 0.1, n),
    })
    k = int(round(nan_share * n))
    df.loc[:k - 1, ["Close", "RS_63d"]] = np.nan
    return df


def _integrity_errors(df, tmp_path, monkeypatch):
    p = tmp_path / "latest_scan.parquet"; df.to_parquet(p)
    monkeypatch.setattr(sic, "PARQUET", p)
    return sic.check_parquet()[0]


@pytest.mark.parametrize("share,blocked", [(0.0, False), (0.18, False), (0.22, False), (0.49, True), (1.0, True)])
def test_integrity_blocks_only_real_damage(tmp_path, monkeypatch, share, blocked):
    errs = [e for e in _integrity_errors(_scan(nan_share=share), tmp_path, monkeypatch) if "price history missing" in e]
    assert bool(errs) is blocked


@pytest.mark.parametrize("share,red", [(0.0, False), (0.18, False), (0.22, True), (0.49, True), (1.0, True)])
def test_monitor_flags_from_20_percent(share, red):
    flags = {c for lvl, c, _ in evaluate(_scan(nan_share=share), vix_real=15.0)["flags"] if lvl == "RED"}
    assert ({"RS_COLUMN_EMPTY", "PRICE_DATA_MISSING"} <= flags) is red
