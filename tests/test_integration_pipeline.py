import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.pipeline_runner import run_scan_pipeline
from typing import Dict, Any


def make_synthetic_history(rows: int = 240) -> pd.DataFrame:
    dates = pd.date_range(end=datetime.utcnow().date(), periods=rows, freq="B")
    base = np.linspace(100.0, 120.0, rows)
    noise = np.random.normal(0, 0.5, rows)
    close = base + noise
    high = close + np.abs(np.random.normal(0, 0.8, rows))
    low = close - np.abs(np.random.normal(0, 0.8, rows))
    open_ = close + np.random.normal(0, 0.3, rows)
    vol = np.random.randint(100000, 2000000, rows)
    df = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol,
    }, index=dates)
    return df


def test_minimal_pipeline_with_synthetic_data():
    # Build synthetic data map to avoid network calls
    df = make_synthetic_history(240)
    universe = ["FAKE"]
    data_map = {"FAKE": df}

    # Configure minimal run (no fundamentals, no beta filter)
    # Minimal config dict expected by pipeline_runner
    config: Dict[str, Any] = {
        "LOOKBACK_DAYS": 200,
        "MA_LONG": 200,
        "BETA_FILTER_ENABLED": False,
        "FUNDAMENTAL_ENABLED": False,
        "BUDGET_TOTAL": 1000,
        "MIN_POSITION": 100,
        "MAX_POSITION_PCT": 20,
        "BETA_BENCHMARK": "SPY",
        "WEIGHTS": {"ma": 0.2, "mom": 0.25, "rsi": 0.2, "near_high_bell": 0.12, "vol": 0.02, "overext": 0.08, "pullback": 0.05, "risk_reward": 0.05, "macd": 0.02, "adx": 0.01},
    }

    results, out_map = run_scan_pipeline(
        universe=universe,
        config=config,
        status_callback=None,
        data_map=data_map,
    )

    # Basic assertions
    assert isinstance(results, pd.DataFrame)
    # Results may be empty due to strict filters; validate schema instead
    assert "Ticker" in results.columns
    assert "Score" in results.columns

    # Allocation column should exist (Hebrew buy amount)
    assert "סכום קנייה ($)" in results.columns
    # If rows exist, ensure non-negative allocation values
    if not results.empty:
        assert results["סכום קנייה ($)"].iloc[0] >= 0.0
    # Data map propagated
    assert "FAKE" in out_map
