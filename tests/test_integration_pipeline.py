import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest

from core.pipeline_runner import run_scan_pipeline
from typing import Dict, Any


def make_synthetic_history(rows: int = 240) -> pd.DataFrame:
    """Create synthetic stock history for testing."""
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


def get_base_config() -> Dict[str, Any]:
    """Get base configuration for pipeline tests."""
    return {
        "LOOKBACK_DAYS": 200,
        "MA_LONG": 200,
        "BETA_FILTER_ENABLED": False,
        "FUNDAMENTAL_ENABLED": False,
        "BUDGET_TOTAL": 1000,
        "MIN_POSITION": 100,
        "MAX_POSITION_PCT": 20,
        "BETA_BENCHMARK": "SPY",
        "WEIGHTS": {
            "ma": 0.2, "mom": 0.25, "rsi": 0.2, "near_high_bell": 0.12,
            "vol": 0.02, "overext": 0.08, "pullback": 0.05,
            "risk_reward": 0.05, "macd": 0.02, "adx": 0.01
        },
    }


def test_minimal_pipeline_with_synthetic_data():
    """Test basic pipeline with synthetic data (no network calls)."""
    # Build synthetic data map to avoid network calls
    df = make_synthetic_history(240)
    universe = ["FAKE"]
    data_map = {"FAKE": df}

    config = get_base_config()

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


def test_pipeline_with_multiple_tickers():
    """Test pipeline with multiple synthetic tickers."""
    universe = ["TICK1", "TICK2", "TICK3"]
    data_map = {ticker: make_synthetic_history(240) for ticker in universe}

    config = get_base_config()

    results, out_map = run_scan_pipeline(
        universe=universe,
        config=config,
        status_callback=None,
        data_map=data_map,
    )

    assert isinstance(results, pd.DataFrame)
    assert "Ticker" in results.columns
    assert "Score" in results.columns
    # All tickers should be in output data map
    for ticker in universe:
        assert ticker in out_map


def test_pipeline_output_schema():
    """Test that pipeline output has expected columns."""
    df = make_synthetic_history(240)
    universe = ["TEST"]
    data_map = {"TEST": df}

    config = get_base_config()

    results, _ = run_scan_pipeline(
        universe=universe,
        config=config,
        status_callback=None,
        data_map=data_map,
    )

    # Required columns
    required_cols = ["Ticker", "Score", "סכום קנייה ($)"]
    for col in required_cols:
        assert col in results.columns, f"Missing required column: {col}"

    # If results exist, check data types
    if not results.empty:
        assert pd.api.types.is_string_dtype(results["Ticker"])
        assert pd.api.types.is_numeric_dtype(results["Score"])


def test_pipeline_status_callback():
    """Test that status callback is called during pipeline execution."""
    df = make_synthetic_history(240)
    universe = ["CALLBACK_TEST"]
    data_map = {"CALLBACK_TEST": df}

    config = get_base_config()
    callback_messages = []

    def status_callback(msg: str):
        callback_messages.append(msg)

    results, _ = run_scan_pipeline(
        universe=universe,
        config=config,
        status_callback=status_callback,
        data_map=data_map,
    )

    # Verify callback was called
    assert len(callback_messages) > 0
    # Check for expected status messages
    status_text = " ".join(callback_messages).lower()
    assert "pipeline" in status_text or "technical" in status_text or "indicators" in status_text


def test_pipeline_empty_universe():
    """Test pipeline with empty universe."""
    config = get_base_config()

    results, out_map = run_scan_pipeline(
        universe=[],
        config=config,
        status_callback=None,
        data_map=None,
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 0
    assert isinstance(out_map, dict)
