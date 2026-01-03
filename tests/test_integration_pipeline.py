"""
Integration guarantees for the unified pipeline
- Verifies canonical schema fields exist after the main pipeline run
- Checks basic invariants for scores, ML prob, RR finiteness, and reliability
"""
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


def test_canonical_schema_and_invariants():
    """After running the pipeline, assert canonical columns and basic invariants.

    This asserts that the refactor centralization surfaced canonical fields
    and that risk-aware classification columns exist for downstream consumers.
    """
    # Use multiple synthetic tickers to increase chance of non-empty output
    universe = ["CANO1", "CANO2", "CANO3", "CANO4"]
    data_map = {t: make_synthetic_history(240) for t in universe}

    config = get_base_config()

    results, _ = run_scan_pipeline(
        universe=universe,
        config=config,
        status_callback=None,
        data_map=data_map,
    )

    # Schema checks (canonical)
    required = [
        "Ticker",
        "FinalScore_20d",
        "ML_20d_Prob",
        "FundamentalScore",
        # Momentum: prefer "MomentumScore" but accept canonical tech rank if used
        # in the current pipeline composition
        # (we still assert presence below via either-or)
        "RR",
        "ReliabilityScore",
        "RiskClass",
        "SafetyBlocked",
        "SafetyReasons",
    ]

    # Ensure base required columns exist
    for col in [c for c in required if c not in {"FundamentalScore", "ReliabilityScore"}]:
        assert col in results.columns, f"Missing required column: {col}"

    # Allow legacy-to-canonical mapping for fundamentals/reliability during migration
    assert (
        ("FundamentalScore" in results.columns)
        or ("Fundamental_Score" in results.columns)
        or ("Fundamental_S" in results.columns)
    ), "Missing FundamentalScore (or compatible legacy alias)"
    assert (
        ("ReliabilityScore" in results.columns)
        or ("Reliability_Score" in results.columns)
        or ("reliability_v2" in results.columns)
        or ("Reliability_v2" in results.columns)
    ), "Missing ReliabilityScore (or compatible legacy alias)"

    # Momentum canonical: prefer explicit MomentumScore, else accept TechScore_20d
    has_momentum = ("MomentumScore" in results.columns) or ("TechScore_20d" in results.columns)
    assert has_momentum, "Missing MomentumScore (or TechScore_20d as canonical momentum proxy)"

    # If no rows made it through filters, skip value-domain checks deterministically
    if results.empty:
        pytest.skip("Pipeline produced no rows for synthetic data")

    # Domain checks on non-rejected and not safety-blocked rows
    mask_ok = (results.get("RiskClass") != "REJECT") & (~results.get("SafetyBlocked").fillna(False))
    if mask_ok.sum() == 0:
        pytest.skip("No eligible rows (all rejected/blocked) in synthetic run")

    df_ok = results[mask_ok].copy()

    # FinalScore_20d in [0,100]
    assert ((df_ok["FinalScore_20d"] >= 0.0) & (df_ok["FinalScore_20d"] <= 100.0)).all()

    # ML_20d_Prob in [0,1] for non-null
    if "ML_20d_Prob" in df_ok.columns:
        ml = df_ok["ML_20d_Prob"].dropna()
        assert ((ml >= 0.0) & (ml <= 1.0)).all()

    # RR finite for eligible rows where a position could be taken
    if "RR" in df_ok.columns:
        rr = df_ok["RR"]
        assert np.isfinite(rr).all(), "RR must be finite for eligible rows"

    # ReliabilityScore within [0,100] where present
    rel_col = None
    for c in ["ReliabilityScore", "Reliability_Score", "reliability_v2", "Reliability_v2"]:
        if c in df_ok.columns:
            rel_col = c
            break
    if rel_col is not None:
        rel_vals = df_ok[rel_col].dropna().astype(float)
        # Handle legacy 0-1 scale if encountered
        if rel_vals.max() <= 1.0:
            rel_vals = rel_vals * 100.0
        assert ((rel_vals >= 0.0) & (rel_vals <= 100.0)).all()

    # Legacy fields are still allowed in tests for compatibility.
    # These are derived from FinalScore_20d, RiskClass, safety filters and reliability.
    for legacy in ["Risk_Level", "Data_Quality", "Should_Display"]:
        assert legacy in results.columns


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
