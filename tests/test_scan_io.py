"""
Unit tests for core/scan_io.py — Scan persistence and loading.
"""
import pytest
import pandas as pd
import json
from pathlib import Path
from core.scan_io import save_scan, load_latest_scan, list_available_scans, get_scan_summary


@pytest.fixture
def sample_results_df():
    """Sample results DataFrame for testing."""
    return pd.DataFrame({
        "Ticker": ["AAPL", "GOOGL", "MSFT"],
        "Score": [85.5, 78.2, 92.1],
        "Risk_Level": ["core", "core", "core"],
        "Classification": ["Strong Buy", "Buy", "Strong Buy"],
        "buy_amount_v2": [500.0, 300.0, 600.0]
    })


@pytest.fixture
def sample_config():
    """Sample config dictionary."""
    return {
        "LOOKBACK_DAYS": 180,
        "UNIVERSE_LIMIT": 100,
        "BUDGET_TOTAL": 10000
    }


@pytest.fixture
def temp_scan_dir(tmp_path):
    """Temporary directory for scan files."""
    scan_dir = tmp_path / "scans"
    scan_dir.mkdir()
    return scan_dir


def test_save_scan_creates_files(sample_results_df, sample_config, temp_scan_dir):
    """Test that save_scan creates both parquet and JSON files."""
    path_latest = temp_scan_dir / "latest_scan.parquet"
    path_timestamped = temp_scan_dir / "scan_20240101_1200.parquet"
    
    metadata = {
        "timestamp": "20240101_1200",
        "universe_size": 3,
        "universe_name": "test",
        "config_version": "v2.0",
        "lookback_days": 180,
        "columns": list(sample_results_df.columns)
    }
    
    save_scan(sample_results_df, sample_config, path_latest, path_timestamped, metadata)
    
    # Check both parquet files exist
    assert path_latest.exists(), "latest_scan.parquet not created"
    assert path_timestamped.exists(), "Timestamped scan not created"
    
    # Check JSON metadata files exist
    assert (temp_scan_dir / "latest_scan.json").exists(), "latest_scan.json not created"
    # Note: save_scan only creates JSON for latest, not for timestamped
    
    # Verify metadata content
    with open(temp_scan_dir / "latest_scan.json", "r") as f:
        loaded_meta = json.load(f)
        assert loaded_meta["universe_size"] == 3
        assert "timestamp" in loaded_meta  # Uses datetime.now(), not passed metadata


def test_load_latest_scan_success(sample_results_df, sample_config, temp_scan_dir):
    """Test loading an existing scan file."""
    path_latest = temp_scan_dir / "latest_scan.parquet"
    path_timestamped = temp_scan_dir / "scan_20240101_1200.parquet"
    
    metadata = {
        "timestamp": "20240101_1200",
        "universe_size": 3,
        "universe_name": "test",
        "config_version": "v2.0",
        "lookback_days": 180,
        "columns": list(sample_results_df.columns)
    }
    
    # Save first
    save_scan(sample_results_df, sample_config, path_latest, path_timestamped, metadata)
    
    # Load
    df, meta = load_latest_scan(path_latest)
    
    assert df is not None, "DataFrame should not be None"
    assert meta is not None, "Metadata should not be None"
    assert len(df) == 3, "Should load 3 rows"
    assert "Ticker" in df.columns, "Ticker column should exist"
    assert meta["universe_size"] == 3, "Metadata should match"
    assert meta["timestamp"] == "20240101_1200"


def test_load_latest_scan_missing_file(temp_scan_dir):
    """Test loading when file doesn't exist returns (None, None)."""
    path_latest = temp_scan_dir / "nonexistent.parquet"
    
    df, meta = load_latest_scan(path_latest)
    
    assert df is None, "DataFrame should be None for missing file"
    assert meta is None, "Metadata should be None for missing file"


def test_load_latest_scan_missing_metadata(sample_results_df, temp_scan_dir):
    """Test loading when parquet exists but JSON metadata is missing."""
    path_latest = temp_scan_dir / "latest_scan.parquet"
    
    # Save parquet without using save_scan (so no JSON)
    sample_results_df.to_parquet(path_latest, engine="pyarrow", compression="snappy")
    
    df, meta = load_latest_scan(path_latest)
    
    assert df is not None, "DataFrame should load even without metadata"
    # load_latest_scan creates minimal metadata dict when JSON missing
    assert meta["timestamp"] == "unknown"
    assert meta["universe_size"] == 3
    assert len(df) == 3


def test_list_available_scans(sample_results_df, sample_config, temp_scan_dir):
    """Test listing timestamped scan files."""
    # Create multiple timestamped scans
    for timestamp in ["20240101_1200", "20240102_1400", "20240103_0900"]:
        path_timestamped = temp_scan_dir / f"scan_{timestamp}.parquet"
        path_latest = temp_scan_dir / "latest_scan.parquet"  # Will be overwritten
        
        metadata = {
            "timestamp": timestamp,
            "universe_size": 3,
            "universe_name": "test",
            "config_version": "v2.0",
            "lookback_days": 180,
            "columns": list(sample_results_df.columns)
        }
        
        save_scan(sample_results_df, sample_config, path_latest, path_timestamped, metadata)
    
    # List scans - returns list of dicts, not filenames
    scans = list_available_scans(temp_scan_dir)
    
    assert len(scans) == 3, "Should find 3 timestamped scans"
    assert all("file_path" in scan for scan in scans), "All should have file_path"
    assert all("timestamp" in scan for scan in scans), "All should have timestamp"
    
    # Should be sorted by timestamp (newest first)
    # Timestamps are extracted from filenames since save_scan doesn't create JSON for timestamped files
    assert "20240103_0900" in scans[0]["timestamp"]
    assert "20240101_1200" in scans[-1]["timestamp"]


def test_list_available_scans_empty_dir(temp_scan_dir):
    """Test listing scans in empty directory."""
    scans = list_available_scans(temp_scan_dir)
    assert scans == [], "Empty directory should return empty list"


def test_get_scan_summary(sample_results_df):
    """Test summary statistics generation."""
    summary = get_scan_summary(sample_results_df)
    
    assert "total_tickers" in summary
    assert summary["total_tickers"] == 3
    # Note: get_scan_summary looks for "Overall_Score", not "Score"
    # Since sample_results_df has "Score", stats won't be computed
    assert "columns" in summary
    assert "classification_counts" in summary
    assert summary["classification_counts"]["Strong Buy"] == 2
    assert summary["classification_counts"]["Buy"] == 1


def test_get_scan_summary_empty_df():
    """Test summary with empty DataFrame."""
    empty_df = pd.DataFrame()
    summary = get_scan_summary(empty_df)
    
    assert summary["total_tickers"] == 0
    # Empty df won't have score columns, so no avg_score key
    assert "columns" in summary


def test_save_scan_preserves_datatypes(sample_results_df, sample_config, temp_scan_dir):
    """Test that save/load preserves DataFrame dtypes."""
    path_latest = temp_scan_dir / "latest_scan.parquet"
    path_timestamped = temp_scan_dir / "scan_test.parquet"
    
    metadata = {
        "timestamp": "test",
        "universe_size": 3,
        "universe_name": "test",
        "config_version": "v2.0",
        "lookback_days": 180,
        "columns": list(sample_results_df.columns)
    }
    
    save_scan(sample_results_df, sample_config, path_latest, path_timestamped, metadata)
    
    df, _ = load_latest_scan(path_latest)
    
    # Check dtypes preserved
    assert df["Score"].dtype == float, "Score should be float"
    assert df["Risk_Level"].dtype == object, "Risk_Level should be object (string)"
    assert df["buy_amount_v2"].dtype == float, "buy_amount_v2 should be float"


def test_save_scan_with_special_characters(temp_scan_dir, sample_config):
    """Test saving DataFrame with Hebrew columns (from actual app)."""
    df_hebrew = pd.DataFrame({
        "Ticker": ["AAPL"],
        "Score": [85.5],
        "סכום קנייה ($)": [500.0],
        "מניות לקנייה": [10]
    })
    
    path_latest = temp_scan_dir / "latest_scan.parquet"
    path_timestamped = temp_scan_dir / "scan_hebrew.parquet"
    
    metadata = {
        "timestamp": "test",
        "universe_size": 1,
        "universe_name": "test",
        "config_version": "v2.0",
        "lookback_days": 180,
        "columns": list(df_hebrew.columns)
    }
    
    # Should not raise
    save_scan(df_hebrew, sample_config, path_latest, path_timestamped, metadata)
    
    df, _ = load_latest_scan(path_latest)
    assert df is not None
    assert "סכום קנייה ($)" in df.columns
    assert "מניות לקנייה" in df.columns


def test_live_style_save_roundtrip(tmp_path: Path):
    """Simulate a 'save from live' call and ensure load works."""
    from core.scan_io import save_scan, load_latest_scan
    df = pd.DataFrame({"Ticker": ["GOOG"], "Overall_Score": [77]})
    config = {"UNIVERSE_NAME": "live_test", "LOOKBACK_DAYS": 180, "VERSION": "v2"}
    latest_path = tmp_path / "latest_scan.parquet"
    ts_path = tmp_path / "scan_20990101_000101.parquet"
    # Perform save (like from Streamlit live run)
    save_scan(df, config, latest_path, ts_path)
    # Load and validate
    loaded_df, meta = load_latest_scan(latest_path)
    assert loaded_df is not None and len(loaded_df) == 1
    assert isinstance(meta, dict)
    assert meta.get("universe_size") == 1
