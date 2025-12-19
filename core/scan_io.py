"""
Scan I/O helpers for batch scanner and Streamlit app.

This module handles saving/loading precomputed scan results to avoid duplication
between batch_scan.py and stock_scout.py.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def save_scan(
    results_df: pd.DataFrame,
    config: Dict[str, Any],
    path_latest: Path,
    path_timestamped: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save scan results and metadata to disk.
    
    Args:
        results_df: Results DataFrame with columns: Ticker, Overall_Score, etc.
        config: Configuration dict used for this scan
        path_latest: Path to save as latest_scan.parquet
        path_timestamped: Optional timestamped backup path
        metadata: Optional additional metadata to include
    
    Raises:
        IOError: If write fails
    """
    try:
        # Ensure parent directory exists
        path_latest.parent.mkdir(parents=True, exist_ok=True)
        
        # Save DataFrame
        results_df.to_parquet(path_latest, index=False, engine="pyarrow")
        logger.info(f"Saved scan results to {path_latest} ({len(results_df)} tickers)")
        
        # Save timestamped backup if requested
        if path_timestamped:
            path_timestamped.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_parquet(path_timestamped, index=False, engine="pyarrow")
            logger.info(f"Saved timestamped backup to {path_timestamped}")
        
        # Build metadata
        meta = {
            "timestamp": datetime.now().isoformat(),
            "universe_size": len(results_df),
            "universe_name": config.get("UNIVERSE_NAME", "default"),
            "config_version": config.get("VERSION", "unknown"),
            "lookback_days": config.get("LOOKBACK_DAYS", 0),
            "columns": list(results_df.columns),
        }
        
        # Merge additional metadata
        if metadata:
            meta.update(metadata)
        
        # Save metadata JSON
        meta_path = path_latest.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved metadata to {meta_path}")
        
    except Exception as e:
        logger.error(f"Failed to save scan: {e}")
        raise IOError(f"Failed to save scan to {path_latest}: {e}")


def load_latest_scan(path_latest: Path) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    Load latest scan results and metadata from disk.
    
    Args:
        path_latest: Path to latest_scan.parquet
    
    Returns:
        Tuple of (results_df, metadata_dict) or (None, None) if not found/invalid
    """
    try:
        if not path_latest.exists():
            logger.warning(f"Scan file not found: {path_latest}")
            return None, None
        
        # Load DataFrame
        results_df = pd.read_parquet(path_latest, engine="pyarrow")
        logger.info(f"Loaded scan from {path_latest} ({len(results_df)} tickers)")
        
        # Load metadata
        meta_path = path_latest.with_suffix(".json")
        metadata = {}
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata from {meta_path}")
        else:
            logger.warning(f"Metadata file not found: {meta_path}")
            # Create minimal metadata from DataFrame
            metadata = {
                "timestamp": "unknown",
                "universe_size": len(results_df),
                "universe_name": "unknown",
                "config_version": "unknown",
            }
        
        # Validate DataFrame has expected columns
        required_cols = ["Ticker"]
        missing = [col for col in required_cols if col not in results_df.columns]
        if missing:
            logger.error(f"Scan missing required columns: {missing}")
            return None, None
        
        return results_df, metadata
        
    except Exception as e:
        logger.error(f"Failed to load scan from {path_latest}: {e}")
        return None, None


def list_available_scans(scan_dir: Path) -> list[Dict[str, Any]]:
    """
    List all available timestamped scans in directory.
    
    Args:
        scan_dir: Directory containing scan files
    
    Returns:
        List of dicts with scan metadata sorted by timestamp (newest first)
    """
    scans = []
    
    if not scan_dir.exists():
        return scans
    
    for parquet_file in scan_dir.glob("scan_*.parquet"):
        meta_file = parquet_file.with_suffix(".json")
        
        # Try to load metadata
        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                    meta["file_path"] = str(parquet_file)
                    scans.append(meta)
            except Exception as e:
                logger.warning(f"Failed to read metadata for {parquet_file}: {e}")
        else:
            # Fallback: extract from filename
            try:
                df = pd.read_parquet(parquet_file)
                scans.append({
                    "file_path": str(parquet_file),
                    "timestamp": parquet_file.stem.replace("scan_", ""),
                    "universe_size": len(df),
                    "universe_name": "unknown",
                })
            except Exception as e:
                logger.warning(f"Failed to read {parquet_file}: {e}")
    
    # Sort by timestamp descending
    scans.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return scans


def get_scan_summary(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for a scan DataFrame.
    
    Args:
        results_df: Scan results DataFrame
    
    Returns:
        Dict with summary metrics
    """
    summary = {
        "total_tickers": len(results_df),
        "columns": list(results_df.columns),
    }
    
    # Check for common columns and compute stats
    if "Overall_Score" in results_df.columns:
        summary["avg_overall_score"] = float(results_df["Overall_Score"].mean())
        summary["max_overall_score"] = float(results_df["Overall_Score"].max())
    
    if "Reliability_v2" in results_df.columns:
        summary["avg_reliability"] = float(results_df["Reliability_v2"].mean())
    
    if "Classification" in results_df.columns:
        summary["classification_counts"] = results_df["Classification"].value_counts().to_dict()
    
    if "Fund_Coverage_Pct" in results_df.columns:
        summary["avg_fundamental_coverage"] = float(results_df["Fund_Coverage_Pct"].mean())
    
    return summary
