"""DuckDB-backed scan store for Stock Scout.

Provides queryable storage for scan history, recommendations, and outcome
tracking.  Sits alongside the existing Parquet export (backward-compatible).

Usage::

    store = get_scan_store()           # singleton, lazy init
    store.save_scan(scan_id, df, config, meta)
    latest = store.get_latest_scan()
    history = store.get_scan_history(days=30)
    custom  = store.query("SELECT ticker, avg(return_20d) FROM outcomes GROUP BY ticker")
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.db.schema import ALL_TABLES, SCHEMA_VERSION

logger = logging.getLogger("stock_scout.db")

_DB_PATH_DEFAULT = os.path.join("data", "stockscout.duckdb")

# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------
_STORE_SINGLETON: Optional["ScanStore"] = None
_STORE_LOCK = threading.Lock()


def get_scan_store(db_path: Optional[str] = None) -> "ScanStore":
    """Return (and lazily create) the global ScanStore singleton."""
    global _STORE_SINGLETON
    if _STORE_SINGLETON is None:
        with _STORE_LOCK:
            if _STORE_SINGLETON is None:
                _STORE_SINGLETON = ScanStore(db_path or _DB_PATH_DEFAULT)
                _STORE_SINGLETON.initialize()
    return _STORE_SINGLETON


# ---------------------------------------------------------------------------
# Column mapping: DataFrame column → DB column
# ---------------------------------------------------------------------------
_REC_COL_MAP: Dict[str, str] = {
    "Ticker":                     "ticker",
    "FinalScore_20d":             "final_score",
    "Score":                      "final_score",
    "TechScore_20d":              "tech_score",
    "Fundamental_Score":          "fundamental_score",
    "Fundamental_S":              "fundamental_score",
    "ML_20d_Prob":                "ml_prob",
    "PatternScore":               "pattern_score",
    "BigWinnerScore_20d":         "big_winner_score",
    "ReliabilityScore":           "reliability_score",
    "Reliability_Score":          "reliability_score",
    "Risk_Meter":                 "risk_meter",
    "risk_meter_v2":              "risk_meter",
    "Entry":                      "entry_price",
    "Close":                      "entry_price",
    "Target_20d":                 "target_price",
    "Stop":                       "stop_price",
    "RR":                         "rr_ratio",
    "RiskClass":                  "risk_class",
    "Risk_Label":                 "risk_label",
    "Market_Regime":              "market_regime",
    "Sector":                     "sector",
    "market_cap":                 "market_cap",
    "MarketCap":                  "market_cap",
    "RSI":                        "rsi",
    "ATR_Pct":                    "atr_pct",
    "VolSurge":                   "volume_surge",
    "Volume_Surge":               "volume_surge",
    "MA_Alignment":               "ma_alignment",
    "RS_vs_SPY_20d":              "rs_vs_spy_20d",
    "relative_strength_20d":      "rs_vs_spy_20d",
    "Fundamental_Coverage_Pct":   "fundamental_coverage_pct",
    "Fundamental_Sources_Count":  "fundamental_sources_count",
    "Data_Quality":               "data_quality",
}


def _safe_float(val: Any) -> Optional[float]:
    """Convert to float, returning None for NaN / non-numeric."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return int(f)
    except (TypeError, ValueError):
        return None


def _safe_str(val: Any) -> Optional[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return str(val)


# ---------------------------------------------------------------------------
# ScanStore
# ---------------------------------------------------------------------------
class ScanStore:
    """DuckDB-backed store for scan history and outcome tracking."""

    def __init__(self, db_path: str = _DB_PATH_DEFAULT):
        self._db_path = db_path
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _connect(self):
        """Return a fresh DuckDB connection (connections are cheap)."""
        import duckdb

        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(self._db_path)

    def initialize(self) -> None:
        """Create tables if they don't exist, apply migrations."""
        if self._initialized:
            return
        con = self._connect()
        try:
            for ddl in ALL_TABLES:
                con.execute(ddl)
            # Record schema version if not present
            existing = con.execute(
                "SELECT max(version) FROM schema_version"
            ).fetchone()[0]
            if existing is None or existing < SCHEMA_VERSION:
                con.execute(
                    "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                    [SCHEMA_VERSION],
                )
            self._initialized = True
            logger.info("ScanStore initialized at %s (schema v%d)",
                        self._db_path, SCHEMA_VERSION)
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save_scan(
        self,
        scan_id: str,
        results_df: pd.DataFrame,
        config: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Persist a scan and its recommendations.  Returns row count."""
        if results_df is None or results_df.empty:
            logger.warning("save_scan: empty results, skipping")
            return 0

        meta = metadata or {}
        config = config or {}
        ts = meta.get("timestamp", datetime.now(timezone.utc).isoformat())
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                ts = datetime.now(timezone.utc)

        con = self._connect()
        try:
            # 1) Insert scan metadata
            con.execute(
                """INSERT OR REPLACE INTO scans
                   (scan_id, timestamp, universe_name, universe_size,
                    market_regime, regime_confidence, config_json,
                    logic_version, ml_model_version,
                    total_scored, total_recommended)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    scan_id,
                    ts,
                    _safe_str(meta.get("universe_name")),
                    _safe_int(meta.get("universe_size", len(results_df))),
                    _safe_str(meta.get("market_regime")),
                    _safe_float(meta.get("regime_confidence")),
                    json.dumps(config, default=str)[:10_000],
                    _safe_str(meta.get("logic_version")),
                    _safe_str(meta.get("ml_model_version")),
                    _safe_int(meta.get("total_scored")),
                    len(results_df),
                ],
            )

            # 2) Insert recommendations
            inserted = 0
            for _, row in results_df.iterrows():
                ticker = _safe_str(
                    row.get("Ticker", row.get("ticker", row.name))
                )
                if not ticker:
                    continue
                rec_id = f"{scan_id}::{ticker}"

                # Map DataFrame columns → DB columns
                mapped: Dict[str, Any] = {}
                for df_col, db_col in _REC_COL_MAP.items():
                    if df_col in row.index and db_col not in mapped:
                        mapped[db_col] = row[df_col]

                con.execute(
                    """INSERT OR REPLACE INTO recommendations
                       (id, scan_id, ticker, scan_timestamp,
                        final_score, tech_score, fundamental_score,
                        ml_prob, pattern_score, big_winner_score,
                        reliability_score, risk_meter,
                        entry_price, target_price, stop_price, rr_ratio,
                        risk_class, risk_label,
                        market_regime, sector, market_cap,
                        rsi, atr_pct, volume_surge, ma_alignment,
                        rs_vs_spy_20d,
                        fundamental_coverage_pct,
                        fundamental_sources_count, data_quality)
                       VALUES (?,?,?,?, ?,?,?, ?,?,?, ?,?, ?,?,?,?, ?,?, ?,?,?, ?,?,?,?,?, ?,?,?)""",
                    [
                        rec_id, scan_id, ticker, ts,
                        _safe_float(mapped.get("final_score")),
                        _safe_float(mapped.get("tech_score")),
                        _safe_float(mapped.get("fundamental_score")),
                        _safe_float(mapped.get("ml_prob")),
                        _safe_float(mapped.get("pattern_score")),
                        _safe_float(mapped.get("big_winner_score")),
                        _safe_float(mapped.get("reliability_score")),
                        _safe_float(mapped.get("risk_meter")),
                        _safe_float(mapped.get("entry_price")),
                        _safe_float(mapped.get("target_price")),
                        _safe_float(mapped.get("stop_price")),
                        _safe_float(mapped.get("rr_ratio")),
                        _safe_str(mapped.get("risk_class")),
                        _safe_str(mapped.get("risk_label")),
                        _safe_str(mapped.get("market_regime")),
                        _safe_str(mapped.get("sector")),
                        _safe_float(mapped.get("market_cap")),
                        _safe_float(mapped.get("rsi")),
                        _safe_float(mapped.get("atr_pct")),
                        _safe_float(mapped.get("volume_surge")),
                        _safe_float(mapped.get("ma_alignment")),
                        _safe_float(mapped.get("rs_vs_spy_20d")),
                        _safe_float(mapped.get("fundamental_coverage_pct")),
                        _safe_int(mapped.get("fundamental_sources_count")),
                        _safe_str(mapped.get("data_quality")),
                    ],
                )
                inserted += 1

            logger.info("Saved scan %s with %d recommendations", scan_id, inserted)
            return inserted
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def get_latest_scan(self) -> Optional[pd.DataFrame]:
        """Load the most recent scan's recommendations as DataFrame."""
        con = self._connect()
        try:
            row = con.execute(
                "SELECT scan_id FROM scans ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            scan_id = row[0]
            df = con.execute(
                "SELECT * FROM recommendations WHERE scan_id = ? ORDER BY final_score DESC",
                [scan_id],
            ).fetchdf()
            return df if not df.empty else None
        finally:
            con.close()

    def get_scan_history(self, days: int = 30) -> pd.DataFrame:
        """List recent scans with metadata."""
        days = int(days)  # sanitise
        con = self._connect()
        try:
            return con.execute(
                f"""SELECT scan_id, timestamp, universe_name, universe_size,
                          market_regime, total_recommended,
                          ml_model_version
                   FROM scans
                   WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '{days}' DAY
                   ORDER BY timestamp DESC"""
            ).fetchdf()
        finally:
            con.close()

    def get_recommendations_for_scan(self, scan_id: str) -> pd.DataFrame:
        """All recommendations for a specific scan."""
        con = self._connect()
        try:
            return con.execute(
                "SELECT * FROM recommendations WHERE scan_id = ? ORDER BY final_score DESC",
                [scan_id],
            ).fetchdf()
        finally:
            con.close()

    def query(self, sql: str, params: Optional[List] = None) -> pd.DataFrame:
        """Execute arbitrary read-only SQL and return DataFrame."""
        con = self._connect()
        try:
            if params:
                return con.execute(sql, params).fetchdf()
            return con.execute(sql).fetchdf()
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def generate_scan_id(self) -> str:
        """Generate a unique scan ID."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        short_uuid = uuid.uuid4().hex[:8]
        return f"scan_{ts}_{short_uuid}"

    def get_stats(self) -> Dict[str, Any]:
        """Quick overview of what's in the database."""
        con = self._connect()
        try:
            scans = con.execute("SELECT count(*) FROM scans").fetchone()[0]
            recs = con.execute("SELECT count(*) FROM recommendations").fetchone()[0]
            outcomes = con.execute("SELECT count(*) FROM outcomes").fetchone()[0]
            pending = con.execute(
                "SELECT count(*) FROM outcomes WHERE status = 'pending'"
            ).fetchone()[0]
            return {
                "total_scans": scans,
                "total_recommendations": recs,
                "total_outcomes": outcomes,
                "pending_outcomes": pending,
                "db_path": self._db_path,
                "schema_version": SCHEMA_VERSION,
            }
        finally:
            con.close()
