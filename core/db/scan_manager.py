"""Supabase-backed scan history manager.

Persists scan results to Supabase alongside local parquet files,
so scan history survives Streamlit Cloud redeployments.

Usage::

    sm = get_scan_manager()
    if sm is not None:
        sm.save_scan(scan_id, results_df, config, metadata)
        history = sm.get_scan_history(days=30)
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from core.db.store import _REC_COL_MAP, _safe_float, _safe_int, _safe_str

# Reverse mapping: DB snake_case → preferred pipeline PascalCase column names.
# _REC_COL_MAP is many-to-one (e.g. "Score" and "FinalScore_20d" both map to
# "final_score"), so we pick the canonical pipeline name for each DB column.
_DB_TO_PIPELINE: Dict[str, str] = {
    "ticker": "Ticker",
    "final_score": "FinalScore_20d",
    "tech_score": "TechScore_20d",
    "fundamental_score": "Fundamental_Score",
    "ml_prob": "ML_20d_Prob",
    "pattern_score": "PatternScore",
    "big_winner_score": "BigWinnerScore_20d",
    "reliability_score": "ReliabilityScore",
    "risk_meter": "Risk_Meter",
    "entry_price": "Entry",
    "target_price": "Target_20d",
    "stop_price": "Stop",
    "rr_ratio": "RR",
    "risk_class": "RiskClass",
    "risk_label": "Risk_Label",
    "market_regime": "Market_Regime",
    "sector": "Sector",
    "market_cap": "MarketCap",
    "rsi": "RSI",
    "atr_pct": "ATR_Pct",
    "volume_surge": "Volume_Surge",
    "ma_alignment": "MA_Alignment",
    "rs_vs_spy_20d": "RS_vs_SPY_20d",
    "fundamental_coverage_pct": "Fund_Coverage_Pct",
    "fundamental_sources_count": "Fundamental_Sources_Count",
    "data_quality": "Data_Quality",
    "holding_days": "Holding_Days",
}

logger = logging.getLogger("stock_scout.db.scan_manager")

DEFAULT_USER = "default"

# ---------------------------------------------------------------------------
# Per-user instance cache (thread-safe, same pattern as portfolio_manager.py)
# ---------------------------------------------------------------------------
_SM_INSTANCES: Dict[str, Optional["SupabaseScanManager"]] = {}
_SM_LOCK = threading.Lock()


def get_scan_manager(user_id: str = DEFAULT_USER) -> Optional["SupabaseScanManager"]:
    """Return a SupabaseScanManager if Supabase is configured, else None.

    Does NOT cache ``None`` — retries on each call until Supabase becomes
    available (e.g. after secrets are configured during a running session).
    """
    cached = _SM_INSTANCES.get(user_id)
    if cached is not None:
        return cached

    with _SM_LOCK:
        # Double-check after acquiring lock
        cached = _SM_INSTANCES.get(user_id)
        if cached is not None:
            return cached
        try:
            from core.db.supabase_client import get_supabase_client

            sb = get_supabase_client()
            if sb is not None:
                mgr = SupabaseScanManager(sb, user_id)
                _SM_INSTANCES[user_id] = mgr
                logger.info("Supabase scan manager ready for user=%s", user_id)
                return mgr
            # Don't cache None — allow retry when credentials become available
            return None
        except Exception as exc:
            logger.warning("Supabase scan manager init failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# SupabaseScanManager
# ---------------------------------------------------------------------------
class SupabaseScanManager:
    """Supabase-backed scan history — persists across Streamlit deploys."""

    def __init__(self, client, user_id: str = DEFAULT_USER):
        self._sb = client
        self._user_id = user_id

    @property
    def _scans_table(self):
        return self._sb.table("scan_history")

    @property
    def _recs_table(self):
        return self._sb.table("scan_recommendations")

    # --- Save ---------------------------------------------------------------

    def save_scan(
        self,
        scan_id: str,
        results_df: pd.DataFrame,
        config: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Persist a scan and its recommendations to Supabase.

        Returns the number of recommendations saved.
        """
        if results_df is None or results_df.empty:
            return 0

        meta = metadata or {}
        config = config or {}
        ts = meta.get("timestamp", datetime.now(timezone.utc).isoformat())

        # 1) Insert scan metadata
        scan_row = {
            "scan_id": scan_id,
            "user_id": self._user_id,
            "timestamp": ts,
            "universe_name": _safe_str(meta.get("universe_name")),
            "universe_size": _safe_int(meta.get("universe_size", len(results_df))),
            "market_regime": _safe_str(meta.get("market_regime")),
            "regime_confidence": _safe_float(meta.get("regime_confidence")),
            "config_json": json.dumps(config, default=str)[:10_000],
            "logic_version": _safe_str(meta.get("logic_version")),
            "ml_model_version": _safe_str(meta.get("ml_model_version")),
            "total_scored": _safe_int(meta.get("total_scored")),
            "total_recommended": len(results_df),
            "scan_type": _safe_str(meta.get("scan_type", "manual")),
            "scan_duration_secs": _safe_float(meta.get("scan_duration_seconds")),
        }
        try:
            self._scans_table.upsert(scan_row).execute()
        except Exception as exc:
            logger.error("Failed to save scan metadata to Supabase: %s", exc)
            return 0

        # 2) Insert recommendations (batch build, individual upsert)
        inserted = 0
        for _, row in results_df.iterrows():
            ticker = _safe_str(row.get("Ticker", row.get("ticker", "")))
            if not ticker:
                continue
            rec_id = f"{scan_id}::{ticker}"

            # Map DataFrame columns to DB columns via _REC_COL_MAP
            mapped: Dict[str, Any] = {}
            for df_col, db_col in _REC_COL_MAP.items():
                if df_col in row.index and db_col not in mapped:
                    mapped[db_col] = row[df_col]

            # Also try pipeline-specific column names not in _REC_COL_MAP
            if "entry_price" not in mapped or mapped["entry_price"] is None:
                for col in ("Entry_Price", "Price_Yahoo", "Close"):
                    if col in row.index and row[col] is not None:
                        mapped["entry_price"] = row[col]
                        break
            if "target_price" not in mapped or mapped["target_price"] is None:
                for col in ("Target_Price",):
                    if col in row.index and row[col] is not None:
                        mapped["target_price"] = row[col]
                        break
            if "stop_price" not in mapped or mapped["stop_price"] is None:
                for col in ("Stop_Loss",):
                    if col in row.index and row[col] is not None:
                        mapped["stop_price"] = row[col]
                        break

            rec_row = {
                "id": rec_id,
                "scan_id": scan_id,
                "user_id": self._user_id,
                "ticker": ticker,
                "scan_timestamp": ts,
                "final_score": _safe_float(mapped.get("final_score")),
                "tech_score": _safe_float(mapped.get("tech_score")),
                "fundamental_score": _safe_float(mapped.get("fundamental_score")),
                "ml_prob": _safe_float(mapped.get("ml_prob")),
                "pattern_score": _safe_float(mapped.get("pattern_score")),
                "reliability_score": _safe_float(mapped.get("reliability_score")),
                "risk_meter": _safe_float(mapped.get("risk_meter")),
                "entry_price": _safe_float(mapped.get("entry_price")),
                "target_price": _safe_float(mapped.get("target_price")),
                "stop_price": _safe_float(mapped.get("stop_price")),
                "rr_ratio": _safe_float(mapped.get("rr_ratio")),
                "holding_days": _safe_int(row.get("Holding_Days", row.get("holding_days"))),
                "risk_class": _safe_str(mapped.get("risk_class")),
                "risk_label": _safe_str(mapped.get("risk_label")),
                "market_regime": _safe_str(mapped.get("market_regime")),
                "sector": _safe_str(mapped.get("sector")),
                "market_cap": _safe_float(mapped.get("market_cap")),
                "rsi": _safe_float(mapped.get("rsi")),
                "atr_pct": _safe_float(mapped.get("atr_pct")),
                "volume_surge": _safe_float(mapped.get("volume_surge")),
                "ma_alignment": _safe_float(mapped.get("ma_alignment")),
                "rs_vs_spy_20d": _safe_float(mapped.get("rs_vs_spy_20d")),
                "fundamental_coverage_pct": _safe_float(mapped.get("fundamental_coverage_pct")),
                "fundamental_sources_count": _safe_int(mapped.get("fundamental_sources_count")),
                "data_quality": _safe_str(mapped.get("data_quality")),
            }
            # Store full pipeline row as JSON blob for lossless historical loading
            try:
                _row_dict = {}
                for col in row.index:
                    val = row[col]
                    if isinstance(val, (np.integer,)):
                        val = int(val)
                    elif isinstance(val, (np.floating,)):
                        val = float(val) if np.isfinite(val) else None
                    elif isinstance(val, (np.bool_,)):
                        val = bool(val)
                    elif pd.isna(val):
                        val = None
                    elif not isinstance(val, (int, float, bool, str, type(None))):
                        val = str(val)
                    _row_dict[col] = val
                rec_row["full_row_json"] = json.dumps(_row_dict, default=str)
            except Exception as _json_exc:
                logger.debug("Could not serialize full row for %s: %s", ticker, _json_exc)
                rec_row["full_row_json"] = None
            try:
                self._recs_table.upsert(rec_row).execute()
                inserted += 1
            except Exception as exc:
                logger.warning("Failed to save rec %s: %s", rec_id, exc)

        logger.info("Saved scan %s to Supabase (%d recommendations)", scan_id, inserted)
        return inserted

    # --- Read ---------------------------------------------------------------

    def get_scan_history(self, days: int = 30, limit: int = 50) -> pd.DataFrame:
        """List recent scans as a DataFrame.

        Queries by user_id first; if empty, retries without user_id filter
        to handle user_id mismatches (local/test/SSO) in single-user mode.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        try:
            resp = (
                self._scans_table
                .select(
                    "scan_id, timestamp, universe_name, universe_size, "
                    "market_regime, total_recommended, ml_model_version, scan_type"
                )
                .eq("user_id", self._user_id)
                .gte("timestamp", cutoff)
                .order("timestamp", desc=True)
                .limit(limit)
                .execute()
            )
            if resp.data:
                return pd.DataFrame(resp.data)
            # Fallback: query without user_id filter (single-user mode)
            logger.info("No scans for user_id=%s, retrying without user filter", self._user_id)
            resp = (
                self._scans_table
                .select(
                    "scan_id, timestamp, universe_name, universe_size, "
                    "market_regime, total_recommended, ml_model_version, scan_type"
                )
                .gte("timestamp", cutoff)
                .order("timestamp", desc=True)
                .limit(limit)
                .execute()
            )
            if not resp.data:
                return pd.DataFrame()
            return pd.DataFrame(resp.data)
        except Exception as exc:
            logger.warning("Failed to load scan history: %s", exc)
            return pd.DataFrame()

    def _latest_scan_id(self) -> Optional[str]:
        """Return the scan_id of the most recent scan.

        Tries user-specific first, falls back to any user (single-user mode).
        """
        for use_filter in (True, False):
            query = self._scans_table.select("scan_id").order("timestamp", desc=True).limit(1)
            if use_filter:
                query = query.eq("user_id", self._user_id)
            resp = query.execute()
            if resp.data:
                return resp.data[0]["scan_id"]
            if use_filter:
                logger.info("No latest scan for user_id=%s, retrying without filter", self._user_id)
        return None

    def get_latest_scan(self) -> Optional[pd.DataFrame]:
        """Load most recent scan's recommendations."""
        try:
            scan_id = self._latest_scan_id()
            if not scan_id:
                return None
            return self.get_recommendations_for_scan(scan_id)
        except Exception as exc:
            logger.warning("Failed to load latest scan: %s", exc)
            return None

    def get_latest_scan_meta(self) -> Optional[Dict[str, Any]]:
        """Return metadata dict for the most recent scan, or None."""
        try:
            resp = (
                self._scans_table
                .select("*")
                .eq("user_id", self._user_id)
                .order("timestamp", desc=True)
                .limit(1)
                .execute()
            )
            # Fallback without user filter
            if not resp.data:
                resp = (
                    self._scans_table
                    .select("*")
                    .order("timestamp", desc=True)
                    .limit(1)
                    .execute()
                )
            if not resp.data:
                return None
            return resp.data[0]
        except Exception as exc:
            logger.warning("Failed to load latest scan meta: %s", exc)
            return None

    def get_recommendations_for_scan(self, scan_id: str) -> Optional[pd.DataFrame]:
        """All recommendations for a specific scan.

        Returns a DataFrame with pipeline-style PascalCase column names
        (e.g. ``FinalScore_20d``, ``TechScore_20d``) so the UI can consume
        it without special-casing Supabase data.
        """
        try:
            resp = (
                self._recs_table
                .select("*")
                .eq("scan_id", scan_id)
                .order("final_score", desc=True)
                .execute()
            )
            if not resp.data:
                return None
            df = pd.DataFrame(resp.data)

            # If full_row_json is available, expand it for lossless data
            if "full_row_json" in df.columns and df["full_row_json"].notna().any():
                expanded = []
                for _, row in df.iterrows():
                    blob = row.get("full_row_json")
                    if blob and isinstance(blob, (str, dict)):
                        d = json.loads(blob) if isinstance(blob, str) else blob
                        expanded.append(d)
                    else:
                        # Fallback: use the explicit columns with rename
                        row_dict = row.to_dict()
                        renamed = {_DB_TO_PIPELINE.get(k, k): v for k, v in row_dict.items()}
                        expanded.append(renamed)
                if expanded:
                    df = pd.DataFrame(expanded)
            else:
                # Legacy path: rename DB columns → pipeline names
                rename = {k: v for k, v in _DB_TO_PIPELINE.items() if k in df.columns}
                df = df.rename(columns=rename)

            # Ensure Score alias exists
            if "FinalScore_20d" in df.columns and "Score" not in df.columns:
                df["Score"] = df["FinalScore_20d"]
            return df
        except Exception as exc:
            logger.warning("Failed to load scan %s: %s", scan_id, exc)
            return None
