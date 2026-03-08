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
                for col in ("Entry_Price", "Unit_Price", "Price_Yahoo"):
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
            try:
                self._recs_table.upsert(rec_row).execute()
                inserted += 1
            except Exception as exc:
                logger.warning("Failed to save rec %s: %s", rec_id, exc)

        logger.info("Saved scan %s to Supabase (%d recommendations)", scan_id, inserted)
        return inserted

    # --- Read ---------------------------------------------------------------

    def get_scan_history(self, days: int = 30, limit: int = 50) -> pd.DataFrame:
        """List recent scans as a DataFrame."""
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
            if not resp.data:
                return pd.DataFrame()
            return pd.DataFrame(resp.data)
        except Exception as exc:
            logger.warning("Failed to load scan history: %s", exc)
            return pd.DataFrame()

    def get_latest_scan(self) -> Optional[pd.DataFrame]:
        """Load most recent scan's recommendations."""
        try:
            resp = (
                self._scans_table
                .select("scan_id")
                .eq("user_id", self._user_id)
                .order("timestamp", desc=True)
                .limit(1)
                .execute()
            )
            if not resp.data:
                return None
            scan_id = resp.data[0]["scan_id"]
            return self.get_recommendations_for_scan(scan_id)
        except Exception as exc:
            logger.warning("Failed to load latest scan: %s", exc)
            return None

    def get_recommendations_for_scan(self, scan_id: str) -> Optional[pd.DataFrame]:
        """All recommendations for a specific scan."""
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
            return pd.DataFrame(resp.data)
        except Exception as exc:
            logger.warning("Failed to load scan %s: %s", scan_id, exc)
            return None
