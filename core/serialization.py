"""
Serialization helpers for contracts.

Transforms `ScanResult` into data structures convenient for UI/scripts
without exposing legacy pipeline internals.
"""
from __future__ import annotations

from typing import Dict, Any
import pandas as pd

from core.contracts import ScanResult, Recommendation

# Column mapping for CSV/UI output; source is snake_case contract fields.
CSV_COLUMN_MAP = {
    "consolidation_ratio": "Consolidation_Ratio",
    "pocket_pivot_ratio": "Pocket_Pivot_Ratio",
    "vcp_ratio": "VCP_Ratio",
}


def _rec_to_row(rec: Recommendation) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "Ticker": rec.ticker,
        "FinalScore_20d": rec.final_score_20d,
        "Score": rec.final_score_20d,
        "Beta": rec.beta,
        "RR": rec.rr,
    }
    # Targets
    if rec.targets is not None:
        row["Entry_Price"] = rec.targets.entry
        row["Target_Price"] = rec.targets.target_20d
        row["Stop_Loss"] = rec.targets.stop_loss
    # Risk sizing
    if rec.risk_sizing is not None:
        if rec.risk_sizing.position_size_usd is not None:
            row["סכום קנייה ($)"] = rec.risk_sizing.position_size_usd
            row["buy_amount_v2"] = rec.risk_sizing.position_size_usd
    # Score breakdown flattened
    for k, v in (rec.scores_breakdown or {}).items():
        # Map common keys to canonical names when possible
        if k == "Fundamental_Score":
            row["Fundamental_Score"] = v
        elif k == "Fundamental_S":
            row["Fundamental_S"] = v
        else:
            row[k] = v
    # Advanced/Meteor signals
    for field, col in CSV_COLUMN_MAP.items():
        val = getattr(rec, field, None)
        if val is not None:
            row[col] = val
    # Classification fields (when present)
    if getattr(rec, "risk_class", None) is not None:
        row["RiskClass"] = rec.risk_class
    if getattr(rec, "safety_blocked", None) is not None:
        row["SafetyBlocked"] = rec.safety_blocked
    if getattr(rec, "safety_reasons", None) is not None:
        row["SafetyReasons"] = rec.safety_reasons
    # Legacy-compatible classification fields
    if getattr(rec, "risk_level", None) is not None:
        row["Risk_Level"] = rec.risk_level
    if getattr(rec, "data_quality", None) is not None:
        row["Data_Quality"] = rec.data_quality
    if getattr(rec, "confidence_level", None) is not None:
        row["Confidence_Level"] = rec.confidence_level
    if getattr(rec, "should_display", None) is not None:
        row["Should_Display"] = rec.should_display
    # Reasons
    if rec.reasons:
        # Keep primary reason in a single column for UI, others in list
        row["RejectionReason"] = rec.reasons[0]
        row["Reasons_List"] = rec.reasons
    return row


def scanresult_to_dataframe(result: ScanResult) -> pd.DataFrame:
    """Convert `ScanResult` to a pandas DataFrame of recommendations.

    Includes commonly used columns in the existing UI, with sensible defaults.
    """
    rows = [_rec_to_row(r) for r in (result.recommendations or [])]
    df = pd.DataFrame(rows)
    # Ensure required columns exist with defaults
    for col in [
        "Ticker", "FinalScore_20d", "Score", "Beta", "RR",
        "Entry_Price", "Target_Price", "Stop_Loss",
    ]:
        if col not in df.columns:
            df[col] = pd.Series([None] * len(df))
    # Ensure allocation column exists for UI/tests, even if empty
    if "סכום קנייה ($)" not in df.columns:
        # Populate with None by default; tests accept presence and non-negative when rows exist
        df["סכום קנייה ($)"] = pd.Series([None] * len(df))
    # Ensure classification columns exist for downstream consumers
    if "RiskClass" not in df.columns:
        df["RiskClass"] = pd.Series([None] * len(df))
    if "SafetyBlocked" not in df.columns:
        df["SafetyBlocked"] = pd.Series([False] * len(df))
    if "SafetyReasons" not in df.columns:
        df["SafetyReasons"] = pd.Series([""] * len(df))
    # Legacy-compatible classification fields
    if "Risk_Level" not in df.columns:
        df["Risk_Level"] = pd.Series([None] * len(df))
    if "Data_Quality" not in df.columns:
        df["Data_Quality"] = pd.Series([None] * len(df))
    if "Should_Display" not in df.columns:
        df["Should_Display"] = pd.Series([False] * len(df))
    return df


# Note: No data map extraction provided; contracts are the single source of truth.


# --- Wrapper Serialization (result + meta) ---
def save_wrapper_json(path: str, wrapper: Dict[str, Any]) -> None:
    """Save a wrapper {"result": ..., "meta": ...} to a JSON file.

    - If result is a tuple (df, data_map), only df is serialized as records.
    - If result is a DataFrame, serialize as records.
    - Meta is stored verbatim under "meta".
    """
    import json
    import os

    if not isinstance(wrapper, dict) or ("result" not in wrapper):
        raise ValueError("Invalid wrapper: missing 'result' key")
    meta = wrapper.get("meta", {})
    payload = wrapper.get("result")
    # Target schema: result is a dict with keys {results_df, data_map}
    # Backward: tuple/list (df, data_map) or bare DataFrame
    df = pd.DataFrame()
    data_map: Dict[str, Any] | None = None
    try:
        if isinstance(payload, dict) and ("results_df" in payload):
            _df = payload.get("results_df")
            df = _df if _df is not None else pd.DataFrame()
            data_map = payload.get("data_map")
        elif isinstance(payload, tuple) and len(payload) >= 1:
            df = payload[0] if isinstance(payload[0], pd.DataFrame) else pd.DataFrame()
            data_map = payload[1] if len(payload) > 1 else None
        elif isinstance(payload, pd.DataFrame):
            df = payload
            data_map = None
    except Exception:
        df = pd.DataFrame()
        data_map = None
    out_obj: Dict[str, Any] = {
        "meta": meta,
        "result": {
            "results_df_records": df.to_dict(orient="records"),
            "data_map": data_map if isinstance(data_map, dict) else None,
        }
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out_obj, f)


def load_wrapper_json(path: str) -> Dict[str, Any]:
    """Load a wrapper from JSON, synthesizing meta when missing for backward compatibility.

    Returns a dict {"result": df, "meta": meta}.
    - If the file is an old format (records only), set meta.engine_version="unknown",
      meta.used_legacy_fallback=True, meta.fallback_reason="loaded_legacy_format".
    """
    import json
    with open(path, "r") as f:
        obj = json.load(f)
    # Backward compatibility: no meta present
    if not isinstance(obj, dict) or ("meta" not in obj):
        records = []
        try:
            if isinstance(obj, list):
                records = obj
            elif isinstance(obj, dict) and "records" in obj:
                records = obj.get("records") or []
        except Exception:
            records = []
        df = pd.DataFrame(records)
        meta = {
            "engine_version": "unknown",
            "used_legacy_fallback": True,
            "fallback_reason": "loaded_legacy_format",
            "sources_used": None,
            "run_timestamp_utc": None,
        }
        return {"result": {"results_df": df, "data_map": None}, "meta": meta}
    # New format
    meta = obj.get("meta", {})
    res = obj.get("result", {})
    # New schema: results_df_records + optional data_map
    if isinstance(res, dict) and ("results_df_records" in res or "results_df" in res):
        records = res.get("results_df_records")
        # Allow older writer: "records"
        if records is None:
            records = res.get("records")
        df = pd.DataFrame(records or [])
        dm = res.get("data_map") if isinstance(res.get("data_map"), dict) else None
        return {"result": {"results_df": df, "data_map": dm}, "meta": meta}
    # Old shape: records under top-level result dict
    records = None
    try:
        if isinstance(res, dict):
            records = res.get("records")
    except Exception:
        records = None
    df = pd.DataFrame(records or [])
    return {"result": {"results_df": df, "data_map": None}, "meta": meta}
