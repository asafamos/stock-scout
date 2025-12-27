"""
core.api_monitor
----------------
Centralized API usage monitoring for Stock Scout.

This module provides a single place for lightweight, structured logging of all external API calls (price, fundamentals, etc.).
It is safe to import from anywhere (live app, offline scripts, tests) and will never raise errors or break the main flow.

Features:
- Appends each API call to logs/api_usage_log.csv (auto-creates logs/ directory)
- Records: timestamp_utc, provider, endpoint, status, latency_sec, extra_json
- Handles all exceptions silently
- Maintains a small in-memory counter for debugging
- Provides a summary helper for diagnostics
"""
import os
import csv
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any

# In-memory counters for debugging (not exposed)
COUNTERS = {}

LOG_PATH = os.path.join("logs", "api_usage_log.csv")

LOG_FIELDS = [
    "timestamp_utc",
    "provider",
    "endpoint",
    "status",
    "latency_sec",
    "disabled_by_preflight",
    "provider_load_order",
    "dataset_used",
    "extra_json",
]

def record_api_call(
    provider: str,
    endpoint: str,
    status: str,
    latency_sec: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    disabled_by_preflight: str = "no",
    provider_load_order: int = -1,
    dataset_used: str = "",
) -> None:
    """
    Record a single API call to the central log. Safe to call from anywhere.
    Silently ignores all errors.
    """
    try:
        # Update in-memory counter
        key = (provider, endpoint, status)
        COUNTERS[key] = COUNTERS.get(key, 0) + 1

        # Prepare log row
        ts = datetime.utcnow().isoformat()
        row = {
            "timestamp_utc": ts,
            "provider": provider,
            "endpoint": endpoint,
            "status": status,
            "latency_sec": round(latency_sec, 4) if latency_sec is not None else 0.0,
            "disabled_by_preflight": disabled_by_preflight,
            "provider_load_order": provider_load_order,
            "dataset_used": dataset_used,
            "extra_json": json.dumps(extra) if extra else "",
        }
        # Ensure logs/ directory exists
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        # Write header if file does not exist
        write_header = not os.path.exists(LOG_PATH)
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        return  # Never break main flow

def get_api_usage_summary(limit: int = 1000) -> dict:
    """
    Return a small in-memory summary of recent API usage,
    grouped by (provider, endpoint, status).
    Safe to call from anywhere.
    """
    summary = {}
    try:
        if not os.path.exists(LOG_PATH):
            return summary
        # Read last 'limit' lines efficiently
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) <= 1:
            return summary  # Only header or empty
        # Only process the last 'limit' records (plus header)
        records = lines[-limit:]
        reader = csv.DictReader([lines[0]] + records)
        for row in reader:
            p = row["provider"]
            e = row["endpoint"]
            s = row["status"]
            summary.setdefault(p, {}).setdefault(e, {}).setdefault(s, 0)
            summary[p][e][s] += 1
    except Exception:
        return summary
    return summary
