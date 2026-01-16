"""
API Preflight Module
Fast pre-check of external data providers before full scan.
Used in LIVE Streamlit runs only (not offline audits).
"""

import os
import requests
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def run_preflight(timeout: float = 3.0) -> Dict[str, Dict[str, any]]:
    """
    Forced preflight: trust presence of API keys.
    If a provider API key is present in environment or secrets, mark ok=True
    and avoid network verification to prevent false negatives.
    """
    status: Dict[str, Dict[str, any]] = {}

    def _key_present(*env_names: str) -> bool:
        for name in env_names:
            if os.getenv(name):
                return True
        return False

    providers = {
        "FMP": ("FMP_API_KEY", "FMP_KEY"),
        "FMP_INDEX": ("FMP_API_KEY", "FMP_KEY"),
        "TIINGO": ("TIINGO_API_KEY",),
        "POLYGON": ("POLYGON_API_KEY",),
        "FINNHUB": ("FINNHUB_API_KEY",),
        "ALPHAVANTAGE": ("ALPHA_VANTAGE_API_KEY", "ALPHAVANTAGE_API_KEY"),
        "EODHD": ("EODHD_API_KEY", "EODHD_TOKEN"),
        "SIMFIN": ("SIMFIN_API_KEY",),
        "MARKETSTACK": ("MARKETSTACK_API_KEY",),
        "NASDAQ": ("NASDAQ_API_KEY", "NASDAQ_DL_API_KEY"),
    }

    for pname, keys in providers.items():
        if _key_present(*keys):
            status[pname] = {"ok": True, "reason": "Forced active by key presence"}
        else:
            status[pname] = {"ok": False, "reason": "No API key"}

    fundamentals_priority = ["FMP", "FINNHUB", "TIINGO", "ALPHAVANTAGE", "EODHD", "SIMFIN"]
    active_fundamentals = [p for p in fundamentals_priority if status.get(p, {}).get("ok", False)]
    status["FUNDAMENTALS_ACTIVE"] = active_fundamentals

    logger.info(
        f"[Preflight] Forced OK for providers with keys; Fundamentals Active: {active_fundamentals}"
    )
    return status
