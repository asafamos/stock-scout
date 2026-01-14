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
    Quick health check of external API providers.
    Returns dict with provider status: {"ok": bool, "reason": str}
    
    Args:
        timeout: Max seconds per provider check
    
    Returns:
        Dict mapping provider name to status dict:
        {
            "FMP": {"ok": True/False, "reason": "..."},
            "TIINGO": {"ok": True/False, "reason": "..."},
            ...
        }
    """
    status = {}
    
    # FMP (Financial Modeling Prep)
    fmp_key = os.getenv("FMP_API_KEY") or os.getenv("FMP_KEY")
    if fmp_key:
        try:
            url = f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={fmp_key}"
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                status["FMP"] = {"ok": True, "reason": "OK"}
            elif resp.status_code in (401, 403):
                status["FMP"] = {"ok": False, "reason": "Auth failed"}
            elif resp.status_code == 429:
                status["FMP"] = {"ok": False, "reason": "Rate limit"}
            else:
                status["FMP"] = {"ok": False, "reason": f"HTTP {resp.status_code}"}
        except Exception as e:
            status["FMP"] = {"ok": False, "reason": f"Error: {str(e)[:50]}"}
    else:
        status["FMP"] = {"ok": False, "reason": "No API key"}

    # FMP Index endpoint specific check (indices/ETFs often blocked on free tier)
    try:
        fmp_key_idx = os.getenv("FMP_API_KEY") or os.getenv("FMP_KEY")
        if fmp_key_idx:
            idx_url = f"https://financialmodelingprep.com/api/v3/historical-chart/1day/SPY?apikey={fmp_key_idx}"
            idx_resp = requests.get(idx_url, timeout=timeout)
            if idx_resp.status_code == 200:
                status["FMP_INDEX"] = {"ok": True, "reason": "OK"}
            elif idx_resp.status_code in (401, 403):
                status["FMP_INDEX"] = {"ok": False, "reason": "Auth/Forbidden"}
            elif idx_resp.status_code == 429:
                status["FMP_INDEX"] = {"ok": False, "reason": "Rate limit"}
            else:
                status["FMP_INDEX"] = {"ok": False, "reason": f"HTTP {idx_resp.status_code}"}
        else:
            status["FMP_INDEX"] = {"ok": False, "reason": "No API key"}
    except Exception as e:
        status["FMP_INDEX"] = {"ok": False, "reason": f"Error: {str(e)[:50]}"}
    
    # Tiingo
    tiingo_key = os.getenv("TIINGO_API_KEY")
    if tiingo_key:
        try:
            url = f"https://api.tiingo.com/tiingo/daily/AAPL/prices?token={tiingo_key}"
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                status["TIINGO"] = {"ok": True, "reason": "OK"}
            elif resp.status_code in (401, 403):
                status["TIINGO"] = {"ok": False, "reason": "Auth failed"}
            elif resp.status_code == 429:
                status["TIINGO"] = {"ok": False, "reason": "Rate limit"}
            else:
                status["TIINGO"] = {"ok": False, "reason": f"HTTP {resp.status_code}"}
        except Exception as e:
            status["TIINGO"] = {"ok": False, "reason": f"Error: {str(e)[:50]}"}
    else:
        status["TIINGO"] = {"ok": False, "reason": "No API key"}
    
    # Polygon
    polygon_key = os.getenv("POLYGON_API_KEY")
    if polygon_key:
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey={polygon_key}"
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                status["POLYGON"] = {"ok": True, "reason": "OK"}
            elif resp.status_code in (401, 403):
                status["POLYGON"] = {"ok": False, "reason": "Auth failed"}
            elif resp.status_code == 429:
                status["POLYGON"] = {"ok": False, "reason": "Rate limit"}
            else:
                status["POLYGON"] = {"ok": False, "reason": f"HTTP {resp.status_code}"}
        except Exception as e:
            status["POLYGON"] = {"ok": False, "reason": f"Error: {str(e)[:50]}"}
    else:
        status["POLYGON"] = {"ok": False, "reason": "No API key"}
    
    # Finnhub
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    if finnhub_key:
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={finnhub_key}"
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                status["FINNHUB"] = {"ok": True, "reason": "OK"}
            elif resp.status_code in (401, 403):
                status["FINNHUB"] = {"ok": False, "reason": "Auth failed"}
            elif resp.status_code == 429:
                status["FINNHUB"] = {"ok": False, "reason": "Rate limit"}
            else:
                status["FINNHUB"] = {"ok": False, "reason": f"HTTP {resp.status_code}"}
        except Exception as e:
            status["FINNHUB"] = {"ok": False, "reason": f"Error: {str(e)[:50]}"}
    else:
        status["FINNHUB"] = {"ok": False, "reason": "No API key"}
    
    # Alpha Vantage
    alpha_key = os.getenv("ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHAVANTAGE_API_KEY")
    if alpha_key:
        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={alpha_key}"
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                if "Global Quote" in data or "Note" not in data:
                    status["ALPHAVANTAGE"] = {"ok": True, "reason": "OK"}
                else:
                    status["ALPHAVANTAGE"] = {"ok": False, "reason": "Rate limit"}
            elif resp.status_code in (401, 403):
                status["ALPHAVANTAGE"] = {"ok": False, "reason": "Auth failed"}
            else:
                status["ALPHAVANTAGE"] = {"ok": False, "reason": f"HTTP {resp.status_code}"}
        except Exception as e:
            status["ALPHAVANTAGE"] = {"ok": False, "reason": f"Error: {str(e)[:50]}"}
    else:
        status["ALPHAVANTAGE"] = {"ok": False, "reason": "No API key"}
    
    # Build Active Providers list for fundamentals in strict priority
    # Priority: FMP → FINNHUB → TIINGO → ALPHAVANTAGE (include only providers with ok=True)
    fundamentals_priority = ["FMP", "FINNHUB", "TIINGO", "ALPHAVANTAGE"]
    active_fundamentals = [p for p in fundamentals_priority if status.get(p, {}).get("ok", False)]

    # Attach sorted active list to the status dict for downstream routing
    status["FUNDAMENTALS_ACTIVE"] = active_fundamentals

    logger.info(
        f"[Preflight] Checked {len(status)-1} providers, "
        f"{sum(1 for k,v in status.items() if k != 'FUNDAMENTALS_ACTIVE' and v.get('ok'))} OK; "
        f"Fundamentals Active: {active_fundamentals}"
    )
    return status
