"""
API Preflight Module
Robust pre-check of external data providers before full scan.
Prioritizes API key presence; only flags providers "down" on clear auth errors.
Network timeouts or 5xx errors are treated as transient (assume provider up).
Used in LIVE Streamlit runs only (not offline audits).
"""

import os
import requests
from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def _key_present(*env_names: str) -> bool:
    """Check for key presence using unified secrets precedence (Streamlit secrets -> env -> .env).

    Falls back to raw env if config is unavailable.
    """
    try:
        from core.config import get_secret
        for name in env_names:
            val = get_secret(name)
            if val and str(val).strip():
                return True
    except Exception:
        for name in env_names:
            if os.getenv(name):
                return True
    return False

def _check_provider(name: str, url: str | None, *, params: Dict[str, Any] | None = None,
                    headers: Dict[str, str] | None = None, timeout: float = 3.0,
                    key_envs: tuple[str, ...] = ()) -> Dict[str, Any]:
    """
    Perform a guarded provider check.
    Logic:
    - If no key present: ok=False, status="no_key"
    - If request returns 401/403: ok=False, status="auth_error"
    - If request returns 429: ok=True, status="rate_limit"
    - If request times out or 5xx: ok=True, status="transient_error"
    - If 200: ok=True, status="ok"
    - If no URL provided: rely purely on key presence.
    """
    if not _key_present(*key_envs):
        return {"ok": False, "status": "no_key", "reason": "No API key", "detail": None}
    if not url:
        return {"ok": True, "status": "ok", "reason": "Key present (no check)", "detail": None}
    try:
        resp = requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
        sc = resp.status_code
        if sc in (401, 403):
            return {"ok": False, "status": "auth_error", "reason": f"HTTP {sc}", "detail": (resp.text or "")[:160]}
        if sc == 429:
            return {"ok": True, "status": "rate_limit", "reason": "Rate limit", "detail": (resp.text or "")[:160]}
        if sc >= 500:
            return {"ok": True, "status": "transient_error", "reason": f"HTTP {sc}", "detail": (resp.text or "")[:160]}
        if sc == 200:
            return {"ok": True, "status": "ok", "reason": "OK", "detail": None}
        # Other non-200 codes: treat as transient
        return {"ok": True, "status": "transient_error", "reason": f"HTTP {sc}", "detail": (resp.text or "")[:160]}
    except requests.Timeout:
        return {"ok": True, "status": "transient_error", "reason": "timeout", "detail": None}
    except Exception as e:
        return {"ok": True, "status": "transient_error", "reason": str(e)[:80], "detail": None}

def run_preflight(timeout: float = 3.0) -> Dict[str, Dict[str, any]]:
    """
    Robust preflight using permissive logic.
    Only marks providers down for missing/invalid keys (auth_error or no_key).
    """
    status: Dict[str, Dict[str, Any]] = {}

    # FMP quote endpoint
    fmp = _check_provider(
        "FMP",
        "https://financialmodelingprep.com/api/v3/quote/AAPL",
        params={"apikey": os.getenv("FMP_API_KEY") or os.getenv("FMP_KEY")},
        timeout=timeout,
        key_envs=("FMP_API_KEY", "FMP_KEY"),
    )
    status["FMP"] = fmp

    # FMP index (SPY daily chart)
    fmp_idx = _check_provider(
        "FMP_INDEX",
        "https://financialmodelingprep.com/api/v3/historical-chart/1day/SPY",
        params={"apikey": os.getenv("FMP_API_KEY") or os.getenv("FMP_KEY")},
        timeout=timeout,
        key_envs=("FMP_API_KEY", "FMP_KEY"),
    )
    status["FMP_INDEX"] = fmp_idx

    tiingo = _check_provider(
        "TIINGO",
        "https://api.tiingo.com/tiingo/daily/AAPL/prices",
        headers={"Authorization": f"Token {os.getenv('TIINGO_API_KEY', '')}"},
        timeout=timeout,
        key_envs=("TIINGO_API_KEY",),
    )
    status["TIINGO"] = tiingo

    polygon = _check_provider(
        "POLYGON",
        f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev",
        params={"apiKey": os.getenv("POLYGON_API_KEY")},
        timeout=timeout,
        key_envs=("POLYGON_API_KEY",),
    )
    status["POLYGON"] = polygon

    finnhub = _check_provider(
        "FINNHUB",
        "https://finnhub.io/api/v1/quote",
        params={"symbol": "AAPL", "token": os.getenv("FINNHUB_API_KEY")},
        timeout=timeout,
        key_envs=("FINNHUB_API_KEY",),
    )
    status["FINNHUB"] = finnhub

    alpha = _check_provider(
        "ALPHAVANTAGE",
        "https://www.alphavantage.co/query",
        params={"function": "GLOBAL_QUOTE", "symbol": "AAPL", "apikey": os.getenv("ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHAVANTAGE_API_KEY")},
        timeout=timeout,
        key_envs=("ALPHA_VANTAGE_API_KEY", "ALPHAVANTAGE_API_KEY"),
    )
    status["ALPHAVANTAGE"] = alpha

    eodhd = _check_provider(
        "EODHD",
        "https://eodhd.com/api/real-time/AAPL",
        params={"api_token": os.getenv("EODHD_API_KEY") or os.getenv("EODHD_TOKEN"), "fmt": "json"},
        timeout=timeout,
        key_envs=("EODHD_API_KEY", "EODHD_TOKEN"),
    )
    status["EODHD"] = eodhd

    simfin = _check_provider(
        "SIMFIN",
        "https://simfin.com/api/v2/companies/statements",
        params={"api-key": os.getenv("SIMFIN_API_KEY"), "ticker": "AAPL", "statement": "pl", "period": "ttm", "fyear": datetime.utcnow().year},
        timeout=timeout,
        key_envs=("SIMFIN_API_KEY",),
    )
    status["SIMFIN"] = simfin

    marketstack = _check_provider(
        "MARKETSTACK",
        "http://api.marketstack.com/v1/eod/latest",
        params={"access_key": os.getenv("MARKETSTACK_API_KEY"), "symbols": "AAPL"},
        timeout=timeout,
        key_envs=("MARKETSTACK_API_KEY",),
    )
    status["MARKETSTACK"] = marketstack

    # NASDAQ: rely on key presence only (no stable public ping)
    nasdaq = _check_provider("NASDAQ", None, key_envs=("NASDAQ_API_KEY", "NASDAQ_DL_API_KEY"))
    status["NASDAQ"] = nasdaq

    # Annotate capabilities per provider (price/fundamentals)
    caps = {
        "FMP": {"can_price": True, "can_fund": True},
        "FMP_INDEX": {"can_price": True, "can_fund": False},
        "TIINGO": {"can_price": True, "can_fund": True},
        "POLYGON": {"can_price": True, "can_fund": False},
        "FINNHUB": {"can_price": True, "can_fund": True},
        "ALPHAVANTAGE": {"can_price": True, "can_fund": True},
        "EODHD": {"can_price": True, "can_fund": True},
        "SIMFIN": {"can_price": False, "can_fund": True},
        "MARKETSTACK": {"can_price": True, "can_fund": False},
        "NASDAQ": {"can_price": False, "can_fund": False},
    }
    for prov, meta in list(status.items()):
        c = caps.get(prov, {"can_price": False, "can_fund": False})
        # If explicit auth/no_key, zero out capabilities
        if meta.get("status") in ("auth_error", "no_key"):
            meta["can_price"] = False
            meta["can_fund"] = False
        else:
            meta["can_price"] = c.get("can_price", False)
            meta["can_fund"] = c.get("can_fund", False)
        status[prov] = meta

    # Active lists: include providers not explicitly blocked by auth/no_key and with capability
    fundamentals_priority = ["FMP", "FINNHUB", "TIINGO", "ALPHAVANTAGE", "EODHD", "SIMFIN"]
    active_fundamentals = [p for p in fundamentals_priority if status.get(p, {}).get("status") not in ("auth_error", "no_key") and status.get(p, {}).get("can_fund", False)]
    status["FUNDAMENTALS_ACTIVE"] = active_fundamentals

    price_priority = ["POLYGON", "EODHD", "FMP", "FINNHUB", "TIINGO", "ALPHAVANTAGE", "MARKETSTACK"]
    active_price = [p for p in price_priority if status.get(p, {}).get("status") not in ("auth_error", "no_key") and status.get(p, {}).get("can_price", False)]
    status["PRICE_ACTIVE"] = active_price

    logger.info(
        f"[Preflight] Providers OK; Fundamentals Active: {active_fundamentals}; Price Active: {active_price}"
    )
    return status
