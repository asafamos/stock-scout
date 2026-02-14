"""Universe construction and API preflight checks for the pipeline."""

import logging
import os
from typing import List

import requests

from core.config import get_secret

logger = logging.getLogger(__name__)

# Public diagnostic: last provider used for universe construction
LAST_UNIVERSE_PROVIDER: str = "Unknown"


def preflight_check() -> None:
    """Assert that critical API keys are present for production scan.

    Required:
    - FMP_API_KEY: Financial Modeling Prep (indices/ETFs, stock list)
    - POLYGON_API_KEY: Polygon (real-time price verification)
    - FINNHUB_API_KEY: Finnhub (earnings, news)

    Raises:
        RuntimeError: When any required key is missing.
    """
    missing = []
    for key in ["FMP_API_KEY", "POLYGON_API_KEY", "FINNHUB_API_KEY"]:
        if not os.getenv(key):
            missing.append(key)
    if missing:
        raise RuntimeError(
            f"Preflight failed: missing environment keys {missing}. "
            "Set them in .env or CI secrets."
        )


def fetch_top_us_tickers_by_market_cap(limit: int = 2000) -> List[str]:
    """Fetch US tickers ordered by market cap with robust fallbacks.

    Priority:
    1) FMP stock/list (fast, preferred)
    2) Local S&P 500 list from data/sp500_tickers.txt (sorted by market cap)
    3) EODHD (API fallback)
    4) Hardcoded Top 100

    Args:
        limit: Hard cap of 2000 tickers (defaults to 2000)

    Returns:
        List of ticker symbols
    """
    global LAST_UNIVERSE_PROVIDER

    # --- Primary: FMP company screener (stable API) ---
    try:
        api_key = get_secret("FMP_API_KEY", "")
        if api_key:
            url = "https://financialmodelingprep.com/stable/company-screener"
            try:
                min_cap = int(os.getenv("MIN_MCAP", "300000000"))  # $300M minimum
            except (TypeError, ValueError):
                min_cap = 300_000_000
            try:
                # Allow full range up to mega-caps for comprehensive scanning
                max_cap = int(os.getenv("MAX_MCAP", "10000000000000"))  # $10T (effectively no limit)
            except (TypeError, ValueError):
                max_cap = 10_000_000_000_000
            params = {
                "marketCapMoreThan": max(min_cap, 0),
                "marketCapLowerThan": max_cap,
                "isActivelyTrading": True,
                "isEtf": False,
                "isFund": False,
                "limit": min(limit * 2, 3000),  # Request extra to filter
                "apikey": api_key,
            }
            resp = requests.get(url, params=params, timeout=8)
            if resp.status_code == 200:
                data = resp.json() or []
                rows = []
                for it in data:
                    sym = it.get("symbol")
                    mcap = it.get("marketCap") or it.get("marketCapitalization") or 0.0
                    if sym and isinstance(mcap, (int, float)) and float(mcap) > 0:
                        rows.append((sym, float(mcap)))
                rows.sort(key=lambda x: x[1], reverse=True)
                out = _normalize_symbols([s for s, _ in rows][:limit])
                if out:
                    logger.info(f"✓ Universe from FMP screener: {len(out)} tickers")
                    LAST_UNIVERSE_PROVIDER = "FMP"
                    return out
            elif resp.status_code == 403:
                logger.warning("FMP screener 403; falling back immediately to Polygon")
            else:
                logger.warning(f"FMP screener failed: HTTP {resp.status_code}")
        else:
            logger.warning("FMP_API_KEY missing; skipping FMP universe fetch")
    except Exception as e:
        logger.warning(f"FMP universe fetch errored: {e}")

    # --- Fallback 1: Polygon (skipped — returns alphabetical list) ---
    logger.info(
        "Skipping Polygon (returns alphabetical list); using local market-cap-sorted fallback"
    )

    # --- Fallback 2: Local S&P 500 (sorted by market cap) - PREFERRED ---
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        # Prefer sorted file (by market cap), then fall back to alphabetical
        candidates = [
            os.path.join(base_dir, "sp500_tickers_sorted.txt"),  # Sorted by market cap
            os.path.join(base_dir, "sp500_tickers.txt"),
            os.path.join(base_dir, "data", "sp500_tickers.txt"),
        ]
        for local_path in candidates:
            if os.path.exists(local_path):
                with open(local_path, "r") as f:
                    syms = [
                        ln.strip()
                        for ln in f
                        if ln.strip() and not ln.strip().startswith("#")
                    ]
                if syms:
                    logger.info(
                        f"✓ Using local S&P 500 fallback: {len(syms)} tickers "
                        f"from {os.path.relpath(local_path, base_dir)}"
                    )
                    LAST_UNIVERSE_PROVIDER = "Local_SP500"
                    return syms[: min(limit, len(syms))]
    except Exception as e:
        logger.debug(f"Local S&P 500 read failed: {e}")

    # --- Fallback 3: EODHD or Nasdaq (API fallback) ---
    try:
        eod_key = get_secret("EODHD_API_KEY", "")
        if eod_key:
            url = "https://eodhd.com/api/exchange-symbol-list/US"
            params = {"api_token": eod_key, "fmt": "json"}
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json() or []
                syms = []
                for it in data:
                    sym = it.get("Code") or it.get("code") or it.get("Symbol")
                    type_ = (it.get("Type") or it.get("type") or "").upper()
                    # Keep common stocks only
                    if sym and (
                        "COMMON" in type_
                        or type_ in ("COMMON STOCK", "CS", "ETF")
                    ):
                        syms.append(sym)
                out = _normalize_symbols(syms[:limit])
                if out:
                    logger.info(f"✓ Universe from EODHD: {len(out)} tickers")
                    LAST_UNIVERSE_PROVIDER = "EODHD"
                    return out
        nasdaq_key = get_secret("NASDAQ_API_KEY", "")
        if nasdaq_key:
            # Placeholder: Attempt Nasdaq symbols endpoint if available
            pass
    except Exception as e:
        logger.warning(f"EODHD/Nasdaq universe fetch errored: {e}")

    # --- Fallback 4: Hardcoded Top 100 US stocks by market cap ---
    try:
        top100_by_mcap = [
            # Mega-cap (>$500B)
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "GOOGL", "META", "BRK-B",
            "TSLA", "AVGO", "LLY", "V", "JPM", "UNH", "XOM", "MA", "COST",
            "WMT", "JNJ", "HD",
            # Large-cap ($100B-$500B)
            "ORCL", "PG", "MRK", "ABBV", "CVX", "BAC", "CRM", "AMD", "KO",
            "NFLX", "PEP", "TMO", "CSCO", "ACN", "MCD", "ABT", "LIN", "ADBE",
            "DIS", "WFC", "PM", "INTC", "INTU", "TXN", "QCOM", "CMCSA", "VZ",
            "NEE", "DHR", "RTX", "AMGN", "NKE", "HON", "SPGI", "LOW", "PFE",
            "IBM", "UNP", "COP", "BA", "CAT", "GS", "AMAT", "BKNG", "AXP",
            "BLK", "ELV", "ISRG", "SYK", "MS", "DE", "MDLZ", "T", "GILD",
            "VRTX", "NOW", "MMC", "REGN", "ADI", "LRCX", "C", "SBUX", "PANW",
            "MU", "BMY", "PGR", "TJX", "CI", "CB", "SO", "ADP", "SCHW",
            "BSX", "KLAC", "DUK", "MO", "ZTS", "SNPS", "PLD", "FI",
        ]
        out = _normalize_symbols(top100_by_mcap)
        logger.warning(
            f"Using hardcoded top-100 by market cap fallback: {len(out)} tickers"
        )
        LAST_UNIVERSE_PROVIDER = "Hardcoded_Top100"
        return out[: min(limit, len(out))]
    except (TypeError, ValueError, KeyError) as fallback_exc:
        logger.debug(f"Top-100 fallback normalization failed: {fallback_exc}")
        return top100_by_mcap


def _normalize_symbols(symbols: List[str]) -> List[str]:
    """Normalize ticker symbols for consistency across providers/yfinance.

    - Convert dots/slashes to dashes (e.g., BRK.B → BRK-B, BRK/B → BRK-B)
    - Uppercase symbols
    - De-duplicate while preserving order
    """
    seen: set = set()
    out: List[str] = []
    for s in symbols:
        if not s:
            continue
        t = str(s).upper().replace('.', '-').replace('/', '-')
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out
