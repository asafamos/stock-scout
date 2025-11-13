"""
Data sources module - Centralized API clients with error handling and fallback logic.
"""
from __future__ import annotations
import time
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import streamlit as st

from core.logging_config import get_logger, mask_sensitive
from core.config import get_api_keys
from core.models import StockData

logger = get_logger("data_sources")


class HTTPClient:
    """HTTP client with retry logic and exponential backoff."""
    
    @staticmethod
    def get_with_retry(
        url: str,
        tries: int = 4,
        timeout: float = 8.0,
        headers: Optional[dict] = None,
        session: Optional[requests.Session] = None,
        backoff_base: float = 0.5,
        max_backoff: float = 10.0,
    ) -> Optional[requests.Response]:
        """
        HTTP GET with exponential backoff + full jitter.
        
        Args:
            url: URL to fetch
            tries: Total attempts (including the first)
            timeout: Requests timeout for each attempt (seconds)
            headers: Optional headers
            session: Optional requests.Session to reuse connections
            backoff_base: Base backoff time
            max_backoff: Maximum backoff time
            
        Returns:
            Response object on success, None otherwise
        """
        sess = session or requests
        for attempt in range(1, max(1, tries) + 1):
            try:
                resp = sess.get(url, timeout=timeout, headers=headers)
                if resp is not None and resp.status_code == 200:
                    return resp
                
                # Treat 429 / 5xx as retryable
                if resp is not None and (
                    resp.status_code == 429 or (500 <= resp.status_code < 600)
                ):
                    logger.debug(
                        f"HTTP {resp.status_code} -> retry attempt {attempt}/{tries}"
                    )
                else:
                    # Non-retryable code (e.g., 400)
                    return resp
                    
            except requests.RequestException as exc:
                logger.debug(f"Request exception on attempt {attempt}/{tries}: {exc}")
            
            # If we'll retry, sleep with full jitter
            if attempt < tries:
                backoff = min(max_backoff, backoff_base * (2 ** (attempt - 1)))
                sleep_time = random.uniform(0, backoff)
                time.sleep(sleep_time)
        
        logger.warning(f"All {tries} attempts failed for URL")
        return None


class AlphaVantageClient:
    """Alpha Vantage API client with rate limiting."""
    
    def __init__(self):
        self.api_keys = get_api_keys()
        self.last_call_time = 0.0
        self.call_count = 0
        self.daily_limit = 25
    
    def _throttle(self, min_gap_seconds: float = 12.0):
        """Throttle API calls to respect rate limits."""
        now = time.time()
        elapsed = now - self.last_call_time
        if elapsed < min_gap_seconds:
            time.sleep(min_gap_seconds - elapsed)
        self.last_call_time = time.time()
        self.call_count += 1
    
    def check_connectivity(self) -> Tuple[bool, str]:
        """Check if Alpha Vantage API is accessible."""
        if not self.api_keys.has_alpha_vantage():
            return False, "Missing API key"
        
        try:
            url = (
                f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&"
                f"symbol=MSFT&apikey={self.api_keys.alpha_vantage}"
            )
            resp = HTTPClient.get_with_retry(url, tries=1, timeout=8)
            
            if not resp:
                return False, "Timeout"
            
            data = resp.json()
            if "Global Quote" in data:
                return True, "OK"
            
            # Check for rate limit messages
            msg = data.get("Note") or data.get("Information") or "Rate-limited"
            return False, msg
            
        except Exception as e:
            logger.error(f"Alpha Vantage connectivity check failed: {e}")
            return False, "Error"
    
    def get_price(self, ticker: str) -> Optional[float]:
        """
        Get current price from Alpha Vantage.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Current price or None if unavailable
        """
        if not self.api_keys.has_alpha_vantage():
            return None
        
        try:
            self._throttle(2.0)
            url = (
                f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&"
                f"symbol={ticker}&apikey={self.api_keys.alpha_vantage}"
            )
            resp = HTTPClient.get_with_retry(url, tries=1, timeout=10)
            
            if not resp:
                return None
            
            data = resp.json()
            if "Global Quote" in data and "05. price" in data["Global Quote"]:
                return float(data["Global Quote"]["05. price"])
                
        except Exception as e:
            logger.debug(f"Failed to get Alpha Vantage price for {ticker}: {e}")
        
        return None
    
    def get_fundamentals(self, ticker: str) -> dict:
        """
        Get fundamental data from Alpha Vantage OVERVIEW.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with fundamental metrics
        """
        if not self.api_keys.has_alpha_vantage():
            return {}
        
        try:
            self._throttle(2.0)
            url = (
                f"https://www.alphavantage.co/query?function=OVERVIEW&"
                f"symbol={ticker}&apikey={self.api_keys.alpha_vantage}"
            )
            resp = HTTPClient.get_with_retry(url, tries=1, timeout=6)
            
            if not resp:
                return {}
            
            data = resp.json()
            if not (isinstance(data, dict) and data.get("Symbol")):
                return {}
            
            def safe_float(key: str) -> float:
                try:
                    val = float(data.get(key, np.nan))
                    return val if np.isfinite(val) else np.nan
                except Exception:
                    return np.nan
            
            # Calculate gross margin
            gp = safe_float("GrossProfitTTM")
            tr = safe_float("TotalRevenueTTM")
            gm_calc = (gp / tr) if (np.isfinite(gp) and np.isfinite(tr) and tr > 0) else np.nan
            pm = safe_float("ProfitMargin")
            
            return {
                "roe": safe_float("ReturnOnEquityTTM"),
                "roic": np.nan,
                "gm": gm_calc if np.isfinite(gm_calc) else pm,
                "ps": safe_float("PriceToSalesTTM"),
                "pe": safe_float("PERatio"),
                "de": safe_float("DebtToEquityTTM"),
                "rev_g_yoy": safe_float("QuarterlyRevenueGrowthYOY"),
                "eps_g_yoy": safe_float("QuarterlyEarningsGrowthYOY"),
                "sector": data.get("Sector") or "Unknown",
                "market_cap": safe_float("MarketCapitalization"),
            }
            
        except Exception as e:
            logger.debug(f"Failed to get Alpha Vantage fundamentals for {ticker}: {e}")
            return {}


class FinnhubClient:
    """Finnhub API client."""
    
    def __init__(self):
        self.api_keys = get_api_keys()
    
    def check_connectivity(self) -> Tuple[bool, str]:
        """Check if Finnhub API is accessible."""
        if not self.api_keys.has_finnhub():
            return False, "Missing API key"
        
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={self.api_keys.finnhub}"
            resp = HTTPClient.get_with_retry(url, tries=1, timeout=6)
            
            if not resp:
                return False, "Timeout"
            
            data = resp.json()
            return ("c" in data), ("OK" if "c" in data else "Bad response")
            
        except Exception as e:
            logger.error(f"Finnhub connectivity check failed: {e}")
            return False, "Error"
    
    def get_price(self, ticker: str) -> Optional[float]:
        """Get current price from Finnhub."""
        if not self.api_keys.has_finnhub():
            return None
        
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={self.api_keys.finnhub}"
            resp = HTTPClient.get_with_retry(url, tries=1, timeout=8)
            
            if not resp:
                return None
            
            data = resp.json()
            return float(data["c"]) if "c" in data else None
            
        except Exception as e:
            logger.debug(f"Failed to get Finnhub price for {ticker}: {e}")
            return None
    
    def get_fundamentals(self, ticker: str) -> dict:
        """Get fundamental data from Finnhub metrics."""
        if not self.api_keys.has_finnhub():
            return {}
        
        try:
            url = (
                f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&"
                f"metric=all&token={self.api_keys.finnhub}"
            )
            resp = HTTPClient.get_with_retry(url, tries=1, timeout=10)
            
            if not resp:
                return {}
            
            data = resp.json()
            metrics = data.get("metric", {})
            
            def get_first(*keys) -> float:
                for key in keys:
                    val = metrics.get(key)
                    if isinstance(val, (int, float)) and np.isfinite(val):
                        return float(val)
                return np.nan
            
            # Calculate debt-to-equity
            de = np.nan
            try:
                total_debt = get_first("totalDebt")
                total_equity = get_first("totalEquity")
                if np.isfinite(total_debt) and np.isfinite(total_equity) and total_equity != 0:
                    de = total_debt / total_equity
            except Exception:
                pass
            
            return {
                "roe": get_first("roeTtm", "roeAnnual"),
                "roic": np.nan,
                "gm": get_first("grossMarginTTM", "grossMarginAnnual"),
                "ps": get_first("psTTM", "priceToSalesTTM"),
                "pe": get_first("peBasicExclExtraTTM", "peNormalizedAnnual", "peTTM"),
                "de": de,
                "rev_g_yoy": get_first("revenueGrowthTTMYoy", "revenueGrowthQuarterlyYoy"),
                "eps_g_yoy": get_first("epsGrowthTTMYoy", "epsGrowthQuarterlyYoy"),
                "sector": self._get_sector(ticker),
                "market_cap": get_first("marketCapitalization"),
            }
            
        except Exception as e:
            logger.debug(f"Failed to get Finnhub fundamentals for {ticker}: {e}")
            return {}
    
    def _get_sector(self, ticker: str) -> str:
        """Get sector from Finnhub profile."""
        try:
            url = (
                f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&"
                f"token={self.api_keys.finnhub}"
            )
            resp = HTTPClient.get_with_retry(url, tries=1, timeout=8)
            
            if not resp:
                return "Unknown"
            
            data = resp.json()
            return data.get("finnhubIndustry") or data.get("sector") or "Unknown"
            
        except Exception:
            return "Unknown"
    
    def get_universe(self, limit: int = 350) -> List[str]:
        """
        Get stock universe from Finnhub.
        
        Args:
            limit: Maximum number of symbols to return
            
        Returns:
            List of ticker symbols
        """
        if not self.api_keys.has_finnhub():
            return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"]
        
        symbols: List[str] = []
        
        for mic in ("XNAS", "XNYS"):
            try:
                url = (
                    f"https://finnhub.io/api/v1/stock/symbol?exchange=US&"
                    f"mic={mic}&token={self.api_keys.finnhub}"
                )
                resp = HTTPClient.get_with_retry(url, tries=1, timeout=14)
                
                if not resp:
                    continue
                
                data = resp.json()
                for item in data:
                    symbol = item.get("symbol", "")
                    typ = item.get("type", "")
                    
                    if not symbol or "." in symbol:
                        continue
                    
                    if typ and "Common Stock" not in typ:
                        continue
                    
                    symbols.append(symbol)
                    
            except Exception as e:
                logger.debug(f"Failed to get symbols from {mic}: {e}")
                continue
        
        symbols = sorted(pd.unique(pd.Series(symbols)))
        
        if not symbols:
            return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"]
        
        # Sample uniformly if too many symbols
        if len(symbols) > limit:
            bins: Dict[str, List[str]] = {}
            for ticker in symbols:
                bins.setdefault(ticker[0], []).append(ticker)
            
            per_bin = max(1, int(limit / max(1, len(bins))))
            sampled: List[str] = []
            
            for _, arr in sorted(bins.items()):
                sampled.extend(sorted(arr)[:per_bin])
            
            if len(sampled) < limit:
                remaining = [t for t in symbols if t not in sampled]
                sampled.extend(remaining[: (limit - len(sampled))])
            
            symbols = sampled
        
        return symbols[:limit]
    
    def get_next_earnings(self, ticker: str) -> Optional[datetime]:
        """Get next earnings date from Finnhub."""
        if not self.api_keys.has_finnhub():
            return None
        
        try:
            today = datetime.utcnow().date()
            url = (
                f"https://finnhub.io/api/v1/calendar/earnings?"
                f"from={today.isoformat()}&"
                f"to={(today + timedelta(days=180)).isoformat()}&"
                f"symbol={ticker}&token={self.api_keys.finnhub}"
            )
            resp = HTTPClient.get_with_retry(url, tries=1, timeout=10)
            
            if not resp:
                return None
            
            data = resp.json()
            for row in data.get("earningsCalendar", []):
                if row.get("symbol") == ticker and row.get("date"):
                    return datetime.fromisoformat(row["date"])
                    
        except Exception as e:
            logger.debug(f"Failed to get Finnhub earnings for {ticker}: {e}")
        
        return None


class PolygonClient:
    """Polygon.io API client."""
    
    def __init__(self):
        self.api_keys = get_api_keys()
    
    def check_connectivity(self) -> Tuple[bool, str]:
        """Check if Polygon API is accessible."""
        if not self.api_keys.has_polygon():
            return False, "Missing API key"
        
        try:
            url = (
                f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?"
                f"adjusted=true&apiKey={self.api_keys.polygon}"
            )
            resp = HTTPClient.get_with_retry(url, tries=1, timeout=6)
            
            if not resp:
                return False, "Timeout"
            
            data = resp.json()
            ok = bool(data.get("resultsCount", 0) > 0 and "results" in data)
            return ok, ("OK" if ok else "Bad response")
            
        except Exception as e:
            logger.error(f"Polygon connectivity check failed: {e}")
            return False, "Error"
    
    def get_price(self, ticker: str) -> Optional[float]:
        """Get current price from Polygon."""
        if not self.api_keys.has_polygon():
            return None
        
        try:
            url = (
                f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?"
                f"adjusted=true&apiKey={self.api_keys.polygon}"
            )
            resp = HTTPClient.get_with_retry(url, tries=1, timeout=8)
            
            if not resp:
                return None
            
            data = resp.json()
            if data.get("resultsCount", 0) > 0 and "results" in data:
                return float(data["results"][0]["c"])
                
        except Exception as e:
            logger.debug(f"Failed to get Polygon price for {ticker}: {e}")
        
        return None


class TiingoClient:
    """Tiingo API client."""
    
    def __init__(self):
        self.api_keys = get_api_keys()
    
    def check_connectivity(self) -> Tuple[bool, str]:
        """Check if Tiingo API is accessible."""
        if not self.api_keys.has_tiingo():
            return False, "Missing API key"
        
        try:
            url = (
                f"https://api.tiingo.com/tiingo/daily/AAPL/prices?"
                f"token={self.api_keys.tiingo}&resampleFreq=daily"
            )
            resp = HTTPClient.get_with_retry(url, tries=1, timeout=6)
            
            if not resp:
                return False, "Timeout"
            
            data = resp.json()
            ok = (
                isinstance(data, list)
                and data
                and isinstance(data[-1], dict)
                and ("close" in data[-1])
            )
            return ok, ("OK" if ok else "Bad response")
            
        except Exception as e:
            logger.error(f"Tiingo connectivity check failed: {e}")
            return False, "Error"
    
    def get_price(self, ticker: str) -> Optional[float]:
        """Get current price from Tiingo."""
        if not self.api_keys.has_tiingo():
            return None
        
        try:
            url = (
                f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?"
                f"token={self.api_keys.tiingo}&resampleFreq=daily"
            )
            resp = HTTPClient.get_with_retry(url, tries=1, timeout=8)
            
            if not resp:
                return None
            
            data = resp.json()
            if isinstance(data, list) and data:
                return float(data[-1].get("close", np.nan))
                
        except Exception as e:
            logger.debug(f"Failed to get Tiingo price for {ticker}: {e}")
        
        return None


class YFinanceClient:
    """Yahoo Finance client using yfinance library."""
    
    @staticmethod
    def download_bulk(
        tickers: List[str],
        start: datetime,
        end: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        Download bulk historical data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            start: Start date
            end: End date
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        out: Dict[str, pd.DataFrame] = {}
        
        if not tickers:
            return out
        
        try:
            # Try bulk download first
            data = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            
            if isinstance(data.columns, pd.MultiIndex):
                # Multiple tickers
                for ticker in tickers:
                    try:
                        df = data[ticker].dropna()
                        if not df.empty:
                            out[ticker] = df
                    except Exception:
                        continue
            else:
                # Single ticker
                df = data.dropna()
                if not df.empty:
                    out[tickers[0]] = df
                    
        except Exception as e:
            logger.debug(f"Bulk download failed: {e}")
        
        # Download missing tickers individually
        missing = [t for t in tickers if t not in out]
        for ticker in missing:
            try:
                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False
                ).dropna()
                
                if not df.empty:
                    out[ticker] = df
                    
            except Exception as e:
                logger.debug(f"Failed to download {ticker}: {e}")
                continue
        
        return out
    
    @staticmethod
    def get_beta(ticker: str, benchmark: str = "SPY", days: int = 252) -> float:
        """
        Calculate beta vs benchmark.
        
        Args:
            ticker: Stock ticker
            benchmark: Benchmark ticker (default SPY)
            days: Lookback period in days
            
        Returns:
            Beta value or np.nan if calculation fails
        """
        try:
            end = datetime.utcnow()
            start = end - timedelta(days=days + 30)
            
            df_ticker = yf.download(
                ticker, start=start, end=end, auto_adjust=True, progress=False
            )
            df_bench = yf.download(
                benchmark, start=start, end=end, auto_adjust=True, progress=False
            )
            
            if df_ticker.empty or df_bench.empty:
                return np.nan
            
            # Merge returns
            returns = pd.concat([
                df_ticker["Close"].pct_change().dropna(),
                df_bench["Close"].pct_change().dropna()
            ], axis=1).dropna()
            
            returns.columns = ["stock", "bench"]
            
            if len(returns) < 40:
                return np.nan
            
            # Calculate beta as slope of linear regression
            slope = np.polyfit(
                returns["bench"].to_numpy(),
                returns["stock"].to_numpy(),
                1
            )[0]
            
            return float(slope)
            
        except Exception as e:
            logger.debug(f"Failed to calculate beta for {ticker}: {e}")
            return np.nan
    
    @staticmethod
    def get_next_earnings(ticker: str) -> Optional[datetime]:
        """Get next earnings date from yfinance."""
        try:
            # Try earnings dates
            ed = yf.Ticker(ticker).get_earnings_dates(limit=4)
            if isinstance(ed, pd.DataFrame) and not ed.empty:
                now = pd.Timestamp.utcnow()
                future = ed[ed.index >= now]
                dt = future.index.min() if not future.empty else ed.index.max()
                if pd.notna(dt):
                    return dt.to_pydatetime()
        except Exception:
            pass
        
        try:
            # Try calendar
            cal = yf.Ticker(ticker).calendar
            if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                vals = cal.loc["Earnings Date"].values
                if len(vals) > 0:
                    dt = pd.to_datetime(str(vals[0]))
                    if pd.notna(dt):
                        return dt.to_pydatetime()
        except Exception:
            pass
        
        return None


class DataSourceManager:
    """
    Manager for all data sources with automatic fallback logic.
    """
    
    def __init__(self):
        self.alpha = AlphaVantageClient()
        self.finnhub = FinnhubClient()
        self.polygon = PolygonClient()
        self.tiingo = TiingoClient()
        self.yfinance = YFinanceClient()
    
    def check_all_connectivity(self) -> Dict[str, Tuple[bool, str]]:
        """Check connectivity to all data sources."""
        return {
            "Alpha Vantage": self.alpha.check_connectivity(),
            "Finnhub": self.finnhub.check_connectivity(),
            "Polygon": self.polygon.check_connectivity(),
            "Tiingo": self.tiingo.check_connectivity(),
        }
    
    def get_price_with_fallback(self, ticker: str) -> Dict[str, Optional[float]]:
        """
        Get price from multiple sources with fallback.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Dictionary with prices from each source
        """
        return {
            "alpha": self.alpha.get_price(ticker),
            "finnhub": self.finnhub.get_price(ticker),
            "polygon": self.polygon.get_price(ticker),
            "tiingo": self.tiingo.get_price(ticker),
        }
    
    def get_fundamentals_with_fallback(self, ticker: str) -> dict:
        """
        Get fundamentals with fallback (Alpha Vantage → Finnhub).
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Dictionary with fundamental metrics
        """
        # Try Alpha Vantage first
        data = self.alpha.get_fundamentals(ticker)
        if data:
            logger.debug(f"Got fundamentals for {ticker} from Alpha Vantage")
            return data
        
        # Fallback to Finnhub
        data = self.finnhub.get_fundamentals(ticker)
        if data:
            logger.debug(f"Got fundamentals for {ticker} from Finnhub (fallback)")
            return data
        
        logger.warning(f"No fundamentals available for {ticker}")
        return {}
    
    def get_next_earnings_with_fallback(self, ticker: str) -> Optional[datetime]:
        """
        Get next earnings date with fallback (Finnhub → yfinance).
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Next earnings date or None
        """
        # Try Finnhub first
        date = self.finnhub.get_next_earnings(ticker)
        if date:
            return date
        
        # Fallback to yfinance
        date = self.yfinance.get_next_earnings(ticker)
        return date
    
    def get_earnings_batch(self, tickers: List[str], max_workers: int = 6) -> Dict[str, Optional[datetime]]:
        """
        Get earnings dates for multiple tickers in parallel.
        
        Args:
            tickers: List of ticker symbols
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping ticker to earnings date
        """
        results: Dict[str, Optional[datetime]] = {}
        
        if not tickers:
            return results
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.get_next_earnings_with_fallback, ticker): ticker
                for ticker in tickers
            }
            
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    results[ticker] = future.result()
                except Exception as e:
                    logger.debug(f"Failed to get earnings for {ticker}: {e}")
                    results[ticker] = None
        
        return results
