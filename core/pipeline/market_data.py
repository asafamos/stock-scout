"""Market context computation and historical data fetching for the pipeline."""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from core.filters import fetch_benchmark_data
from core.market_context import get_benchmark_series
from core.sector_mapping import SECTOR_ETFS, get_sector_etf, get_stock_sector

logger = logging.getLogger(__name__)


# --- Global Market Context (computed once per pipeline run) ---
_GLOBAL_MARKET_CONTEXT: Dict[str, float] = {}
_SECTOR_ETF_RETURNS: Dict[str, float] = {}  # sector ETF -> 20d return
_SECTOR_CONTEXT_CACHE: Dict[str, Dict[str, float]] = {}  # ticker -> sector context


# ---------------------------------------------------------------------------
# Market context helpers
# ---------------------------------------------------------------------------

def _compute_global_market_context() -> Dict[str, float]:
    """Compute global market context from SPY/VIX once per pipeline run.

    Returns dict with:
        - Market_Regime: 1 (bull), 0 (neutral), -1 (bear)
        - Market_Volatility: VIX-based volatility (0.0-0.5)
        - Market_Trend: 1 if SPY > MA20 > MA50, else 0
        - SPY_20d_ret: SPY 20-day return
    """
    context = {
        'Market_Regime': 0.0,
        'Market_Volatility': 0.15,
        'Market_Trend': 0.0,
        'SPY_20d_ret': 0.0,
    }
    try:
        spy_df = get_benchmark_series("SPY", period="3mo")
        if spy_df is None or len(spy_df) < 50:
            return context

        close = spy_df["close"] if "close" in spy_df.columns else spy_df.get("Close", pd.Series())
        if len(close) < 50:
            return context

        # Compute SPY 20d return
        if len(close) >= 20:
            spy_ret = (close.iloc[-1] / close.iloc[-20] - 1.0)
            context['SPY_20d_ret'] = float(spy_ret)

        # Market trend: SPY > MA20 > MA50
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        current = close.iloc[-1]
        if current > ma20 > ma50:
            context['Market_Trend'] = 1.0
            context['Market_Regime'] = 1.0  # Bullish
        elif current < ma20 < ma50:
            context['Market_Trend'] = 0.0
            context['Market_Regime'] = -1.0  # Bearish
        else:
            context['Market_Trend'] = 0.5
            context['Market_Regime'] = 0.0  # Neutral

        # Market volatility from returns std
        returns = close.pct_change().dropna()
        if len(returns) >= 20:
            vol_20d = returns.tail(20).std() * np.sqrt(252)
            context['Market_Volatility'] = float(min(vol_20d, 0.5))

        logger.info(
            f"[ML] Market context: regime={context['Market_Regime']}, "
            f"trend={context['Market_Trend']:.2f}, vol={context['Market_Volatility']:.3f}"
        )
    except Exception as e:
        logger.warning(f"Failed to compute market context: {e}")

    return context


def _compute_sector_etf_returns() -> Dict[str, float]:
    """Compute 20d returns for all sector ETFs once per pipeline run.

    Uses yfinance batch download to avoid rate limits.
    """
    etf_returns: Dict[str, float] = {}
    etf_symbols = list(SECTOR_ETFS.values())

    try:
        # Batch download all sector ETFs at once
        df = yf.download(etf_symbols, period="2mo", progress=False, threads=False)
        if df.empty:
            logger.warning("[ML] Sector ETF batch download returned empty")
            return etf_returns

        # Handle multi-index columns if multiple symbols
        if isinstance(df.columns, pd.MultiIndex):
            close_df = df['Close'] if 'Close' in df.columns.get_level_values(0) else df['Adj Close']
        else:
            close_df = df[['Close']] if 'Close' in df.columns else df

        for sector, etf in SECTOR_ETFS.items():
            try:
                if etf in close_df.columns:
                    close = close_df[etf].dropna()
                    if len(close) >= 20:
                        ret = float(close.iloc[-1] / close.iloc[-20] - 1.0)
                        etf_returns[sector] = ret
                        etf_returns[etf] = ret  # Also store by ETF symbol
            except Exception:
                pass

        if etf_returns:
            logger.info(f"[ML] Computed sector ETF returns for {len(etf_returns) // 2} sectors")
    except Exception as e:
        logger.warning(f"[ML] Sector ETF batch download failed: {e}")

    return etf_returns


def _get_sector_context_for_ticker(
    ticker: str, stock_20d_return: float,
) -> Dict[str, float]:
    """Get sector context for a specific ticker.

    Returns dict with:
        - Sector_RS: stock return - sector ETF return
        - Sector_Momentum: sector ETF 20d return
        - Sector_Rank: percentile of sector vs other sectors
    """
    global _SECTOR_ETF_RETURNS
    context = {
        'Sector_RS': 0.0,
        'Sector_Momentum': 0.0,
        'Sector_Rank': 0.5,
    }

    try:
        sector = get_stock_sector(ticker)
        if sector == "Unknown":
            return context

        sector_etf = get_sector_etf(sector)
        if not sector_etf or sector not in _SECTOR_ETF_RETURNS:
            return context

        sector_ret = _SECTOR_ETF_RETURNS.get(sector, 0.0)
        context['Sector_Momentum'] = float(sector_ret)
        context['Sector_RS'] = float(stock_20d_return - sector_ret)

        # Compute sector rank (percentile of this sector vs others)
        all_returns = [r for s, r in _SECTOR_ETF_RETURNS.items() if not s.startswith("XL")]
        if all_returns and len(all_returns) >= 3:
            rank = sum(1 for r in all_returns if r <= sector_ret) / len(all_returns)
            context['Sector_Rank'] = float(rank)
    except Exception as e:
        logger.debug(f"Sector context computation failed for {ticker}: {e}")

    return context


def _initialize_ml_context() -> None:
    """Initialize global market and sector context for ML features."""
    global _GLOBAL_MARKET_CONTEXT, _SECTOR_ETF_RETURNS, _SECTOR_CONTEXT_CACHE

    _GLOBAL_MARKET_CONTEXT = _compute_global_market_context()
    _SECTOR_ETF_RETURNS = _compute_sector_etf_returns()
    _SECTOR_CONTEXT_CACHE = {}  # Clear cache for new run


# ---------------------------------------------------------------------------
# Historical data fetching
# ---------------------------------------------------------------------------

def fetch_history_bulk(
    tickers: List[str], period_days: int, ma_long: int,
) -> Dict[str, pd.DataFrame]:
    """Batch-download historical OHLCV via yfinance.

    Always fetches 365 calendar days regardless of *period_days* / *ma_long*
    to ensure enough history for VCP / 52-week calculations.
    """
    # Phase 14: Hard override for lookback – ignore args and fetch 365-calendar days
    days_to_fetch = 365
    end = datetime.utcnow()
    start = end - timedelta(days=days_to_fetch)
    data_map: Dict[str, pd.DataFrame] = {}
    # Relax minimum rows requirement: allow proceeding with 60 rows
    min_rows = 60

    # Batch in groups of 50 to mitigate timeouts; sleep 1s between batches
    CHUNK = 50
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i : i + CHUNK]
        try:
            df_all = yf.download(
                chunk,
                start=start,
                end=end,
                group_by='ticker',
                progress=False,
                threads=True,
                auto_adjust=True,
            )
            if len(chunk) == 1:
                tkr = chunk[0]
                # For single ticker, yfinance with group_by='ticker' still returns MultiIndex
                # Access via ticker key to get flat columns
                try:
                    df = df_all[tkr].dropna(how='all')
                except (KeyError, TypeError):
                    # Fallback: flatten MultiIndex columns if direct access fails
                    df = df_all.copy()
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(-1)
                    df = df.dropna(how='all')
                if not df.empty and len(df) >= min_rows:
                    data_map[tkr] = df
                    logger.info(f"Fetched {len(df)} rows for {tkr}")
                else:
                    logger.warning(
                        f"Insufficient data for {tkr}: {len(df)} rows < {min_rows} required"
                    )
            else:
                for tkr in chunk:
                    try:
                        df = df_all[tkr].dropna(how='all')
                        if len(df) >= min_rows:
                            data_map[tkr] = df
                            logger.info(f"Fetched {len(df)} rows for {tkr}")
                        else:
                            logger.warning(
                                f"Insufficient data for {tkr}: {len(df)} rows < {min_rows} required"
                            )
                    except KeyError:
                        logger.warning(f"No data for {tkr} in bulk download")
        except Exception as e:
            logger.warning(f"Batch fetch failed for {len(chunk)} tickers: {e}")
        # Sleep between batches to ease provider load
        time.sleep(1.0)
    return data_map


def fetch_beta_vs_benchmark(
    ticker: str, bench: str = "SPY", days: int = 252,
) -> float:
    """Compute beta of *ticker* against *bench* using daily returns."""
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=days + 30)
        df_t = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        df_b = yf.download(bench, start=start, end=end, progress=False, auto_adjust=True)

        if df_t.empty or df_b.empty:
            return np.nan

        # Handle MultiIndex columns from newer yfinance versions
        if isinstance(df_t.columns, pd.MultiIndex):
            df_t.columns = df_t.columns.get_level_values(0)
        if isinstance(df_b.columns, pd.MultiIndex):
            df_b.columns = df_b.columns.get_level_values(0)

        j = pd.concat(
            [df_t["Close"].pct_change(), df_b["Close"].pct_change()], axis=1,
        ).dropna()
        j.columns = ["rt", "rb"]
        if len(j) < 40:
            return np.nan

        slope = np.polyfit(j["rb"].to_numpy(), j["rt"].to_numpy(), 1)[0]
        return float(slope)
    except (KeyError, TypeError, ValueError) as exc:
        logger.debug(f"Beta calculation failed: {exc}")
        return np.nan


def _step_fetch_and_prepare_base_data(
    universe: List[str],
    config: Dict[str, Any],
    status_callback: Optional[Callable[[str], None]],
    data_map: Optional[Dict[str, pd.DataFrame]],
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Fetch historical OHLCV data and benchmark DataFrame."""
    if not data_map:
        if status_callback:
            status_callback("Fetching historical data...")
        # Phase 13: Hard-code minimum lookback to 250 days for VCP/52-week calculations
        data_map = fetch_history_bulk(universe, 250, config.get("ma_long", 200))

    # Ensure benchmark also uses at least ~250 days (252 acceptable)
    benchmark_df = fetch_benchmark_data(
        config.get("beta_benchmark", "SPY"),
        max(250, int(config.get("lookback_days", 252))),
    )
    return data_map, benchmark_df
