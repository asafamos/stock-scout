"""Full-pipeline walk-forward backtest engine.

Runs the **exact same** scoring pipeline as the live system
(technical + fundamental + ML + pattern + risk) against historical data.

Unlike ``backtest_ml.py`` which only evaluates ML predictions, this tests
the complete end-to-end system: universe selection → indicator calculation
→ scoring → ranking → portfolio simulation → performance measurement.

Usage::

    engine = FullPipelineBacktest(
        start_date="2024-01-01",
        end_date="2025-12-31",
        top_k=10,
        holding_days=20,
    )
    result = engine.run()
    print(result.summary())
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from core.backtest.portfolio_sim import PortfolioSimulator
from core.backtest.result import BacktestResult

logger = logging.getLogger("stock_scout.backtest.engine")

# Default S&P 500 tickers — a representative subset used when a full
# universe list isn't provided.  Keeping ~50 for speed; the full 500
# can be supplied via the ``universe`` parameter.
_DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
    "UNH", "JNJ", "V", "XOM", "JPM", "MA", "PG", "HD", "CVX", "MRK",
    "ABBV", "PEP", "KO", "COST", "AVGO", "LLY", "TMO", "MCD", "WMT",
    "CSCO", "ACN", "ABT", "CRM", "DHR", "NKE", "TXN", "NEE", "PM",
    "UPS", "RTX", "LOW", "INTC", "QCOM", "AMAT", "INTU", "ISRG",
    "AMD", "BKNG", "ADI", "MDLZ", "ADP", "GILD",
]


class FullPipelineBacktest:
    """Walk-forward backtest of the complete Stock Scout scoring pipeline.

    For each rebalance date the engine:
      1. Fetches historical OHLCV data (lookback from rebalance date)
      2. Computes all technical indicators via ``build_technical_indicators``
      3. Optionally fetches fundamental data
      4. Optionally runs ML inference
      5. Runs pattern matching and big-winner detection
      6. Computes final scores via ``compute_final_score_20d``
      7. Selects top-K by final score
      8. Passes selections to the PortfolioSimulator
      9. Simulates holding for ``holding_days`` with stop/target exits
    """

    def __init__(
        self,
        start_date: str = "2024-01-01",
        end_date: str = "2025-12-31",
        rebalance_freq: str = "monthly",
        top_k: int = 10,
        holding_days: int = 20,
        universe: Optional[List[str]] = None,
        initial_capital: float = 100_000,
        max_positions: int = 15,
        enable_ml: bool = True,
        enable_fundamentals: bool = True,
        enable_patterns: bool = True,
        lookback_days: int = 252,
        status_callback: Optional[Callable[[str], None]] = None,
    ):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        self.rebalance_freq = rebalance_freq
        self.top_k = top_k
        self.holding_days = holding_days
        self.universe = universe or _DEFAULT_UNIVERSE
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.enable_ml = enable_ml
        self.enable_fundamentals = enable_fundamentals
        self.enable_patterns = enable_patterns
        self.lookback_days = lookback_days
        self.status_callback = status_callback or (lambda _: None)

        # Cached price data (fetched once)
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._spy_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """Execute the walk-forward backtest."""
        self.status_callback("Fetching historical price data...")
        self._prefetch_prices()

        rebalance_dates = self._generate_rebalance_dates()
        logger.info(
            "Backtest %s → %s: %d rebalance dates, universe=%d, top_k=%d",
            self.start_date, self.end_date, len(rebalance_dates),
            len(self.universe), self.top_k,
        )

        sim = PortfolioSimulator(
            initial_capital=self.initial_capital,
            max_positions=self.max_positions,
            holding_days=self.holding_days,
        )

        # All trading days in range for daily equity updates
        trading_days = self._get_trading_days()

        rebal_set = set(rebalance_dates)
        for i, day in enumerate(trading_days):
            prices_today = self._get_prices_on_date(day)
            if not prices_today:
                continue

            # Rebalance: score universe and open new positions
            if day in rebal_set:
                self.status_callback(
                    f"Rebalancing {day} ({i+1}/{len(trading_days)})"
                )
                try:
                    scored = self._score_universe_at_date(day)
                    if scored is not None and not scored.empty:
                        selections = scored.head(self.top_k)
                        sim.open_positions(day, selections, prices_today)
                except Exception as e:
                    logger.warning("Scoring failed for %s: %s", day, e)

            # Daily update: check stops, targets, expiry
            sim.update(day, prices_today)

        # Close all remaining positions at end
        final_prices = self._get_prices_on_date(self.end_date) or {}
        sim.close_all(self.end_date, final_prices)

        # Build result
        equity_curve = sim.get_equity_curve()
        trade_log = sim.get_trade_log()
        benchmark_equity = self._build_benchmark_equity(trading_days)

        config = {
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
            "rebalance_freq": self.rebalance_freq,
            "top_k": self.top_k,
            "holding_days": self.holding_days,
            "universe_size": len(self.universe),
            "initial_capital": self.initial_capital,
            "enable_ml": self.enable_ml,
            "enable_fundamentals": self.enable_fundamentals,
            "enable_patterns": self.enable_patterns,
        }

        if equity_curve.empty or len(equity_curve) < 2:
            logger.warning("Backtest produced no equity curve data")
            return BacktestResult(
                start_date=self.start_date,
                end_date=self.end_date,
                config=config,
            )

        return BacktestResult.from_trades(
            trade_log=trade_log,
            equity_curve=equity_curve,
            benchmark_equity=benchmark_equity,
            config=config,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_universe_at_date(self, as_of: date) -> Optional[pd.DataFrame]:
        """Score all universe stocks using the production scoring pipeline.

        Matches the live pipeline (ticker_scoring.py) as closely as possible:
        1. Build technical indicators
        2. Score via compute_recommendation_scores (tech + fund + ML)
        3. Enhance with pattern matching + big-winner detection
        4. Recompute FinalScore_20d with all 5 components

        Returns DataFrame sorted by FinalScore_20d descending.
        """
        from core.indicators import build_technical_indicators
        from core.scoring.recommendation import compute_recommendation_scores

        results = []
        for ticker in self.universe:
            try:
                df = self._get_historical_df(ticker, as_of)
                if df is None or len(df) < 50:
                    continue

                # Build technical indicators (same as production)
                df_ind = build_technical_indicators(df)
                if df_ind.empty:
                    continue

                row = df_ind.iloc[-1].copy()
                row["Ticker"] = ticker

                # Add relative strength vs SPY
                try:
                    from core.market_context import compute_relative_strength_vs_spy
                    spy_df = self._get_historical_df("SPY", as_of)
                    if spy_df is not None and len(spy_df) > 20:
                        rs = compute_relative_strength_vs_spy(df, spy_df)
                        if rs is not None and len(rs) > 0:
                            row["RS_vs_SPY_20d"] = rs.iloc[-1] if isinstance(rs, pd.Series) else rs
                except Exception:
                    pass

                # Score using production canonical function
                scored = compute_recommendation_scores(
                    row,
                    ticker=ticker,
                    as_of_date=as_of if self.enable_fundamentals else None,
                    enable_ml=self.enable_ml,
                    use_multi_source=False,  # avoid API calls in backtest
                )

                if scored is None:
                    continue

                # ── Pattern/BW enhancement + authoritative FinalScore_20d ──
                # Uses the same compute_final_score_20d as the live pipeline
                # so backtest and live produce identical scores.
                if self.enable_patterns:
                    try:
                        from core.unified_logic import compute_big_winner_signal_20d
                        from core.pattern_matcher import PatternMatcher
                        from core.scoring_engine import compute_final_score_20d

                        bw_dict = compute_big_winner_signal_20d(row)
                        bw_score = float(bw_dict.get("BigWinnerScore_20d", 0.0)) if isinstance(bw_dict, dict) else 0.0
                        bw_flag = int(bw_dict.get("BigWinnerFlag_20d", 0)) if isinstance(bw_dict, dict) else 0
                        patt_eval = PatternMatcher.evaluate_stock(row)

                        # Store pattern/BW columns first so compute_final_score_20d reads them
                        scored["Pattern_Score"] = float(patt_eval.get("pattern_score", 0.0))
                        scored["Pattern_Count"] = int(patt_eval.get("pattern_count", 0))
                        scored["Big_Winner_Signal"] = bw_score
                        scored["BigWinnerFlag_20d"] = bw_flag

                        # Single authoritative scoring — same function as runner.py:1290
                        scored["FinalScore_20d"] = float(compute_final_score_20d(pd.Series(scored)))
                    except Exception as e:
                        logger.debug("Pattern/BW enhancement failed for %s: %s", ticker, e)

                results.append(scored)
            except Exception as e:
                logger.debug("Skipping %s at %s: %s", ticker, as_of, e)
                continue

        if not results:
            return None

        df_scored = pd.DataFrame(results)

        # Ensure we have the FinalScore column
        score_col = None
        for col in ["FinalScore_20d", "final_score", "Score"]:
            if col in df_scored.columns:
                score_col = col
                break

        if score_col is None:
            return None

        df_scored = df_scored.sort_values(score_col, ascending=False)
        return df_scored

    # ------------------------------------------------------------------
    # Price data
    # ------------------------------------------------------------------

    def _prefetch_prices(self) -> None:
        """Download all price data upfront for speed."""
        import yfinance as yf

        start = self.start_date - timedelta(days=self.lookback_days + 30)
        end = self.end_date + timedelta(days=5)

        all_symbols = list(set(self.universe + ["SPY"]))

        logger.info("Downloading price data for %d symbols...", len(all_symbols))
        self.status_callback(f"Downloading {len(all_symbols)} symbols...")

        try:
            data = yf.download(
                all_symbols,
                start=start.isoformat(),
                end=end.isoformat(),
                progress=False,
                threads=True,
                auto_adjust=True,
            )
        except Exception as e:
            logger.error("yfinance bulk download failed: %s", e)
            data = pd.DataFrame()

        if data.empty:
            logger.warning("No price data downloaded — backtest will be empty")
            return

        # Parse multi-symbol result
        for sym in all_symbols:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    # Multi-symbol: columns are (Price, Symbol)
                    sym_df = data.xs(sym, level=1, axis=1) if sym in data.columns.get_level_values(1) else None
                    if sym_df is None:
                        continue
                else:
                    # Single symbol
                    sym_df = data

                sym_df = sym_df.dropna(subset=["Close"] if "Close" in sym_df.columns else [])
                if not sym_df.empty:
                    self._price_cache[sym] = sym_df

                    if sym == "SPY":
                        self._spy_df = sym_df
            except Exception as e:
                logger.debug("Failed to parse data for %s: %s", sym, e)

        logger.info("Cached price data for %d symbols", len(self._price_cache))

    def _get_historical_df(self, ticker: str, as_of: date) -> Optional[pd.DataFrame]:
        """Get OHLCV up to as_of date from cache."""
        df = self._price_cache.get(ticker)
        if df is None:
            return None
        mask = df.index.date <= as_of
        subset = df[mask].tail(self.lookback_days)
        return subset if len(subset) >= 50 else None

    def _get_prices_on_date(self, dt: date) -> Dict[str, float]:
        """Get close prices for all cached symbols on a date."""
        prices: Dict[str, float] = {}
        for sym, df in self._price_cache.items():
            close_col = "Close" if "Close" in df.columns else df.columns[0]
            mask = df.index.date <= dt
            if mask.any():
                prices[sym] = float(df.loc[mask, close_col].iloc[-1])
        return prices

    def _get_trading_days(self) -> List[date]:
        """List of trading days in range from SPY data."""
        spy = self._price_cache.get("SPY")
        if spy is None:
            # Fallback: generate business days
            dates = pd.bdate_range(self.start_date, self.end_date)
            return [d.date() for d in dates]
        mask = (spy.index.date >= self.start_date) & (spy.index.date <= self.end_date)
        return sorted(set(spy.index[mask].date))

    # ------------------------------------------------------------------
    # Rebalance schedule
    # ------------------------------------------------------------------

    def _generate_rebalance_dates(self) -> List[date]:
        """Generate rebalance dates between start and end."""
        trading_days = self._get_trading_days()
        if not trading_days:
            return []

        if self.rebalance_freq == "monthly":
            # First trading day of each month
            rebalance = []
            current_month = None
            for d in trading_days:
                m = (d.year, d.month)
                if m != current_month:
                    rebalance.append(d)
                    current_month = m
            return rebalance
        elif self.rebalance_freq == "biweekly":
            return trading_days[::10]  # Every ~2 weeks
        elif self.rebalance_freq == "weekly":
            return trading_days[::5]
        else:
            return trading_days[::21]  # Default: monthly

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def _build_benchmark_equity(self, trading_days: List[date]) -> pd.DataFrame:
        """Build SPY buy-and-hold equity curve for comparison."""
        spy = self._price_cache.get("SPY")
        if spy is None or spy.empty:
            # Flat benchmark
            idx = pd.DatetimeIndex([pd.Timestamp(d) for d in trading_days])
            return pd.DataFrame(
                {"equity": [self.initial_capital] * len(idx)}, index=idx
            )

        close_col = "Close" if "Close" in spy.columns else spy.columns[0]
        mask = (spy.index.date >= self.start_date) & (spy.index.date <= self.end_date)
        spy_period = spy.loc[mask, close_col].dropna()

        if spy_period.empty:
            idx = pd.DatetimeIndex([pd.Timestamp(d) for d in trading_days])
            return pd.DataFrame(
                {"equity": [self.initial_capital] * len(idx)}, index=idx
            )

        # Normalise to initial capital
        spy_equity = spy_period / spy_period.iloc[0] * self.initial_capital
        return pd.DataFrame({"equity": spy_equity})
