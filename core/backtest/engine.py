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

# Default universe: dynamically fetched at runtime via fetch_top_us_tickers.
# Fallback to S&P 500 subset if fetch fails.
_FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
    "UNH", "JNJ", "V", "XOM", "JPM", "MA", "PG", "HD", "CVX", "MRK",
    "ABBV", "PEP", "KO", "COST", "AVGO", "LLY", "TMO", "MCD", "WMT",
    "CSCO", "ACN", "ABT", "CRM", "DHR", "NKE", "TXN", "NEE", "PM",
    "UPS", "RTX", "LOW", "INTC", "QCOM", "AMAT", "INTU", "ISRG",
    "AMD", "BKNG", "ADI", "MDLZ", "ADP", "GILD",
]

_DEFAULT_UNIVERSE_SIZE = 200


def _fetch_default_universe(size: int = _DEFAULT_UNIVERSE_SIZE) -> List[str]:
    """Fetch universe dynamically, same as live pipeline."""
    try:
        from core.pipeline_runner import fetch_top_us_tickers_by_market_cap
        tickers = fetch_top_us_tickers_by_market_cap(limit=size)
        if tickers and len(tickers) >= 20:
            logger.info("Fetched %d tickers for backtest universe", len(tickers))
            return tickers
    except Exception as e:
        logger.warning("Dynamic universe fetch failed: %s — using fallback", e)
    return _FALLBACK_UNIVERSE


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
        self.universe = universe or _fetch_default_universe()
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
        # Cached fundamentals: fetched ONCE, reused across all rebalance dates
        # (fundamentals don't change meaningfully within a backtest window)
        self._fundamentals_cache: Dict[str, Any] = {}
        # Scored universes per rebalance date (for weight optimization)
        self._scored_universes: Dict[date, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """Execute the walk-forward backtest."""
        self.status_callback("Fetching historical price data...")
        self._prefetch_prices()

        # Pre-fetch fundamentals ONCE for all tickers (saves ~6 hours of API calls)
        if self.enable_fundamentals:
            self._prefetch_fundamentals()

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
                        # Save full scored universe for weight optimization
                        self._scored_universes[day] = scored.copy()
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

                # Tier 1 OHLCV filter (same as live pipeline)
                last = df.iloc[-1]
                price = float(last.get("Close", 0))
                volume = float(last.get("Volume", 0))
                avg_vol_20 = float(df["Volume"].tail(20).mean()) if "Volume" in df.columns else 0
                if price < 3.0 or price > 10000:
                    continue
                if avg_vol_20 < 500_000:
                    continue
                if price * avg_vol_20 < 5_000_000:
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

                # Build v3.6 ML features explicitly (same as ticker_scoring.py)
                try:
                    from core.ml_feature_builder import build_all_ml_features_v3_6
                    ml_feats = build_all_ml_features_v3_6(df_ind)
                    if ml_feats is not None:
                        for k, v in ml_feats.items():
                            row[k] = v
                except Exception:
                    pass

                # Score using production canonical function.
                # Use cached fundamentals (pre-fetched once) to avoid
                # repeated API calls that cause 6+ hour backtests.
                ms_override = self._fundamentals_cache.get(ticker)
                scored = compute_recommendation_scores(
                    row,
                    ticker=ticker,
                    as_of_date=as_of,
                    enable_ml=self.enable_ml,
                    use_multi_source=False,  # Don't fetch — use cache
                    multi_source_override=ms_override,
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

        # --- Sync with live pipeline (_phase_finalize equivalent) ---

        # 0. Blended RS ranking (same as runner.py:365-404, applied for >100 tickers)
        if len(df_scored) >= 20 and "RS_vs_SPY_20d" in df_scored.columns:
            try:
                rs_20 = pd.to_numeric(df_scored["RS_vs_SPY_20d"], errors="coerce")
                # Compute 63d RS from cached price data
                rs_63_vals = []
                for _, row in df_scored.iterrows():
                    ticker = row.get("Ticker", "")
                    hist = self._get_historical_df(ticker, as_of)
                    spy_hist = self._get_historical_df("SPY", as_of)
                    if hist is not None and spy_hist is not None and len(hist) >= 63 and len(spy_hist) >= 63:
                        stock_ret = float(hist["Close"].iloc[-1] / hist["Close"].iloc[-63] - 1)
                        spy_ret = float(spy_hist["Close"].iloc[-1] / spy_hist["Close"].iloc[-63] - 1)
                        rs_63_vals.append(stock_ret - spy_ret)
                    else:
                        rs_63_vals.append(np.nan)
                rs_63 = pd.Series(rs_63_vals, index=df_scored.index)
                # Blend: 70% 63d + 30% 21d (same as live)
                rs_blend = 0.7 * rs_63.fillna(0) + 0.3 * rs_20.fillna(0)
                df_scored["RS_Blend"] = rs_blend
                df_scored["RS_Pctile"] = rs_blend.rank(pct=True)
            except Exception as e:
                logger.debug("Blended RS computation failed: %s", e)

        # 1. Detect market regime for this date (look-ahead-safe)
        market_regime = self._detect_regime_for_date(as_of)
        df_scored["Market_Regime"] = market_regime.upper()

        # 2. Apply classification (safety filters + risk class)
        # Fill fundamentals defaults only for stocks where API fetch failed
        for _f_col, _f_default in [("ROE", 5.0), ("MarketCap", 1e10)]:
            if _f_col not in df_scored.columns:
                df_scored[_f_col] = _f_default
            else:
                df_scored[_f_col] = df_scored[_f_col].fillna(_f_default)

        try:
            from core.classification import apply_classification

            df_scored = apply_classification(df_scored)
            # Filter REJECT / SafetyBlocked stocks
            pre_count = len(df_scored)
            reject_mask = (
                df_scored.get("RiskClass", pd.Series("CORE", index=df_scored.index))
                == "REJECT"
            )
            safety_mask = (
                df_scored.get(
                    "SafetyBlocked", pd.Series(False, index=df_scored.index)
                )
                == True  # noqa: E712
            )
            n_rejected = (reject_mask | safety_mask).sum()
            if n_rejected > 0:
                logger.debug(
                    "Classification at %s: %d/%d rejected",
                    as_of, n_rejected, pre_count,
                )
            df_scored = df_scored[~(reject_mask | safety_mask)]
            if df_scored.empty:
                logger.info(
                    "All %d stocks rejected by classification at %s",
                    pre_count,
                    as_of,
                )
                return None
        except Exception as e:
            logger.warning("Classification failed at %s: %s", as_of, e)

        # 3. Dynamic R:R with regime-aware ATR multipliers
        try:
            from core.pipeline.helpers import _compute_rr_for_row

            tickers = df_scored["Ticker"].unique().tolist()
            data_map = self._build_data_map_for_date(tickers, as_of)

            _WYCKOFF_TO_ATR = {
                "trend_up": "bullish",
                "moderate_up": "bullish",
                "sideways": "neutral",
                "distribution": "bearish",
                "correction": "bearish",
                "panic": "bearish",
                "bullish": "bullish",
                "neutral": "neutral",
                "bearish": "bearish",
            }
            rr_regime = _WYCKOFF_TO_ATR.get(market_regime.lower(), "neutral")

            rr_updates = df_scored.apply(
                lambda row: _compute_rr_for_row(
                    row, data_map, market_regime=rr_regime
                ),
                axis=1,
                result_type="expand",
            )
            for col in [
                "Entry_Price",
                "Target_Price",
                "Stop_Loss",
                "RewardRisk",
                "RR_Ratio",
                "RR",
                "Target_Source",
            ]:
                if col in rr_updates.columns:
                    df_scored[col] = rr_updates[col]
        except Exception as e:
            logger.warning("Dynamic RR computation failed at %s: %s", as_of, e)

        # 4. Recompute reliability post-enrichment (matches runner.py:1256-1301)
        try:
            def _recompute_reliability(row):
                fund_sources = row.get("Fundamental_Sources_Count", 0)
                if pd.isna(fund_sources):
                    fund_sources = 0
                fund_sources = int(fund_sources)
                fund_score = min(fund_sources, 4) * 15 + 20
                price_bonus = 0
                if pd.notna(row.get("Price_Yahoo", row.get("Close"))):
                    price_bonus += 10
                if pd.notna(row.get("ATR")):
                    price_bonus += 5
                if pd.notna(row.get("RSI")):
                    price_bonus += 5
                fund_data_bonus = 0
                fund_s = row.get("Fundamental_S", 50.0)
                has_real_fund = any(
                    pd.notna(row.get(f))
                    for f in ["pe", "roe", "pb", "margin", "debt_equity",
                              "PE_Ratio", "ROE", "PB_Ratio", "Debt_to_Equity"]
                )
                if has_real_fund and fund_s != 50.0:
                    fund_data_bonus = 10
                return min(fund_score + price_bonus + fund_data_bonus, 100)

            df_scored["ReliabilityScore"] = df_scored.apply(_recompute_reliability, axis=1)
            df_scored["Reliability_Score"] = df_scored["ReliabilityScore"]
        except Exception as e:
            logger.debug("Reliability recompute skipped at %s: %s", as_of, e)

        # 5. Recompute FinalScore_20d with updated R:R, reliability, and market regime
        try:
            from core.scoring_engine import compute_final_score_20d

            df_scored["FinalScore_20d"] = df_scored.apply(
                lambda row: compute_final_score_20d(pd.Series(row)), axis=1
            )
            if "Score" in df_scored.columns:
                df_scored["Score"] = df_scored["FinalScore_20d"]
        except Exception as e:
            logger.warning("FinalScore recompute failed at %s: %s", as_of, e)

        # 6. Post-RR safety re-check with VIX-adjusted min R:R (matches runner.py)
        try:
            from core.scoring_config import HARD_FILTERS, get_vix_min_rr, REGIME_RR_FLOOR

            # Get VIX value from SPY data for this date
            _vix_val = None
            try:
                spy_hist = self._get_historical_df("SPY", as_of)
                if spy_hist is not None and len(spy_hist) > 20:
                    # Use ATR-based VIX proxy if real VIX not available
                    _spy_ret = spy_hist["Close"].pct_change().dropna()
                    _vix_val = float(_spy_ret.tail(20).std() * np.sqrt(252) * 100)
            except Exception:
                pass

            # RR as scoring component (20% of conviction), NOT hard gate
            # Aligned with live pipeline (runner.py:1481-1485) which removed
            # hard RR gating to eliminate double-gating.
            min_rr = get_vix_min_rr(_vix_val)
            regime_floor = REGIME_RR_FLOOR.get(market_regime.upper(), 0.0)
            min_rr = max(min_rr, regime_floor)
            if min_rr > 0 and not df_scored.empty:
                for idx in df_scored.index:
                    rr_val = None
                    for rr_col in ("RR", "RR_Ratio", "RewardRisk"):
                        if rr_col in df_scored.columns:
                            v = df_scored.at[idx, rr_col]
                            if (
                                v is not None
                                and isinstance(v, (int, float))
                                and np.isfinite(v)
                            ):
                                rr_val = float(v)
                                break
                    # Apply RR as 20% penalty to FinalScore instead of hard reject
                    if rr_val is not None and "FinalScore_20d" in df_scored.columns:
                        rr_ratio = min(rr_val / max(min_rr, 0.5), 1.5)  # cap at 1.5x
                        rr_factor = 0.8 + 0.2 * min(rr_ratio, 1.0)  # 0.8-1.0 range
                        df_scored.at[idx, "FinalScore_20d"] = (
                            float(df_scored.at[idx, "FinalScore_20d"]) * rr_factor
                        )
        except Exception as e:
            logger.debug("Post-RR safety re-check skipped: %s", e)

        # 7. Dynamic holding days per stock (matches runner.py:1478-1498)
        try:
            if "ATR_Pct" in df_scored.columns and not df_scored.empty:
                atr_vals = df_scored["ATR_Pct"].dropna()
                median_atr = (
                    float(atr_vals.median()) if len(atr_vals) > 0 else 0.025
                )

                def _dynamic_holding(row):
                    atr = row.get("ATR_Pct", median_atr)
                    if (
                        not isinstance(atr, (int, float))
                        or pd.isna(atr)
                        or atr <= 0
                    ):
                        atr = median_atr
                    if atr < 0.015:
                        return 28
                    if atr < 0.025:
                        return 22
                    if atr < 0.04:
                        return 18
                    return 12

                df_scored["Holding_Days"] = df_scored.apply(
                    _dynamic_holding, axis=1
                )
            elif "Holding_Days" not in df_scored.columns:
                df_scored["Holding_Days"] = self.holding_days
        except Exception:
            if "Holding_Days" not in df_scored.columns:
                df_scored["Holding_Days"] = self.holding_days

        # --- End: Sync with live pipeline ---

        if df_scored.empty:
            return None

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

    def _prefetch_fundamentals(self) -> None:
        """Pre-fetch fundamental data for all universe tickers ONCE.

        Fundamentals (PE, ROE, margins, etc.) don't change meaningfully
        between monthly rebalance dates.  Fetching them once instead of
        per-ticker-per-rebalance eliminates ~90% of API calls and reduces
        backtest runtime from 6+ hours to ~30-60 minutes.
        """
        from core.scoring.recommendation import MultiSourceData

        tickers_to_fetch = [t for t in self.universe if t not in self._fundamentals_cache]
        if not tickers_to_fetch:
            return

        self.status_callback(
            f"Pre-fetching fundamentals for {len(tickers_to_fetch)} tickers (one-time)..."
        )
        logger.info("Pre-fetching fundamentals for %d tickers...", len(tickers_to_fetch))

        try:
            from core import data_sources_v2

            for i, ticker in enumerate(tickers_to_fetch):
                if (i + 1) % 20 == 0:
                    self.status_callback(
                        f"Fundamentals: {i+1}/{len(tickers_to_fetch)} ({ticker})"
                    )
                try:
                    raw = data_sources_v2.fetch_multi_source_data(ticker)
                    self._fundamentals_cache[ticker] = MultiSourceData.from_dict(raw)
                except Exception as e:
                    logger.debug("Fundamentals fetch failed for %s: %s", ticker, e)
                    self._fundamentals_cache[ticker] = MultiSourceData()

        except ImportError:
            logger.warning("data_sources_v2 not available — using empty fundamentals")
            for ticker in tickers_to_fetch:
                self._fundamentals_cache[ticker] = MultiSourceData()

        logger.info(
            "Cached fundamentals for %d/%d tickers",
            sum(1 for v in self._fundamentals_cache.values()
                if v is not None),
            len(tickers_to_fetch),
        )

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

    def _detect_regime_for_date(self, as_of: date) -> str:
        """Detect market regime using backtest-cached SPY data (no look-ahead).

        Returns a simple regime string: 'bullish', 'neutral', or 'bearish'.
        """
        from core.market_regime import detect_market_regime

        spy_df = self._get_historical_df("SPY", as_of)
        if spy_df is None or len(spy_df) < 50:
            return "neutral"

        try:
            regime_info = detect_market_regime(spy_data=spy_df)
            return regime_info.get("regime", "neutral")
        except Exception as e:
            logger.debug("Regime detection failed for %s: %s", as_of, e)
            return "neutral"

    def _build_data_map_for_date(
        self, tickers: List[str], as_of: date,
    ) -> Dict[str, pd.DataFrame]:
        """Build ticker → historical DataFrame map for dynamic R:R computation.

        All DataFrames are filtered to <= as_of to prevent look-ahead bias.
        """
        data_map: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            hist = self._get_historical_df(ticker, as_of)
            if hist is not None and len(hist) >= 5:
                data_map[ticker] = hist
        return data_map

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
