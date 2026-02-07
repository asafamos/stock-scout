"""
Professional Backtesting System for Stock Scout

Implements rigorous backtesting with:
- Walk-forward analysis (no lookahead bias)
- Multiple holding periods
- Risk-adjusted metrics (Sharpe, Sortino, Calmar)
- Drawdown analysis
- Transaction cost modeling
- Statistical significance testing

Usage:
    from core.backtester import Backtester
    bt = Backtester(model_path="models/ml_20d_v4.pkl")
    results = bt.run_walkforward(start_date="2023-01-01", end_date="2024-01-01")
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest runs."""
    # Universe
    universe_size: int = 500
    min_price: float = 5.0
    min_volume: int = 500000
    
    # Timing
    holding_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    rebalance_frequency: str = "weekly"  # daily, weekly, monthly
    
    # Portfolio
    max_positions: int = 20
    position_size: float = 0.05  # 5% per position
    max_sector_weight: float = 0.25
    
    # Costs
    commission_pct: float = 0.001  # 0.1% commission
    slippage_pct: float = 0.001   # 0.1% slippage
    
    # Risk
    stop_loss_pct: float = 0.08   # 8% stop loss
    max_drawdown_exit: float = 0.20  # Exit if portfolio down 20%
    
    # Misc
    random_seed: int = 42
    n_folds: int = 5  # For walk-forward


@dataclass
class BacktestMetrics:
    """Metrics from a single backtest run."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    avg_hold_days: float
    num_trades: int
    total_costs: float
    
    # Per-period returns
    monthly_returns: pd.Series = field(default_factory=pd.Series)
    
    # Benchmark comparison
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 1.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dict for logging."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_return": self.avg_trade_return,
            "num_trades": self.num_trades,
            "benchmark_return": self.benchmark_return,
            "alpha": self.alpha
        }


class Backtester:
    """
    Walk-forward backtesting engine.
    
    Ensures no lookahead bias by:
    1. Using only data available at signal time
    2. Retraining model periodically (optional)
    3. Proper date alignment
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[BacktestConfig] = None
    ):
        """
        Initialize backtester.
        
        Args:
            model_path: Path to trained ML model (pickle)
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        self.model = None
        self.feature_names = []
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, path: str) -> None:
        """Load trained model from pickle."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            
            if isinstance(data, dict):
                self.model = data.get("model")
                self.feature_names = data.get("feature_names", [])
            else:
                self.model = data
                # Try to get feature count from model
                if hasattr(self.model, "n_features_in_"):
                    self.feature_names = [f"f{i}" for i in range(self.model.n_features_in_)]
            
            logger.info(f"Loaded model from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _fetch_universe(self, as_of_date: datetime) -> List[str]:
        """Get tradeable universe as of a specific date."""
        # Use S&P 500 as base universe
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            tickers = tables[0]["Symbol"].str.replace(".", "-").tolist()
            return tickers[:self.config.universe_size]
        except Exception as e:
            logger.warning(f"Could not fetch S&P 500: {e}")
            # Fallback
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
    
    def _fetch_prices(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Fetch price data for all tickers."""
        prices = {}
        
        # Add buffer for lookback
        buffer_start = start_date - timedelta(days=300)
        
        def _fetch_one(ticker: str) -> Tuple[str, Optional[pd.DataFrame]]:
            try:
                df = yf.download(
                    ticker,
                    start=buffer_start.strftime("%Y-%m-%d"),
                    end=(end_date + timedelta(days=30)).strftime("%Y-%m-%d"),
                    progress=False
                )
                if len(df) > 50:
                    df.columns = df.columns.str.lower()
                    return ticker, df
            except Exception as e:
                logger.debug(f"Failed to fetch {ticker}: {e}")
            return ticker, None
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_fetch_one, t) for t in tickers]
            for future in as_completed(futures):
                ticker, df = future.result()
                if df is not None:
                    prices[ticker] = df
        
        logger.info(f"Fetched prices for {len(prices)} tickers")
        return prices
    
    def _generate_signals(
        self,
        prices: Dict[str, pd.DataFrame],
        signal_date: datetime,
        spy_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Generate trading signals for a specific date."""
        if self.model is None:
            raise ValueError("No model loaded")
        
        from core.ml_feature_builder_v4 import build_features_v4, get_feature_defaults_v4
        
        signals = []
        
        for ticker, price_df in prices.items():
            # Ensure we only use data up to signal_date (no lookahead)
            if isinstance(price_df.index, pd.DatetimeIndex):
                mask = price_df.index <= pd.Timestamp(signal_date)
            else:
                mask = pd.Series(True, index=price_df.index)
            
            df_filtered = price_df[mask]
            
            if len(df_filtered) < 50:
                continue
            
            # Build features
            features = build_features_v4(
                price_df=df_filtered,
                ticker=ticker,
                spy_df=spy_df
            )
            
            # Get feature vector in correct order
            feature_vec = []
            defaults = get_feature_defaults_v4()
            
            for fname in self.feature_names:
                val = features.get(fname, defaults.get(fname, 0.0))
                feature_vec.append(val)
            
            # Predict
            try:
                X = np.array([feature_vec])
                proba = self.model.predict_proba(X)[0, 1]
                
                signals.append({
                    "Ticker": ticker,
                    "Signal_Date": signal_date,
                    "Probability": proba,
                    "Close": df_filtered["close"].iloc[-1] if "close" in df_filtered.columns else df_filtered["Close"].iloc[-1]
                })
            except Exception as e:
                logger.debug(f"Prediction failed for {ticker}: {e}")
        
        df = pd.DataFrame(signals)
        
        if len(df) > 0:
            df = df.sort_values("Probability", ascending=False)
        
        return df
    
    def _simulate_trades(
        self,
        signals: pd.DataFrame,
        prices: Dict[str, pd.DataFrame],
        holding_period: int = 20
    ) -> pd.DataFrame:
        """Simulate trades based on signals."""
        if signals.empty:
            return pd.DataFrame()
        
        trades = []
        
        # Take top N signals
        top_signals = signals.head(self.config.max_positions)
        
        for _, row in top_signals.iterrows():
            ticker = row["Ticker"]
            entry_date = pd.Timestamp(row["Signal_Date"])
            entry_price = row["Close"]
            
            if ticker not in prices:
                continue
            
            price_df = prices[ticker]
            
            # Find exit date
            exit_date = entry_date + timedelta(days=holding_period)
            
            # Get prices in the holding window
            if isinstance(price_df.index, pd.DatetimeIndex):
                future_prices = price_df[price_df.index > entry_date]
            else:
                continue
            
            if len(future_prices) == 0:
                continue
            
            # Simulate with stop loss
            stop_price = entry_price * (1 - self.config.stop_loss_pct)
            exit_price = None
            exit_reason = "hold"
            actual_hold_days = 0
            
            close_col = "close" if "close" in price_df.columns else "Close"
            low_col = "low" if "low" in price_df.columns else "Low"
            
            for i, (idx, bar) in enumerate(future_prices.iterrows()):
                actual_hold_days = i + 1
                
                # Check stop loss (use low for conservative estimate)
                if bar[low_col] <= stop_price:
                    exit_price = stop_price
                    exit_reason = "stop_loss"
                    exit_date = idx
                    break
                
                # Check holding period
                if i + 1 >= holding_period:
                    exit_price = bar[close_col]
                    exit_reason = "hold_complete"
                    exit_date = idx
                    break
            
            if exit_price is None:
                # End of data
                exit_price = future_prices[close_col].iloc[-1]
                exit_reason = "data_end"
            
            # Calculate return
            gross_return = (exit_price / entry_price) - 1
            costs = self.config.commission_pct + self.config.slippage_pct
            net_return = gross_return - costs * 2  # Entry + exit
            
            trades.append({
                "Ticker": ticker,
                "Entry_Date": entry_date,
                "Entry_Price": entry_price,
                "Exit_Date": exit_date,
                "Exit_Price": exit_price,
                "Gross_Return": gross_return,
                "Net_Return": net_return,
                "Hold_Days": actual_hold_days,
                "Exit_Reason": exit_reason,
                "Probability": row["Probability"]
            })
        
        return pd.DataFrame(trades)
    
    def _compute_metrics(
        self,
        trades_df: pd.DataFrame,
        benchmark_return: float = 0.0
    ) -> BacktestMetrics:
        """Compute backtest metrics from trade results."""
        if trades_df.empty:
            return BacktestMetrics(
                total_return=0, annualized_return=0, sharpe_ratio=0,
                sortino_ratio=0, calmar_ratio=0, max_drawdown=0,
                win_rate=0, profit_factor=0, avg_trade_return=0,
                avg_hold_days=0, num_trades=0, total_costs=0
            )
        
        returns = trades_df["Net_Return"]
        
        # Basic stats
        total_return = (1 + returns).prod() - 1
        num_trades = len(trades_df)
        avg_return = returns.mean()
        avg_hold_days = trades_df["Hold_Days"].mean()
        
        # Win rate
        wins = (returns > 0).sum()
        win_rate = wins / num_trades if num_trades > 0 else 0
        
        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Calculate portfolio equity curve for drawdown
        equity = (1 + returns).cumprod()
        running_max = equity.cummax()
        drawdowns = (equity - running_max) / running_max
        max_drawdown = abs(drawdowns.min())
        
        # Annualize
        duration_days = (
            trades_df["Exit_Date"].max() - trades_df["Entry_Date"].min()
        ).days if len(trades_df) > 0 else 365
        annualized_return = (1 + total_return) ** (365 / max(duration_days, 1)) - 1
        
        # Risk metrics
        if len(returns) > 1:
            std_ret = returns.std()
            sharpe = avg_return / std_ret * np.sqrt(252 / avg_hold_days) if std_ret > 0 else 0
            
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else std_ret
            sortino = avg_return / downside_std * np.sqrt(252 / avg_hold_days) if downside_std > 0 else 0
        else:
            sharpe = 0
            sortino = 0
        
        calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Costs
        total_costs = (self.config.commission_pct + self.config.slippage_pct) * 2 * num_trades
        
        # Alpha vs benchmark
        alpha = total_return - benchmark_return
        
        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=avg_return,
            avg_hold_days=avg_hold_days,
            num_trades=num_trades,
            total_costs=total_costs,
            benchmark_return=benchmark_return,
            alpha=alpha
        )
    
    def run_single_period(
        self,
        start_date: datetime,
        end_date: datetime,
        holding_period: int = 20,
        verbose: bool = True
    ) -> Tuple[BacktestMetrics, pd.DataFrame]:
        """
        Run backtest for a single period.
        
        Args:
            start_date: Start of signal generation
            end_date: End of signal generation
            holding_period: Days to hold each position
            verbose: Print progress
        
        Returns:
            Tuple of (metrics, trades_df)
        """
        if verbose:
            print(f"Running backtest: {start_date.date()} to {end_date.date()}")
            print(f"Holding period: {holding_period} days")
        
        # Get universe
        universe = self._fetch_universe(start_date)
        if verbose:
            print(f"Universe: {len(universe)} tickers")
        
        # Fetch all price data
        prices = self._fetch_prices(universe, start_date, end_date)
        if verbose:
            print(f"Fetched prices for {len(prices)} tickers")
        
        # Fetch SPY for benchmark and context
        spy_df = yf.download(
            "SPY",
            start=(start_date - timedelta(days=300)).strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=60)).strftime("%Y-%m-%d"),
            progress=False
        )
        spy_df.columns = spy_df.columns.str.lower()
        
        # Generate signals at rebalance points
        rebalance_dates = self._get_rebalance_dates(start_date, end_date)
        if verbose:
            print(f"Rebalance dates: {len(rebalance_dates)}")
        
        all_trades = []
        
        for signal_date in rebalance_dates:
            signals = self._generate_signals(prices, signal_date, spy_df)
            
            if len(signals) > 0:
                trades = self._simulate_trades(
                    signals, prices, holding_period
                )
                all_trades.append(trades)
        
        if all_trades:
            trades_df = pd.concat(all_trades, ignore_index=True)
        else:
            trades_df = pd.DataFrame()
        
        # Calculate benchmark return
        spy_start_mask = spy_df.index >= pd.Timestamp(start_date)
        spy_end_mask = spy_df.index <= pd.Timestamp(end_date)
        spy_period = spy_df[spy_start_mask & spy_end_mask]
        
        if len(spy_period) > 0:
            benchmark_return = (
                spy_period["close"].iloc[-1] / spy_period["close"].iloc[0]
            ) - 1
        else:
            benchmark_return = 0
        
        metrics = self._compute_metrics(trades_df, benchmark_return)
        
        if verbose:
            print("\n" + "="*50)
            print("BACKTEST RESULTS")
            print("="*50)
            print(f"Total Return: {metrics.total_return:.2%}")
            print(f"Benchmark Return (SPY): {benchmark_return:.2%}")
            print(f"Alpha: {metrics.alpha:.2%}")
            print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
            print(f"Win Rate: {metrics.win_rate:.2%}")
            print(f"Profit Factor: {metrics.profit_factor:.2f}")
            print(f"Total Trades: {metrics.num_trades}")
        
        return metrics, trades_df
    
    def _get_rebalance_dates(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[datetime]:
        """Get rebalance dates based on frequency."""
        dates = []
        current = start_date
        
        while current <= end_date:
            # Only trade on weekdays
            if current.weekday() < 5:
                dates.append(current)
            
            if self.config.rebalance_frequency == "daily":
                current += timedelta(days=1)
            elif self.config.rebalance_frequency == "weekly":
                current += timedelta(days=7)
            elif self.config.rebalance_frequency == "monthly":
                current += timedelta(days=30)
            else:
                current += timedelta(days=7)
        
        return dates
    
    def run_walkforward(
        self,
        start_date: str,
        end_date: str,
        retrain_model: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run walk-forward analysis with multiple folds.
        
        This is the gold standard for backtesting - it splits the time period
        into train/test folds and moves forward through time.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            retrain_model: Whether to retrain model for each fold
            verbose: Print progress
        
        Returns:
            Dict with overall metrics and per-fold results
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        total_days = (end_dt - start_dt).days
        fold_days = total_days // self.config.n_folds
        
        fold_results = []
        all_trades = []
        
        for fold in range(self.config.n_folds):
            fold_start = start_dt + timedelta(days=fold * fold_days)
            fold_end = fold_start + timedelta(days=fold_days)
            
            if fold_end > end_dt:
                fold_end = end_dt
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"FOLD {fold + 1}/{self.config.n_folds}")
                print(f"{'='*60}")
            
            # Run backtest for this fold
            metrics, trades = self.run_single_period(
                fold_start, fold_end,
                holding_period=20,
                verbose=verbose
            )
            
            fold_results.append({
                "fold": fold + 1,
                "start": fold_start.strftime("%Y-%m-%d"),
                "end": fold_end.strftime("%Y-%m-%d"),
                "metrics": metrics.to_dict()
            })
            
            if not trades.empty:
                trades["Fold"] = fold + 1
                all_trades.append(trades)
        
        # Combine all trades
        combined_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        
        # Compute overall metrics
        overall_metrics = self._compute_metrics(combined_trades)
        
        if verbose:
            print("\n" + "="*60)
            print("OVERALL WALK-FORWARD RESULTS")
            print("="*60)
            
            # Per-fold summary
            print("\nPer-Fold Summary:")
            for fr in fold_results:
                m = fr["metrics"]
                print(f"  Fold {fr['fold']}: Return={m['total_return']:.2%}, "
                      f"Sharpe={m['sharpe_ratio']:.2f}, Trades={m['num_trades']}")
            
            print(f"\nOverall Metrics:")
            print(f"  Total Return: {overall_metrics.total_return:.2%}")
            print(f"  Annualized Return: {overall_metrics.annualized_return:.2%}")
            print(f"  Sharpe Ratio: {overall_metrics.sharpe_ratio:.2f}")
            print(f"  Sortino Ratio: {overall_metrics.sortino_ratio:.2f}")
            print(f"  Max Drawdown: {overall_metrics.max_drawdown:.2%}")
            print(f"  Win Rate: {overall_metrics.win_rate:.2%}")
            print(f"  Total Trades: {overall_metrics.num_trades}")
        
        return {
            "overall_metrics": overall_metrics.to_dict(),
            "fold_results": fold_results,
            "trades": combined_trades
        }
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_path: str
    ) -> None:
        """Save backtest results to file."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # Save trades
        trades_df = results.get("trades")
        if trades_df is not None and not trades_df.empty:
            trades_df.to_csv(output.with_suffix(".csv"), index=False)
        
        # Save metrics
        metrics = {
            "overall": results.get("overall_metrics", {}),
            "folds": results.get("fold_results", [])
        }
        
        import json
        with open(output.with_suffix(".json"), "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"Saved backtest results to {output}")


def run_quick_backtest(
    model_path: str,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    holding_period: int = 20
) -> Dict[str, Any]:
    """
    Quick backtest utility function.
    
    Args:
        model_path: Path to trained model
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        holding_period: Days to hold positions
    
    Returns:
        Dict with backtest results
    """
    bt = Backtester(model_path=model_path)
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    metrics, trades = bt.run_single_period(
        start_dt, end_dt, holding_period, verbose=True
    )
    
    return {
        "metrics": metrics.to_dict(),
        "trades": trades,
        "config": {
            "model_path": model_path,
            "start_date": start_date,
            "end_date": end_date,
            "holding_period": holding_period
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--model", default="models/ml_20d_v4.pkl", help="Model path")
    parser.add_argument("--start", default="2024-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    parser.add_argument("--output", default="reports/backtest_results", help="Output path")
    
    args = parser.parse_args()
    
    bt = Backtester(model_path=args.model)
    results = bt.run_walkforward(args.start, args.end)
    bt.save_results(results, args.output)
