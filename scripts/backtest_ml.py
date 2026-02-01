"""
Simple Walk-Forward Backtest for Stock Scout ML.

Simulates monthly retraining and evaluates out-of-sample performance.
This provides a realistic assessment of how the ML model would have
performed historically with periodic retraining.

Usage:
    python scripts/backtest_ml.py
    python scripts/backtest_ml.py --start 2023-06-01 --end 2025-01-31 --top-k 20
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# Import training utilities
from scripts.train_rolling_ml_20d import (
    fetch_polygon_history,
    calculate_features,
    calculate_market_regime,
    fetch_sector_etf_data,
    get_universe_tickers,
    precision_at_k,
    POLYGON_KEY,
)

# Directories
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
DATA_DIR = Path("data")


def _check_api_key():
    """Verify POLYGON_API_KEY is set."""
    if not POLYGON_KEY:
        print("❌ ERROR: POLYGON_API_KEY environment variable is required.")
        print("   export POLYGON_API_KEY=your_api_key_here")
        sys.exit(1)


class SimpleBacktest:
    """Walk-forward backtesting with monthly retraining."""
    
    # Feature list (must match training script)
    FEATURES = [
        'RSI', 'ATR_Pct', 'Return_20d', 'Return_10d', 'Return_5d',
        'VCP_Ratio', 'Tightness_Ratio', 'Dist_From_52w_High', 'MA_Alignment',
        'Volume_Surge', 'Up_Down_Volume_Ratio',
        'Momentum_Consistency', 'RS_vs_SPY_20d',
        'Market_Regime', 'Market_Volatility', 'Market_Trend', 'High_Volatility',
        'Sector_RS', 'Sector_Momentum', 'Sector_Rank',
        'Volume_Ratio_20d', 'Volume_Trend', 'Up_Volume_Ratio',
        'Volume_Price_Confirm', 'Relative_Volume_Rank',
        'Distance_From_52w_Low', 'Consolidation_Tightness', 'Days_Since_52w_High',
        'Price_vs_SMA50', 'Price_vs_SMA200', 'SMA50_vs_SMA200', 'MA_Slope_20d',
        'Distance_To_Resistance', 'Support_Strength',
    ]
    
    def __init__(self, 
                 start_date: str = '2023-01-01',
                 end_date: str = '2024-12-31',
                 retrain_frequency: str = 'monthly',
                 top_k: int = 20,
                 holding_period: int = 20,
                 train_lookback_days: int = 365,
                 universe_limit: int = 500,
                 verbose: bool = True):
        """
        Initialize walk-forward backtest.
        
        Args:
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            retrain_frequency: How often to retrain ('monthly', 'quarterly')
            top_k: Number of top stocks to "buy" each period
            holding_period: Days to hold positions (for measuring returns)
            train_lookback_days: Days of historical data for training
            universe_limit: Max stocks to include in universe
            verbose: Print progress updates
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.retrain_frequency = retrain_frequency
        self.top_k = top_k
        self.holding_period = holding_period
        self.train_lookback_days = train_lookback_days
        self.universe_limit = universe_limit
        self.verbose = verbose
        
        self.results: List[Dict] = []
        self.period_details: List[Dict] = []  # Detailed per-period info
        
        # Data cache
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._spy_data: Optional[pd.DataFrame] = None
        self._sector_etf_data: Optional[pd.DataFrame] = None
    
    def _log(self, msg: str):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(msg)
    
    def _get_rebalance_dates(self) -> List[pd.Timestamp]:
        """Generate list of rebalance dates based on frequency."""
        dates = []
        current = self.start_date
        
        while current <= self.end_date:
            dates.append(current)
            
            if self.retrain_frequency == 'monthly':
                # Move to first trading day of next month
                if current.month == 12:
                    current = pd.Timestamp(year=current.year + 1, month=1, day=1)
                else:
                    current = pd.Timestamp(year=current.year, month=current.month + 1, day=1)
            elif self.retrain_frequency == 'quarterly':
                # Move to first day of next quarter
                next_quarter_month = ((current.month - 1) // 3 + 1) * 3 + 1
                if next_quarter_month > 12:
                    current = pd.Timestamp(year=current.year + 1, month=next_quarter_month - 12, day=1)
                else:
                    current = pd.Timestamp(year=current.year, month=next_quarter_month, day=1)
            else:
                raise ValueError(f"Unknown retrain_frequency: {self.retrain_frequency}")
        
        return dates
    
    def _fetch_data(self, start_str: str, end_str: str) -> pd.DataFrame:
        """Fetch and process data for a date range."""
        # Fetch SPY data (cache it)
        if self._spy_data is None:
            self._log("   📥 Fetching SPY benchmark data...")
            self._spy_data = fetch_polygon_history("SPY", start_str, end_str)
        
        spy_returns = None
        market_regime_df = None
        
        if self._spy_data is not None and len(self._spy_data) > 50:
            spy_returns = self._spy_data['Close'].pct_change(20)
            market_regime_df = calculate_market_regime(self._spy_data)
        
        # Fetch sector ETF data (cache it)
        if self._sector_etf_data is None:
            self._log("   📥 Fetching sector ETF data...")
            self._sector_etf_data = fetch_sector_etf_data(start_str, end_str)
        
        # Get universe
        tickers = get_universe_tickers(self.universe_limit)
        
        # Fetch stock data (with caching)
        all_data = []
        tickers_to_fetch = [t for t in tickers if t not in self._data_cache]
        
        if tickers_to_fetch:
            self._log(f"   📥 Fetching {len(tickers_to_fetch)} stocks...")
            with ThreadPoolExecutor(max_workers=15) as executor:
                future_to_ticker = {
                    executor.submit(fetch_polygon_history, t, start_str, end_str): t 
                    for t in tickers_to_fetch
                }
                for future in as_completed(future_to_ticker):
                    t = future_to_ticker[future]
                    df = future.result()
                    if df is not None and len(df) > 50:
                        self._data_cache[t] = df
        
        # Process all tickers
        for t in tickers:
            if t in self._data_cache:
                df = self._data_cache[t].copy()
                # Filter to date range
                df = df[(df.index >= start_str) & (df.index <= end_str)]
                if len(df) > 50:
                    df = calculate_features(
                        df, spy_returns, market_regime_df, 
                        self._sector_etf_data, t
                    )
                    df['Ticker'] = t
                    all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.concat(all_data)
    
    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> HistGradientBoostingClassifier:
        """Train a model on the training data."""
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=20,
            l2_regularization=0.1,
            class_weight='balanced',
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=42,
        )
        model.fit(X_train, y_train)
        return model
    
    def evaluate_period(self, 
                       model, 
                       X_test: pd.DataFrame, 
                       y_test: pd.Series,
                       returns: pd.Series,
                       period_start: pd.Timestamp,
                       period_end: pd.Timestamp) -> Dict:
        """
        Evaluate model performance for one period.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels (binary: 1=winner, 0=loser)
            returns: Actual forward returns
            period_start: Start of evaluation period
            period_end: End of evaluation period
        
        Returns:
            dict with metrics: precision_at_k, avg_return_top_k, avg_return_all,
                              lift, hit_rate, n_predictions, auc
        """
        if len(X_test) == 0:
            return None
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Determine actual K (may be less than self.top_k if small test set)
        actual_k = min(self.top_k, len(X_test))
        
        # Get top K predictions (highest probability)
        top_k_idx = np.argsort(y_pred_proba)[-actual_k:]
        
        # Calculate metrics
        precision = y_test.iloc[top_k_idx].mean() if len(top_k_idx) > 0 else 0.0
        avg_return_top_k = returns.iloc[top_k_idx].mean() if len(top_k_idx) > 0 else 0.0
        avg_return_all = returns.mean()
        
        lift = avg_return_top_k / avg_return_all if avg_return_all != 0 else 1.0
        hit_rate = (returns.iloc[top_k_idx] > 0).mean() if len(top_k_idx) > 0 else 0.0
        
        # AUC if we have both classes
        auc = 0.5
        try:
            if len(y_test.unique()) > 1:
                auc = roc_auc_score(y_test, y_pred_proba)
        except:
            pass
        
        return {
            'period_start': period_start,
            'period_end': period_end,
            'precision_at_k': precision,
            'avg_return_top_k': avg_return_top_k,
            'avg_return_all': avg_return_all,
            'lift': lift,
            'hit_rate': hit_rate,
            'n_predictions': len(top_k_idx),
            'n_test_samples': len(X_test),
            'auc': auc,
        }
    
    def run(self) -> List[Dict]:
        """
        Execute walk-forward backtest.
        
        For each rebalance date:
        1. Train on data up to that date
        2. Make predictions for the next period
        3. Measure actual performance
        
        Returns:
            List of period results
        """
        _check_api_key()
        
        rebalance_dates = self._get_rebalance_dates()
        self._log(f"\n🔄 Running Walk-Forward Backtest")
        self._log(f"   Period: {self.start_date.date()} to {self.end_date.date()}")
        self._log(f"   Rebalance frequency: {self.retrain_frequency}")
        self._log(f"   Rebalance dates: {len(rebalance_dates)}")
        self._log(f"   Top K: {self.top_k}")
        self._log(f"   Holding period: {self.holding_period} days")
        
        # Need to fetch all data first
        full_start = self.start_date - timedelta(days=self.train_lookback_days + 60)
        full_end = self.end_date + timedelta(days=self.holding_period + 30)
        
        self._log(f"\n📥 Fetching data ({full_start.date()} to {full_end.date()})...")
        all_data = self._fetch_data(
            full_start.strftime("%Y-%m-%d"),
            full_end.strftime("%Y-%m-%d")
        )
        
        if len(all_data) == 0:
            self._log("❌ No data available!")
            return []
        
        self._log(f"   Total rows: {len(all_data):,}")
        self._log(f"   Unique tickers: {all_data['Ticker'].nunique()}")
        
        # Walk through each period
        self.results = []
        
        for i, rebalance_date in enumerate(rebalance_dates[:-1]):
            next_rebalance = rebalance_dates[i + 1]
            
            self._log(f"\n📅 Period {i+1}/{len(rebalance_dates)-1}: {rebalance_date.date()} → {next_rebalance.date()}")
            
            # Training data: from lookback_days before rebalance up to rebalance
            train_start = rebalance_date - timedelta(days=self.train_lookback_days)
            train_end = rebalance_date - timedelta(days=1)
            
            # Test data: from rebalance to next rebalance
            test_start = rebalance_date
            test_end = next_rebalance - timedelta(days=1)
            
            # Filter training data
            train_mask = (all_data.index >= train_start) & (all_data.index <= train_end)
            train_df = all_data[train_mask].copy()
            
            # Filter test data
            test_mask = (all_data.index >= test_start) & (all_data.index <= test_end)
            test_df = all_data[test_mask].copy()
            
            # Check data availability
            if len(train_df) < 100:
                self._log(f"   ⚠️  Insufficient training data ({len(train_df)} rows), skipping")
                continue
            
            if len(test_df) < 10:
                self._log(f"   ⚠️  Insufficient test data ({len(test_df)} rows), skipping")
                continue
            
            # Create labels for training (top 20% performers)
            train_df['Label'] = (
                train_df['Forward_Return_20d'] >= 
                train_df['Forward_Return_20d'].quantile(0.80)
            ).astype(int)
            
            # Create labels for test (using actual forward returns)
            test_df['Label'] = (
                test_df['Forward_Return_20d'] >= 
                test_df['Forward_Return_20d'].quantile(0.80)
            ).astype(int)
            
            # Prepare features
            X_train = train_df[self.FEATURES]
            y_train = train_df['Label']
            X_test = test_df[self.FEATURES]
            y_test = test_df['Label']
            returns = test_df['Forward_Return_20d']
            
            # Handle missing values
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
            
            # Train model
            self._log(f"   🧠 Training on {len(X_train):,} samples...")
            try:
                model = self._train_model(X_train, y_train)
            except Exception as e:
                self._log(f"   ❌ Training failed: {e}")
                continue
            
            # Evaluate
            self._log(f"   📊 Evaluating on {len(X_test):,} samples...")
            result = self.evaluate_period(
                model, X_test, y_test, returns,
                rebalance_date, next_rebalance
            )
            
            if result:
                self.results.append(result)
                self._log(f"   ✅ P@{self.top_k}={result['precision_at_k']:.1%}, "
                         f"Return={result['avg_return_top_k']*100:.2f}%, "
                         f"Lift={result['lift']:.2f}x, "
                         f"Hit={result['hit_rate']:.1%}")
        
        self._log(f"\n✅ Backtest complete: {len(self.results)} periods evaluated")
        return self.results
    
    def generate_report(self) -> Dict:
        """Generate backtest summary report."""
        if not self.results:
            return {'error': 'No results to report'}
        
        df = pd.DataFrame(self.results)
        
        # Calculate aggregate metrics
        report = {
            'backtest_period': f"{self.start_date.date()} to {self.end_date.date()}",
            'total_periods': len(self.results),
            'retrain_frequency': self.retrain_frequency,
            'top_k': self.top_k,
            'holding_period_days': self.holding_period,
            
            # Performance metrics
            'avg_precision_at_k': float(df['precision_at_k'].mean()),
            'std_precision_at_k': float(df['precision_at_k'].std()),
            'avg_return_top_k': float(df['avg_return_top_k'].mean()),
            'std_return_top_k': float(df['avg_return_top_k'].std()),
            'avg_return_all': float(df['avg_return_all'].mean()),
            'avg_lift': float(df['lift'].mean()),
            'avg_hit_rate': float(df['hit_rate'].mean()),
            'avg_auc': float(df['auc'].mean()),
            
            # Risk metrics
            'sharpe_ratio': self._calculate_sharpe(df['avg_return_top_k']),
            'max_drawdown': self._calculate_max_drawdown(df['avg_return_top_k']),
            'win_rate': float((df['avg_return_top_k'] > 0).mean()),
            
            # Total return
            'total_return_top_k': float((1 + df['avg_return_top_k']).prod() - 1),
            'total_return_market': float((1 + df['avg_return_all']).prod() - 1),
            
            # Consistency
            'pct_periods_with_lift': float((df['lift'] > 1.0).mean()),
            'pct_periods_positive': float((df['avg_return_top_k'] > 0).mean()),
        }
        
        # Best and worst periods
        best_idx = df['avg_return_top_k'].idxmax()
        worst_idx = df['avg_return_top_k'].idxmin()
        
        report['best_period'] = {
            'start': str(df.loc[best_idx, 'period_start'].date()),
            'return': float(df.loc[best_idx, 'avg_return_top_k']),
            'precision': float(df.loc[best_idx, 'precision_at_k']),
        }
        
        report['worst_period'] = {
            'start': str(df.loc[worst_idx, 'period_start'].date()),
            'return': float(df.loc[worst_idx, 'avg_return_top_k']),
            'precision': float(df.loc[worst_idx, 'precision_at_k']),
        }
        
        return report
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.05) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        # Assume monthly returns for this backtest
        periods_per_year = 12 if self.retrain_frequency == 'monthly' else 4
        
        excess_returns = returns - risk_free_rate / periods_per_year
        if excess_returns.std() == 0:
            return 0.0
        
        return float((excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year))
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) < 2:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        return float(drawdown.min())
    
    def plot_results(self, save_path: Optional[str] = None):
        """Plot backtest results."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("   matplotlib not available, skipping plots")
            return
        
        if not self.results:
            print("   No results to plot")
            return
        
        df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Cumulative returns comparison
        cum_returns_strategy = (1 + df['avg_return_top_k']).cumprod()
        cum_returns_market = (1 + df['avg_return_all']).cumprod()
        
        axes[0, 0].plot(cum_returns_strategy.values, label='Top K Strategy', linewidth=2)
        axes[0, 0].plot(cum_returns_market.values, label='Market Average', linewidth=2, linestyle='--')
        axes[0, 0].set_title('Cumulative Returns', fontsize=12)
        axes[0, 0].set_ylabel('Growth of $1')
        axes[0, 0].set_xlabel('Period')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Precision@K over time
        x_pos = range(len(df))
        bars = axes[0, 1].bar(x_pos, df['precision_at_k'], color='steelblue', alpha=0.7)
        axes[0, 1].axhline(y=0.2, color='red', linestyle='--', linewidth=2, label='Random baseline (20%)')
        axes[0, 1].axhline(y=df['precision_at_k'].mean(), color='green', linestyle='-', 
                          linewidth=2, label=f'Mean ({df["precision_at_k"].mean():.1%})')
        axes[0, 1].set_title(f'Precision@{self.top_k} by Period', fontsize=12)
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_xlabel('Period')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Lift over time
        colors = ['green' if l > 1 else 'red' for l in df['lift']]
        axes[1, 0].bar(x_pos, df['lift'], color=colors, alpha=0.7)
        axes[1, 0].axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='No edge (1.0x)')
        axes[1, 0].set_title('Lift vs Market Average', fontsize=12)
        axes[1, 0].set_ylabel('Lift (X times better)')
        axes[1, 0].set_xlabel('Period')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Hit rate distribution
        axes[1, 1].bar(x_pos, df['hit_rate'], color='teal', alpha=0.7)
        axes[1, 1].axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='50% baseline')
        axes[1, 1].axhline(y=df['hit_rate'].mean(), color='green', linestyle='-',
                          linewidth=2, label=f'Mean ({df["hit_rate"].mean():.1%})')
        axes[1, 1].set_title('Hit Rate (% Profitable Picks)', fontsize=12)
        axes[1, 1].set_ylabel('Hit Rate')
        axes[1, 1].set_xlabel('Period')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   📊 Chart saved to {save_path}")
        
        plt.show()
    
    def save_detailed_results(self, filepath: str):
        """Save detailed period-by-period results to CSV."""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False)
        print(f"   📄 Detailed results saved to {filepath}")


def main():
    """Main entry point for backtest."""
    parser = argparse.ArgumentParser(description='Walk-Forward ML Backtest')
    parser.add_argument('--start', type=str, default='2023-06-01', 
                       help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-01-31',
                       help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--top-k', type=int, default=20,
                       help='Number of top stocks to select each period')
    parser.add_argument('--holding-period', type=int, default=20,
                       help='Holding period in days')
    parser.add_argument('--frequency', type=str, default='monthly',
                       choices=['monthly', 'quarterly'],
                       help='Retraining frequency')
    parser.add_argument('--universe', type=int, default=500,
                       help='Max stocks in universe')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔄 Stock Scout ML Walk-Forward Backtest")
    print("=" * 60)
    
    # Initialize backtest
    backtest = SimpleBacktest(
        start_date=args.start,
        end_date=args.end,
        top_k=args.top_k,
        holding_period=args.holding_period,
        retrain_frequency=args.frequency,
        universe_limit=args.universe,
        verbose=True,
    )
    
    # Run backtest
    backtest.run()
    
    # Generate and print report
    report = backtest.generate_report()
    
    print("\n" + "=" * 60)
    print("📊 BACKTEST RESULTS SUMMARY")
    print("=" * 60)
    
    if 'error' in report:
        print(f"   ❌ {report['error']}")
        return
    
    # Print key metrics
    print(f"\n   📅 Period: {report['backtest_period']}")
    print(f"   📈 Total Periods: {report['total_periods']}")
    print(f"   🔄 Rebalancing: {report['retrain_frequency']}")
    
    print(f"\n   ── Performance Metrics ──")
    print(f"   Avg Precision@{args.top_k}: {report['avg_precision_at_k']:.1%} ± {report['std_precision_at_k']:.1%}")
    print(f"   Avg Return (Top K): {report['avg_return_top_k']*100:.2f}% ± {report['std_return_top_k']*100:.2f}%")
    print(f"   Avg Return (Market): {report['avg_return_all']*100:.2f}%")
    print(f"   Avg Lift: {report['avg_lift']:.2f}x")
    print(f"   Avg Hit Rate: {report['avg_hit_rate']:.1%}")
    print(f"   Avg AUC: {report['avg_auc']:.4f}")
    
    print(f"\n   ── Risk Metrics ──")
    print(f"   Sharpe Ratio: {report['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {report['max_drawdown']*100:.1f}%")
    print(f"   Win Rate: {report['win_rate']:.1%}")
    
    print(f"\n   ── Total Returns ──")
    print(f"   Strategy: {report['total_return_top_k']*100:.1f}%")
    print(f"   Market: {report['total_return_market']*100:.1f}%")
    print(f"   Alpha: {(report['total_return_top_k'] - report['total_return_market'])*100:.1f}%")
    
    print(f"\n   ── Consistency ──")
    print(f"   Periods with Lift > 1: {report['pct_periods_with_lift']:.1%}")
    print(f"   Positive Return Periods: {report['pct_periods_positive']:.1%}")
    
    print(f"\n   ── Best Period ──")
    print(f"   Start: {report['best_period']['start']}")
    print(f"   Return: {report['best_period']['return']*100:.2f}%")
    print(f"   Precision: {report['best_period']['precision']:.1%}")
    
    print(f"\n   ── Worst Period ──")
    print(f"   Start: {report['worst_period']['start']}")
    print(f"   Return: {report['worst_period']['return']*100:.2f}%")
    print(f"   Precision: {report['worst_period']['precision']:.1%}")
    
    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report_path = REPORTS_DIR / f"backtest_report_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n   📄 Report saved to {report_path}")
    
    # Save detailed results
    details_path = REPORTS_DIR / f"backtest_details_{timestamp}.csv"
    backtest.save_detailed_results(str(details_path))
    
    # Plot results
    if not args.no_plot:
        try:
            chart_path = REPORTS_DIR / f"backtest_chart_{timestamp}.png"
            backtest.plot_results(str(chart_path))
        except Exception as e:
            print(f"   (Plotting skipped: {e})")
    
    print("\n" + "=" * 60)
    print("✅ Backtest Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
