#!/usr/bin/env python3
"""
ML 20D Training Script (Local/Free Data) — v3.1
=================================================

Uses yfinance (free) to download data and train the ML model with all 39 features
from feature_registry v3.1 (stock-picking focused, no market-timing features).

Key improvements over v3:
  - Removed Market_Trend, Market_Volatility, High_Volatility (caused market-timer bias)
  - Added 8 stock-specific features (streaks, squeezes, relative strength momentum)
  - Purged Walk-Forward CV with 20-day embargo (prevents label leakage)
  - Expanding training window (walk-forward realistic simulation)

Run: python scripts/train_ml_20d_v3_local.py

Output: models/model_20d_v3.pkl (+ metadata.json)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import yfinance as yf

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV

# Import from project
from core.feature_registry import get_feature_names, get_feature_defaults
from core.sector_mapping import get_stock_sector, get_sector_etf, get_all_sector_etfs
from core.ensemble import EnsembleClassifier  # Shared module for pickle compatibility

# --- CONFIG ---
MODELS_DIR = Path("models")
DATA_DIR = Path("data")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Use feature registry as SINGLE SOURCE OF TRUTH
FEATURE_NAMES_V3 = get_feature_names("v3.1")  # 39 features (stock-picker focused)
print(f"📋 Feature Registry v3.1: {len(FEATURE_NAMES_V3)} features")


def precision_at_k(y_true, y_pred_proba, k=20):
    """Calculate precision for top K predictions."""
    if len(y_true) < k:
        k = len(y_true)
    if k == 0:
        return 0.0
    top_k_idx = np.argsort(y_pred_proba)[-k:]
    if hasattr(y_true, 'iloc'):
        precision = y_true.iloc[top_k_idx].mean()
    else:
        precision = y_true[top_k_idx].mean()
    return float(precision)


def get_sp500_tickers(limit=None):
    """Get S&P 500 tickers from local file or Wikipedia."""
    local_file = DATA_DIR / "sp500_tickers.txt"
    if local_file.exists():
        with open(local_file, 'r') as f:
            tickers = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        tickers = [t for t in tickers if t.isalpha() or '-' in t or '.' in t]
        print(f"📁 Loaded {len(tickers)} tickers from {local_file}")
    else:
        try:
            df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            tickers = df['Symbol'].str.replace('.', '-').tolist()
            print(f"🌐 Fetched {len(tickers)} tickers from Wikipedia")
        except Exception as e:
            print(f"⚠️ Failed to fetch S&P 500 list: {e}")
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 
                      'JPM', 'V', 'WMT', 'JNJ', 'PG', 'MA', 'HD', 'UNH']
    
    if limit:
        tickers = tickers[:limit]
    return tickers


def fetch_yf_history(ticker, start_str, end_str):
    """Fetch OHLCV data from yfinance."""
    try:
        df = yf.download(ticker, start=start_str, end=end_str, progress=False, auto_adjust=True)
        if df.empty or len(df) < 50:
            return None
        # Flatten multi-index columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        return None


def calculate_all_features(df, spy_returns=None, spy_returns_60d=None,
                          market_regime_df=None,
                          sector_etf_returns=None, ticker=None):
    """
    Calculate ALL 39 features from feature_registry v3.1.
    This ensures training uses the exact same features as inference.
    
    v3.1 changes:
      - Removed: Market_Trend, Market_Volatility, High_Volatility (market-timing)
      - Added: Vol_Contraction_Ratio, Squeeze_On_Flag, RS_vs_SPY_60d, RS_Momentum,
              UpStreak_Days, DownStreak_Days, Range_Pct_10d, OvernightGap_Avg
    """
    df = df.copy()
    
    # === BASIC TECHNICAL ===
    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rs = ma_up / ma_down
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ATR_Pct
    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ], axis=1).max(axis=1)
    df['ATR_Pct'] = tr.rolling(14).mean() / df['Close']
    
    # Returns
    df['Return_20d'] = df['Close'].pct_change(20)
    df['Return_10d'] = df['Close'].pct_change(10)
    df['Return_5d'] = df['Close'].pct_change(5)
    
    # === VOLATILITY PATTERNS ===
    # VCP_Ratio: ATR(10) / ATR(30)
    atr_10 = tr.rolling(10).mean()
    atr_30 = tr.rolling(30).mean()
    df['VCP_Ratio'] = atr_10 / atr_30.replace(0, np.nan)
    
    # Tightness_Ratio: range_5d / range_20d
    range_5d = df['High'].rolling(5).max() - df['Low'].rolling(5).min()
    range_20d = df['High'].rolling(20).max() - df['Low'].rolling(20).min()
    df['Tightness_Ratio'] = range_5d / range_20d.replace(0, np.nan)
    
    # Dist_From_52w_High: (Close / 52w_High) - 1
    high_52w = df['High'].rolling(252, min_periods=100).max()
    df['Dist_From_52w_High'] = (df['Close'] / high_52w.replace(0, np.nan)) - 1
    
    # MA_Alignment: 1 if Close > MA20 > MA50 > MA200
    ma20 = df['Close'].rolling(20).mean()
    ma50 = df['Close'].rolling(50).mean()
    ma200 = df['Close'].rolling(200, min_periods=100).mean()
    df['MA_Alignment'] = ((df['Close'] > ma20) & (ma20 > ma50) & (ma50 > ma200)).astype(float)
    
    # === VOLUME BASIC ===
    vol_5d = df['Volume'].rolling(5).mean()
    vol_20d = df['Volume'].rolling(20).mean()
    df['Volume_Surge'] = vol_5d / vol_20d.replace(0, np.nan)
    
    # Up_Down_Volume_Ratio
    daily_return = df['Close'].pct_change()
    up_day = daily_return > 0
    down_day = daily_return < 0
    up_volume = (df['Volume'] * up_day).rolling(20).sum()
    down_volume = (df['Volume'] * down_day).rolling(20).sum()
    up_days_count = up_day.rolling(20).sum()
    down_days_count = down_day.rolling(20).sum()
    avg_up_vol = up_volume / up_days_count.replace(0, np.nan)
    avg_down_vol = down_volume / down_days_count.replace(0, np.nan)
    df['Up_Down_Volume_Ratio'] = avg_up_vol / avg_down_vol.replace(0, np.nan)
    
    # === MOMENTUM ===
    # Momentum_Consistency: % of last 20 days with positive returns
    positive_return = (daily_return > 0).astype(int)
    df['Momentum_Consistency'] = positive_return.rolling(20).mean()
    
    # RS_vs_SPY_20d
    if spy_returns is not None:
        spy_aligned = spy_returns.reindex(df.index, method='ffill')
        df['RS_vs_SPY_20d'] = df['Return_20d'] - spy_aligned.fillna(0)
    else:
        df['RS_vs_SPY_20d'] = 0.0
    
    # RS_vs_SPY_60d (longer-horizon relative strength)
    df['Return_60d'] = df['Close'].pct_change(60)
    if spy_returns_60d is not None:
        spy_60_aligned = spy_returns_60d.reindex(df.index, method='ffill')
        df['RS_vs_SPY_60d'] = df['Return_60d'] - spy_60_aligned.fillna(0)
    else:
        df['RS_vs_SPY_60d'] = 0.0
    
    # RS_Momentum (acceleration — short-term RS vs long-term RS)
    df['RS_Momentum'] = df['RS_vs_SPY_20d'] - df['RS_vs_SPY_60d']
    
    # === MARKET REGIME === (v3.1: only Market_Regime — removed market-timing)
    if market_regime_df is not None and 'Market_Regime' in market_regime_df.columns:
        df['Market_Regime'] = market_regime_df['Market_Regime'].reindex(df.index, method='ffill').fillna(0)
    else:
        df['Market_Regime'] = 0
    
    # === SECTOR RELATIVE ===
    if sector_etf_returns is not None and ticker is not None:
        sector = get_stock_sector(ticker)
        sector_etf = get_sector_etf(sector)
        if sector_etf is not None and sector_etf in sector_etf_returns.columns:
            sector_ret = sector_etf_returns[sector_etf].reindex(df.index, method='ffill').fillna(0)
            df['Sector_RS'] = df['Return_20d'] - sector_ret
            df['Sector_Momentum'] = sector_ret
            sector_ret_5d = sector_etf_returns[sector_etf].pct_change(5).reindex(df.index, method='ffill').fillna(0)
            df['Sector_Rank'] = (df['Return_5d'] > sector_ret_5d).astype(float)
        else:
            df['Sector_RS'] = 0.0
            df['Sector_Momentum'] = 0.0
            df['Sector_Rank'] = 0.5
    else:
        df['Sector_RS'] = 0.0
        df['Sector_Momentum'] = 0.0
        df['Sector_Rank'] = 0.5
    
    # === VOLUME ADVANCED ===
    df['Volume_Ratio_20d'] = df['Volume'] / vol_20d.replace(0, np.nan)
    
    # Volume_Trend: slope of volume
    def volume_slope(x):
        if len(x) < 20:
            return 0.0
        try:
            slope = np.polyfit(range(len(x)), x, 1)[0]
            return slope / (x.mean() + 1e-8)
        except:
            return 0.0
    df['Volume_Trend'] = df['Volume'].rolling(20).apply(volume_slope, raw=True)
    
    # Up_Volume_Ratio
    total_vol_20d = df['Volume'].rolling(20).sum()
    df['Up_Volume_Ratio'] = up_volume / total_vol_20d.replace(0, np.nan)
    
    # Volume_Price_Confirm
    price_up = df['Close'] > df['Close'].shift(1)
    vol_up = df['Volume'] > df['Volume'].shift(1)
    df['Volume_Price_Confirm'] = (price_up & vol_up).astype(float).rolling(5).mean()
    
    # Relative_Volume_Rank
    df['Relative_Volume_Rank'] = df['Volume'].rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 60 else 0.5, raw=False
    )
    
    # === PRICE ACTION ===
    low_52w = df['Low'].rolling(252, min_periods=100).min()
    df['Distance_From_52w_Low'] = (df['Close'] - low_52w) / low_52w.replace(0, np.nan)
    
    h20 = df['High'].rolling(20).max()
    l20 = df['Low'].rolling(20).min()
    avg20 = df['Close'].rolling(20).mean()
    df['Consolidation_Tightness'] = (h20 - l20) / avg20.replace(0, np.nan)
    
    # Days_Since_52w_High (normalized 0-1)
    def days_since_high(s):
        if len(s) < 252:
            return 126
        try:
            return len(s) - np.argmax(s.values) - 1
        except:
            return 126
    df['Days_Since_52w_High'] = df['High'].rolling(252, min_periods=100).apply(days_since_high, raw=False)
    df['Days_Since_52w_High'] = df['Days_Since_52w_High'] / 252.0
    
    sma50 = df['Close'].rolling(50).mean()
    sma200 = df['Close'].rolling(200, min_periods=100).mean()
    df['Price_vs_SMA50'] = (df['Close'] - sma50) / sma50.replace(0, np.nan)
    df['Price_vs_SMA200'] = (df['Close'] - sma200) / sma200.replace(0, np.nan)
    df['SMA50_vs_SMA200'] = (sma50 - sma200) / sma200.replace(0, np.nan)
    
    df['MA_Slope_20d'] = ma20.diff(20) / ma20.shift(20).replace(0, np.nan)
    df['Distance_To_Resistance'] = (h20 - df['Close']) / df['Close'].replace(0, np.nan)
    
    # Support_Strength
    near_low = (df['Low'] <= l20 * 1.02).astype(float)
    df['Support_Strength'] = near_low.rolling(20).mean()
    
    # === VOLATILITY ADDITIONS (v3.1) ===
    # Vol_Contraction_Ratio: ATR(20) / ATR(50) — tighter = breakout setup
    atr_20 = tr.rolling(20).mean()
    atr_50 = tr.rolling(50).mean()
    df['Vol_Contraction_Ratio'] = atr_20 / atr_50.replace(0, np.nan)
    
    # Squeeze_On_Flag: Bollinger Bands inside Keltner Channels (pre-breakout)
    bb_std = df['Close'].rolling(20).std()
    bb_upper = ma20 + 2 * bb_std
    bb_lower = ma20 - 2 * bb_std
    kc_atr = atr_20
    kc_upper = ma20 + 1.5 * kc_atr
    kc_lower = ma20 - 1.5 * kc_atr
    df['Squeeze_On_Flag'] = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).astype(float)
    
    # === STREAK & PATTERN FEATURES (v3.1) ===
    # UpStreak_Days: consecutive up-close days (capped at 10)
    up_close = (df['Close'] > df['Close'].shift(1)).astype(int)
    streak_break_up = (up_close == 0).cumsum()
    df['UpStreak_Days'] = up_close.groupby(streak_break_up).cumsum().clip(upper=10)
    
    # DownStreak_Days: consecutive down-close days (capped at 10)
    down_close = (df['Close'] < df['Close'].shift(1)).astype(int)
    streak_break_down = (down_close == 0).cumsum()
    df['DownStreak_Days'] = down_close.groupby(streak_break_down).cumsum().clip(upper=10)
    
    # Range_Pct_10d: average intraday range as pct of close (10d)
    intraday_range_pct = (df['High'] - df['Low']) / df['Close'].replace(0, np.nan)
    df['Range_Pct_10d'] = intraday_range_pct.rolling(10).mean()
    
    # OvernightGap_Avg: average overnight gap pct over 5 days
    overnight_gap = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1).replace(0, np.nan)
    df['OvernightGap_Avg'] = overnight_gap.rolling(5).mean()
    
    # === TARGET ===
    df['Forward_Return_20d'] = df['Close'].shift(-20) / df['Close'] - 1.0
    
    # Handle inf/nan
    for col in FEATURE_NAMES_V3:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # Clip features to valid ranges (must match inference clipping in ml_20d_inference.py)
    if 'ATR_Pct' in df.columns:
        df['ATR_Pct'] = df['ATR_Pct'].clip(0.0, 0.2)
    if 'RSI' in df.columns:
        df['RSI'] = df['RSI'].clip(5.0, 95.0)
    
    return df.dropna(subset=['Forward_Return_20d'] + [f for f in FEATURE_NAMES_V3 if f in df.columns])


def calculate_market_regime(spy_df):
    """Calculate market regime from SPY.
    
    v3.1: Only computes Market_Regime (bull/bear/sideways).
    Removed: Market_Volatility, Market_Trend, High_Volatility (market-timing features
    that caused the model to be a market timer instead of stock picker).
    """
    if spy_df is None or len(spy_df) < 200:
        return None
    
    regime_df = pd.DataFrame(index=spy_df.index)
    spy_close = spy_df['Close']
    ma50 = spy_close.rolling(50).mean()
    ma200 = spy_close.rolling(200, min_periods=100).mean()
    ret_20d = spy_close.pct_change(20)
    
    regime_df['Market_Regime'] = 0
    bull = (spy_close > ma50) & (ma50 > ma200) & (ret_20d > 0.02)
    bear = (spy_close < ma50) & (ma50 < ma200) & (ret_20d < -0.02)
    regime_df.loc[bull, 'Market_Regime'] = 1
    regime_df.loc[bear, 'Market_Regime'] = -1
    
    return regime_df


def fetch_sector_etf_data(start_str, end_str):
    """Fetch all sector ETF returns."""
    sector_etfs = get_all_sector_etfs()
    sector_data = {}
    
    print(f"📥 Fetching {len(sector_etfs)} sector ETFs...")
    for etf in sector_etfs:
        df = fetch_yf_history(etf, start_str, end_str)
        if df is not None and len(df) > 30:
            sector_data[etf] = df['Close'].pct_change(20)
    
    if not sector_data:
        return None
    
    return pd.DataFrame(sector_data)


# =============================================================================
# PURGED WALK-FORWARD CROSS-VALIDATION
# =============================================================================

class PurgedWalkForwardCV:
    """Walk-forward cross-validation with purge gap.
    
    Expanding training window with embargo period between train and validation
    to prevent information leakage from overlapping forward-looking targets.
    
    The 20-day embargo is critical because the target (Forward_Return_20d) means
    a row at date T uses price data up to T+20. Without the embargo, train rows
    near the boundary contaminate validation rows via overlapping future prices.
    
    Split layout (5 folds, 30% min train):
      Fold 1: Train [d0..d_30%]  ~~20d gap~~  Val [d_30%+20 .. d_44%]
      Fold 2: Train [d0..d_44%]  ~~20d gap~~  Val [d_44%+20 .. d_58%]
      ...
      Fold 5: Train [d0..d_86%]  ~~20d gap~~  Val [d_86%+20 .. d_100%]
    
    Each fold's training window EXPANDS (walk-forward), giving later folds
    more training data — just like real-world model retraining.
    """
    
    def __init__(self, n_splits=5, embargo_td=20, min_train_pct=0.3):
        """
        Args:
            n_splits: Number of validation folds
            embargo_td: Trading days to purge between train and val
            min_train_pct: Minimum fraction of dates for first training set
        """
        self.n_splits = n_splits
        self.embargo_td = embargo_td
        self.min_train_pct = min_train_pct
    
    def split(self, X, y=None, groups=None, dates=None):
        """Generate (train_idx, val_idx) tuples.
        
        Args:
            X: Feature matrix
            dates: DatetimeIndex aligned with X (required for date-based embargo)
        
        Yields:
            (train_indices, val_indices) numpy arrays
        """
        n = len(X)
        
        if dates is not None and hasattr(dates, 'unique'):
            # Date-based splitting: proper embargo in calendar days
            unique_dates = np.sort(dates.unique())
            n_dates = len(unique_dates)
            
            # Calculate sizes
            min_train_dates = int(n_dates * self.min_train_pct)
            remaining_dates = n_dates - min_train_dates
            val_size = remaining_dates // self.n_splits
            
            if val_size < 5:
                raise ValueError(f"Not enough dates for {self.n_splits} folds "
                               f"(only {remaining_dates} dates after min_train)")
            
            for fold in range(self.n_splits):
                train_end = min_train_dates + fold * val_size
                val_start = train_end + self.embargo_td
                val_end = min(val_start + val_size, n_dates)
                
                if val_start >= n_dates or val_end <= val_start:
                    break
                
                train_date_set = set(unique_dates[:train_end])
                val_date_set = set(unique_dates[val_start:val_end])
                
                train_idx = np.where(dates.isin(train_date_set))[0]
                val_idx = np.where(dates.isin(val_date_set))[0]
                
                if len(train_idx) > 0 and len(val_idx) > 0:
                    yield train_idx, val_idx
        else:
            # Fallback: index-based splitting with row embargo
            min_train = int(n * self.min_train_pct)
            remaining = n - min_train
            val_size = remaining // self.n_splits
            
            for fold in range(self.n_splits):
                train_end = min_train + fold * val_size
                val_start = train_end + self.embargo_td
                val_end = min(val_start + val_size, n)
                
                if val_start >= n or val_end <= val_start:
                    break
                
                yield np.arange(train_end), np.arange(val_start, val_end)
    
    def get_n_splits(self):
        return self.n_splits


def train_and_save():
    """Main training pipeline."""
    print("🚀 Starting ML 20D Training Pipeline (yfinance)...")
    print(f"📋 Target Features: {len(FEATURE_NAMES_V3)} (from feature_registry v3.1)")
    
    # 1. Get Tickers
    tickers = get_sp500_tickers(limit=500)  # S&P 500 full universe
    print(f"📊 Universe: {len(tickers)} tickers")
    
    # 2. Download Data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 5)  # 5 years history
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # Fetch SPY for relative strength
    print("📥 Fetching SPY benchmark...")
    spy_df = fetch_yf_history("SPY", start_str, end_str)
    spy_returns = None
    spy_returns_60d = None
    market_regime_df = None
    
    if spy_df is not None:
        spy_returns = spy_df['Close'].pct_change(20)
        spy_returns_60d = spy_df['Close'].pct_change(60)
        market_regime_df = calculate_market_regime(spy_df)
        print(f"   ✅ SPY: {len(spy_df)} days")
    
    # Fetch sector ETFs
    sector_etf_returns = fetch_sector_etf_data(start_str, end_str)
    
    # 3. Download and process all tickers
    all_data = []
    print(f"📥 Downloading {len(tickers)} tickers (yfinance)...")
    
    completed = 0
    for ticker in tickers:
        completed += 1
        if completed % 25 == 0:
            print(f"   ... processed {completed}/{len(tickers)}")
        
        df = fetch_yf_history(ticker, start_str, end_str)
        if df is not None and len(df) > 250:
            df = calculate_all_features(df, spy_returns, spy_returns_60d, market_regime_df, sector_etf_returns, ticker)
            if len(df) > 50:
                df['Ticker'] = ticker
                all_data.append(df)
    
    if not all_data:
        raise RuntimeError("No data downloaded!")
    
    full_df = pd.concat(all_data)
    print(f"📊 Total rows: {len(full_df)}, Tickers: {full_df['Ticker'].nunique()}")
    
    # 4. Labeling — Fixed economic threshold (not percentile-based)
    # A "winner" must deliver ≥5% absolute return over 20 trading days.
    # This gives the label a stable real-world meaning regardless of market regime.
    # Previous P80 approach labeled exactly 20% as winners, making "win" = "top quintile
    # of whatever happened", which could mean -2% in a bear market.
    WINNER_THRESHOLD = 0.05  # 5% absolute return in 20 trading days
    full_df['Label'] = (full_df['Forward_Return_20d'] >= WINNER_THRESHOLD).astype(int)
    
    winner_pct = full_df['Label'].mean() * 100
    print(f"🎯 Winner threshold: ≥{WINNER_THRESHOLD*100:.0f}% return (fixed economic target)")
    print(f"📊 Class: {full_df['Label'].sum()} winners ({winner_pct:.1f}%)")
    if winner_pct < 5 or winner_pct > 40:
        print(f"⚠️  Class imbalance outside 5-40% range — verify data quality")
    
    # 5. Prepare features
    # Only use features that exist in the data
    available_features = [f for f in FEATURE_NAMES_V3 if f in full_df.columns]
    missing_features = [f for f in FEATURE_NAMES_V3 if f not in full_df.columns]
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")
    print(f"📊 Using {len(available_features)} features")
    
    full_df = full_df.sort_index()
    X = full_df[available_features].fillna(0).astype(np.float32)  # float32 halves memory
    y = full_df['Label']
    dates = full_df.index  # DatetimeIndex for purged walk-forward CV
    n_training_samples = len(X)  # Save before freeing
    n_tickers = full_df['Ticker'].nunique() if 'Ticker' in full_df.columns else 0
    
    # Free the large DataFrame — we only need X, y, dates from here
    del full_df
    gc.collect()
    print(f"💾 Feature matrix: {X.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    
    # 6. Purged Walk-Forward Cross-Validation
    print("\n📊 Purged Walk-Forward CV (5 folds, 20-day embargo)...")
    pwf_cv = PurgedWalkForwardCV(n_splits=5, embargo_td=20, min_train_pct=0.3)
    cv_aucs = []
    cv_p20 = []
    
    for fold, (train_idx, val_idx) in enumerate(pwf_cv.split(X, dates=dates), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if y_train.sum() < 10 or y_val.sum() < 5:
            print(f"   Fold {fold}: SKIPPED (insufficient positive samples)")
            continue
        
        train_dates = dates[train_idx]
        val_dates = dates[val_idx]
        print(f"   Fold {fold}: Train {train_dates.min().date()} → {train_dates.max().date()} "
              f"({len(train_idx):,} rows), "
              f"Val {val_dates.min().date()} → {val_dates.max().date()} ({len(val_idx):,} rows)")
        
        model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=4,
            min_samples_leaf=20, l2_regularization=0.1,
            class_weight='balanced', early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        p20 = precision_at_k(y_val, y_pred, k=20)
        
        cv_aucs.append(auc)
        cv_p20.append(p20)
        baseline = y_val.mean()
        lift = p20 / baseline if baseline > 0 else 0
        print(f"            AUC={auc:.4f}, P@20={p20:.1%} ({lift:.1f}x lift)")
    
    mean_auc = np.mean(cv_aucs)
    mean_p20 = np.mean(cv_p20)
    print(f"\n📈 Walk-Forward CV: AUC={mean_auc:.4f} ± {np.std(cv_aucs):.4f}, P@20={mean_p20:.1%}")
    
    # 7. Train final ensemble
    print("\n🧠 Training final ensemble model...")
    split_idx = int(len(X) * 0.8)
    X_train, X_calib = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_calib = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Model 1: HistGradientBoosting
    model1 = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4,
        min_samples_leaf=20, l2_regularization=0.1,
        class_weight='balanced', early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=15,
        random_state=42
    )
    model1.fit(X_train, y_train)
    print(f"   HistGB stopped at iteration {model1.n_iter_}")
    gc.collect()
    
    # Model 2: RandomForest (reduced for memory-constrained environments)
    model2 = RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_leaf=30,
        max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=1
    )
    model2.fit(X_train, y_train)
    print(f"   RandomForest: {model2.n_estimators} trees")
    gc.collect()
    
    # Model 3: LogisticRegression (with scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values.astype(np.float64))  # LR needs float64
    model3 = LogisticRegression(class_weight='balanced', max_iter=1000, C=0.1, random_state=42)
    model3.fit(X_train_scaled, y_train)
    model3._needs_scaling = True
    del X_train_scaled
    gc.collect()
    print("   LogisticRegression converged")
    
    # Ensemble
    weights = [0.45, 0.35, 0.20]
    base_model = EnsembleClassifier([model1, model2, model3], weights, scaler)
    
    # For sklearn 1.8+, skip explicit calibration and use the ensemble directly
    # The ensemble already averages probabilities from multiple models which provides
    # reasonable calibration through model averaging
    calibrated = base_model
    print("   ✅ Ensemble created (using model averaging for calibration)")
    
    # 8. Evaluate
    y_pred_calib = calibrated.predict_proba(X_calib)[:, 1]
    calib_auc = roc_auc_score(y_calib, y_pred_calib)
    calib_p20 = precision_at_k(y_calib, y_pred_calib, k=20)
    print(f"\n📊 Calibration Set: AUC={calib_auc:.4f}, P@20={calib_p20:.1%}")
    
    # 9. Feature importance (use RandomForest which has stable importance API)
    print("\n📊 Top 15 Feature Importance (from RandomForest):")
    imp_df = pd.DataFrame({
        'feature': available_features,
        'importance': model2.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in imp_df.head(15).iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"   {row['feature']:28s} {row['importance']:.4f} {bar}")
    
    # Warn about zero-importance features
    zero_imp = imp_df[imp_df['importance'] == 0]['feature'].tolist()
    if zero_imp:
        print(f"\n⚠️ Zero-importance features ({len(zero_imp)}): {zero_imp}")
    
    # Save feature importance CSV (keep synced with model)
    imp_path = MODELS_DIR / "model_20d_v3_feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"   💾 Feature importance saved: {imp_path}")

    # 10. Save
    import sklearn
    bundle = {
        "model": calibrated,
        "feature_names": available_features,
        "full_feature_names": FEATURE_NAMES_V3,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now().strftime("%Y%m%d_%H%M"),
        "training_samples": n_training_samples,
        "metrics": {
            "oos_auc": mean_auc,
            "cv_auc_mean": mean_auc,
            "cv_auc_std": np.std(cv_aucs),
            "precision_at_20": mean_p20,
            "note": f"Trained on {len(available_features)} features ({n_training_samples} samples)"
        }
    }
    
    path = MODELS_DIR / "model_20d_v3.pkl"
    joblib.dump(bundle, path)
    print(f"\n✅ Model saved: {path}")
    
    # Save metadata.json
    import json
    meta = {
        "sklearn_version": sklearn.__version__,
        "feature_version": "v3.1",
        "feature_list": available_features,
        "training_timestamp_utc": datetime.utcnow().isoformat(),
        "model_type": "Ensemble(HistGB+RF+LR, no calibration)",
        "cv_method": "PurgedWalkForwardCV(embargo=20d, expanding_window)",
        "target": "Forward_Return_20d >= 5%",
        "metrics": bundle["metrics"]
    }
    with open(path.with_suffix(".metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    # Also save to ml/bundles/latest
    latest_dir = Path("ml/bundles/latest")
    latest_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, latest_dir / "model.joblib")
    with open(latest_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"🗂️ Latest bundle: {latest_dir}")
    
    print(f"\n🏆 Training Complete!")
    print(f"   OOS AUC: {mean_auc:.4f}")
    print(f"   P@20: {mean_p20:.1%}")
    
    return path, bundle


if __name__ == "__main__":
    train_and_save()
