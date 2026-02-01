import sys, os
# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import requests
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.calibration import CalibratedClassifierCV

# Sector mapping for sector-relative features
from core.sector_mapping import get_stock_sector, get_sector_etf, get_all_sector_etfs

# --- CONFIG ---
POLYGON_KEY = os.environ.get("POLYGON_API_KEY")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
DATA_DIR = Path("data")


def precision_at_k(y_true, y_pred_proba, k=20):
    """
    Calculate precision for top K predictions.
    
    This is the key metric for trading signals - we only trade the top K,
    so we need to know what fraction of those are actually winners.
    
    Args:
        y_true: Actual labels (0/1)
        y_pred_proba: Predicted probabilities for positive class
        k: Number of top predictions to evaluate
    
    Returns:
        Precision: fraction of top K predictions that are actually positive
    """
    if len(y_true) < k:
        k = len(y_true)
    if k == 0:
        return 0.0
    
    # Get indices of top K predictions (highest probability)
    top_k_idx = np.argsort(y_pred_proba)[-k:]
    
    # Calculate precision: how many of top K are actually winners?
    if hasattr(y_true, 'iloc'):
        precision = y_true.iloc[top_k_idx].mean()
    else:
        precision = y_true[top_k_idx].mean()
    
    return float(precision)


# --- FEATURE IMPORTANCE ANALYSIS ---
def analyze_feature_importance(model, feature_names, X_test, y_test, report_lines=None):
    """Analyze and display feature importance from multiple perspectives.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        X_test: Test features for permutation importance
        y_test: Test labels
        report_lines: Optional list to append report text to
    
    Returns:
        DataFrame with permutation importance results
    """
    from sklearn.inspection import permutation_importance
    
    if report_lines is None:
        report_lines = []
    
    header = "\n" + "="*60 + "\n📊 FEATURE IMPORTANCE ANALYSIS\n" + "="*60
    print(header)
    report_lines.append(header)
    
    # 1. Built-in importance (from HistGradientBoosting or ensemble average)
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'builtin_importance': model.feature_importances_
        }).sort_values('builtin_importance', ascending=False)
        
        section = "\n🔹 Built-in Feature Importance (Top 15):"
        print(section)
        report_lines.append(section)
        
        for _, row in importance_df.head(15).iterrows():
            bar = "█" * int(row['builtin_importance'] * 50)
            line = f"   {row['feature']:28s} {row['builtin_importance']:.4f} {bar}"
            print(line)
            report_lines.append(line)
    
    # 2. Permutation importance (more reliable for model-agnostic analysis)
    print("\n   Computing permutation importance (this may take a moment)...")
    perm_importance = permutation_importance(
        model, X_test, y_test, 
        n_repeats=10, 
        random_state=42,
        n_jobs=-1,
        scoring='roc_auc'
    )
    
    perm_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    section = "\n🔹 Permutation Importance (AUC-based, Top 15):"
    print(section)
    report_lines.append(section)
    
    for _, row in perm_df.head(15).iterrows():
        bar = "█" * int(max(0, row['importance_mean']) * 200)  # Scale for visibility
        line = f"   {row['feature']:28s} {row['importance_mean']:+.4f} ± {row['importance_std']:.4f} {bar}"
        print(line)
        report_lines.append(line)
    
    # 3. Identify useless features (negative or near-zero importance)
    useless = perm_df[perm_df['importance_mean'] < 0.001]['feature'].tolist()
    harmful = perm_df[perm_df['importance_mean'] < 0]['feature'].tolist()
    
    if harmful:
        warning = f"\n⚠️  Harmful features (negative importance, consider removing): {harmful}"
        print(warning)
        report_lines.append(warning)
    
    if useless and not harmful:
        warning = f"\n⚠️  Low-value features (importance < 0.001): {[f for f in useless if f not in harmful]}"
        print(warning)
        report_lines.append(warning)
    
    # 4. Feature importance tiers
    section = "\n🔹 Feature Importance Tiers:"
    print(section)
    report_lines.append(section)
    
    tier1 = perm_df[perm_df['importance_mean'] >= 0.01]['feature'].tolist()
    tier2 = perm_df[(perm_df['importance_mean'] >= 0.005) & (perm_df['importance_mean'] < 0.01)]['feature'].tolist()
    tier3 = perm_df[(perm_df['importance_mean'] >= 0.001) & (perm_df['importance_mean'] < 0.005)]['feature'].tolist()
    tier4 = perm_df[perm_df['importance_mean'] < 0.001]['feature'].tolist()
    
    line = f"   🥇 High impact (≥0.01):     {len(tier1)} features"
    print(line)
    report_lines.append(line)
    line = f"   🥈 Medium impact (0.005-0.01): {len(tier2)} features"
    print(line)
    report_lines.append(line)
    line = f"   🥉 Low impact (0.001-0.005):   {len(tier3)} features"
    print(line)
    report_lines.append(line)
    line = f"   ❌ Negligible (<0.001):       {len(tier4)} features"
    print(line)
    report_lines.append(line)
    
    return perm_df, report_lines


def select_top_features(importance_df, threshold=0.001, min_features=15):
    """Select features above importance threshold.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance_mean' columns
        threshold: Minimum importance to keep feature
        min_features: Minimum number of features to keep regardless of threshold
    
    Returns:
        Tuple of (selected_features, removed_features)
    """
    sorted_df = importance_df.sort_values('importance_mean', ascending=False)
    
    # Always keep at least min_features
    above_threshold = sorted_df[sorted_df['importance_mean'] >= threshold]['feature'].tolist()
    
    if len(above_threshold) < min_features:
        selected = sorted_df.head(min_features)['feature'].tolist()
    else:
        selected = above_threshold
    
    removed = [f for f in sorted_df['feature'].tolist() if f not in selected]
    
    print(f"\n✅ Feature Selection: Kept {len(selected)}, Removed {len(removed)}")
    if removed:
        print(f"   Removed features: {removed}")
    
    return selected, removed


def remove_correlated_features(df, features, threshold=0.95, report_lines=None):
    """Remove highly correlated features to reduce redundancy.
    
    Args:
        df: DataFrame containing the features
        features: List of feature names to analyze
        threshold: Correlation threshold above which to remove features
        report_lines: Optional list to append report text to
    
    Returns:
        List of features with highly correlated ones removed
    """
    if report_lines is None:
        report_lines = []
    
    corr_matrix = df[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find pairs of highly correlated features
    to_drop = set()
    correlated_pairs = []
    
    for column in upper.columns:
        for idx in upper.index:
            if upper.loc[idx, column] > threshold:
                correlated_pairs.append((idx, column, upper.loc[idx, column]))
                # Drop the one that appears later in the list (arbitrary choice)
                to_drop.add(column)
    
    if correlated_pairs:
        section = f"\n🔄 Highly Correlated Features (r > {threshold}):"
        print(section)
        report_lines.append(section)
        
        for f1, f2, corr in correlated_pairs:
            line = f"   {f1} <-> {f2}: r={corr:.3f}"
            print(line)
            report_lines.append(line)
        
        line = f"   Removing: {list(to_drop)}"
        print(line)
        report_lines.append(line)
    else:
        line = f"\n✅ No highly correlated features found (threshold={threshold})"
        print(line)
        report_lines.append(line)
    
    return [f for f in features if f not in to_drop], report_lines


def save_feature_importance_report(report_lines, importance_df, output_path):
    """Save feature importance analysis to a text report file.
    
    Args:
        report_lines: List of report text lines
        importance_df: DataFrame with importance results
        output_path: Path to save the report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Feature Importance Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        # Write report lines
        for line in report_lines:
            f.write(line + "\n")
        
        # Write full importance table
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("FULL PERMUTATION IMPORTANCE TABLE\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"{'Feature':<30} {'Importance':>12} {'Std':>10}\n")
        f.write("-" * 55 + "\n")
        
        for _, row in importance_df.sort_values('importance_mean', ascending=False).iterrows():
            f.write(f"{row['feature']:<30} {row['importance_mean']:>+12.6f} {row['importance_std']:>10.6f}\n")
    
    print(f"\n📄 Feature importance report saved to: {output_path}")

def _check_api_key():
    """Verify POLYGON_API_KEY is set before proceeding."""
    if not POLYGON_KEY:
        print("❌ ERROR: POLYGON_API_KEY environment variable is required but not set.")
        print("   Please set it before running this script:")
        print("   export POLYGON_API_KEY=your_api_key_here")
        sys.exit(1)

# --- DATA INGESTION (POLYGON) ---
def fetch_polygon_history(ticker, start_str, end_str):
    """Fetch adjusted daily bars from Polygon.io (Fast & Reliable)."""
    # Using 'aggs' endpoint which allows fetching a range in one go
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_str}/{end_str}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": POLYGON_KEY
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if "results" not in data or not data["results"]:
            return None
        
        # Parse
        df = pd.DataFrame(data["results"])
        # Map Polygon cols: v=Volume, o=Open, c=Close, h=High, l=Low, t=UnixMS
        df = df.rename(columns={"v": "Volume", "o": "Open", "c": "Close", "h": "High", "l": "Low", "t": "Date"})
        df["Date"] = pd.to_datetime(df["Date"], unit="ms")
        df = df.set_index("Date").sort_index()
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        return None

def get_universe_tickers(limit=2000):
    """Try to get full universe, fallback to S&P 500."""
    try:
        from core.data import get_universe
        return get_universe(limit=limit)
    except:
        # Fallback to standard list if core fails
        try:
            return pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
        except:
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD", "NFLX", "INTC"] # Minimal fallback

# --- FEATURE ENGINEERING ---
def calculate_features(df, spy_returns: pd.Series = None, market_regime_df: pd.DataFrame = None,
                       sector_etf_returns: pd.DataFrame = None, ticker: str = None):
    """Generate 20d ML features (expanded set).
    
    Args:
        df: DataFrame with OHLCV data
        spy_returns: Series of SPY 20-day returns indexed by date (for relative strength)
        market_regime_df: DataFrame with market regime indicators (optional)
        sector_etf_returns: DataFrame with sector ETF returns, columns=ETF symbols (optional)
        ticker: Stock ticker for sector lookup (optional)
    """
    df = df.copy()
    
    # === BASIC RETURNS ===
    df['Return_20d'] = df['Close'].pct_change(20)
    df['Return_10d'] = df['Close'].pct_change(10)
    df['Return_5d'] = df['Close'].pct_change(5)
    
    # === VOLATILITY ===
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['ATR_Pct'] = df['ATR'] / df['Close']
    
    # VCP_Ratio: ATR(10) / ATR(30) - volatility contraction pattern
    atr_10 = (df['High'] - df['Low']).rolling(10).mean()
    atr_30 = (df['High'] - df['Low']).rolling(30).mean()
    df['VCP_Ratio'] = atr_10 / atr_30.replace(0, np.nan)
    
    # Tightness_Ratio: (High-Low range last 5 days) / (range last 20 days)
    range_5d = df['High'].rolling(5).max() - df['Low'].rolling(5).min()
    range_20d = df['High'].rolling(20).max() - df['Low'].rolling(20).min()
    df['Tightness_Ratio'] = range_5d / range_20d.replace(0, np.nan)
    
    # Dist_From_52w_High: (Close / 52-week High) - 1
    high_52w = df['High'].rolling(252).max()
    df['Dist_From_52w_High'] = (df['Close'] / high_52w.replace(0, np.nan)) - 1
    
    # === MOVING AVERAGE ALIGNMENT ===
    ma20 = df['Close'].rolling(20).mean()
    ma50 = df['Close'].rolling(50).mean()
    ma200 = df['Close'].rolling(200).mean()
    # MA_Alignment: 1 if Close > MA20 > MA50 > MA200, else 0
    df['MA_Alignment'] = ((df['Close'] > ma20) & (ma20 > ma50) & (ma50 > ma200)).astype(int)
    
    # === VOLUME ===
    # Volume_Surge: Volume(5-day avg) / Volume(20-day avg)
    vol_5d = df['Volume'].rolling(5).mean()
    vol_20d = df['Volume'].rolling(20).mean()
    df['Volume_Surge'] = vol_5d / vol_20d.replace(0, np.nan)
    
    # Up_Down_Volume_Ratio: avg volume on up days / avg volume on down days (last 20 days)
    daily_return = df['Close'].pct_change()
    up_day = daily_return > 0
    down_day = daily_return < 0
    # Calculate rolling sums for up/down volume
    up_volume = (df['Volume'] * up_day).rolling(20).sum()
    down_volume = (df['Volume'] * down_day).rolling(20).sum()
    up_days_count = up_day.rolling(20).sum()
    down_days_count = down_day.rolling(20).sum()
    avg_up_vol = up_volume / up_days_count.replace(0, np.nan)
    avg_down_vol = down_volume / down_days_count.replace(0, np.nan)
    df['Up_Down_Volume_Ratio'] = avg_up_vol / avg_down_vol.replace(0, np.nan)
    
    # === INSTITUTIONAL ACCUMULATION VOLUME FEATURES ===
    # Volume_Ratio_20d: current volume / 20-day avg (spike detection)
    df['Volume_Ratio_20d'] = df['Volume'] / vol_20d.replace(0, np.nan)
    
    # Volume_Trend: linear regression slope of volume over 20 days (accumulation trend)
    def volume_slope(x):
        if len(x) < 20:
            return 0.0
        try:
            slope = np.polyfit(range(len(x)), x, 1)[0]
            return slope / (x.mean() + 1e-8)  # Normalize by mean volume
        except:
            return 0.0
    df['Volume_Trend'] = df['Volume'].rolling(20).apply(volume_slope, raw=True)
    
    # Up_Volume_Ratio: sum of volume on up days / total volume (buying pressure)
    total_vol_20d = df['Volume'].rolling(20).sum()
    df['Up_Volume_Ratio'] = up_volume / total_vol_20d.replace(0, np.nan)
    
    # Volume_Price_Confirm: price up AND volume up over last 5 days avg
    price_up = df['Close'] > df['Close'].shift(1)
    vol_up = df['Volume'] > df['Volume'].shift(1)
    df['Volume_Price_Confirm'] = (price_up & vol_up).astype(float).rolling(5).mean()
    
    # Relative_Volume_Rank: percentile rank of today's volume vs last 60 days
    def vol_percentile_rank(x):
        if len(x) < 60:
            return 0.5
        return (x.rank(pct=True).iloc[-1] if hasattr(x, 'rank') else 
                pd.Series(x).rank(pct=True).iloc[-1])
    df['Relative_Volume_Rank'] = df['Volume'].rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 60 else 0.5, raw=False
    )
    
    # Handle inf/nan in volume features
    volume_features = ['Volume_Ratio_20d', 'Volume_Trend', 'Up_Volume_Ratio', 
                       'Volume_Price_Confirm', 'Relative_Volume_Rank']
    for vf in volume_features:
        df[vf] = df[vf].replace([np.inf, -np.inf], np.nan)
        df[vf] = df[vf].fillna(df[vf].median() if df[vf].notna().any() else 0.5)
    
    # === RSI ===
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rs = ma_up / ma_down
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # === MOMENTUM ===
    # Momentum_Consistency: % of last 20 days with positive returns
    positive_return = (daily_return > 0).astype(int)
    df['Momentum_Consistency'] = positive_return.rolling(20).mean()
    
    # RS_vs_SPY_20d: stock return(20d) - SPY return(20d)
    if spy_returns is not None:
        # Ensure SPY series has datetime index
        spy_series = spy_returns.copy()
        if not isinstance(spy_series.index, pd.DatetimeIndex):
            spy_series.index = pd.to_datetime(spy_series.index)
        
        # Reindex to stock's dates with forward-fill for missing SPY dates
        # This handles holidays, trading halts, and minor date mismatches
        spy_aligned = spy_series.reindex(df.index, method='ffill')
        
        df['RS_vs_SPY_20d'] = df['Return_20d'] - spy_aligned.fillna(0)
        
        # Log alignment quality (only warn if significant mismatch)
        aligned_pct = (spy_aligned.notna().sum() / len(df)) * 100
        if aligned_pct < 90:
            import warnings
            warnings.warn(f"SPY alignment: {aligned_pct:.1f}% of dates matched")
    else:
        df['RS_vs_SPY_20d'] = 0.0  # Fallback if no SPY data
    
    # === MARKET REGIME FEATURES ===
    if market_regime_df is not None:
        # Align market regime data with stock dates using forward-fill
        for col in ['Market_Regime', 'Market_Volatility', 'Market_Trend', 'High_Volatility']:
            if col in market_regime_df.columns:
                aligned = market_regime_df[col].reindex(df.index, method='ffill')
                default_val = 0.15 if col == 'Market_Volatility' else 0.0
                df[col] = aligned.fillna(default_val)
    else:
        # Defaults if no regime data available
        df['Market_Regime'] = 0
        df['Market_Volatility'] = 0.15
        df['Market_Trend'] = 0.0
        df['High_Volatility'] = 0
    
    # === SECTOR-RELATIVE FEATURES ===
    # Compare stock performance to its sector ETF
    if sector_etf_returns is not None and ticker is not None:
        sector = get_stock_sector(ticker)
        sector_etf = get_sector_etf(sector)
        
        if sector_etf is not None and sector_etf in sector_etf_returns.columns:
            # Get sector ETF returns aligned to stock dates
            sector_ret = sector_etf_returns[sector_etf].reindex(df.index, method='ffill')
            
            # Sector_RS: stock 20d return - sector ETF 20d return
            df['Sector_RS'] = df['Return_20d'] - sector_ret.fillna(0)
            
            # Sector_Momentum: sector ETF 20d return (absolute sector strength)
            df['Sector_Momentum'] = sector_ret.fillna(0)
            
            # Sector_Rank: rolling rank of stock vs sector (simplified proxy)
            # This measures if stock is outperforming sector consistently
            # 1 if stock beats sector in return_5d, else 0
            sector_ret_5d = sector_etf_returns[sector_etf].pct_change(5).reindex(df.index, method='ffill')
            df['Sector_Rank'] = (df['Return_5d'] > sector_ret_5d.fillna(0)).astype(float)
        else:
            # Unknown sector or missing ETF data - use neutral defaults
            df['Sector_RS'] = 0.0
            df['Sector_Momentum'] = 0.0
            df['Sector_Rank'] = 0.5
    else:
        # No sector data available - use neutral defaults
        df['Sector_RS'] = 0.0
        df['Sector_Momentum'] = 0.0
        df['Sector_Rank'] = 0.5
    
    # === PRICE ACTION PATTERN FEATURES (Breakout Detection) ===
    # 52-week features for breakout proximity
    low_52w = df['Low'].rolling(252).min()
    df['Distance_From_52w_Low'] = (df['Close'] - low_52w) / low_52w.replace(0, np.nan)
    
    # Consolidation tightness (lower = tighter = more explosive potential)
    h20 = df['High'].rolling(20).max()
    l20 = df['Low'].rolling(20).min()
    avg20 = df['Close'].rolling(20).mean()
    df['Consolidation_Tightness'] = (h20 - l20) / avg20.replace(0, np.nan)
    
    # Days since 52w high (lower = momentum, higher = recovery potential)
    def days_since_high(s):
        if len(s) < 252:
            return 126  # Default to half year
        try:
            max_idx = np.argmax(s.values)
            return len(s) - max_idx - 1
        except:
            return 126
    df['Days_Since_52w_High'] = df['High'].rolling(252).apply(days_since_high, raw=False)
    # Normalize to 0-1 range (0 = just hit high, 1 = 252 days ago)
    df['Days_Since_52w_High'] = df['Days_Since_52w_High'] / 252.0
    
    # Moving average trend features
    sma50 = df['Close'].rolling(50).mean()
    sma200 = df['Close'].rolling(200).mean()
    df['Price_vs_SMA50'] = (df['Close'] - sma50) / sma50.replace(0, np.nan)
    df['Price_vs_SMA200'] = (df['Close'] - sma200) / sma200.replace(0, np.nan)
    df['SMA50_vs_SMA200'] = (sma50 - sma200) / sma200.replace(0, np.nan)  # Golden/death cross
    
    # MA slope (trend strength/direction)
    df['MA_Slope_20d'] = ma20.diff(20) / ma20.shift(20).replace(0, np.nan)
    
    # Distance to resistance (recent 20d high)
    df['Distance_To_Resistance'] = (h20 - df['Close']) / df['Close'].replace(0, np.nan)
    
    # Support strength: count of times price touched 20d low level (within 1%)
    def support_touches(prices, lows):
        if len(prices) < 20:
            return 0
        support_level = lows.iloc[-1]
        tolerance = support_level * 0.01  # 1% tolerance
        touches = ((prices >= support_level - tolerance) & 
                   (prices <= support_level + tolerance)).sum()
        return min(touches / 5.0, 1.0)  # Normalize: 5+ touches = 1.0
    
    # Simpler approach: rolling count of days near 20d low
    near_low = (df['Low'] <= l20 * 1.02).astype(float)  # Within 2% of 20d low
    df['Support_Strength'] = near_low.rolling(20).mean()  # Fraction of days near support
    
    # Handle inf/nan in price action features
    price_action_features = ['Distance_From_52w_Low', 'Consolidation_Tightness', 
                             'Days_Since_52w_High', 'Price_vs_SMA50', 'Price_vs_SMA200',
                             'SMA50_vs_SMA200', 'MA_Slope_20d', 'Distance_To_Resistance',
                             'Support_Strength']
    for pf in price_action_features:
        df[pf] = df[pf].replace([np.inf, -np.inf], np.nan)
        df[pf] = df[pf].fillna(df[pf].median() if df[pf].notna().any() else 0.0)
    
    # === TARGET ===
    df['Forward_Return_20d'] = df['Close'].shift(-20) / df['Close'] - 1.0
    
    return df.dropna()


def calculate_market_regime(spy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate market regime indicators from SPY data.
    
    Identifies Bull, Bear, or Sideways market conditions to provide
    context for the model - meteoric rises behave differently in each regime.
    
    Returns DataFrame with:
    - Market_Regime: 1=Bull, 0=Sideways, -1=Bear
    - Market_Volatility: 20-day rolling volatility (annualized)
    - Market_Trend: 50-day return (momentum)
    - High_Volatility: 1 if volatility > 75th percentile
    """
    if spy_df is None or len(spy_df) < 200:
        return None
    
    regime_df = pd.DataFrame(index=spy_df.index)
    
    # Moving averages
    spy_close = spy_df['Close']
    ma50 = spy_close.rolling(50).mean()
    ma200 = spy_close.rolling(200).mean()
    
    # Regime based on MA alignment and momentum
    # Bull: SPY > MA50 > MA200 and positive 20d return
    # Bear: SPY < MA50 < MA200 and negative 20d return
    # Sideways: Everything else
    ret_20d = spy_close.pct_change(20)
    
    regime_df['Market_Regime'] = 0  # Default: Sideways
    
    bull_condition = (spy_close > ma50) & (ma50 > ma200) & (ret_20d > 0.02)
    bear_condition = (spy_close < ma50) & (ma50 < ma200) & (ret_20d < -0.02)
    
    regime_df.loc[bull_condition, 'Market_Regime'] = 1
    regime_df.loc[bear_condition, 'Market_Regime'] = -1
    
    # Market volatility (annualized)
    daily_returns = spy_close.pct_change()
    regime_df['Market_Volatility'] = daily_returns.rolling(20).std() * np.sqrt(252)
    
    # Market trend (50-day momentum)
    regime_df['Market_Trend'] = spy_close.pct_change(50)
    
    # VIX proxy: high volatility regime
    vol_75th = regime_df['Market_Volatility'].quantile(0.75)
    regime_df['High_Volatility'] = (regime_df['Market_Volatility'] > vol_75th).astype(int)
    
    return regime_df


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Ensemble of multiple classifiers with weighted averaging.
    
    Combines predictions from multiple models (HistGB, RF, LR) to reduce
    variance and improve robustness across different market conditions.
    """
    
    def __init__(self, models, weights=None, scaler=None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)
        self.scaler = scaler  # For models that need scaling (e.g., LogisticRegression)
        self.classes_ = np.array([0, 1])
    
    def fit(self, X, y):
        # Models should already be fitted externally
        return self
    
    def predict_proba(self, X):
        """Average probability predictions from all models."""
        X_arr = np.array(X)
        weighted_probas = np.zeros((len(X_arr), 2))
        
        for model, weight in zip(self.models, self.weights):
            # Check if this model needs scaled input
            if hasattr(model, '_needs_scaling') and model._needs_scaling and self.scaler:
                X_input = self.scaler.transform(X_arr)
            else:
                X_input = X_arr
            
            proba = model.predict_proba(X_input)
            weighted_probas += proba * weight
        
        return weighted_probas
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    @property
    def n_iter_(self):
        """Return n_iter_ from first model (HistGB) for metadata."""
        if hasattr(self.models[0], 'n_iter_'):
            return self.models[0].n_iter_
        return None


# --- MAIN PIPELINE ---
def fetch_sector_etf_data(start_str: str, end_str: str) -> pd.DataFrame:
    """Fetch all sector ETF data and calculate 20d returns.
    
    Returns DataFrame with columns = ETF symbols, rows = dates,
    values = 20-day returns.
    """
    sector_etfs = get_all_sector_etfs()  # ['XLK', 'XLF', 'XLE', ...]
    sector_data = {}
    
    print(f"📥 Fetching {len(sector_etfs)} sector ETFs...")
    for etf in sector_etfs:
        df = fetch_polygon_history(etf, start_str, end_str)
        if df is not None and len(df) > 30:
            # Calculate 20-day returns for sector ETF
            sector_data[etf] = df['Close'].pct_change(20)
    
    if not sector_data:
        print("   ⚠️  No sector ETF data available")
        return None
    
    sector_df = pd.DataFrame(sector_data)
    print(f"   ✅ Sector ETF data loaded: {len(sector_data)} sectors, {len(sector_df)} days")
    return sector_df


def train_and_save_bundle():
    # Verify API key is available before proceeding
    _check_api_key()
    
    print("🚀 Starting Polygon-Powered Training Pipeline...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Fetch Universe
    tickers = get_universe_tickers(2000)
    print(f"📋 Universe size: {len(tickers)} tickers")
    
    # 2. Download Data (Parallel)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2) # 2 Years history
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # Fetch SPY data once for relative strength calculation
    print("📥 Fetching SPY benchmark data...")
    spy_df = fetch_polygon_history("SPY", start_str, end_str)
    spy_returns = None
    market_regime_df = None
    
    if spy_df is not None and len(spy_df) > 50:
        # Create SPY returns as a proper Series for date-aligned merging
        spy_returns = spy_df['Close'].pct_change(20)
        spy_returns.name = 'SPY_Return_20d'
        print(f"   ✅ SPY data loaded ({len(spy_df)} days, {spy_returns.notna().sum()} valid returns)")
        
        # Calculate market regime
        market_regime_df = calculate_market_regime(spy_df)
        if market_regime_df is not None:
            bull_days = (market_regime_df['Market_Regime'] == 1).sum()
            bear_days = (market_regime_df['Market_Regime'] == -1).sum()
            sideways_days = (market_regime_df['Market_Regime'] == 0).sum()
            print(f"   📊 Market regime distribution: Bull={bull_days}, Sideways={sideways_days}, Bear={bear_days}")
    else:
        print("   ⚠️  SPY data unavailable, RS_vs_SPY_20d will be 0")
    
    # Fetch sector ETF data for sector-relative features
    sector_etf_returns = fetch_sector_etf_data(start_str, end_str)
    
    all_data = []
    print("📥 Downloading data from Polygon (Threads=15)...")
    
    with ThreadPoolExecutor(max_workers=15) as executor:
        future_to_ticker = {executor.submit(fetch_polygon_history, t, start_str, end_str): t for t in tickers}
        completed = 0
        for future in as_completed(future_to_ticker):
            completed += 1
            if completed % 100 == 0: print(f"   ... processed {completed}/{len(tickers)}")
            
            t = future_to_ticker[future]
            df = future.result()
            if df is not None and len(df) > 50:
                df = calculate_features(df, spy_returns, market_regime_df, sector_etf_returns, t)
                df['Ticker'] = t
                all_data.append(df)

    if not all_data:
        raise RuntimeError("No data downloaded! Check API Key or Network.")
    
    full_df = pd.concat(all_data)
    print(f"📊 Total Training Rows: {len(full_df)}")
    
    # 3. Labeling (Percentile-based ranking for balanced classes)
    # Label top 20% performers as winners (relative ranking)
    # This guarantees ~20% positive class regardless of market conditions
    WINNER_PERCENTILE = 80  # Top 20% are winners
    percentile_threshold = full_df['Forward_Return_20d'].quantile(WINNER_PERCENTILE / 100.0)
    full_df['Label'] = (full_df['Forward_Return_20d'] >= percentile_threshold).astype(int)
    
    # Store threshold for metadata
    threshold = float(percentile_threshold)
    
    print(f"🎯 Winner threshold (top {100 - WINNER_PERCENTILE}%): {percentile_threshold*100:.1f}% return")
    
    # Print class distribution
    pos_count = full_df['Label'].sum()
    neg_count = len(full_df) - pos_count
    print(f"📊 Class distribution: {pos_count} winners ({pos_count/len(full_df)*100:.1f}%), "
          f"{neg_count} losers ({neg_count/len(full_df)*100:.1f}%)")
    
    # Print performance by market regime
    print("\n📊 Performance by Market Regime:")
    for regime_val, regime_name in [(1, 'Bull'), (0, 'Sideways'), (-1, 'Bear')]:
        mask = full_df['Market_Regime'] == regime_val
        if mask.sum() > 100:
            regime_winners = full_df.loc[mask, 'Label'].mean()
            print(f"   {regime_name:10s}: {mask.sum():6d} samples, {regime_winners:.1%} winners")

    # 4. Time-Series Cross-Validation (proper OOS evaluation)
    features = [
        # Original features (13)
        'RSI', 'ATR_Pct', 'Return_20d', 'Return_10d', 'Return_5d',
        # Technical - Volatility patterns
        'VCP_Ratio', 'Tightness_Ratio', 'Dist_From_52w_High', 'MA_Alignment',
        # Volume (basic)
        'Volume_Surge', 'Up_Down_Volume_Ratio',
        # Momentum
        'Momentum_Consistency', 'RS_vs_SPY_20d',
        # Market regime features (4)
        'Market_Regime', 'Market_Volatility', 'Market_Trend', 'High_Volatility',
        # Sector-relative features (3)
        'Sector_RS', 'Sector_Momentum', 'Sector_Rank',
        # Institutional accumulation volume features (5)
        'Volume_Ratio_20d', 'Volume_Trend', 'Up_Volume_Ratio',
        'Volume_Price_Confirm', 'Relative_Volume_Rank',
        # Price action pattern features (10)
        'Distance_From_52w_Low', 'Consolidation_Tightness', 'Days_Since_52w_High',
        'Price_vs_SMA50', 'Price_vs_SMA200', 'SMA50_vs_SMA200', 'MA_Slope_20d',
        'Distance_To_Resistance', 'Support_Strength',
    ]
    
    # Sort by date for proper time-series split
    full_df = full_df.sort_index()
    X = full_df[features]
    y = full_df['Label']
    
    print("📊 Running Time-Series Cross-Validation (5 folds)...")
    tscv = TimeSeriesSplit(n_splits=5)
    oos_aucs = []
    oos_logloss = []
    oos_p20 = []  # Precision@20
    oos_p50 = []  # Precision@50
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Skip folds with insufficient positive samples
        if y_train.sum() < 10 or y_val.sum() < 5:
            print(f"   Fold {fold}: ⚠️  Skipped (insufficient positive samples)")
            continue
        
        fold_model = HistGradientBoostingClassifier(
            max_iter=300,              # Maximum iterations (will stop early)
            learning_rate=0.05,
            max_depth=4,               # Slightly deeper for complex patterns
            min_samples_leaf=20,       # Prevent overfitting on small groups
            l2_regularization=0.1,     # L2 regularization
            class_weight='balanced',   # Built-in class balancing
            early_stopping=True,       # Enable early stopping
            validation_fraction=0.15,  # Use 15% for early stopping validation
            n_iter_no_change=10,       # Stop if no improvement for 10 iterations
            random_state=42,
        )
        fold_model.fit(X_train, y_train)
        
        # Out-of-sample predictions
        y_pred_proba = fold_model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        fold_auc = roc_auc_score(y_val, y_pred_proba)
        fold_ll = log_loss(y_val, y_pred_proba)
        fold_p20 = precision_at_k(y_val, y_pred_proba, k=20)
        fold_p50 = precision_at_k(y_val, y_pred_proba, k=50)
        
        oos_aucs.append(fold_auc)
        oos_logloss.append(fold_ll)
        oos_p20.append(fold_p20)
        oos_p50.append(fold_p50)
        
        # Calculate baseline (random) precision for comparison
        baseline = y_val.mean()  # ~20% if using percentile labeling
        lift_20 = fold_p20 / baseline if baseline > 0 else 0
        
        print(f"   Fold {fold}: AUC={fold_auc:.4f}, P@20={fold_p20:.1%} ({lift_20:.1f}x lift), P@50={fold_p50:.1%}")
    
    if not oos_aucs:
        raise RuntimeError("No valid CV folds! Check data quality.")
    
    mean_oos_auc = np.mean(oos_aucs)
    std_oos_auc = np.std(oos_aucs)
    mean_oos_ll = np.mean(oos_logloss)
    mean_p20 = np.mean(oos_p20) if oos_p20 else 0.0
    mean_p50 = np.mean(oos_p50) if oos_p50 else 0.0
    
    # Calculate baseline and lift
    baseline_precision = 0.20  # Since we label top 20% as winners
    lift_20 = mean_p20 / baseline_precision if baseline_precision > 0 else 0
    lift_50 = mean_p50 / baseline_precision if baseline_precision > 0 else 0
    
    print(f"\n📈 Cross-Validation Results:")
    print(f"   Out-of-Sample AUC: {mean_oos_auc:.4f} ± {std_oos_auc:.4f}")
    print(f"   Out-of-Sample LogLoss: {mean_oos_ll:.4f}")
    print(f"   Precision@20: {mean_p20:.1%} (lift: {lift_20:.2f}x vs random)")
    print(f"   Precision@50: {mean_p50:.1%} (lift: {lift_50:.2f}x vs random)")
    print(f"   ")
    print(f"   🎯 Target: P@20 > 40% (2x lift) = meaningful signal")
    print(f"   🏆 Excellent: P@20 > 60% (3x lift) = strong alpha")
    
    # 4.5. Feature Importance Analysis (using last fold's model)
    report_lines = []
    report_lines.append(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Features: {len(features)}")
    report_lines.append(f"Total Training Samples: {len(full_df)}")
    report_lines.append(f"CV Folds: {len(oos_aucs)}")
    report_lines.append(f"Mean OOS AUC: {mean_oos_auc:.4f} ± {std_oos_auc:.4f}")
    report_lines.append(f"Mean P@20: {mean_p20:.1%}")
    
    # Train a quick model on last 80% for importance analysis
    print("\n🔍 Analyzing feature importance...")
    split_for_analysis = int(len(X) * 0.8)
    X_train_analysis = X.iloc[:split_for_analysis]
    X_test_analysis = X.iloc[split_for_analysis:]
    y_train_analysis = y.iloc[:split_for_analysis]
    y_test_analysis = y.iloc[split_for_analysis:]
    
    analysis_model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        class_weight='balanced',
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    analysis_model.fit(X_train_analysis, y_train_analysis)
    
    # Run feature importance analysis
    importance_df, report_lines = analyze_feature_importance(
        analysis_model, features, X_test_analysis, y_test_analysis, report_lines
    )
    
    # Check for correlated features
    _, report_lines = remove_correlated_features(full_df, features, threshold=0.95, report_lines=report_lines)
    
    # Recommend feature selection
    selected_features, removed_features = select_top_features(importance_df, threshold=0.001, min_features=20)
    report_lines.append(f"\nRecommended Features ({len(selected_features)}): {selected_features}")
    report_lines.append(f"Candidates for Removal ({len(removed_features)}): {removed_features}")
    
    # Save report
    report_path = MODELS_DIR / "feature_importance_report.txt"
    save_feature_importance_report(report_lines, importance_df, report_path)
    
    # 5. Train final model with probability calibration
    # Split: 80% for training, 20% for calibration (preserving time order)
    print("\n🧠 Training final model with probability calibration...")
    
    # Use time-based split (not random) to preserve temporal ordering
    split_idx = int(len(X) * 0.8)
    X_train_final, X_calib = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_final, y_calib = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"   Training samples: {len(X_train_final)}, Calibration samples: {len(X_calib)}")
    
    # Train base model 1: HistGradientBoostingClassifier (non-linear patterns)
    print("   Training Model 1/3: HistGradientBoostingClassifier...")
    model1 = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        l2_regularization=0.1,
        class_weight='balanced',
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
    )
    model1.fit(X_train_final, y_train_final)
    print(f"      HistGB stopped at iteration {model1.n_iter_}")
    
    # Train base model 2: RandomForest (feature interactions, robust to outliers)
    print("   Training Model 2/3: RandomForestClassifier...")
    model2 = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=20,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    model2.fit(X_train_final, y_train_final)
    print(f"      RandomForest trained with {model2.n_estimators} trees")
    
    # Train base model 3: LogisticRegression (linear baseline, needs scaling)
    print("   Training Model 3/3: LogisticRegression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    model3 = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        C=0.1,  # Regularization
        random_state=42,
    )
    model3.fit(X_train_scaled, y_train_final)
    model3._needs_scaling = True  # Mark for ensemble
    print("      LogisticRegression converged")
    
    # Create ensemble with weights favoring tree models
    ensemble_weights = [0.45, 0.35, 0.20]  # HistGB, RF, LR
    base_model = EnsembleClassifier(
        models=[model1, model2, model3],
        weights=ensemble_weights,
        scaler=scaler,
    )
    print(f"   Ensemble created with weights: HistGB={ensemble_weights[0]}, RF={ensemble_weights[1]}, LR={ensemble_weights[2]}")
    
    # Calibrate using isotonic regression on held-out 20%
    # Note: CalibratedClassifierCV with cv='prefit' uses provided data for calibration only
    calibrated_model = CalibratedClassifierCV(
        base_model, method='isotonic', cv='prefit'
    )
    calibrated_model.fit(X_calib, y_calib)
    
    # Use calibrated model as final model
    model = calibrated_model
    
    # Evaluate calibration quality with reliability diagram data
    print("\n📊 Calibration Quality Check:")
    y_pred_calib = calibrated_model.predict_proba(X_calib)[:, 1]
    y_pred_uncalib = base_model.predict_proba(X_calib)[:, 1]
    
    # Compute reliability metrics (binned calibration)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    print("   Predicted Prob  |  Actual Freq  |  Count  |  Calibration")
    print("   " + "-" * 55)
    
    calibration_errors = []
    for i in range(n_bins):
        mask = (y_pred_calib >= bin_edges[i]) & (y_pred_calib < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_pred = y_pred_calib[mask].mean()
            bin_actual = y_calib.iloc[mask.values].mean()
            count = mask.sum()
            error = abs(bin_pred - bin_actual)
            calibration_errors.append(error * count)  # Weighted by count
            status = "✓" if error < 0.1 else "⚠️" if error < 0.2 else "✗"
            print(f"   {bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}      |    {bin_actual:.3f}     |  {count:5d}  |  {status} (err={error:.3f})")
    
    mean_calibration_error = sum(calibration_errors) / len(X_calib) if calibration_errors else 0
    print(f"\n   Mean Calibration Error (MCE): {mean_calibration_error:.4f}")
    
    # Compare AUCs (calibration shouldn't hurt AUC much)
    auc_uncalib = roc_auc_score(y_calib, y_pred_uncalib)
    auc_calib = roc_auc_score(y_calib, y_pred_calib)
    print(f"   Calibration set AUC (uncalibrated): {auc_uncalib:.4f}")
    print(f"   Calibration set AUC (calibrated):   {auc_calib:.4f}")
    
    # Precision@K on calibration set (key trading metric)
    p20_calib = precision_at_k(y_calib, y_pred_calib, k=20)
    p50_calib = precision_at_k(y_calib, y_pred_calib, k=50)
    print(f"   Calibration set P@20: {p20_calib:.1%}")
    print(f"   Calibration set P@50: {p50_calib:.1%}")
    
    # In-sample AUC on full data (for reference only)
    insample_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    print(f"   Full data AUC (reference): {insample_auc:.4f}")
    
    # Feature Importance (averaged across ensemble)
    print("\n📊 Feature Importance (averaged across ensemble):")
    try:
        # HistGB importance
        imp1 = model1.feature_importances_
        # RF importance  
        imp2 = model2.feature_importances_
        # LR coefficients (absolute value as importance proxy)
        imp3 = np.abs(model3.coef_[0])
        imp3 = imp3 / imp3.sum()  # Normalize to sum to 1
        
        # Weighted average matching ensemble weights
        avg_importance = imp1 * ensemble_weights[0] + imp2 * ensemble_weights[1] + imp3 * ensemble_weights[2]
        
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': avg_importance,
            'histgb': imp1,
            'rf': imp2,
            'lr': imp3,
        }).sort_values('importance', ascending=False)
        
        for _, row in importance_df.iterrows():
            bar = '█' * int(row['importance'] * 50)
            print(f"   {row['feature']:22s} {row['importance']:.4f} {bar}")
    except Exception as e:
        print(f"   (Feature importance calculation failed: {e})")
        importance_df = None
    
    # 6. Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    bundle = {
        "model": model,  # Calibrated model
        "feature_names": features,
        "metrics": {
            "oos_auc_mean": mean_oos_auc,
            "oos_auc_std": std_oos_auc,
            "oos_logloss_mean": mean_oos_ll,
            "precision_at_20_mean": mean_p20,
            "precision_at_50_mean": mean_p50,
            "lift_at_20": lift_20,
            "lift_at_50": lift_50,
            "baseline_precision": baseline_precision,
            "insample_auc": insample_auc,
            "cv_folds": len(oos_aucs),
            "cv_aucs": oos_aucs,
            "cv_p20": oos_p20,
            "cv_p50": oos_p50,
            "calibration_method": "isotonic",
            "calibration_samples": len(X_calib),
            "mean_calibration_error": mean_calibration_error,
            "calibration_auc": auc_calib,
            "calibration_p20": p20_calib,
            "calibration_p50": p50_calib,
        },
        "trained_at": timestamp
    }
    
    path = MODELS_DIR / "model_20d_v3.pkl"
    joblib.dump(bundle, path)
    # Write metadata.json next to model and to canonical latest path
    meta = {
        "sklearn_version": __import__("sklearn").__version__,
        "feature_list": features,
        "training_timestamp_utc": datetime.utcnow().isoformat(),
        "label_spec": {
            "horizon_days": 20,
            "threshold_type": "percentile",
            "threshold_percentile": WINNER_PERCENTILE,
            "threshold_return": threshold,
            "label_name": f"Forward_Return_20d>=p{WINNER_PERCENTILE}",
            "class_weighting": "balanced",
        },
        "model_type": "CalibratedEnsemble(HistGB+RF+LR)",
        "model_name": "ml_20d_v3_ensemble_calibrated",
        "ensemble": {
            "models": ["HistGradientBoostingClassifier", "RandomForestClassifier", "LogisticRegression"],
            "weights": ensemble_weights,
        },
        "early_stopping": {
            "enabled": True,
            "histgb_stopped_at_iter": model1.n_iter_,
            "max_iter": 300,
            "n_iter_no_change": 15,
        },
        "regularization": {
            "histgb_l2": 0.1,
            "histgb_max_depth": 4,
            "rf_max_depth": 6,
            "lr_C": 0.1,
            "min_samples_leaf": 20,
        },
        "calibration": {
            "method": "isotonic",
            "samples": len(X_calib),
            "mean_calibration_error": mean_calibration_error,
        },
        "evaluation": {
            "precision_at_20": mean_p20,
            "precision_at_50": mean_p50,
            "lift_at_20": lift_20,
            "lift_at_50": lift_50,
            "baseline_precision": baseline_precision,
            "target_p20": 0.40,
            "excellent_p20": 0.60,
        },
    }
    try:
        with open(path.with_suffix(".metadata.json"), "w", encoding="utf-8") as f:
            import json
            json.dump(meta, f)
    except Exception:
        pass

    # Also publish to canonical latest path for inference determinism
    latest_dir = Path("ml/bundles/latest")
    latest_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, latest_dir / "model.joblib")
    try:
        with open(latest_dir / "metadata.json", "w", encoding="utf-8") as f:
            import json
            json.dump(meta, f)
    except Exception:
        pass

    print(f"✅ Model saved to: {path}")
    print(f"🗂️  Latest bundle written to: {latest_dir}")
    print(f"🏆 Out-of-Sample AUC: {bundle['metrics']['oos_auc_mean']:.4f} ± {bundle['metrics']['oos_auc_std']:.4f}")
    
    return path, bundle

if __name__ == "__main__":
    train_and_save_bundle()