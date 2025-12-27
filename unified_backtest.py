"""
Unified Backtest Script for Stock Scout

Uses unified_logic module to ensure exact consistency with live app.
Generates signals across date range and computes forward returns.
"""

from __future__ import annotations
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pickle
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
import yfinance as yf

from core.config import get_config
from core.classifier import apply_classification
from core.scoring.fundamental import compute_fundamental_score_with_breakdown
from core.unified_logic import (
    build_technical_indicators,
    apply_technical_filters,
    score_with_ml_model,
    compute_forward_returns,
    fetch_stock_data,
    compute_technical_score,
    compute_final_score,
)
from core.market_context import (
    fetch_spy_context,
    compute_relative_strength_vs_spy,
    get_market_cap_decile,
    compute_price_distance_from_52w_high,
)


def build_universe(limit: int = 100) -> List[str]:
    """Build stock universe from S&P 500."""
    try:
        sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500_df = sp500_table[0]
        tickers = sp500_df['Symbol'].str.replace('.', '-').tolist()
        return tickers[:limit]
    except Exception as e:
        print(f"⚠ Could not fetch S&P 500 list: {e}")
        # Fallback to common tickers
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'WMT'][:limit]


def run_backtest(
    tickers: List[str],
    start_date: datetime,
    end_date: datetime,
    horizons: List[int],
    model_data: Optional[Dict],
    config: Dict
) -> pd.DataFrame:
    """
    Run backtest across tickers and date range.
    
    Returns DataFrame with signals and forward returns.
    """
    # Download benchmark (SPY) - needed for context
    print("Downloading benchmark (SPY)...")
    lookback_start = (start_date - timedelta(days=400)).strftime('%Y-%m-%d')
    end_str = (end_date + timedelta(days=30)).strftime('%Y-%m-%d')
    spy_df = fetch_stock_data('SPY', lookback_start, end_str)
    
    if spy_df is None:
        print("⚠ Could not download SPY benchmark")
        spy_df = pd.DataFrame()
    
    # Fetch market context once (reuse for all stocks)
    print("Computing market context features...")
    spy_context_cache = {}  # Cache by date
    
    all_signals = []
    total_tickers = len(tickers)
    
    for idx, ticker in enumerate(tickers, 1):
        print(f"[{idx}/{total_tickers}] Processing {ticker}...", end=' ')
        
        # Download stock data with lookback for indicators
        df = fetch_stock_data(ticker, lookback_start, end_str)
        if df is None or len(df) < 250:
            print("SKIP (insufficient data)")
            continue
        
        # Build technical indicators
        ind_df = build_technical_indicators(df)
        
        # Filter to backtest date range
        mask = (ind_df.index >= start_date) & (ind_df.index <= end_date)
        test_df = ind_df[mask].copy()
        
        if len(test_df) == 0:
            print("SKIP (no data in range)")
            continue
        
        signals_found = 0
        
        # Scan each trading day
        for date_idx, (date, row) in enumerate(test_df.iterrows()):
            # Check if all indicators are valid
            if pd.isna(row['RSI']) or pd.isna(row['ATR_Pct']) or pd.isna(row['MA50']):
                continue
            
            # Apply Core & Spec filters (relaxed flag not used in backtest scanning)
            passes_core = apply_technical_filters(row, strict=True, relaxed=False)
            passes_spec = apply_technical_filters(row, strict=False, relaxed=False)
            
            if not (passes_core or passes_spec):
                continue
            
            # Compute ML probability (calibrated model preferred by loader)
            ml_prob = score_with_ml_model(row, model_data)

            # Technical score (0-100)
            tech_score = compute_technical_score({
                'MA_Aligned': float(row.get('Close', np.nan) > row.get('MA50', np.nan)) if pd.notna(row.get('MA50', np.nan)) else 0.0,
                'Momentum_Consistency': float(row.get('MomCons', 0.0)) if pd.notna(row.get('MomCons', np.nan)) else 0.0,
                'RSI': row.get('RSI', np.nan),
                'VolSurge': row.get('VolSurge', np.nan),
                'Overext': row.get('Overext', np.nan),
                'Near52w': row.get('Near52w', np.nan),
                'RR': row.get('RR', np.nan),
                'MACD_Pos': False,  # MACD not implemented in unified builder yet
                'ADX14': np.nan,
            })

            # Fundamental score placeholder (0 if unavailable to keep formula stable)
            fundamental_score = None
            final_score = compute_final_score(tech_score, fundamental_score, ml_prob)
            
            # Compute forward returns
            forward_rets = compute_forward_returns(
                df=ind_df,
                date=date,
                horizons=horizons,
                benchmark_df=spy_df
            )
            
            # Check if we have valid forward returns
            if all(pd.isna(forward_rets.get(f'R_{h}d')) for h in horizons):
                continue
            
            # Compute market context features for this date
            date_str = date.strftime('%Y-%m-%d')
            if date_str not in spy_context_cache:
                # Get SPY data up to this date
                spy_up_to_date = spy_df[spy_df.index <= date]
                if len(spy_up_to_date) >= 50:
                    sma20 = float(spy_up_to_date['Close'].tail(20).mean())
                    sma50 = float(spy_up_to_date['Close'].tail(50).mean())
                    current_spy = float(spy_up_to_date['Close'].iloc[-1])
                    spy_returns = spy_up_to_date['Close'].pct_change(fill_method=None)
                    spy_vol = float(spy_returns.tail(20).std() * np.sqrt(252))
                    
                    # Simple SPY RSI
                    spy_rsi_val = 50.0
                    if len(spy_up_to_date) >= 14:
                        delta = spy_up_to_date['Close'].diff()
                        gain_val = float(delta.where(delta > 0, 0.0).tail(14).mean())
                        loss_val = float(-delta.where(delta < 0, 0.0).tail(14).mean())
                        if not np.isnan(loss_val) and loss_val > 0:
                            spy_rsi_val = 100 - (100 / (1 + gain_val/loss_val))
                    
                    spy_context_cache[date_str] = {
                        'market_trend': 1.0 if current_spy > sma20 > sma50 else 0.0,
                        'market_volatility': min(spy_vol, 0.5),
                        'spy_rsi': spy_rsi_val,
                    }
                else:
                    spy_context_cache[date_str] = {
                        'market_trend': 0.5,
                        'market_volatility': 0.2,
                        'spy_rsi': 50.0,
                    }
            
            context = spy_context_cache[date_str]
            
            # Relative strength vs SPY (20-day)
            ticker_df_up_to_date = ind_df[ind_df.index <= date]
            spy_up_to_date = spy_df[spy_df.index <= date]
            if len(ticker_df_up_to_date) >= 21 and len(spy_up_to_date) >= 21:
                ticker_ret = float((ticker_df_up_to_date['Close'].iloc[-1] / ticker_df_up_to_date['Close'].iloc[-21]) - 1)
                spy_ret = float((spy_up_to_date['Close'].iloc[-1] / spy_up_to_date['Close'].iloc[-21]) - 1)
                rel_strength = ticker_ret - spy_ret
            else:
                rel_strength = 0.0
            
            # Distance from 52w high
            if len(ticker_df_up_to_date) >= 252:
                high_52w = ticker_df_up_to_date['High'].tail(252).max()
            else:
                high_52w = ticker_df_up_to_date['High'].max()
            dist_52w = (row['Close'] / high_52w) - 1
            
            # Record signal with context features
            signal = {
                'Ticker': ticker,
                'Date': date.strftime('%Y-%m-%d'),
                'Close': row['Close'],
                'RSI': row['RSI'],
                'ATR_Pct': row['ATR_Pct'],
                'Overext': row['Overext'],
                'RR': row['RR'],
                'MomCons': row['MomCons'],
                'VolSurge': row['VolSurge'],
                'ML_Prob': ml_prob,
                'PassesCore': passes_core,
                'PassesSpec': passes_spec,
                'Tech_Score': tech_score,
                'Final_Score': final_score,
                # Context features
                'Market_Trend': context['market_trend'],
                'Market_Volatility': context['market_volatility'],
                'SPY_RSI': context['spy_rsi'],
                'Relative_Strength_20d': rel_strength,
                'Dist_From_52w_High': dist_52w,
                'Vol_Breakout': float(row.get('Vol_Breakout', False)),
                'Price_Breakout': float(row.get('Price_Breakout', False)),
                'Mom_Acceleration': float(row.get('Mom_Acceleration', False)),
            }
            
            # Add forward returns
            signal.update(forward_rets)
            
            all_signals.append(signal)
            signals_found += 1
        
        print(f"✓ {signals_found} signals")
    
    # Convert to DataFrame
    signals_df = pd.DataFrame(all_signals)
    if not signals_df.empty and 'Final_Score' in signals_df.columns:
        signals_df = signals_df.sort_values('Final_Score', ascending=False).reset_index(drop=True)
    return signals_df


def print_summary(signals_df: pd.DataFrame, horizons: List[int]):
    """Print backtest summary statistics."""
    print(f"\n{'='*70}")
    print(f"BACKTEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total Signals: {len(signals_df)}")
    print(f"Unique Tickers: {signals_df['Ticker'].nunique()}")
    print(f"Date Range: {signals_df['Date'].min()} to {signals_df['Date'].max()}")
    print()
    
    # Core vs Speculative split
    n_core = signals_df['PassesCore'].sum()
    n_spec = (~signals_df['PassesCore'] & signals_df['PassesSpec']).sum()
    print(f"Core Signals: {n_core} ({n_core/len(signals_df)*100:.1f}%)")
    print(f"Speculative Signals: {n_spec} ({n_spec/len(signals_df)*100:.1f}%)")
    print()
    
    # ML distribution
    print(f"ML Probability: mean={signals_df['ML_Prob'].mean():.3f}, "
          f"median={signals_df['ML_Prob'].median():.3f}")
    print()
    
    # Forward returns analysis
    for h in horizons:
        ret_col = f'R_{h}d'
        excess_col = f'Excess_{h}d'
        
        if ret_col not in signals_df.columns:
            continue
        
        rets = signals_df[ret_col].dropna()
        excess = signals_df[excess_col].dropna() if excess_col in signals_df.columns else pd.Series(dtype=float)
        
        if len(rets) == 0:
            continue
        
        hit_rate = (rets > 0).mean() * 100
        outperform_rate = (excess > 0).mean() * 100 if len(excess) > 0 else 0
        
        print(f"{h}-Day Forward Returns:")
        print(f"  Hit Rate (>0%): {hit_rate:.1f}%")
        print(f"  Outperform SPY: {outperform_rate:.1f}%")
        print(f"  Mean Return: {rets.mean():.2f}%")
        print(f"  Median Return: {rets.median():.2f}%")
        print(f"  Mean Excess: {excess.mean():.2f}% (vs SPY)" if len(excess) > 0 else "")
        print(f"  Worst Return: {rets.min():.2f}%")
        print(f"  Best Return: {rets.max():.2f}%")
        print()
    
    # ML stratification (5-day horizon)
    if 'R_5d' in signals_df.columns and len(signals_df) > 0:
        print("ML Probability Stratification (5-day returns):")
        signals_df['ML_Tier'] = pd.cut(
            signals_df['ML_Prob'],
            bins=[0, 0.3, 0.5, 1.0],
            labels=['Low(<30%)', 'Medium(30-50%)', 'High(>50%)']
        )
        
        for tier in ['Low(<30%)', 'Medium(30-50%)', 'High(>50%)']:
            tier_df = signals_df[signals_df['ML_Tier'] == tier]
            if len(tier_df) == 0:
                continue
            
            tier_rets = tier_df['R_5d'].dropna()
            if len(tier_rets) > 0:
                tier_hit = (tier_rets > 0).mean() * 100
                print(f"  {tier}: {len(tier_df)} signals, "
                      f"hit rate {tier_hit:.1f}%, "
                      f"mean return {tier_rets.mean():.2f}%")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Unified Stock Scout Backtest")
    parser.add_argument("--limit", type=int, default=100, help="Max stocks in universe")
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--horizons", type=str, default="5,10,20", help="Forward horizons (comma-separated)")
    parser.add_argument("--model", type=str, default="model_xgboost_5d_calibrated.pkl", help="ML model file")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")
    horizons = [int(h) for h in args.horizons.split(",")]
    
    # Load configuration
    config = get_config()
    
    # Load ML model
    model_data = None
    model_path = Path(__file__).parent / args.model
    if model_path.exists():
        try:
            with open(model_path, 'rb') as f:
                loaded = pickle.load(f)
            if isinstance(loaded, dict) and 'model' in loaded:
                model_data = loaded
                print(f"✓ Loaded ML model: {args.model} ({len(loaded.get('feature_names', []))} features)")
            else:
                print(f"⚠ Model format not recognized")
        except Exception as e:
            print(f"⚠ Could not load model: {e}")
    else:
        print(f"⚠ Model file not found: {model_path}")
    
    print(f"\n{'='*70}")
    print(f"STOCK SCOUT UNIFIED BACKTEST")
    print(f"{'='*70}")
    print(f"Date Range: {args.start} to {args.end}")
    print(f"Universe Limit: {args.limit}")
    print(f"Forward Horizons: {horizons}")
    print(f"{'='*70}\n")
    
    # Build universe
    print("Building stock universe...")
    tickers = build_universe(args.limit)
    print(f"✓ Universe: {len(tickers)} stocks\n")
    
    # Run backtest
    signals_df = run_backtest(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        horizons=horizons,
        model_data=model_data,
        config=config
    )
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"backtest_results_{timestamp}.csv"
    
    # Compute unified technical and final scores
    if not signals_df.empty:
        from core.unified_logic import compute_technical_score, compute_final_score
        # Technical score
        signals_df['Tech_Score'] = signals_df.apply(lambda r: compute_technical_score(r), axis=1)
        # Fundamental not available in this historical-only backtest; keep NaN
        signals_df['Fund_Score'] = np.nan
        # Final score
        signals_df['Final_Score'] = signals_df.apply(lambda r: compute_final_score(r['Tech_Score'], r['Fund_Score'], r['ML_Prob']), axis=1)
        signals_df = signals_df.sort_values('Final_Score', ascending=False).reset_index(drop=True)

    signals_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved results to: {output_path}")
    
    # Print summary
    print_summary(signals_df, horizons)
    
    return signals_df


if __name__ == "__main__":
    main()
