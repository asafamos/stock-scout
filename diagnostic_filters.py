"""
Diagnostic script to analyze filter rejection behavior across historical dates.
Shows exactly where stocks fail and helps calibrate thresholds empirically.
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from stock_scout import build_universe, fetch_history_bulk, CONFIG
from advanced_filters import (
    compute_advanced_score,
    should_reject_ticker,
    fetch_benchmark_data,
    compute_relative_strength,
    detect_volume_surge,
    compute_momentum_quality,
    calculate_risk_reward_ratio,
)

def analyze_rejection_stats(test_dates: list[str], sample_size: int = 50):
    """
    Analyze rejection patterns across multiple historical dates.
    Returns distribution statistics for each signal.
    """
    all_signals = []
    
    for test_date in test_dates:
        print(f"\n📅 Testing {test_date}...")
        
        # Build universe as of this date
        universe = build_universe(CONFIG)[:sample_size]
        
        # Fetch historical data up to test date
        lookback = CONFIG['LOOKBACK_DAYS']
        end_date = datetime.strptime(test_date, '%Y-%m-%d')
        
        data = fetch_history_bulk(universe, lookback)
        if not data:
            continue
            
        # Fetch benchmark
        benchmark_df = fetch_benchmark_data(CONFIG["BETA_BENCHMARK"], lookback)
        if benchmark_df.empty:
            continue
        
        # Collect signals for each stock
        from stock_scout import data_map
        for ticker, hist_data in data_map.items():
            if hist_data.empty or len(hist_data) < 50:
                continue
                
            # Compute all signals
            rs_scores = compute_relative_strength(hist_data, benchmark_df)
            vol_data = detect_volume_surge(hist_data)
            mom_data = compute_momentum_quality(hist_data)
            rr_data = calculate_risk_reward_ratio(hist_data)
            
            signal_row = {
                'date': test_date,
                'ticker': ticker,
                'rs_63d': rs_scores.get('rs_63d', np.nan),
                'momentum_consistency': mom_data.get('momentum_consistency', 0.0),
                'risk_reward_ratio': rr_data.get('risk_reward_ratio', np.nan),
                'volume_surge': vol_data.get('volume_surge', 0.0),
            }
            
            # Check if would be rejected
            signals_dict = {**rs_scores, **mom_data, **rr_data, **vol_data}
            should_reject, reason = should_reject_ticker(signals_dict)
            signal_row['rejected'] = should_reject
            signal_row['rejection_reason'] = reason
            
            all_signals.append(signal_row)
    
    df = pd.DataFrame(all_signals)
    
    if df.empty:
        print("❌ No data collected")
        return None
    
    print(f"\n📊 Collected {len(df)} stock samples across {len(test_dates)} dates")
    print(f"   Rejection rate: {df['rejected'].mean()*100:.1f}%")
    
    # Show distribution of each signal
    print("\n📈 Signal Distributions (percentiles):")
    for col in ['rs_63d', 'momentum_consistency', 'risk_reward_ratio']:
        valid = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) == 0:
            continue
        print(f"\n{col}:")
        print(f"  5th:  {valid.quantile(0.05):.3f}")
        print(f"  10th: {valid.quantile(0.10):.3f}")
        print(f"  25th: {valid.quantile(0.25):.3f}")
        print(f"  50th: {valid.quantile(0.50):.3f}")
        print(f"  75th: {valid.quantile(0.75):.3f}")
        print(f"  90th: {valid.quantile(0.90):.3f}")
    
    # Rejection reason breakdown
    print("\n🚫 Rejection Reasons:")
    rej_df = df[df['rejected']]
    if len(rej_df) > 0:
        reason_counts = rej_df['rejection_reason'].value_counts()
        for reason, count in reason_counts.items():
            pct = count / len(df) * 100
            print(f"  {reason}: {count} ({pct:.1f}% of all stocks)")
    
    # Save results
    output_path = f"filter_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved detailed results to: {output_path}")
    
    return df


if __name__ == "__main__":
    # Test across multiple historical dates spanning market conditions
    test_dates = [
        "2024-01-15",  # Q1 2024
        "2024-04-15",  # Q2 2024
        "2024-07-15",  # Q3 2024
        "2024-10-15",  # Q4 2024
        "2025-01-15",  # Q1 2025
    ]
    
    print("🔬 FILTER DIAGNOSTICS — Analyzing rejection patterns")
    print("=" * 70)
    
    results = analyze_rejection_stats(test_dates, sample_size=30)
    
    if results is not None:
        print("\n" + "=" * 70)
        print("💡 RECOMMENDATION:")
        print("   Based on these distributions, consider setting thresholds at:")
        
        for col, desc in [
            ('rs_63d', 'Relative Strength (reject if below)'),
            ('momentum_consistency', 'Momentum Consistency (reject if below)'),
            ('risk_reward_ratio', 'Risk/Reward (reject if below)'),
        ]:
            valid = results[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) > 0:
                # Suggest 10th percentile (rejects bottom 10%)
                threshold = valid.quantile(0.10)
                print(f"   {desc}: {threshold:.3f}")
