"""
Unified Time-Travel Validation Script

Loads manual test cases (ticker + date) and validates:
1. ML probability on that date
2. Whether signal passes all filters
3. Realized forward returns

Uses unified_logic module for consistency with backtest and live app.
"""

from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import pickle
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

from core.config import get_config
from core.unified_logic import (
    build_technical_indicators,
    apply_technical_filters,
    compute_forward_returns,
    fetch_stock_data,
    compute_technical_score,
    compute_final_score,
    build_market_context_table,
)
from core.ml_20d_inference import (
    ML_20D_AVAILABLE,
    compute_ml_20d_probabilities_raw,
    calibrate_ml_20d_prob,
)


def validate_case(
    ticker: str,
    event_date: datetime,
    pre_days: int,
    model_data: Optional[Dict],
    horizons: List[int] = [5, 10, 20]
) -> Dict:
    """
    Validate a single test case.
    
    Args:
        ticker: Stock symbol
        event_date: Event date (e.g., when we know stock moved)
        pre_days: Days before event to evaluate signal
        model_data: ML model dictionary
        horizons: Forward return horizons
        
    Returns:
        Dict with validation results
    """
    eval_date = event_date - timedelta(days=pre_days)
    
    # Download data with lookback
    lookback_start = (eval_date - timedelta(days=400)).strftime('%Y-%m-%d')
    end_date = (event_date + timedelta(days=30)).strftime('%Y-%m-%d')
    
    df = fetch_stock_data(ticker, lookback_start, end_date)
    
    result = {
        'Ticker': ticker,
        'EventDate': event_date.strftime('%Y-%m-%d'),
        'EvalDate': eval_date.strftime('%Y-%m-%d'),
        'PreDays': pre_days,
        'DataAvailable': False,
        'PassesCore': False,
        'PassesSpec': False,
        'ML_Prob': np.nan,
        'RSI': np.nan,
        'ATR_Pct': np.nan,
        'Overext': np.nan,
        'RR': np.nan,
        'MomCons': np.nan,
        'Volatility_Contraction_Score': np.nan,
        'Market_Regime': None,
        'ML_Confidence_Status': 'NEUTRAL',
        'Data_Integrity': 'OK',
        'Primary_Index_Source': None,
    }
    
    if df is None or len(df) < 250:
        result['Error'] = 'Insufficient data'
        return result
    
    # Build indicators
    ind_df = build_technical_indicators(df)
    
    # Find evaluation date
    eval_dates = ind_df.index[ind_df.index <= eval_date]
    if len(eval_dates) == 0:
        result['Error'] = 'Eval date not in data'
        return result
    
    actual_eval_date = eval_dates[-1]
    row = ind_df.loc[actual_eval_date]
    
    result['DataAvailable'] = True
    result['EvalDate'] = actual_eval_date.strftime('%Y-%m-%d')
    
    # Check if indicators are valid
    if pd.isna(row['RSI']) or pd.isna(row['ATR_Pct']):
        result['Error'] = 'Invalid indicators'
        return result
    
    # Extract indicators
    result['RSI'] = row['RSI']
    result['ATR_Pct'] = row['ATR_Pct']
    result['Overext'] = row['Overext']
    result['RR'] = row['RR']
    result['MomCons'] = row['MomCons']
    result['Close'] = row['Close']
    # Volatility Contraction Pattern score from indicators
    if 'Volatility_Contraction_Score' in row.index:
        result['Volatility_Contraction_Score'] = float(row['Volatility_Contraction_Score']) if pd.notna(row['Volatility_Contraction_Score']) else np.nan
    
    # Apply filters
    result['PassesCore'] = apply_technical_filters(row, strict=True)
    result['PassesSpec'] = apply_technical_filters(row, strict=False)
    
    # Compute ML probability (raw + calibration if model available)
    if ML_20D_AVAILABLE:
        try:
            prob_raw = compute_ml_20d_probabilities_raw(row)
            atr_pct_pct = row.get("ATR_Pct_percentile", np.nan)
            price_as_of = row.get("Price_As_Of_Date", np.nan)
            reliability_factor = row.get("ReliabilityFactor", np.nan)
            ml_prob = calibrate_ml_20d_prob(
                prob_raw,
                atr_pct_percentile=float(atr_pct_pct) if pd.notna(atr_pct_pct) else None,
                price_as_of=float(price_as_of) if pd.notna(price_as_of) else None,
                reliability_factor=float(reliability_factor) if pd.notna(reliability_factor) else None,
            )
        except Exception:
            ml_prob = 0.5
    else:
        ml_prob = 0.5
    result['ML_Prob'] = ml_prob

    tech_score = compute_technical_score({
        'MA_Aligned': float(row.get('Close', np.nan) > row.get('MA50', np.nan)) if pd.notna(row.get('MA50', np.nan)) else 0.0,
        'Momentum_Consistency': float(row.get('MomCons', 0.0)) if pd.notna(row.get('MomCons', np.nan)) else 0.0,
        'RSI': row.get('RSI', np.nan),
        'VolSurge': row.get('VolSurge', np.nan),
        'Overext': row.get('Overext', np.nan),
        'Near52w': row.get('Near52w', np.nan),
        'RR': row.get('RR', np.nan),
        'MACD_Pos': False,
        'ADX14': np.nan,
    })
    # Derive market regime via context and pass into final score
    market_regime = result['Market_Regime'] if 'Market_Regime' in result else None
    final_score = compute_final_score(tech_score, None, ml_prob, market_regime=market_regime)
    result['Tech_Score'] = tech_score
    result['Final_Score'] = final_score

    # ML Confidence Status (gatekeeper)
    try:
        ml_val = float(ml_prob) if ml_prob is not None else np.nan
    except Exception:
        ml_val = np.nan
    if pd.isna(ml_val):
        result['ML_Confidence_Status'] = 'NEUTRAL'
    elif ml_val < 0.15:
        result['ML_Confidence_Status'] = 'PENALIZED'
    elif ml_val > 0.62:
        result['ML_Confidence_Status'] = 'BOOSTED'
    else:
        result['ML_Confidence_Status'] = 'NEUTRAL'

    # Market regime using breadth-aware classification
    try:
        start_ctx = (actual_eval_date - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
        end_ctx = (actual_eval_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        ctx = build_market_context_table(start_ctx, end_ctx)
        if not ctx.empty:
            # Find the regime for actual_eval_date or last available
            if actual_eval_date in pd.to_datetime(ctx['date']).values:
                idx = ctx.index[pd.to_datetime(ctx['date']) == actual_eval_date]
                if len(idx) > 0:
                    result['Market_Regime'] = str(ctx.loc[idx[0], 'Market_Regime'])
                else:
                    result['Market_Regime'] = str(ctx['Market_Regime'].iloc[-1])
            else:
                result['Market_Regime'] = str(ctx['Market_Regime'].iloc[-1])
        # Primary source hint comes from last index fetch
        from core.data_sources_v2 import get_last_index_source
        result['Primary_Index_Source'] = get_last_index_source('SPY')
    except Exception:
        result['Data_Integrity'] = 'DATA_INCOMPLETE'
    
    # Compute forward returns
    forward_rets = compute_forward_returns(
        df=ind_df,
        date=actual_eval_date,
        horizons=horizons
    )
    result.update(forward_rets)
    
    return result


def load_test_cases(csv_path: str) -> pd.DataFrame:
    """
    Load test cases from CSV.
    
    Expected columns: Ticker, EventDate, PreDays (optional, default 3)
    """
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    if 'Ticker' not in df.columns or 'EventDate' not in df.columns:
        raise ValueError("CSV must have 'Ticker' and 'EventDate' columns")
    
    # Add PreDays if missing
    if 'PreDays' not in df.columns:
        df['PreDays'] = 3
    
    return df


def print_results_table(results_df: pd.DataFrame):
    """Print formatted results table."""
    print(f"\n{'='*80}")
    print(f"TIME-TRAVEL VALIDATION RESULTS")
    print(f"{'='*80}\n")

    print(f"{'Ticker':<8} {'EvalDate':<12} {'Core':<6} {'Spec':<6} {'ML_Prob':<10} "
          f"{'VCP':<8} {'Regime':<13} {'Final':<8} {'ML_Status':<12} {'R_5d':<10} {'Primary':<10} {'Integrity':<14} {'Status':<12}")
    print(f"{'-'*80}")
    
    for _, row in results_df.iterrows():
        ticker = row['Ticker']
        eval_date = row.get('EvalDate', 'N/A')[:10]
        passes_core = '✓' if row.get('PassesCore') else '✗'
        passes_spec = '✓' if row.get('PassesSpec') else '✗'
        ml_prob = f"{row.get('ML_Prob', 0):.1%}" if pd.notna(row.get('ML_Prob')) else 'N/A'
        r_5d = f"{row.get('R_5d', 0):.1f}%" if pd.notna(row.get('R_5d')) else 'N/A'
        vcp = row.get('Volatility_Contraction_Score')
        vcp_s = f"{vcp:.2f}" if pd.notna(vcp) else 'N/A'
        regime = str(row.get('Market_Regime', 'UNKNOWN'))
        final_s = f"{row.get('Final_Score', 0):.1f}" if pd.notna(row.get('Final_Score')) else 'N/A'
        ml_status = str(row.get('ML_Confidence_Status', 'NEUTRAL'))
        primary_src = str(row.get('Primary_Index_Source', 'N/A'))
        integrity = str(row.get('Data_Integrity', 'OK'))
        
        # Determine status
        if not row.get('DataAvailable'):
            status = '⚠ No Data'
        elif 'Error' in row:
            status = f"⚠ {row['Error']}"
        elif row.get('PassesCore') and pd.notna(row.get('R_5d')) and row.get('R_5d') > 0:
            status = '✅ Success'
        elif row.get('PassesSpec'):
            status = '🟡 Spec Only'
        else:
            status = '❌ Filtered'

        print(f"{ticker:<8} {eval_date:<12} {passes_core:<6} {passes_spec:<6} {ml_prob:<10} "
              f"{vcp_s:<8} {regime:<13} {final_s:<8} {ml_status:<12} {r_5d:<10} {primary_src:<10} {integrity:<14} {status:<12}")
    
    print(f"{'-'*80}\n")
    
    # Summary statistics
    valid_cases = results_df[results_df['DataAvailable']]
    if len(valid_cases) > 0:
        print("SUMMARY:")
        print(f"  Total Cases: {len(results_df)}")
        print(f"  Valid Data: {len(valid_cases)}")
        print(f"  Passed Core: {valid_cases['PassesCore'].sum()} "
              f"({valid_cases['PassesCore'].mean()*100:.1f}%)")
        print(f"  Passed Spec: {valid_cases['PassesSpec'].sum()} "
              f"({valid_cases['PassesSpec'].mean()*100:.1f}%)")
        print(f"  Mean ML Prob: {valid_cases['ML_Prob'].mean():.1%}")
        
        if 'R_5d' in valid_cases.columns:
            rets_5d = valid_cases['R_5d'].dropna()
            if len(rets_5d) > 0:
                print(f"  5-Day Returns: mean={rets_5d.mean():.2f}%, "
                      f"median={rets_5d.median():.2f}%, "
                      f"hit rate={(rets_5d > 0).mean()*100:.1f}%")
    
    print(f"{'='*80}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Time-Travel Validation")
    parser.add_argument("--cases", type=str, required=True, help="CSV file with test cases")
    parser.add_argument("--model", type=str, default="models/model_20d_v3.pkl", help="ML model file (sklearn bundle)")
    parser.add_argument("--horizons", type=str, default="5,10,20", help="Forward horizons")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    
    args = parser.parse_args()
    
    horizons = [int(h) for h in args.horizons.split(",")]

    # Prefetch global market context to minimize per-case provider calls
    try:
        from core.market_context import initialize_market_context
        initialize_market_context()
        print("✓ Initialized global market context (SPY/VIX/Sectors)")
    except Exception as e:
        print(f"⚠ Global market context initialization failed: {e}")
    
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
    
    # Load test cases
    print(f"\nLoading test cases from: {args.cases}")
    cases_df = load_test_cases(args.cases)
    print(f"✓ Loaded {len(cases_df)} test cases\n")
    
    # Validate each case
    results = []
    for idx, row in cases_df.iterrows():
        ticker = row['Ticker']
        event_date = pd.to_datetime(row['EventDate'])
        pre_days = int(row.get('PreDays', 3))
        
        print(f"[{idx+1}/{len(cases_df)}] Validating {ticker} @ {event_date.strftime('%Y-%m-%d')} "
              f"(eval {pre_days}d before)...", end=' ')
        
        result = validate_case(
            ticker=ticker,
            event_date=event_date,
            pre_days=pre_days,
            model_data=model_data,
            horizons=horizons
        )
        
        results.append(result)
        
        if result.get('DataAvailable'):
            status = '✓' if result.get('PassesCore') else '✗'
            ml_prob = result.get('ML_Prob', 0)
            print(f"{status} ML:{ml_prob:.1%}")
        else:
            print(f"⚠ {result.get('Error', 'Failed')}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"time_test_results_{timestamp}.csv"
    
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved results to: {output_path}")
    
    # Print results table
    print_results_table(results_df)


if __name__ == "__main__":
    main()
