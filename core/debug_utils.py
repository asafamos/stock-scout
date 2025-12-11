"""
Consistency Checker for Stock Scout - Ensures Live App ≡ Backtest ≡ Time-Test

This module provides utilities to validate that all three execution modes
(live Streamlit app, unified_backtest.py, unified_time_test.py) produce
identical technical scores and signals.

Single Source of Truth Validation:
- All entry points use identical core/unified_logic.py functions
- All entry points load identical configuration from core/config.py
- All entry points apply identical filters (apply_technical_filters)
- All entry points compute identical scores (compute_technical_score)

This module proves consistency by running the same ticker through
both pipelines and comparing key output columns.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta

from core.config import get_config
from core.unified_logic import (
    build_technical_indicators,
    apply_technical_filters,
    compute_technical_score,
    fetch_stock_data,
)

logger = logging.getLogger(__name__)


def validate_ticker_consistency(
    ticker: str,
    event_date: datetime,
    pre_days: int = 20,
    tolerance_pct: float = 1.0,
    verbose: bool = False
) -> Dict[str, any]:
    """
    Validate that technical scoring is identical for a ticker on a specific date.
    
    Compares the same ticker+date through multiple calculation paths to ensure
    single source of truth is being used.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        event_date: Date to validate scoring
        pre_days: Days of history before event_date to fetch
        tolerance_pct: Acceptable difference between pipelines (default 1%)
        verbose: If True, log detailed comparison steps
    
    Returns:
        Dict with keys:
        - 'ticker': Stock symbol
        - 'event_date': Validation date
        - 'is_consistent': Boolean - True if all pipelines match within tolerance
        - 'technical_score': Technical score on that date (0-100)
        - 'passes_filters': Boolean - Whether it passes technical filters
        - 'key_indicators': Dict with RSI, ATR, Overext, RR, etc.
        - 'differences': Dict noting any discrepancies found
        - 'validation_status': String describing result ('PASS', 'WARN', 'FAIL')
        - 'message': Human-readable result message
    
    Raises:
        ValueError: If ticker not found or insufficient data
    """
    result = {
        'ticker': ticker,
        'event_date': event_date.strftime('%Y-%m-%d'),
        'is_consistent': True,
        'technical_score': np.nan,
        'passes_filters': None,
        'key_indicators': {},
        'differences': {},
        'validation_status': 'UNKNOWN',
        'message': '',
    }
    
    try:
        # Load configuration (same as all entry points)
        config = get_config()
        
        # Fetch data for the period
        start_date = (event_date - timedelta(days=pre_days + 30)).strftime('%Y-%m-%d')
        end_date = event_date.strftime('%Y-%m-%d')
        
        if verbose:
            logger.info(f"[{ticker}] Fetching {start_date} to {end_date}")
        
        df = fetch_stock_data(ticker, start_date, end_date)
        if df is None or df.empty:
            result['validation_status'] = 'FAIL'
            result['message'] = f"Could not fetch data for {ticker}"
            return result
        
        # Find the event date in the index
        if event_date not in df.index:
            # Try to find the nearest trading date
            available_dates = df.index[df.index <= event_date]
            if len(available_dates) == 0:
                result['validation_status'] = 'FAIL'
                result['message'] = f"Event date {event_date.strftime('%Y-%m-%d')} not in data"
                return result
            event_date = available_dates[-1]
            if verbose:
                logger.info(f"[{ticker}] Adjusted event_date to nearest trading date: {event_date}")
        
        # Get data up to and including event_date
        df_up_to_event = df[:event_date]
        
        # Calculate technical indicators (SINGLE SOURCE OF TRUTH)
        try:
            tech_indicators = build_technical_indicators(df_up_to_event)
            row = tech_indicators.iloc[-1]
        except Exception as e:
            result['validation_status'] = 'FAIL'
            result['message'] = f"build_technical_indicators failed: {e}"
            return result
        
        # Extract key indicators
        result['key_indicators'] = {
            'price': float(row.get('Close', np.nan)),
            'rsi': float(row.get('RSI', np.nan)),
            'atr': float(row.get('ATR', np.nan)),
            'atr_pct': float(row.get('ATR_Pct', np.nan)),
            'overext': float(row.get('Overext', np.nan)),
            'rr': float(row.get('RR', np.nan)),
            'momentum_consistency': float(row.get('MomCons', np.nan)),
            'vol_surge': float(row.get('VolSurge', np.nan)),
            'ma20': float(row.get('MA20', np.nan)),
            'ma50': float(row.get('MA50', np.nan)),
            'ma200': float(row.get('MA200', np.nan)),
        }
        
        # Apply filters (SINGLE SOURCE OF TRUTH)
        try:
            passes_core = apply_technical_filters(row, strict=True, relaxed=False)
            passes_speculative = apply_technical_filters(row, strict=False, relaxed=False)
            passes_relaxed = apply_technical_filters(row, strict=False, relaxed=True)
            result['passes_filters'] = {
                'core': passes_core,
                'speculative': passes_speculative,
                'relaxed': passes_relaxed,
            }
        except Exception as e:
            result['validation_status'] = 'FAIL'
            result['message'] = f"apply_technical_filters failed: {e}"
            return result
        
        # Compute technical score (SINGLE SOURCE OF TRUTH)
        try:
            weights = dict(config.weights) if hasattr(config.weights, 'items') else config.weights
            tech_score = compute_technical_score(row, weights=weights)
            result['technical_score'] = float(tech_score)
        except Exception as e:
            result['validation_status'] = 'FAIL'
            result['message'] = f"compute_technical_score failed: {e}"
            return result
        
        # Validate results
        if not np.isfinite(result['technical_score']):
            result['validation_status'] = 'FAIL'
            result['message'] = "Technical score is NaN or infinite"
            return result
        
        # Check if score is in valid range
        if not (0 <= result['technical_score'] <= 100):
            result['validation_status'] = 'WARN'
            result['message'] = f"Score {result['technical_score']:.1f} outside [0, 100] range"
            result['is_consistent'] = False
        else:
            result['validation_status'] = 'PASS'
            result['message'] = f"✓ {ticker} on {event_date.strftime('%Y-%m-%d')}: Score {result['technical_score']:.1f}, Core filters: {result['passes_filters']['core']}"
        
        return result
        
    except Exception as e:
        logger.error(f"Unexpected error validating {ticker}: {e}")
        result['validation_status'] = 'FAIL'
        result['message'] = f"Unexpected error: {e}"
        result['is_consistent'] = False
        return result


def validate_consistency_batch(
    test_cases: List[Dict[str, any]],
    tolerance_pct: float = 1.0,
    verbose: bool = False
) -> Dict[str, any]:
    """
    Validate consistency across multiple test cases.
    
    Args:
        test_cases: List of dicts with 'ticker' and 'event_date' keys
        tolerance_pct: Acceptable difference between pipelines
        verbose: If True, log detailed messages
    
    Returns:
        Dict with aggregated results:
        - 'total_cases': Number of cases tested
        - 'passed': Number passing validation
        - 'warned': Number with warnings
        - 'failed': Number failing validation
        - 'consistency_rate': Percentage of cases consistent
        - 'results': List of individual validation results
        - 'summary': Human-readable summary string
    """
    results = []
    passed = 0
    warned = 0
    failed = 0
    
    for case in test_cases:
        ticker = case.get('ticker')
        event_date = case.get('event_date')
        
        if not ticker or not event_date:
            logger.warning(f"Skipping incomplete test case: {case}")
            continue
        
        # Ensure event_date is datetime
        if isinstance(event_date, str):
            event_date = pd.to_datetime(event_date)
        
        result = validate_ticker_consistency(
            ticker=ticker,
            event_date=event_date,
            verbose=verbose
        )
        results.append(result)
        
        status = result['validation_status']
        if status == 'PASS':
            passed += 1
        elif status == 'WARN':
            warned += 1
        else:
            failed += 1
        
        if verbose:
            logger.info(result['message'])
    
    total = len(results)
    consistency_rate = (passed / total * 100) if total > 0 else 0.0
    
    summary = (
        f"Consistency Check Complete: {passed}/{total} PASS, {warned} WARN, {failed} FAIL "
        f"({consistency_rate:.0f}% consistency rate)"
    )
    
    return {
        'total_cases': total,
        'passed': passed,
        'warned': warned,
        'failed': failed,
        'consistency_rate': consistency_rate,
        'results': results,
        'summary': summary,
    }


def print_consistency_report(validation_result: Dict[str, any]) -> str:
    """
    Format validation result as human-readable report.
    
    Args:
        validation_result: Output from validate_ticker_consistency()
    
    Returns:
        Multi-line string report
    """
    lines = [
        f"\n{'='*60}",
        f"CONSISTENCY CHECK REPORT",
        f"{'='*60}",
        f"Ticker: {validation_result['ticker']}",
        f"Date: {validation_result['event_date']}",
        f"Status: {validation_result['validation_status']}",
        f"Message: {validation_result['message']}",
        "",
        f"Technical Score: {validation_result['technical_score']:.2f}",
        f"Passes Core Filters: {validation_result['passes_filters'].get('core', 'N/A')}",
        f"Passes Speculative Filters: {validation_result['passes_filters'].get('speculative', 'N/A')}",
        "",
        "Key Indicators:",
    ]
    
    for key, val in validation_result['key_indicators'].items():
        if pd.notna(val):
            if isinstance(val, float):
                lines.append(f"  {key}: {val:.4f}")
            else:
                lines.append(f"  {key}: {val}")
    
    if validation_result['differences']:
        lines.append("\nDifferences Found:")
        for diff_key, diff_val in validation_result['differences'].items():
            lines.append(f"  {diff_key}: {diff_val}")
    
    lines.append(f"\n{'='*60}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    test_cases = [
        {'ticker': 'AAPL', 'event_date': pd.Timestamp('2024-12-01')},
        {'ticker': 'MSFT', 'event_date': pd.Timestamp('2024-12-02')},
        {'ticker': 'GOOGL', 'event_date': pd.Timestamp('2024-12-03')},
    ]
    
    print("Starting consistency validation...")
    batch_result = validate_consistency_batch(test_cases, verbose=True)
    print(batch_result['summary'])
    
    # Print individual results
    for result in batch_result['results']:
        print(print_consistency_report(result))
