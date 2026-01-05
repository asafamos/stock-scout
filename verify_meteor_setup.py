#!/usr/bin/env python3
"""
Quick verification for Meteor setup:
- Confirms RSI mapping changes under METEOR_MODE=1 vs 0
- Computes core Meteor signals (VCP ratio, pocket pivot proxy, RS) on a sample ticker

Usage:
  python verify_meteor_setup.py [ticker]
Default ticker: AAPL
"""
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf

# Ensure project root on path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from core.unified_logic import build_technical_indicators, compute_technical_score, compute_atr


def fetch_ohlcv(ticker: str, lookback_days: int = 300) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    return df if df is not None else pd.DataFrame()


def compute_vcp_ratio(df: pd.DataFrame) -> float:
    """Compute ATR(10)/ATR(30) as VCP tightness proxy."""
    if df is None or df.empty or len(df) < 40:
        return np.nan
    atr10 = float(compute_atr(df, period=10).iloc[-1])
    atr30 = float(compute_atr(df, period=30).iloc[-1])
    if not np.isfinite(atr10) or not np.isfinite(atr30) or atr30 <= 0:
        return np.nan
    return float(atr10 / atr30)


def pocket_pivot_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
    """Up-day vs down-day mean volume ratio (proxy)."""
    if df is None or df.empty or len(df) < lookback + 5:
        return np.nan
    price_chg = df['Close'].diff()
    up_mask = price_chg > 0
    down_mask = price_chg < 0
    up_vols = df['Volume'].where(up_mask).dropna().tail(lookback)
    down_vols = df['Volume'].where(down_mask).dropna().tail(lookback)
    up_vol = float(up_vols.mean()) if len(up_vols) > 0 else np.nan
    down_vol = float(down_vols.mean()) if len(down_vols) > 0 else np.nan
    if np.isfinite(up_vol) and np.isfinite(down_vol) and down_vol > 0:
        return float(up_vol / down_vol)
    return np.nan


def dist_from_52w_high(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return np.nan
    lookback = min(len(df), 252)
    hi = float(df['High'].tail(lookback).max())
    close = float(df['Close'].iloc[-1])
    if hi <= 0:
        return np.nan
    return float((close / hi) - 1.0)  # e.g., -0.06 = 6% below 52w high


def rs_diff(df: pd.DataFrame, bench: pd.DataFrame, period: int = 63) -> float:
    if df is None or bench is None or len(df) < period or len(bench) < period:
        return np.nan
    t_now = float(df['Close'].iloc[-1]); t_prev = float(df['Close'].iloc[-period])
    b_now = float(bench['Close'].iloc[-1]); b_prev = float(bench['Close'].iloc[-period])
    if t_prev == 0 or b_prev == 0:
        return np.nan
    t_ret = (t_now / t_prev) - 1.0
    b_ret = (b_now / b_prev) - 1.0
    return float(t_ret - b_ret)


def verify_rsi_mapping(df: pd.DataFrame) -> dict:
    """Compare technical score with METEOR_MODE=0 vs 1 to show RSI mapping impact."""
    # Build indicators once
    tech = build_technical_indicators(df)
    row = tech.iloc[-1]

    # Meteor off
    os.environ['METEOR_MODE'] = '0'
    score_off = float(compute_technical_score(row))

    # Meteor on
    os.environ['METEOR_MODE'] = '1'
    score_on = float(compute_technical_score(row))

    return {
        'RSI': float(row.get('RSI', np.nan)) if pd.notna(row.get('RSI')) else np.nan,
        'TechScore_MeteorOff': score_off,
        'TechScore_MeteorOn': score_on,
        'Delta': score_on - score_off,
    }


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    print(f"\nVerifying Meteor setup for {ticker}\n" + "="*60)

    # Fetch data
    df = fetch_ohlcv(ticker)
    if df.empty or len(df) < 60:
        print("Insufficient data fetched.")
        return

    # RSI mapping impact
    rsi_check = verify_rsi_mapping(df)
    print("RSI Mapping Check:")
    print(f"  RSI: {rsi_check['RSI']:.2f}")
    print(f"  TechScore (Meteor OFF): {rsi_check['TechScore_MeteorOff']:.2f}")
    print(f"  TechScore (Meteor ON):  {rsi_check['TechScore_MeteorOn']:.2f}")
    print(f"  Delta:                  {rsi_check['Delta']:.2f}")

    # Core Meteor signals
    spy = fetch_ohlcv('SPY')
    vcp = compute_vcp_ratio(df)
    pocket = pocket_pivot_ratio(df)
    dist = dist_from_52w_high(df)
    rs63 = rs_diff(df, spy, period=63)

    print("\nMeteor Signals:")
    print(f"  VCP_Ratio (ATR10/ATR30): {vcp if np.isfinite(vcp) else np.nan:.3f}")
    print(f"  Pocket_Pivot_Ratio:      {pocket if np.isfinite(pocket) else np.nan:.3f}")
    print(f"  Dist_From_52w_High:      {dist if np.isfinite(dist) else np.nan:.3f}")
    print(f"  RS_63d (vs SPY):         {rs63 if np.isfinite(rs63) else np.nan:.3f}")

    # Simple pass heuristic mirroring should_pass_meteor
    passed = (
        np.isfinite(vcp) and vcp < 1.0 and vcp <= 0.75 and
        np.isfinite(pocket) and pocket > 1.3 and
        np.isfinite(dist) and (-0.10 <= dist <= -0.05) and
        (not np.isfinite(rs63) or rs63 > 0.0)
    )
    reason = "meteor_pass" if passed else "criteria_not_met"
    print(f"\nMeteor Pass Heuristic: {passed} ({reason})")


if __name__ == '__main__':
    main()
