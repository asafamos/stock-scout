"""Backtest recommendation logic using historical technical filters.

Goal:
    Emulate (simplified) daily recommendation generation over a past period
    using only price/volume data (no historical fundamentals) and evaluate
    forward returns vs benchmark (SPY).

Features Implemented:
    - Indicator set: RSI(14), ATR(14), MA20, MA50, MA200, Overextension ratio,
      Momentum consistency (fraction of up days last 14), Volume surge (vol / avg20),
      Reward/Risk ratio (20d high vs 20d low relative to current close).
    - Core style filters approximated from current CONFIG thresholds:
        * RSI between 40 and 75
        * Overextension <= 0.12 (close/MA50 - 1)
        * ATR/Price <= 0.09
        * Reward/Risk >= 1.3
        * Momentum consistency >= 0.45 (soft technical quality floor)
    - Daily selection excludes first 60 trading days (to build lookback).
    - Forward performance horizons: 5, 10, 20 trading days (configurable).
    - Metrics: Hit rate (>0% return), Outperformance rate (> SPY), Mean excess
      return, Median return, Worst return, Average reward/risk at selection.
    - Simple position sizing assumption: equal weight per recommended ticker.

Usage:
    python backtest_recommendations.py \
        --tickers AAPL,MSFT,AMD,NVDA,GOOGL,META,AMZN \
        --start 2024-01-01 --end 2024-12-31 \
        --horizons 5,10,20

    Or load tickers from file:
        python backtest_recommendations.py --tickers-file tickers.txt --start 2024-01-01 --end 2024-12-31

Notes:
    - This is a simplified backtest: it does not reproduce full scoring or
      fundamentals; its purpose is diagnostic (does technical filter set find
      reasonable forward performers historically?).
    - Benchmark forward returns computed from SPY over same horizons.
    - For reliability, download all needed data once per ticker.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ------------------------------- Indicator Helpers -------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df['Close'].shift()
    range1 = df['High'] - df['Low']
    range2 = (df['High'] - prev_close).abs()
    range3 = (df['Low'] - prev_close).abs()
    true_range = np.maximum(range1, np.maximum(range2, range3))
    # Ensure 1D
    true_range = np.asarray(true_range).reshape(-1)
    return pd.Series(true_range, index=df.index).rolling(period).mean()


def momentum_consistency(close: pd.Series, lookback: int = 14) -> pd.Series:
    up = (close.diff() > 0).astype(int)
    return up.rolling(lookback).mean()


def volume_surge(volume: pd.Series, lookback: int = 20) -> pd.Series:
    avg = volume.rolling(lookback).mean()
    return volume / avg


def reward_risk(close: pd.Series, lookback: int = 20) -> pd.Series:
    highest = close.rolling(lookback).max()
    lowest = close.rolling(lookback).min()
    # Potential reward: recent high - current; risk: current - recent low
    reward = (highest - close).clip(lower=0)
    raw_risk = (close - lowest)
    # Guard against micro risk distances creating huge ratios
    risk = raw_risk.where(raw_risk > close * 0.002, close * 0.002)
    rr = (reward / risk).clip(upper=10)  # cap unrealistic extremes
    return rr.replace([np.inf, -np.inf], np.nan)


# ------------------------------- Core Filter Logic -------------------------------

@dataclass
class FilterThresholds:
    rsi_min: float = 40.0
    rsi_max: float = 75.0
    max_overext: float = 0.12  # close/MA50 - 1
    max_atr_price: float = 0.09
    min_rr: float = 1.3
    min_mom_consistency: float = 0.45


def apply_filters(row: pd.Series, t: FilterThresholds) -> bool:
    return (
        t.rsi_min <= row['RSI'] <= t.rsi_max and
        row['Overext'] <= t.max_overext and
        row['ATR_Pct'] <= t.max_atr_price and
        row['RR'] >= t.min_rr and
        row['MomCons'] >= t.min_mom_consistency
    )


# ------------------------------- Backtest Engine -------------------------------

def fetch_history(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    data = {}
    for tk in tickers:
        try:
            df = yf.download(tk, start=start, end=end, auto_adjust=True, progress=False)
            if df.empty:
                continue
            df.dropna(inplace=True)
            data[tk] = df
        except Exception:
            continue
    return data


def build_indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out['Close'] = df['Close']
    out['RSI'] = rsi(df['Close'])
    out['ATR'] = atr(df)
    out['ATR_Pct'] = out['ATR'] / out['Close']
    out['MA20'] = df['Close'].rolling(20).mean()
    out['MA50'] = df['Close'].rolling(50).mean()
    out['MA200'] = df['Close'].rolling(200).mean()
    out['Overext'] = (out['Close'] / out['MA50']) - 1
    out['MomCons'] = momentum_consistency(df['Close'])
    out['VolSurge'] = volume_surge(df['Volume'])
    out['RR'] = reward_risk(df['Close'])
    return out


def generate_daily_recommendations(indicators: Dict[str, pd.DataFrame], thresholds: FilterThresholds) -> Dict[pd.Timestamp, List[Tuple[str, pd.Series]]]:
    # Align common date index
    all_dates = sorted(set().union(*[df.index for df in indicators.values()]))
    recommendations = {}
    for date in all_dates:
        daily = []
        for tk, df in indicators.items():
            if date not in df.index:
                continue
            # Skip early period without enough lookback (need MA50 & MA200)
            if df.index.get_loc(date) < 200:
                continue
            row = df.loc[date]
            if row[['RSI', 'Overext', 'ATR_Pct', 'RR', 'MomCons']].isna().any():
                continue
            if apply_filters(row, thresholds):
                daily.append((tk, row))
        if daily:
            recommendations[date] = daily
    return recommendations


def compute_forward_returns(df: pd.DataFrame, date: pd.Timestamp, horizons: List[int]) -> Dict[int, float]:
    returns = {}
    for h in horizons:
        # Forward date index position
        idx = df.index.get_loc(date)
        forward_idx = idx + h
        if forward_idx >= len(df.index):
            returns[h] = np.nan
            continue
        start_price = df.loc[date, 'Close']
        end_price = df.iloc[forward_idx]['Close']
        returns[h] = (end_price - start_price) / start_price
    return returns


def aggregate_metrics(recs: Dict[pd.Timestamp, List[Tuple[str, pd.Series]]], data: Dict[str, pd.DataFrame], benchmark: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    rows = []
    for date, items in recs.items():
        for tk, row in items:
            fwd = compute_forward_returns(data[tk], date, horizons)
            bench_fwd = compute_forward_returns(benchmark, date, horizons)
            base = {
                'Date': date,
                'Ticker': tk,
                'RSI': float(row['RSI']),
                'ATR_Pct': float(row['ATR_Pct']),
                'Overext': float(row['Overext']),
                'RR': float(row['RR']),
                'MomCons': float(row['MomCons']),
                'VolSurge': float(row.get('VolSurge', np.nan)),
            }
            for h in horizons:
                r = fwd[h]
                b = bench_fwd[h]
                def _scalar(x):
                    if isinstance(x, (pd.Series, pd.DataFrame)):
                        if hasattr(x, 'size') and x.size > 0:
                            try:
                                return float(np.asarray(x).reshape(-1)[0])
                            except Exception:
                                return np.nan
                        return np.nan
                    try:
                        return float(x)
                    except Exception:
                        return np.nan
                r_val = _scalar(r)
                b_val = _scalar(b)
                base[f'R_{h}d'] = r_val if np.isfinite(r_val) else np.nan
                base[f'Excess_{h}d'] = (r_val - b_val) if np.isfinite(r_val) and np.isfinite(b_val) else np.nan
            rows.append(base)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    summary_rows = []
    for h in horizons:
        col = f'R_{h}d'
        excess_col = f'Excess_{h}d'
        series = pd.to_numeric(df[col], errors='coerce')
        valid = series.dropna()
        if valid.empty:
            continue
        summary_rows.append({
            'Horizon': h,
            'Count': int(valid.count()),
            'HitRate_%': valid.gt(0).mean() * 100,
            'Outperform_%': pd.to_numeric(df[excess_col], errors='coerce').gt(0).mean() * 100,
            'MeanReturn_%': valid.mean() * 100,
            'MedianReturn_%': valid.median() * 100,
            'MeanExcess_%': pd.to_numeric(df[excess_col], errors='coerce').mean() * 100,
            'WorstReturn_%': valid.min() * 100,
            'AvgRewardRisk': df['RR'].mean(),
        })
    return pd.DataFrame(summary_rows)


# ------------------------------- CLI Execution -------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest technical recommendation filters")
    p.add_argument('--tickers', type=str, help='Comma separated list of tickers')
    p.add_argument('--tickers-file', type=str, help='File with one ticker per line')
    p.add_argument('--start', type=str, required=True, help='Start date YYYY-MM-DD')
    p.add_argument('--end', type=str, required=True, help='End date YYYY-MM-DD')
    p.add_argument('--horizons', type=str, default='5,10,20', help='Forward day horizons')
    p.add_argument('--benchmark', type=str, default='SPY', help='Benchmark ticker (default SPY)')
    p.add_argument('--use-finnhub', action='store_true', help='Fetch US universe via Finnhub API keys')
    p.add_argument('--limit', type=int, default=150, help='Universe size limit when using Finnhub')
    p.add_argument('--sp500', action='store_true', help='Use current S&P500 constituents from Wikipedia')
    return p.parse_args()


def load_ticker_list(args: argparse.Namespace) -> List[str]:
    tickers: List[str] = []
    if args.tickers:
        tickers.extend([t.strip().upper() for t in args.tickers.split(',') if t.strip()])
    if args.tickers_file:
        with open(args.tickers_file) as f:
            tickers.extend([line.strip().upper() for line in f if line.strip()])
    if args.sp500 and not tickers:
        try:
            import pandas as pd
            tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            sp = tables[0]
            tickers = sp['Symbol'].astype(str).str.replace('.', '-', regex=False).tolist()
            print(f"Fetched {len(tickers)} S&P500 tickers from Wikipedia.")
        except Exception as e:
            print(f"S&P500 fetch failed: {e}")
    if args.use_finnhub and not tickers:
        try:
            from core.pipeline_runner import fetch_top_us_tickers_by_market_cap
            tickers = fetch_top_us_tickers_by_market_cap(limit=args.limit)
            print(f"Fetched {len(tickers)} tickers from universe.")
        except Exception as e:
            print(f"Universe fetch failed: {e}. Falling back to manual list if provided.")
    # Deduplicate
    tickers = sorted(set(tickers))
    if not tickers:
        raise SystemExit("No tickers provided. Use --tickers or --tickers-file.")
    return tickers


def main():
    args = parse_args()
    tickers = load_ticker_list(args)
    horizons = [int(h) for h in args.horizons.split(',') if h.strip()]
    print(f"🔄 Downloading data for {len(tickers)} tickers …")
    data = fetch_history(tickers + [args.benchmark], args.start, args.end)
    if args.benchmark not in data:
        raise SystemExit("Benchmark data missing.")
    benchmark_df = data.pop(args.benchmark)
    print("✅ Data download complete.")

    print("📐 Computing indicators …")
    indicators = {tk: build_indicator_frame(df) for tk, df in data.items()}
    print("✅ Indicators ready.")

    thresholds = FilterThresholds()
    print("🧪 Generating daily recommendations …")
    recs = generate_daily_recommendations(indicators, thresholds)
    total_days = len(recs)
    total_signals = sum(len(v) for v in recs.values())
    print(f"✅ {total_signals} signals on {total_days} days.")
    if not recs:
        print("No recommendations generated. Consider broadening tickers or period.")
        return

    print("📊 Evaluating forward performance …")
    perf = aggregate_metrics(recs, data, benchmark_df, horizons)
    summary = summarize(perf, horizons)
    print("\n=== Summary Metrics ===")
    if not summary.empty:
        print(summary.to_string(index=False, justify='center'))
    else:
        print("No valid forward returns to summarize.")

    print("\n=== Sample Signals (first 10) ===")
    print(perf.head(10).to_string(index=False))

    # Save outputs
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    perf.to_csv(f'backtest_signals_{ts}.csv', index=False)
    summary.to_csv(f'backtest_summary_{ts}.csv', index=False)
    print(f"\n📝 Saved per-signal performance to backtest_signals_{ts}.csv")
    print(f"📝 Saved summary metrics to backtest_summary_{ts}.csv")

    # Basic diagnostics
    avg_rr = perf['RR'].mean()
    avg_mom = perf['MomCons'].mean()
    print(f"\nDiagnostics: Avg RR={avg_rr:.2f}, Avg MomentumConsistency={avg_mom:.2f}")


if __name__ == '__main__':
    main()
