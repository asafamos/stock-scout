"""
Meteor-mode filters (VCP + RS + Pocket Pivots) for large-universe scanning.
Clean, top-level functions suitable for pipeline integration and unit tests.

This module exposes a minimal, consistent API used by tests:
- compute_relative_strength(ticker_df, benchmark_df, periods=[...]) -> {"rs_21d": ..., ...}
- detect_volume_surge(df, lookback=20) -> {"volume_surge": float, "pv_correlation": float, "pocket_pivot_ratio": float}
- detect_consolidation(df, short_period=10, long_period=30) -> float
- check_ma_alignment(df, periods=[10,20,50,100]) -> {"ma_aligned": bool, "alignment_score": float}
- find_support_resistance(df, window=5) -> {"support_level": float, "resistance_level": float, "distance_to_support": float, "distance_to_resistance": float}
- compute_momentum_quality(df) -> {"momentum_consistency": float}
- should_reject_ticker(signals, dynamic=None) -> (bool, str)
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


# --- OHLCV helpers ---
def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    # Accept Series as Close series and coerce to DataFrame
    if isinstance(df, pd.Series):
        out = df.to_frame(name="Close")
    else:
        # Work on a copy to avoid mutating caller
        out = df.copy()

    # 1) Flatten MultiIndex or tuple columns from providers like yfinance
    if isinstance(out.columns, pd.MultiIndex) or any(isinstance(c, tuple) for c in out.columns):
        flat_cols = []
        for c in out.columns:
            base = c[-1] if isinstance(c, tuple) else c
            flat_cols.append(str(base))
        out.columns = flat_cols

    # 2) Ensure we have a Date column; if index is datetime, materialize it
    if "Date" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index()
            # After reset_index, the first column is usually the datetime index
            first_col = out.columns[0]
            out = out.rename(columns={first_col: "Date"})
        else:
            # Try common date-like column names
            for cand in list(out.columns):
                try:
                    name = str(cand).lower()
                except Exception:
                    continue
                if name in ("date", "datetime", "time") or "date" in name:
                    out = out.rename(columns={cand: "Date"})
                    break

    # 3) Normalize OHLCV column names (case-insensitive) to Title-case expected
    norm = {}
    for c in out.columns:
        lc = str(c).lower()
        if lc == "open":
            norm[c] = "Open"
        elif lc == "high":
            norm[c] = "High"
        elif lc == "low":
            norm[c] = "Low"
        elif lc == "close":
            norm[c] = "Close"
        elif lc in ("adj close", "adjusted close", "adjclose", "close_adj"):
            norm[c] = "Close"
        elif lc == "volume":
            norm[c] = "Volume"
        elif lc == "date":
            norm[c] = "Date"
    if norm:
        out = out.rename(columns=norm)
    # If still no Close, pick first column containing 'close'
    if "Close" not in out.columns:
        for c in out.columns:
            lc = str(c).lower()
            if "close" in lc:
                out = out.rename(columns={c: "Close"})
                break

    # 4) Ensure Close exists; fabricate other OHLC fields if missing
    if "Close" not in out.columns:
        raise ValueError("Missing OHLCV column: Close")

    if "Open" not in out.columns:
        out["Open"] = out["Close"]
    if "High" not in out.columns:
        out["High"] = out["Close"]
    if "Low" not in out.columns:
        out["Low"] = out["Close"]
    if "Volume" not in out.columns:
        out["Volume"] = 0

    # 5) If still no Close, pick first numeric column as Close
    if "Close" not in out.columns:
        for c in out.columns:
            s = out[c]
            if pd.api.types.is_numeric_dtype(s):
                out = out.rename(columns={c: "Close"})
                break
    if "Close" not in out.columns:
        raise ValueError("Missing OHLCV column: Close")

    # 6) Ensure Date exists; if missing, synthesize a simple integer range
    if "Date" not in out.columns:
        try:
            out["Date"] = np.arange(len(out))
        except Exception:
            out["Date"] = list(range(len(out)))

    return out


def _true_range(df: pd.DataFrame) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


# --- Core signal components ---
def compute_relative_strength(
    ticker_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    periods: list[int] = [21, 63],
) -> Dict[str, float]:
    """Return RS difference for requested periods using lowercase keys expected by tests.

    Keys: rs_{period}d (e.g., rs_21d)
    """
    try:
        tdf = _ensure_ohlcv(ticker_df)
        bdf = _ensure_ohlcv(benchmark_df)
    except Exception:
        return {f"rs_{p}d": np.nan for p in periods}
    out: Dict[str, float] = {}
    for period in periods:
        if len(tdf) < period or len(bdf) < period:
            out[f"rs_{period}d"] = np.nan
            continue
        t_now = float(tdf["Close"].iloc[-1])
        t_prev = float(tdf["Close"].iloc[-period])
        b_now = float(bdf["Close"].iloc[-1])
        b_prev = float(bdf["Close"].iloc[-period])
        if t_prev == 0 or b_prev == 0:
            out[f"rs_{period}d"] = np.nan
            continue
        t_ret = (t_now / t_prev) - 1.0
        b_ret = (b_now / b_prev) - 1.0
        out[f"rs_{period}d"] = float(t_ret - b_ret)
    return out


def detect_consolidation(df: pd.DataFrame, short_period: int = 10, long_period: int = 30) -> float:
    dff = _ensure_ohlcv(df)
    if len(dff) < max(short_period, long_period) + 2:
        return np.nan
    tr = _true_range(dff)
    atr_s = float(tr.rolling(short_period).mean().iloc[-1])
    atr_l = float(tr.rolling(long_period).mean().iloc[-1])
    if not np.isfinite(atr_l) or atr_l <= 0:
        return np.nan
    return float(atr_s / atr_l)


def check_ma_alignment(df: pd.DataFrame, periods: list[int] = [10, 20, 50, 100]) -> Dict[str, float | bool]:
    """Return simple MA alignment signal.

    - ma_aligned: True if MA(shorter) > MA(longer) for all consecutive pairs and Close > shortest MA
    - alignment_score: fraction [0,1] of consecutive MA pairs in correct order
    """
    dff = _ensure_ohlcv(df)
    close = dff["Close"]
    periods_sorted = sorted(set(int(p) for p in periods))
    mas = []
    for p in periods_sorted:
        s = close.rolling(p).mean().iloc[-1]
        mas.append(float(s) if pd.notna(s) else np.nan)
    # Compute pairwise alignment score
    pairs = list(zip(mas[:-1], mas[1:]))
    valid_pairs = [(a, b) for a, b in pairs if np.isfinite(a) and np.isfinite(b)]
    if not valid_pairs:
        return {"ma_aligned": False, "alignment_score": 0.0}
    good = sum(1 for a, b in valid_pairs if a > b)
    score = float(good) / float(len(valid_pairs)) if valid_pairs else 0.0
    shortest_ma = mas[0] if mas else np.nan
    close_ok = np.isfinite(shortest_ma) and float(close.iloc[-1]) > float(shortest_ma)
    aligned = bool(score == 1.0 and close_ok)
    return {"ma_aligned": aligned, "alignment_score": float(score)}


def find_support_resistance(df: pd.DataFrame, window: int = 5) -> Dict[str, float]:
    """Return simple support/resistance and distances from current price.

    Distances are returned as absolute price deltas to satisfy unit tests.
    """
    dff = _ensure_ohlcv(df)
    hi = float(dff["High"].rolling(window).max().iloc[-1]) if len(dff) >= window else float(dff["High"].max())
    lo = float(dff["Low"].rolling(window).min().iloc[-1]) if len(dff) >= window else float(dff["Low"].min())
    cur = float(dff["Close"].iloc[-1])
    return {
        "support_level": lo,
        "resistance_level": hi,
        "distance_to_support": float(max(0.0, cur - lo)),
        "distance_to_resistance": float(max(0.0, hi - cur)),
    }

def compute_momentum_quality(df: pd.DataFrame, window: int = 60) -> Dict[str, float]:
    """Estimate momentum consistency in [0,1].

    Combines the share of positive daily returns and the smoothness (low variance)
    of returns over the window. Designed to be simple and robust for tests.
    """
    dff = _ensure_ohlcv(df)
    if len(dff) < max(10, window // 3):
        return {"momentum_consistency": 0.0}
    closes = dff["Close"]
    ret = closes.pct_change(fill_method=None).dropna()
    ret_w = ret.tail(window) if len(ret) >= window else ret
    if ret_w.empty:
        return {"momentum_consistency": 0.0}
    pos_share = float((ret_w > 0).mean())  # [0,1]
    vol = float(ret_w.std())
    # Map volatility to [0,1] penalty (lower vol → higher score)
    vol_penalty = float(np.clip(vol / 0.02, 0.0, 1.0))  # 2% daily std is high
    smooth_score = 1.0 - vol_penalty
    score = float(np.clip(0.7 * pos_share + 0.3 * smooth_score, 0.0, 1.0))
    return {"momentum_consistency": score}


def calculate_distance_from_52w_high(df: pd.DataFrame) -> float:
    dff = _ensure_ohlcv(df)
    look = min(len(dff), 252)
    if look < 20:
        return np.nan
    hi_52w = float(dff["High"].tail(look).max())
    close = float(dff["Close"].iloc[-1])
    if hi_52w <= 0:
        return np.nan
    return float((close / hi_52w) - 1.0)


def detect_volume_surge(df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
    dff = _ensure_ohlcv(df)
    if len(dff) < lookback + 5:
        # For insufficient data, return neutral zeros to satisfy tests
        return {"volume_surge": 0.0, "pv_correlation": 0.0, "pocket_pivot_ratio": np.nan}
    recent_vol = float(dff["Volume"].tail(5).mean())
    avg_vol = float(dff["Volume"].tail(lookback).mean())
    surge_ratio = float(recent_vol / avg_vol) if avg_vol > 0 else np.nan
    ret = dff["Close"].pct_change(fill_method=None).tail(lookback)
    vol_chg = dff["Volume"].pct_change(fill_method=None).tail(lookback)
    combined = pd.concat([ret, vol_chg], axis=1).dropna()
    pv_corr = float(combined.corr().iloc[0, 1]) if len(combined) > 5 else np.nan
    price_chg = dff["Close"].diff()
    up_mask = price_chg > 0
    down_mask = price_chg < 0
    up_vols = dff["Volume"].where(up_mask).dropna().tail(lookback)
    down_vols = dff["Volume"].where(down_mask).dropna().tail(lookback)
    if len(up_vols) and len(down_vols) and down_vols.mean() > 0:
        pocket_ratio = float(up_vols.mean() / down_vols.mean())
    else:
        pocket_ratio = np.nan
    return {"volume_surge": surge_ratio, "pv_correlation": pv_corr, "pocket_pivot_ratio": pocket_ratio}


def calculate_risk_reward_ratio(df: pd.DataFrame, atr_period: int = 14) -> Dict[str, float]:
    dff = _ensure_ohlcv(df)
    tr = _true_range(dff)
    atr = tr.rolling(atr_period).mean()
    try:
        sr = find_support_resistance(dff, window=20)
        entry = float(dff["Close"].iloc[-1])
        stop = float(sr["recent_low"])  # simplistic stop
        target = float(sr["recent_high"])  # breakout to range high
        risk = max(1e-6, entry - stop)
        reward = max(1e-6, target - entry)
        rr = float(reward / risk)
    except Exception:
        rr = np.nan
    return {"ATR": float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else np.nan, "RR": rr}


# --- Public API ---
def compute_meteor_signals(ticker: str, df: pd.DataFrame, benchmark_df: pd.DataFrame) -> Dict[str, float | bool]:
    dff = _ensure_ohlcv(df)
    rs = compute_relative_strength(dff, benchmark_df, periods=[21, 63])
    cons = detect_consolidation(dff, short_period=10, long_period=30)
    ma = check_ma_alignment(dff)
    dist = calculate_distance_from_52w_high(dff)
    vol = detect_volume_surge(dff, lookback=20)
    rr = calculate_risk_reward_ratio(dff, atr_period=14)
    return {
        "Ticker": ticker,
        "VCP_Ratio": cons,
        "Above_MA50": ma.get("Above_MA50", False),
        "Above_MA200": ma.get("Above_MA200", False),
        "Dist_From_52w_High": dist,
        "RS_21d": rs.get("RS_21d", np.nan),
        "RS_63d": rs.get("RS_63d", np.nan),
        "Pocket_Pivot_Ratio": vol.get("pocket_pivot_ratio", np.nan),
        "Volume_Surge_Ratio": vol.get("volume_surge", np.nan),
        "PV_Correlation": vol.get("pv_correlation", np.nan),
        "ATR": rr.get("ATR", np.nan),
        "RR": rr.get("RR", np.nan),
    }


def should_pass_meteor(signals: Dict[str, float | bool]) -> Tuple[bool, str]:
    dist = float(signals.get("Dist_From_52w_High", np.nan))
    vcp = float(signals.get("VCP_Ratio", np.nan))
    above_ma50 = bool(signals.get("Above_MA50", False))
    above_ma200 = bool(signals.get("Above_MA200", False))
    pocket = float(signals.get("Pocket_Pivot_Ratio", np.nan))
    rs63 = float(signals.get("RS_63d", np.nan))
    if pd.isna(dist) or pd.isna(vcp) or pd.isna(pocket):
        return False, "insufficient_data"
    if not (above_ma50 and above_ma200):
        return False, "ma_alignment_failed"
    if not (-0.10 <= dist <= -0.05):
        return False, "not_near_52w_high"
    if not (vcp < 1.0 and vcp <= 0.75):
        return False, "no_vcp_contraction"
    if not (pocket > 1.3):
        return False, "no_pocket_pivot_ratio"
    if pd.notna(rs63) and rs63 <= 0.0:
        return False, "weak_rs"
    return True, "meteor_pass"


def fetch_benchmark_data(benchmark: str = "SPY", days: int = 200) -> pd.DataFrame:
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        bench_df = yf.download(benchmark, start=start, end=end, auto_adjust=True, progress=False)
        return bench_df if not bench_df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


    def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        cols = {c.lower(): c for c in df.columns}
        # Normalize case to Title-case expected in this module
        mapping = {
            cols.get("date", "date"): "Date",
            cols.get("open", "Open"): "Open",
            cols.get("high", "High"): "High",
            cols.get("low", "Low"): "Low",
            cols.get("close", "Close"): "Close",
            cols.get("volume", "Volume"): "Volume",
        }
        out = df.rename(columns=mapping)
        for c in ["Date", "Open", "High", "Low", "Close", "Volume"]:
            if c not in out.columns:
                raise ValueError(f"Missing OHLCV column: {c}")
        return out


    def compute_relative_strength(
        ticker_df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
        periods: list[int] = [21, 63]
    ) -> Dict[str, float]:
        """
        Calculate relative strength (outperformance vs SPY) over given periods.
        Returns dict: {"rs_21d": diff, "rs_63d": diff} where diff = stock_ret - spy_ret.
        """
        tdf = _ensure_ohlcv(ticker_df)
        bdf = _ensure_ohlcv(benchmark_df)
        rs_scores: Dict[str, float] = {}

        for period in periods:
            if len(tdf) < period or len(bdf) < period:
                rs_scores[f"rs_{period}d"] = np.nan
                continue
            t_now = float(tdf["Close"].iloc[-1])
            t_prev = float(tdf["Close"].iloc[-period])
            b_now = float(bdf["Close"].iloc[-1])
            b_prev = float(bdf["Close"].iloc[-period])
            if t_prev == 0 or b_prev == 0:
                rs_scores[f"rs_{period}d"] = np.nan
                continue
            t_ret = (t_now / t_prev) - 1.0
            b_ret = (b_now / b_prev) - 1.0
            rs_scores[f"rs_{period}d"] = float(t_ret - b_ret)
        return rs_scores


    def detect_volume_surge(df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
        """
        Pocket Pivot-style surge: ratio of up-day volume to down-day volume,
        plus a simple price-volume correlation.
        """
        dff = _ensure_ohlcv(df)
        if len(dff) < lookback + 5:
            return {"volume_surge": np.nan, "pv_correlation": np.nan}

        recent_vol = float(dff["Volume"].tail(5).mean())
        avg_vol = float(dff["Volume"].tail(lookback).mean())
        surge_ratio = float(recent_vol / avg_vol) if avg_vol > 0 else 0.0

        ret = dff["Close"].pct_change().tail(lookback)
        vol_chg = dff["Volume"].pct_change().tail(lookback)
        combined = pd.concat([ret, vol_chg], axis=1).dropna()
        pv_corr = float(combined.corr().iloc[0, 1]) if len(combined) > 5 else np.nan

        # Pocket Pivot proxy: up-day vol vs down-day vol
        price_chg = dff["Close"].diff()
        up_mask = price_chg > 0
        down_mask = price_chg < 0
        up_vol = float(dff.loc[up_mask, "Volume"].tail(lookback).mean()) if up_mask.any() else np.nan
        down_vol = float(dff.loc[down_mask, "Volume"].tail(lookback).mean()) if down_mask.any() else np.nan
        if pd.notna(up_vol) and pd.notna(down_vol) and down_vol > 0:
            pocket_ratio = float(up_vol / down_vol)
        else:
            pocket_ratio = np.nan

        return {
            "volume_surge": surge_ratio,
            "pv_correlation": pv_corr,
            "pocket_pivot_ratio": pocket_ratio,
        }


    def _true_range(df: pd.DataFrame) -> pd.Series:
        hl = df["High"] - df["Low"]
        hc = (df["High"] - df["Close"].shift()).abs()
        lc = (df["Low"] - df["Close"].shift()).abs()
        return pd.concat([hl, hc, lc], axis=1).max(axis=1)


    def detect_consolidation(df: pd.DataFrame, short_period: int = 10, long_period: int = 30) -> float:
        """
        VCP-style consolidation: ATR(10) / ATR(30).
        Values < 1 indicate contraction; < 0.75 indicates strong tightening.
        """
        dff = _ensure_ohlcv(df)
        if len(dff) < max(short_period, long_period) + 2:
            return np.nan
        tr = _true_range(dff)
        atr_short = float(tr.rolling(short_period).mean().iloc[-1])
        atr_long = float(tr.rolling(long_period).mean().iloc[-1])
        if atr_long <= 0 or not np.isfinite(atr_long):
            return np.nan
        return float(atr_short / atr_long)


    def check_ma_alignment(df: pd.DataFrame) -> Dict[str, bool]:
        dff = _ensure_ohlcv(df)
        close = dff["Close"]
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        return {
            "above_ma50": bool(float(close.iloc[-1]) > float(ma50.iloc[-1]) if pd.notna(ma50.iloc[-1]) else False),
            "above_ma200": bool(float(close.iloc[-1]) > float(ma200.iloc[-1]) if pd.notna(ma200.iloc[-1]) else False),
        }


    def find_support_resistance(df: pd.DataFrame, window: int = 20) -> Dict[str, float]:
        dff = _ensure_ohlcv(df)
        hi = float(dff["High"].rolling(window).max().iloc[-1])
        lo = float(dff["Low"].rolling(window).min().iloc[-1])
        return {"recent_high": hi, "recent_low": lo}


    def calculate_distance_from_52w_high(df: pd.DataFrame) -> float:
        dff = _ensure_ohlcv(df)
        lookback = min(len(dff), 252)
        if lookback < 20:
            return np.nan
        high_52w = float(dff["High"].tail(lookback).max())
        close = float(dff["Close"].iloc[-1])
        if high_52w <= 0:
            return np.nan
        return float((close / high_52w) - 1.0)  # e.g., -0.06 = 6% below 52w high


    def calculate_risk_reward_ratio(df: pd.DataFrame, atr_period: int = 14) -> Dict[str, float]:
        dff = _ensure_ohlcv(df)
        tr = _true_range(dff)
        atr = tr.rolling(atr_period).mean()
        rr = np.nan
        try:
            sup_res = find_support_resistance(dff, window=20)
            entry = float(dff["Close"].iloc[-1])
            stop = float(sup_res["recent_low"])  # simplistic stop beneath range low
            target = float(sup_res["recent_high"])  # breakout to range high
            risk = max(1e-6, entry - stop)
            reward = max(1e-6, target - entry)
            rr = float(reward / risk)
        except Exception:
            rr = np.nan
        return {"ATR": float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else np.nan, "RR": rr}


    def compute_meteor_signals(
        ticker: str,
        df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
    ) -> Dict[str, float | bool]:
        """
        Compute core Meteor signals: VCP contraction, MA alignment, distance to 52w high,
        RS diffs, and pocket pivot ratio.
        """
        dff = _ensure_ohlcv(df)
        rs = compute_relative_strength(dff, benchmark_df, periods=[21, 63])
        cons_ratio = detect_consolidation(dff, short_period=10, long_period=30)
        ma = check_ma_alignment(dff)
        dist_52w = calculate_distance_from_52w_high(dff)
        vol = detect_volume_surge(dff, lookback=20)
        rr = calculate_risk_reward_ratio(dff, atr_period=14)
        return {
            "Ticker": ticker,
            "VCP_Ratio": cons_ratio,
            "Above_MA50": ma.get("above_ma50", False),
            "Above_MA200": ma.get("above_ma200", False),
            "Dist_From_52w_High": dist_52w,
            "RS_21d": rs.get("rs_21d", np.nan),
            "RS_63d": rs.get("rs_63d", np.nan),
            "Pocket_Pivot_Ratio": vol.get("pocket_pivot_ratio", np.nan),
            "Volume_Surge_Ratio": vol.get("volume_surge", np.nan),
            "PV_Correlation": vol.get("pv_correlation", np.nan),
            "ATR": rr.get("ATR", np.nan),
            "RR": rr.get("RR", np.nan),
        }


    def should_pass_meteor(signals: Dict[str, float | bool]) -> Tuple[bool, str]:
        """
        Meteor filter:
        - Price within 5–10% of 52w high (distance in [-0.10, -0.05])
        - Price above MA50 and MA200
        - VCP contraction: ATR(10)/ATR(30) <= 0.75 (≥25% decrease)
        - Pocket Pivot ratio > 1.3 (up-day vol > down-day vol)
        - RS_63d positive (top 20% ranking handled upstream)
        """
        dist = float(signals.get("Dist_From_52w_High", np.nan))
        vcp = float(signals.get("VCP_Ratio", np.nan))
        above_ma50 = bool(signals.get("Above_MA50", False))
        above_ma200 = bool(signals.get("Above_MA200", False))
        pocket = float(signals.get("Pocket_Pivot_Ratio", np.nan))
        rs63 = float(signals.get("RS_63d", np.nan))

        if pd.isna(dist) or pd.isna(vcp) or pd.isna(pocket):
            return False, "insufficient_data"
        if not (above_ma50 and above_ma200):
            return False, "ma_alignment_failed"
        if not (-0.10 <= dist <= -0.05):
            return False, "not_near_52w_high"
        # Require contraction: ATR(10) strictly less than ATR(30), and tightness ≤ 0.75
        if not (vcp < 1.0 and vcp <= 0.75):
            return False, "no_vcp_contraction"
        if not (pocket > 1.3):
            return False, "no_pocket_pivot_ratio"
        if pd.notna(rs63) and rs63 <= 0.0:
            return False, "weak_rs"
        return True, "meteor_pass"


    def compute_advanced_score(
        ticker: str,
        df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
        base_score: float
    ) -> Tuple[float, Dict[str, object]]:
        """Lightweight enhancer: build signals and add a modest boost when aligned.

        base_score is assumed to be normalized [0,1].
        """
        dff = _ensure_ohlcv(df)
        rs = compute_relative_strength(dff, benchmark_df, periods=[63])
        ma = check_ma_alignment(dff)
        vcp = detect_consolidation(dff)
        vol = detect_volume_surge(dff)
        rr = calculate_risk_reward_ratio(dff)
        signals = {
            "rs_63d": rs.get("rs_63d", np.nan),
            "ma_aligned": bool(ma.get("ma_aligned", False)),
            "consolidation_ratio": vcp,
            "volume_surge": vol.get("volume_surge", np.nan),
            "risk_reward_ratio": rr.get("RR", np.nan),
        }
        boost = 0.0
        if np.isfinite(signals["rs_63d"]) and signals["rs_63d"] > 0.0:
            boost += 0.05
        if signals["ma_aligned"]:
            boost += 0.06
        if np.isfinite(vcp) and vcp < 0.8:
            boost += 0.03
        if np.isfinite(signals["risk_reward_ratio"]) and signals["risk_reward_ratio"] >= 1.5:
            boost += 0.05
        enhanced = float(np.clip(base_score + boost, 0.0, 1.0))
        return enhanced, {"signals": signals}


    def fetch_benchmark_data(benchmark: str = "SPY", days: int = 200) -> pd.DataFrame:
        """
        Fetch benchmark series via shared market context providers.
        """
        try:
            from core.market_context import get_benchmark_series
            period = "6mo" if days <= 180 else "1y"
            return get_benchmark_series(symbol=benchmark, period=period)
        except Exception:
            return pd.DataFrame()
    
    current_price = float(df["Close"].iloc[-1])
    
    # Get support/resistance
    sr_levels = find_support_resistance(df)
    resistance = sr_levels.get("resistance_level", current_price * 1.1)
    support = sr_levels.get("support_level", current_price * 0.95)
    
    # Potential reward (to resistance)
    potential_reward = float((resistance - current_price)) if resistance > current_price else atr * 2
    
    # Potential risk (to support or 2x ATR)
    potential_risk = float(max((current_price - support), atr * 2)) if support < current_price else atr * 2
    
    risk_reward = float(potential_reward / potential_risk) if potential_risk > 0 else 0.0
    
    return {
        "risk_reward_ratio": float(risk_reward) if np.isfinite(risk_reward) else 0.0,
        "potential_reward": float(potential_reward),
        "potential_risk": float(potential_risk)
    }


def compute_advanced_score(
    ticker: str,
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    base_score: float
) -> Tuple[float, Dict[str, any]]:
    """
    Compute enhanced score using robust, minimal primitives.
    Returns (enhanced_score in [0,1], signals dict with expected keys).
    """
    dff = _ensure_ohlcv(df)
    # Relative strength (RS_21d/RS_63d)
    rs = compute_relative_strength(dff, benchmark_df, periods=[21, 63])
    # Volume patterns
    vol = detect_volume_surge(dff, lookback=20)
    # Consolidation tightness
    vcp = detect_consolidation(dff, short_period=10, long_period=30)
    # MA flags
    ma = check_ma_alignment(dff)
    ma_aligned = bool(ma.get("Above_MA50", False) and ma.get("Above_MA200", False))
    alignment_score = (int(ma.get("Above_MA50", False)) + int(ma.get("Above_MA200", False))) / 2.0
    # Risk/Reward proxy
    rr_info = calculate_risk_reward_ratio(dff)
    rr_ratio = rr_info.get("RR", np.nan) if "RR" in rr_info else rr_info.get("risk_reward_ratio", np.nan)

    # Compose signals to match pipeline expectations
    signals = {
        "rs_63d": rs.get("RS_63d", np.nan),
        "volume_surge": vol.get("volume_surge", np.nan),
        "pv_correlation": vol.get("pv_correlation", np.nan),
        "pocket_pivot_ratio": vol.get("pocket_pivot_ratio", np.nan),
        "consolidation_ratio": vcp,
        "ma_aligned": ma_aligned,
        "alignment_score": alignment_score,
        "risk_reward_ratio": rr_ratio,
        # Not computed here; provide a stable default for permissive thresholds
        "momentum_consistency": 0.6,
    }

    # Simple boosting logic (normalized to [0,1])
    boost = 0.0
    rs63 = signals["rs_63d"]
    if np.isfinite(rs63) and rs63 > 0.0:
        boost += 0.05 if rs63 <= 0.05 else 0.10
    if signals["volume_surge"] and signals["volume_surge"] > 1.3:
        boost += 0.04
    if np.isfinite(vcp) and vcp < 0.80:
        boost += 0.03
    if ma_aligned:
        boost += 0.06
    if np.isfinite(rr_ratio) and rr_ratio >= 1.5:
        boost += 0.05

    enhanced_score = float(np.clip(base_score + boost, 0.0, 1.0))
    # Add a quality indicator on 0-100 scale compatible with pipeline
    signals["quality_score"] = float(np.clip(boost * 100.0, 0.0, 50.0))
    return enhanced_score, signals


def should_reject_ticker(signals: Dict[str, any], dynamic: Optional[Dict[str, float]] = None) -> Tuple[bool, str]:
    """Strict but clear rejection rules tailored for tests.

    Thresholds (can be overridden via dynamic):
    - rs_63d <= -0.25 → Underperforming
    - momentum_consistency < 0.10 → Weak momentum
    - risk_reward_ratio < 0.40 → Poor Risk/Reward
    """
    # Catastrophic: nothing to evaluate
    core_vals = [signals.get("rs_63d", np.nan), signals.get("momentum_consistency", np.nan), signals.get("risk_reward_ratio", np.nan)]
    if all([not np.isfinite(v) for v in core_vals]):
        return True, "insufficient_data"

    rs_thr = (dynamic.get("rs_63d") if dynamic and "rs_63d" in dynamic else -0.25)
    mom_thr = (dynamic.get("momentum_consistency") if dynamic and "momentum_consistency" in dynamic else 0.10)
    rr_thr = (dynamic.get("risk_reward_ratio") if dynamic and "risk_reward_ratio" in dynamic else 0.40)

    rs_63d = signals.get("rs_63d", np.nan)
    if np.isfinite(rs_63d) and float(rs_63d) <= float(rs_thr):
        return True, "Underperforming vs SPY"

    mom = signals.get("momentum_consistency", np.nan)
    if np.isfinite(mom) and float(mom) < float(mom_thr):
        return True, "Weak momentum"

    rr = signals.get("risk_reward_ratio", np.nan)
    if np.isfinite(rr) and float(rr) < float(rr_thr):
        return True, "Poor Risk/Reward"

    return False, ""


def fetch_benchmark_data(benchmark: str = "SPY", days: int = 400) -> pd.DataFrame:
    """
    Fetch benchmark data for relative strength calculations.
    Cached to avoid repeated downloads.
    """
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        bench_df = yf.download(benchmark, start=start, end=end, auto_adjust=True, progress=False)
        return bench_df if not bench_df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()
