import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    series = pd.to_numeric(series.squeeze(), errors="coerce")
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, prev_close = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def macd_line(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute ADX with Wilder smoothing and return ADX, +DI and -DI as a DataFrame.

    Returns a DataFrame with columns: ['ADX', 'PLUS_DI', 'MINUS_DI'] where 'ADX' is
    the smoothed average of DX using Wilder's smoothing (EMA with alpha=1/period).
    This keeps compatibility when callers expect a Series (they can take the first
    column which is ADX).
    """
    high, low, close = df["High"], df["Low"], df["Close"]

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    # Directional movements
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > 0) & (up_move > down_move), 0.0)
    minus_dm = down_move.where((down_move > 0) & (down_move > up_move), 0.0)

    # Wilder's smoothing (equivalent to EMA with alpha=1/period, adjust=False)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_dm_sm = plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    minus_dm_sm = minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Directional indicators
    plus_di = 100.0 * (plus_dm_sm / (atr + 1e-9))
    minus_di = 100.0 * (minus_dm_sm / (atr + 1e-9))

    # DX and ADX
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx_series = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    out = pd.DataFrame({"ADX": adx_series, "PLUS_DI": plus_di, "MINUS_DI": minus_di})
    return out


def _sigmoid(x, k: float = 3.0) -> float:
    try:
        return 1.0 / (1.0 + np.exp(-k * x))
    except Exception:
        return 0.5
