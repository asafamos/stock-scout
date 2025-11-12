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


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm <= plus_dm] = 0
    tr = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(period, min_periods=period).mean()
    plus_di = 100 * (
        plus_dm.rolling(period, min_periods=period).mean() / (atr14 + 1e-9)
    )
    minus_di = 100 * (
        minus_dm.rolling(period, min_periods=period).mean() / (atr14 + 1e-9)
    )
    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9)
    return dx.rolling(period, min_periods=period).mean()


def _sigmoid(x, k: float = 3.0) -> float:
    try:
        return 1.0 / (1.0 + np.exp(-k * x))
    except Exception:
        return 0.5
