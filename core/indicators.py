"""Indicator utilities for 1m futures trading.

Lightweight, vectorized implementations using pandas/numpy only.
All functions are NaN-safe where applicable and intended for reuse
across modules (strategy, backtest, etc).
"""
import numpy as np
import pandas as pd
from typing import Callable, Optional, Tuple

EPS: float = 1e-12

__all__ = [
    "EPS",
    "ema",
    "sma",
    "rsi",
    "bbands",
    "bb_bandwidth",
    "atr",
    "vwap",
    "avwap",
    "vol_ma",
    "recent_high",
    "recent_low",
    "_rolling_safe",
    "pct_change_safe",
]

def _rolling_safe(series: pd.Series, window: int, fn: Callable[[np.ndarray], float]) -> pd.Series:
    """Apply a rolling function safely with full-window requirement.

    - Returns all-NaN if series shorter than window or invalid window.
    - Uses min_periods=window to avoid partial-window bias.
    - The provided function should accept a numpy array and return a scalar.
    """
    if window is None or window < 1 or len(series) < window:
        return pd.Series(np.nan, index=series.index)
    try:
        return series.rolling(window, min_periods=window).apply(fn, raw=True)
    except Exception:
        # Fallback: wrap in Series if fn expects a Series
        return series.rolling(window, min_periods=window).apply(lambda x: fn(np.asarray(x)), raw=True)

def pct_change_safe(series: pd.Series) -> pd.Series:
    """Percent change with inf handling -> NaN."""
    out = series.pct_change()
    return out.replace([np.inf, -np.inf], np.nan)

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (Wilder-compatible smoothing)."""
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average with full-window requirement."""
    return series.rolling(period, min_periods=period).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing).

    Returns values in [0, 100]. NaN until enough history accumulates.
    """
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_up / (avg_down.replace(0, np.nan) + EPS)
    return 100 - (100 / (1 + rs))

def bbands(series: pd.Series, period: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands (upper, mid, lower) using population std (ddof=0)."""
    mid = sma(series, period)
    stddev = series.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + std * stddev
    lower = mid - std * stddev
    return upper, mid, lower

def bb_bandwidth(upper: pd.Series, mid: pd.Series, lower: pd.Series) -> pd.Series:
    """Bollinger Bandwidth helper used for squeeze filters."""
    return (upper - lower) / (mid.replace(0, np.nan) + EPS)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder's smoothing)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def vwap(price: pd.Series, volume: pd.Series) -> pd.Series:
    """Cumulative VWAP starting from the beginning of series."""
    pv = price * volume
    cum_pv = pv.cumsum()
    cum_vol = volume.cumsum()
    return cum_pv / (cum_vol + EPS)

def avwap(df: pd.DataFrame, anchor_ts: Optional[int] = None) -> pd.Series:
    """Anchored VWAP by timestamp.

    If `anchor_ts` is None, behaves like cumulative VWAP.
    Otherwise, returns NaN before the anchor and cumulative VWAP from the
    first row where df["timestamp"] >= anchor_ts.
    """
    required = {"timestamp", "close", "volume"}
    if not required.issubset(df.columns):
        return pd.Series(np.nan, index=df.index)

    pv = df["close"] * df["volume"]
    if anchor_ts is None:
        cum_pv = pv.cumsum()
        cum_vol = df["volume"].cumsum()
        return cum_pv / (cum_vol + EPS)

    mask = df["timestamp"] >= anchor_ts
    if not mask.any():
        return pd.Series(np.nan, index=df.index)

    # Compute anchored cumulative sums using masked zeros pre-anchor
    cum_pv = (pv.where(mask, 0.0)).cumsum()
    cum_vol = (df["volume"].where(mask, 0.0)).cumsum()
    out = cum_pv / (cum_vol + EPS)
    out = out.where(mask, np.nan)
    return out

def vol_ma(volume: pd.Series, period: int) -> pd.Series:
    """Volume moving average (full window)."""
    return volume.rolling(period, min_periods=period).mean()

def recent_high(series: pd.Series, lookback: int) -> pd.Series:
    """Rolling recent high with full-window requirement."""
    return series.rolling(lookback, min_periods=lookback).max()

def recent_low(series: pd.Series, lookback: int) -> pd.Series:
    """Rolling recent low with full-window requirement."""
    return series.rolling(lookback, min_periods=lookback).min()

if __name__ == "__main__":
    # Mini self-test: print last 3 rows for each computed indicator
    n = 60
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "timestamp": np.arange(n),
        "open": np.linspace(100, 110, n) + rng.normal(0, 0.2, n),
        "high": np.linspace(101, 111, n) + rng.normal(0, 0.2, n),
        "low": np.linspace(99, 109, n) + rng.normal(0, 0.2, n),
        "close": np.linspace(100, 110, n) + rng.normal(0, 0.5, n),
        "volume": rng.integers(100, 1000, n),
    })

    ema_fast = ema(df["close"], 9)
    ema_slow = ema(df["close"], 21)
    print("EMA(9) tail:\n", ema_fast.tail(3))
    print("EMA(21) tail:\n", ema_slow.tail(3))

    print("SMA(20) tail:\n", sma(df["close"], 20).tail(3))
    print("RSI(14) tail:\n", rsi(df["close"], 14).tail(3))

    u, m, l = bbands(df["close"], 20, 2.0)
    print("BB upper tail:\n", u.tail(3))
    print("BB mid   tail:\n", m.tail(3))
    print("BB lower tail:\n", l.tail(3))
    print("BB bandwidth tail:\n", bb_bandwidth(u, m, l).tail(3))

    print("ATR(14) tail:\n", atr(df["high"], df["low"], df["close"], 14).tail(3))
    print("VWAP tail:\n", vwap(df["close"], df["volume"]).tail(3))
    print("AVWAP (anchored@ts=30) tail:\n", avwap(df, anchor_ts=30).tail(3))
    print("Vol MA(20) tail:\n", vol_ma(df["volume"], 20).tail(3))
    print("Recent High(5) tail:\n", recent_high(df["high"], 5).tail(3))
    print("Recent Low(5) tail:\n", recent_low(df["low"], 5).tail(3))