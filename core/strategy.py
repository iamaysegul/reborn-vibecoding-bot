"""Strategy module

Provides a small, async-first Strategy class that computes ATR and a long/short
signal relative to a long-period SMA. The signal shape matches repository
conventions:

    {"side": "long"|"short", "entry": float, "atr": float, "timestamp": int}

The implementation is intentionally compact and dependency-light (pandas/numpy
used if available). It's written to be dry-run friendly and easy to extend.
"""
from __future__ import annotations

import asyncio

from typing import Optional, Dict, Any
import pandas as pd
from core.indicators import ema, rsi, bbands, bb_bandwidth, atr, vwap, avwap, vol_ma, recent_high, recent_low, sma


class Strategy:
    """Simple ATR + SMA breakout strategy.

    Contract
    - Input: candles: pd.DataFrame with columns ["open","high","low","close","volume","timestamp"], ascending time.
    - Output: Optional[dict] with keys: side, entry, atr, timestamp (or None if no signal).

    Configurable parameters are exposed via the constructor.
    """

    def __init__(
        self,
        atr_period: int = 14,
        sma_period: int = 200,
        atr_multiplier: float = 1.5,
    ) -> None:
        self.atr_period = int(atr_period)
        self.sma_period = int(sma_period)
        self.atr_multiplier = float(atr_multiplier)

    async def generate(self, candles: pd.DataFrame, timestamp: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Asynchronously generate a signal from OHLCV candles.

        Returns None when no actionable signal is present.
        """
        # Input validation
        if candles is None or len(candles) < max(self.atr_period, self.sma_period) + 1:
            return None

        # Compute indicators
        atr_series = self._atr(candles["high"], candles["low"], candles["close"], self.atr_period)
        if atr_series.isna().all():
            return None

        sma = candles["close"].rolling(self.sma_period).mean().iloc[-1]
        last = candles.iloc[-1]
        last_close = float(last["close"])
        last_atr = float(atr_series.iloc[-1])

        entry: Optional[float] = None
        side: Optional[str] = None

        # Simple breakout relative to SMA using ATR for buffer
        long_threshold = sma + self.atr_multiplier * last_atr
        short_threshold = sma - self.atr_multiplier * last_atr

        if last_close > long_threshold:
            side = "long"
            entry = last_close
        elif last_close < short_threshold:
            side = "short"
            entry = last_close

        if side is None or entry is None:
            return None

        sig_ts = int(timestamp if timestamp is not None else int(last["timestamp"]))
        return {"side": side, "entry": float(entry), "atr": float(last_atr), "timestamp": sig_ts}

    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Compute ATR using simple moving average of True Range.

        Formula:
            TR = max(high-low, abs(high - close.shift(1)), abs(low - close.shift(1)))
            ATR = TR.rolling(period, min_periods=1).mean()
        """
        tr = pd.concat(
            [
                (high - low),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(period, min_periods=1).mean()
        return atr

    def generate_sync(self, candles: pd.DataFrame, timestamp: Optional[int] = None, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
        """Synchronous wrapper for convenience (runs the async generate)."""
        try:
            return asyncio.run(asyncio.wait_for(self.generate(candles, timestamp), timeout=timeout))
        except Exception:
            return None


__all__ = ["Strategy"]
