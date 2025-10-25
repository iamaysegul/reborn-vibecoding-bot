"""Backtest engine for Reborn - simple CSV-based backtester.

Usage:
    python -m core.backtest --csv data/AVAX_1m.csv --config config.yaml

This module attempts to reuse `core.strategy`, `core.indicators`, and `core.risk` when
available. If those modules are missing, it falls back to local implementations
compatible with the acceptance criteria in Reborn_Codex_Context.md.

Outputs JSON-like summary to stdout with metrics and prints a small tabular
summary of trades.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yaml
import sys
import os

# Try to import project modules; fall back to local implementations if needed
try:
    from core import indicators as project_indicators  # type: ignore
except Exception:
    project_indicators = None

try:
    from core import strategy as project_strategy  # type: ignore
except Exception:
    project_strategy = None

try:
    from core import risk as project_risk  # type: ignore
except Exception:
    project_risk = None


@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    exit_reason: str


# ----------------------------- Utilities ---------------------------------

def load_config(path: Optional[str]) -> Dict:
    defaults = {
        "balance": 1000.0,
        "risk_per_trade": 0.01,
        "leverage": 1,
        "price_step": 0.01,
        "qty_step": 0.1,
        "ema_fast": 9,
        "ema_slow": 21,
        "atr_period": 14,
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_period": 14,
        "rsi_min": 50.0,
        "rsi_max": 70.0,
        "vol_ma_period": 20,
        "vol_multiplier": 1.5,
        "bb_bw_threshold": 0.06,
        "breakout_n": 5,
        "sl_atr_mult": 1.0,
    }
    if not path:
        return defaults
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        print(f"Warning: failed to read config {path}, using defaults", file=sys.stderr)
        cfg = {}
    merged = defaults.copy()
    merged.update(cfg)
    return merged


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure we have standard columns
    expected = ["timestamp", "open", "high", "low", "close", "volume"]
    for c in expected:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")
    # Convert timestamp if numeric
    if np.issubdtype(df["timestamp"].dtype, np.number):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce").fillna(pd.to_datetime(df["timestamp"], unit="ms", errors="coerce"))
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# --------------------- Minimal indicators (fallback) ----------------------
def compute_indicators(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    df = df.copy()

    # EMA
    ema_fast = cfg.get("ema_fast", 9)
    ema_slow = cfg.get("ema_slow", 21)
    df[f"ema_{ema_fast}"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
    df[f"ema_{ema_slow}"] = df["close"].ewm(span=ema_slow, adjust=False).mean()

    # ATR
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_period = cfg.get("atr_period", 14)
    df["atr"] = tr.rolling(atr_period, min_periods=1).mean()

    # Bollinger Bands and bandwidth
    bb_period = cfg.get("bb_period", 20)
    bb_std = cfg.get("bb_std", 2.0)
    ma = close.rolling(bb_period, min_periods=1).mean()
    sd = close.rolling(bb_period, min_periods=1).std()
    df["bb_upper"] = ma + bb_std * sd
    df["bb_lower"] = ma - bb_std * sd
    df["bb_middle"] = ma
    # bandwidth: (upper - lower) / middle
    df["bb_bw"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"].replace(0, np.nan)

    # RSI
    rsi_period = cfg.get("rsi_period", 14)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / rsi_period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # AVWAP (session cumulative)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_num = (typical * df["volume"]).cumsum()
    cum_den = df["volume"].cumsum().replace(0, np.nan)
    df["avwap"] = cum_num / cum_den

    # Volume moving average
    vol_ma_period = cfg.get("vol_ma_period", 20)
    df["vol_ma"] = df["volume"].rolling(vol_ma_period, min_periods=1).mean()

    return df


# ---------------------- Fallback strategy --------------------------------
def generate_signal_from_df(df: pd.DataFrame, idx: int, cfg: Dict, mode: str = "Calm") -> Optional[Dict]:
    """Return signal dict or None.

    Applies the acceptance criteria from Reborn_Codex_Context.md to the row at `idx`.
    """
    if idx <= 0 or idx >= len(df):
        return None
    row = df.iloc[idx]

    # Must have indicators
    required = ["ema_{}".format(cfg.get("ema_fast", 9)), "ema_{}".format(cfg.get("ema_slow", 21)), "atr", "bb_bw", "rsi", "avwap", "vol_ma"]
    for r in required:
        if r not in df.columns:
            return None
    # Breakout: close > recent high of last N highs
    n = int(cfg.get("breakout_n", 5))
    if idx - n < 0:
        return None
    recent_high = df.iloc[idx - n:idx]["high"].max()
    if not (row["close"] > recent_high):
        return None
    # Trend: close > ema_fast > ema_slow
    ema_fast_col = f"ema_{cfg.get('ema_fast',9)}"
    ema_slow_col = f"ema_{cfg.get('ema_slow',21)}"
    if not (row["close"] > row[ema_fast_col] > row[ema_slow_col]):
        return None
    # Bollinger squeeze
    bb_bw_threshold = cfg.get("bb_bw_threshold", 0.06)
    if mode != "Hype":
        if not (row["bb_bw"] < bb_bw_threshold):
            return None
    # Volume surge
    vol_mult = float(cfg.get("vol_multiplier", 1.5))
    vol_cond = row["volume"] > row["vol_ma"] * vol_mult
    if mode == "Calm":
        vol_cond = row["volume"] > row["vol_ma"] * (vol_mult + 0.1)
    if not vol_cond:
        return None
    # RSI band
    rsi_min = float(cfg.get("rsi_min", 50.0))
    rsi_max = float(cfg.get("rsi_max", 70.0))
    if not (rsi_min <= row["rsi"] <= rsi_max):
        return None
    # AVWAP
    if not (row["close"] >= row["avwap"]):
        return None

    signal = {"side": "long", "entry": float(row["close"]), "atr": float(row["atr"]), "timestamp": int(row["timestamp"].timestamp())}
    return signal


# ---------------------- Position sizing & SL/TP ---------------------------

def position_size(entry: float, stop: float, balance: float, risk_pct: float, leverage: float, price_step: float, qty_step: float, max_margin_frac: float = 0.8) -> Tuple[float, float]:
    # qty_by_risk = (balance * risk_pct) / (entry - stop)
    denom = (entry - stop)
    if denom <= 0:
        return 0.0, entry
    qty_by_risk = (balance * risk_pct) / denom
    max_qty_by_margin = (balance * leverage * max_margin_frac) / entry
    qty = min(qty_by_risk, max_qty_by_margin)
    # round down to qty_step
    if qty_step > 0:
        qty = math.floor(qty / qty_step) * qty_step
    # round entry to price_step
    if price_step > 0:
        entry_rounded = round(round(entry / price_step) * price_step, 8)
    else:
        entry_rounded = entry
    return float(max(qty, 0.0)), float(entry_rounded)


def sl_tp_levels(entry: float, atr: float, sl_atr_mult: float) -> Tuple[float, List[float]]:
    sl = entry - max(atr * sl_atr_mult, entry * 0.004)
    R = entry - sl
    tp1 = entry + R * 1
    tp2 = entry + R * 2
    tp3 = entry + R * 3
    return float(sl), [float(tp1), float(tp2), float(tp3)]


# ---------------------- Backtest simulation ------------------------------

def run_backtest(df: pd.DataFrame, cfg: Dict, mode: str = "Calm") -> Dict:
    # Compute indicators (use project_indicators if available)
    if project_indicators is not None:
        try:
            df_ind = project_indicators.compute_indicators(df.copy(), cfg)  # type: ignore
        except Exception:
            df_ind = compute_indicators(df, cfg)
    else:
        df_ind = compute_indicators(df, cfg)

    trades: List[Trade] = []
    equity = [float(cfg.get("balance", 1000.0))]
    balance = float(cfg.get("balance", 1000.0))
    in_position = False

    i = 0
    n = len(df_ind)
    while i < n - 1:
        # Generate signal at i
        signal = None
        if project_strategy is not None:
            try:
                # Try to use project strategy interface
                strat = getattr(project_strategy, "BreakoutStrategy", None)
                if strat is not None:
                    bs = strat(cfg)  # type: ignore
                    # Some strategies may expect a dataframe window; try a few call patterns
                    try:
                        signal = bs.generate_signal(df_ind.iloc[: i + 1], None, mode)  # type: ignore
                    except Exception:
                        try:
                            signal = bs.generate_signal(df_ind.iloc[i], None, mode)  # type: ignore
                        except Exception:
                            signal = None
            except Exception:
                signal = None
        if signal is None:
            signal = generate_signal_from_df(df_ind, i, cfg, mode)

        if signal is None:
            i += 1
            equity.append(balance)
            continue

        # We have a long signal. Enter at next bar open if available
        entry_idx = i + 1
        if entry_idx >= n:
            break
        entry_price = float(df_ind.iloc[entry_idx]["open"])
        atr = float(df_ind.iloc[i]["atr"]) if not math.isnan(df_ind.iloc[i]["atr"]) else float(cfg.get("atr_period", 14))
        sl, tps = sl_tp_levels(entry_price, atr, float(cfg.get("sl_atr_mult", 1.0)))

        qty, entry_rounded = position_size(entry_price, sl, balance, float(cfg.get("risk_per_trade", 0.01)), float(cfg.get("leverage", 1)), float(cfg.get("price_step", 0.01)), float(cfg.get("qty_step", 0.1)))
        if qty <= 0:
            # skip if no size
            i = entry_idx
            equity.append(balance)
            continue

        exited = False
        exit_idx = entry_idx
        exit_price = entry_price
        exit_reason = "timeout"

        # scan forward for exit
        j = entry_idx
        while j < n:
            high = float(df_ind.iloc[j]["high"])
            low = float(df_ind.iloc[j]["low"])
            # Check TP/SL hits. Determine first hit if both hit same candle.
            tp_hit = None
            for tp in tps:
                if high >= tp:
                    tp_hit = tp
                    break
            sl_hit = low <= sl

            if tp_hit is None and not sl_hit:
                # continue
                j += 1
                continue

            # If both hit, approximate which happened first by distance from entry
            if tp_hit is not None and sl_hit:
                dist_to_sl = entry_price - sl
                dist_to_tp = tp_hit - entry_price
                if dist_to_sl < dist_to_tp:
                    # SL first
                    exit_price = sl
                    exit_reason = "SL"
                else:
                    exit_price = tp_hit
                    exit_reason = "TP1+"
            elif tp_hit is not None:
                exit_price = tp_hit
                exit_reason = "TP1+"
            elif sl_hit:
                exit_price = sl
                exit_reason = "SL"
            exited = True
            exit_idx = j
            break
        if not exited:
            # exit at last close
            exit_idx = n - 1
            exit_price = float(df_ind.iloc[-1]["close"])
            exit_reason = "EOD"

        pnl = (exit_price - entry_price) * qty
        balance += pnl
        trades.append(Trade(entry_idx=entry_idx, exit_idx=exit_idx, entry_price=entry_price, exit_price=exit_price, qty=qty, pnl=pnl, exit_reason=exit_reason))
        equity.append(balance)
        # Continue after exit bar
        i = max(exit_idx, entry_idx) + 1

    # Metrics
    total_pnl = balance - float(cfg.get("balance", 1000.0))
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    winrate = (len(wins) / len(trades)) if trades else 0.0
    gross_win = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in losses) if losses else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)

    # max drawdown
    eq = np.array(equity)
    running_max = np.maximum.accumulate(eq)
    dd = (running_max - eq) / running_max
    max_dd = float(np.nanmax(dd)) if len(dd) > 0 else 0.0

    metrics = {
        "trades": len(trades),
        "winrate": winrate,
        "total_pnl": float(total_pnl),
        "gross_win": float(gross_win),
        "gross_loss": float(gross_loss),
        "profit_factor": float(profit_factor) if not math.isinf(profit_factor) else "inf",
        "max_drawdown": float(max_dd),
        "equity_final": float(balance),
    }

    # Also collect per-trade summary
    trades_summary = [t.__dict__ for t in trades]
    return {"metrics": metrics, "trades": trades_summary, "equity_curve": equity}


# ---------------------- CLI Entrypoint ----------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run CSV backtest for Reborn strategy")
    p.add_argument("--csv", required=True, help="Path to OHLCV CSV file with timestamp,open,high,low,close,volume columns")
    p.add_argument("--config", required=False, help="Path to config.yaml (optional)")
    p.add_argument("--mode", required=False, default="Calm", choices=["Calm", "Hype"], help="Vibe mode to apply during backtest")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    df = load_csv(args.csv)

    result = run_backtest(df, cfg, mode=args.mode)

    # Print concise metrics
    print(json.dumps(result["metrics"], indent=2))
    print("\nTrades: {}".format(len(result["trades"])))
    if result["trades"]:
        print("idx | entry -> exit | qty | pnl | reason")
        for t in result["trades"][:50]:
            print(f"{t['entry_idx']}->{t['exit_idx']} | {t['entry_price']:.4f}->{t['exit_price']:.4f} | {t['qty']:.4f} | {t['pnl']:.4f} | {t['exit_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
