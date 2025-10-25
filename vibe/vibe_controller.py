"""Vibe controller: adaptive Calm/Hype mode manager.

Provides a small state machine to detect and update the bot "vibe" (Calm vs Hype)
based on volatility and volume signals. Designed to be simple, deterministic
and easy to tune.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Dict

# Tunable constants
VOLATILITY_UP_THRESHOLD = 1.0
VOLATILITY_DOWN_THRESHOLD = 0.7
VOLUME_MULTIPLIER = 1.5


class VibeMode(Enum):
    CALM = "Calm"
    HYPE = "Hype"


@dataclass
class VibeState:
    mode: VibeMode = VibeMode.CALM
    switch_count: int = 0
    streak: int = 0
    last_switch_ts: float = 0.0
    volatility: float = 0.0
    volume_ratio: float = 0.0


def detect_mode(volatility: float, volume: float, avg_vol: float) -> VibeMode:
    """Detect target vibe mode from recent stats.

    Rules (simple):
    - Calm -> Hype when volatility is elevated AND volume > VOLUME_MULTIPLIER * avg_vol
    - Hype -> Calm when volatility has dropped below VOLATILITY_DOWN_THRESHOLD AND volume < avg_vol

    This function is stateless and returns the *detected* target mode; callers may
    choose whether to apply it immediately (see update_vibe).
    """
    if avg_vol <= 0:
        return VibeMode.CALM

    if volatility >= VOLATILITY_UP_THRESHOLD and volume > VOLUME_MULTIPLIER * avg_vol:
        return VibeMode.HYPE

    if volatility <= VOLATILITY_DOWN_THRESHOLD and volume < avg_vol:
        return VibeMode.CALM

    # Default to Calm when ambiguous
    return VibeMode.CALM


def update_vibe(state: VibeState, new_mode: VibeMode) -> VibeState:
    """Update VibeState with a newly detected mode.

    - If mode changed: increment switch_count, reset streak to 0, update last_switch_ts.
    - If mode unchanged: increment streak (indicates sustained conditions).
    Returns updated VibeState (mutates the passed object and returns it for convenience).
    """
    now_ts = time.time()
    if new_mode != state.mode:
        state.mode = new_mode
        state.switch_count += 1
        state.streak = 0
        state.last_switch_ts = now_ts
    else:
        state.streak += 1

    # Update observed stats timestamp fields if caller wants to store them
    state.volatility = float(state.volatility)
    state.volume_ratio = float(state.volume_ratio)
    return state


def vibe_feedback(state: VibeState) -> Dict[str, float]:
    """Return dynamic tuning parameters based on current vibe state.

    Returns a dict with keys (example):
    - risk_multiplier: float (applied to base risk)
    - cooldown_s: seconds to wait between aggressive entries
    - aggressiveness: 0..1 float controlling order sizing/entry frequency
    """
    if state.mode == VibeMode.HYPE:
        # More aggressive but slightly higher risk; scale with short positive streaks
        base_risk = 1.2
        cooldown = 30.0
        aggressiveness = 0.9
    else:
        base_risk = 0.8
        cooldown = 120.0
        aggressiveness = 0.4

    # Slightly amplify risk/aggressiveness when streaks are positive (cap to avoid runaway)
    streak_bonus = min(state.streak * 0.02, 0.2)
    return {
        "risk_multiplier": base_risk * (1.0 + streak_bonus),
        "cooldown_s": cooldown,
        "aggressiveness": min(1.0, aggressiveness * (1.0 + streak_bonus)),
    }


__all__ = ["VibeMode", "VibeState", "detect_mode", "update_vibe", "vibe_feedback"]
