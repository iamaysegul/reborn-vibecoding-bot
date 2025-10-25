"""ModeSwitcher: adds hysteresis on top of VibeController to avoid flapping.

Provides ModeSwitcher class that chooses Calm/Hype with configurable
hysteresis knobs read from a cfg dict.
"""
from __future__ import annotations

from typing import Dict, Tuple
from vibe.vibe_controller import (
    VibeMode,
    VibeState,
    detect_mode,
    update_vibe,
    vibe_feedback,
)


class ModeSwitcher:
    """Simple hysteresis wrapper for VibeController.

    Config knobs (from cfg dict under key 'vibe'):
      - enter_hype_vol_mult (default 1.6)
      - exit_hype_vol_mult (default 1.0)
    """

    def __init__(self, cfg: Dict | None = None) -> None:
        cfg = cfg or {}
        vibe_cfg = cfg.get("vibe", {}) if isinstance(cfg, dict) else {}
        self.enter_hype_vol_mult: float = float(vibe_cfg.get("enter_hype_vol_mult", 1.6))
        self.exit_hype_vol_mult: float = float(vibe_cfg.get("exit_hype_vol_mult", 1.0))
        self.state: VibeState = VibeState()

    def select(self, volatility: float, volume: float, avg_vol: float) -> Tuple[VibeMode, Dict[str, float]]:
        """Decide and apply mode using detect_mode plus hysteresis.

        Returns tuple (mode, feedback_dict).
        """
        # naive detection from controller
        detected = detect_mode(volatility, volume, avg_vol)

        # Hysteresis rules on top of detected
        current = self.state.mode

        # If currently CALM, be stricter to enter HYPE
        if current == VibeMode.CALM:
            enter_condition = (
                volatility >= 0.0 and volume > self.enter_hype_vol_mult * avg_vol
            )
            if detected == VibeMode.HYPE and enter_condition:
                new_mode = VibeMode.HYPE
            else:
                new_mode = VibeMode.CALM

        # If currently HYPE, be stricter to exit to CALM
        elif current == VibeMode.HYPE:
            exit_condition = volume < self.exit_hype_vol_mult * avg_vol and volatility <= 0.0
            # Allow detected CALM to force exit if volume and volatility indicate drop
            if detected == VibeMode.CALM and exit_condition:
                new_mode = VibeMode.CALM
            else:
                new_mode = VibeMode.HYPE

        else:
            new_mode = detected

        # Apply update
        self.state = update_vibe(self.state, new_mode)
        fb = vibe_feedback(self.state)
        return self.state.mode, fb

    def get_state(self) -> VibeState:
        """Return current VibeState."""
        return self.state


if __name__ == "__main__":
    # Tiny self-test
    cfg = {"vibe": {"enter_hype_vol_mult": 1.6, "exit_hype_vol_mult": 1.0}}
    ms = ModeSwitcher(cfg)
    # simulate low vol -> high vol
    for vol, avg in [(10, 10), (20, 10), (5, 10)]:
        mode, fb = ms.select(volatility=1.2, volume=vol, avg_vol=avg)
        print("input vol,avg:", vol, avg, "-> mode:", mode, "fb:", fb)