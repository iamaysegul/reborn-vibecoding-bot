"""
Daily loss limiter for Reborn VibeCoding Bot.
Monitors cumulative realized PnL and stops trading when daily total loss exceeds config threshold.
"""
import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict
import pandas as pd
import yaml

MEMORY_PATH = os.path.join(os.path.dirname(__file__), '../memory/daily_risk.json')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config.yaml')

# Load config
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
MAX_DAILY_LOSS_PCT = float(config.get('safety', {}).get('max_daily_loss_pct', 6))

# Internal state
_state = {
    'total_loss': 0.0,
    'total_gain': 0.0,
    'last_reset': None
}

def _load_state():
    global _state
    if os.path.exists(MEMORY_PATH):
        try:
            with open(MEMORY_PATH, 'r') as f:
                _state.update(json.load(f))
        except Exception:
            pass
    if not _state['last_reset'] or _is_new_day(_state['last_reset']):
        reset_daily_limits()

def _save_state():
    with open(MEMORY_PATH, 'w') as f:
        json.dump(_state, f)

def _is_new_day(last_reset: str) -> bool:
    if not last_reset:
        return True
    last = datetime.fromisoformat(last_reset)
    now = datetime.now(timezone.utc)
    return now.date() != last.date()

def reset_daily_limits() -> None:
    """Reset daily loss/gain counters at UTC midnight."""
    _state['total_loss'] = 0.0
    _state['total_gain'] = 0.0
    _state['last_reset'] = datetime.now(timezone.utc).isoformat()
    _save_state()
    logging.info("Daily risk limits reset.")

_load_state()

def check_loss_limit(current_pnl: float, total_pnl: float) -> bool:
    """Return True if daily loss limit exceeded; update totals."""
    _load_state()
    if current_pnl < 0:
        _state['total_loss'] += abs(current_pnl)
    else:
        _state['total_gain'] += current_pnl
    _save_state()
    # Calculate buffer
    buffer = MAX_DAILY_LOSS_PCT - _state['total_loss']
    if _state['total_loss'] >= MAX_DAILY_LOSS_PCT:
        logging.info(f"Daily loss limit ({MAX_DAILY_LOSS_PCT}%) exceeded. Trading stopped.")
        return True
    return False

def get_status() -> Dict[str, float]:
    """Get current totals and remaining buffer."""
    _load_state()
    buffer = MAX_DAILY_LOSS_PCT - _state['total_loss']
    return {
        'total_loss': _state['total_loss'],
        'total_gain': _state['total_gain'],
        'remaining_buffer': buffer
    }

# Example usage:
# from safety.risk_guard import check_loss_limit, get_status
# if check_loss_limit(current_pnl, total_pnl):
#     logger.warning("Daily loss limit reached — trading paused.")
#     break
