from safety.risk_guard import check_loss_limit, get_status, reset_daily_limits
from vibe.mode_switcher import ModeSwitcher
from vibe.vibe_controller import VibeMode, VibeState, vibe_feedback
import yaml

# Load config and instantiate ModeSwitcher with cfg
with open("config.yaml", "r") as _f:
	cfg = yaml.safe_load(_f)

# Vibe object (keeps state: streaks/switch_count)
mode = ModeSwitcher(cfg)


def vibe_loop() -> None:
	# 1) gün başı reset (UTC)
	reset_daily_limits()

	# 2) örnek PnL değerleri (dry mode / test için)
	current_pnl = 0.0     # o anki gerçekleşen PnL delta (emir kapanınca update et)
	total_pnl = 0.0       # gün içi kümülatif gerçekleşen PnL (trade log’dan)

	if check_loss_limit(current_pnl, total_pnl):
		print("[RiskGuard] Günlük %6 kayıp eşiği aşıldı, trading durdu.")
		return

	# --- Vibe integration: compute volatility & volume from incoming candle ---
	# Simulated incoming candle (replace with real feed in integration)
	candle = {
		"timestamp": 0,
		"open": 100.0,
		"high": 101.2,
		"low": 99.8,
		"close": 100.5,
		"volume": 1200.0,
	}

	# basit volatilite tahmini: (high - low) / max(close, 1e-12)
	volatility = (candle["high"] - candle["low"]) / max(candle["close"], 1e-12)
	volume = candle["volume"]
	# geçici avg_vol; ileride core.indicators.vol_ma ile 20-bar ortalama bağlanacak
	avg_vol = max(volume, 1.0)

	selected_mode, feedback = mode.select(volatility=volatility, volume=volume, avg_vol=avg_vol)

	# Support both key names: 'risk_mult' (requested) and 'risk_multiplier' (vibe feedback)
	risk_mult = float(feedback.get("risk_mult", feedback.get("risk_multiplier", 1.0)))
	cooldown = feedback.get("cooldown_s", 0.0)
	aggressiveness = feedback.get("aggressiveness", 0.5)

	# Example: apply risk multiplier to base risk_per_trade from config
	base_risk = float(cfg.get("risk_per_trade", 0.01))
	risk_per_trade = base_risk * risk_mult

	print(f"[Vibe] mode={selected_mode.value} feedback={feedback}")
	print(f"[Risk] base_risk={base_risk} risk_mult={risk_mult} => risk_per_trade={risk_per_trade}")


if __name__ == "__main__":
	vibe_loop()