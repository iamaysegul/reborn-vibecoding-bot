# Reborn – Codex Agent Context (VibeCoding Futures Bot)

**Primary Goal:** Build a 1-minute futures breakout bot (EMA9/21 + AVWAP + RSI + Bollinger + Volume) with strict risk and safety rules.  
**Style:** Modular, async-first, VibeCoding (Calm/Hype modes).  
**Exchange:** Binance USD-M (perpetual), isolated margin, configurable leverage.  
**Critical Constraint:** Daily total loss < **2%** (hard stop).

---

## 0) Repo & Environment
- **Repo name:** `reborn-vibecoding-bot`
- **Python:** 3.11+
- **Packages:** `pandas, numpy, ccxt, pyyaml, websockets`
- **Structure (must keep):**
```
reborn/
 ├── core/
 │   ├── data_feed.py
 │   ├── indicators.py
 │   ├── strategy.py
 │   ├── risk.py
 │   └── order_manager.py
 ├── vibe/
 │   ├── vibe_controller.py
 │   ├── mode_switcher.py
 │   └── energy_logger.py
 ├── safety/
 │   ├── risk_guard.py
 │   └── stop_trigger.py
 ├── interface/
 │   └── vibe_console.py
 ├── memory/
 │   ├── trades.json
 │   └── state.json
 ├── config.yaml
 ├── requirements.txt
 └── main.py
```

**Always remember:**  
> Reborn operates on 1-minute candles. Detect breakouts using EMA9/21 + volume surge. Respect daily loss < 2% and leverage/margin rules at all times.

---

## 1) Global Coding Standards
- Use **async/await** where I/O bound (exchange calls, websockets).
- Add type hints and minimal docstrings for public functions.
- Centralize parameters in `config.yaml` (no magic numbers in code).
- Respect tick/lot size, rounding (`price_step`, `qty_step`).
- Every order is logged; all exits use **reduceOnly**.
- Fail-safe defaults (if config missing, run in **dry mode**).

---

## 2) Indicators & Signal – Acceptance Criteria
**File:** `core/strategy.py` (+ `core/indicators.py`)  
**Signal (LONG) must return**: `{"side":"long","entry":float,"atr":float,"timestamp":int}`

Rules:
1. **Breakout:** close > recent high of last **N=5** highs (configurable).
2. **Trend:** close > EMA9 > EMA21.
3. **Bollinger squeeze:** `bandwidth < 0.06` (configurable; relax in Hype mode).
4. **Volume surge:** `volume > vol_ma * 1.5` (configurable).
5. **RSI band:** 50 ≤ RSI ≤ 70 (configurable).
6. **AVWAP:** close ≥ AVWAP.
7. Vibe Mode:
   - **Calm:** apply stricter volume condition (`*1.1` extra).
   - **Hype:** allow skipping BB squeeze check.

Edge Cases:
- Compute indicators with a rolling window; drop NaN rows.
- Return `None` if data is insufficient or conditions fail.

---

## 3) Position Sizing & Levels – Acceptance Criteria
**File:** `core/risk.py`

- `position_size(entry, stop, balance, risk_pct, leverage, price_step, qty_step, max_margin_frac=0.8)`  
  - **qty_by_risk = (balance*risk_pct) / (entry-stop)**  
  - **max_qty_by_margin = (balance*leverage*max_margin_frac)/entry**  
  - **qty = min(qty_by_risk, max_qty_by_margin)** → round down to `qty_step`  
  - Round entry to `price_step`  
  - Return `qty, entry_rounded`
- `sl_tp_levels(entry, atr, sl_atr_mult)`  
  - **SL = entry - max(atr*mult, entry*0.004)**  
  - **R = entry - SL** → **TP1/TP2/TP3 = entry + 1R/2R/3R**

---

## 4) Risk Guard & Safety – Acceptance Criteria
**File:** `safety/risk_guard.py`

- Daily PnL tracking; block trading when `(day_start_balance - balance)/day_start_balance >= max_daily_loss`.
- Loss streak control: 3 consecutive losing trades ⇒ cooldown period (configurable).
- Funding filter: block new entries within **±10 min** of funding times (configurable).
- Provide `is_safe_to_trade(candle)` returning bool.

---

## 5) Order Manager (Binance USD-M) – Acceptance Criteria
**File:** `core/order_manager.py`

- Connect with `ccxt.binanceusdm` using API keys from `config.yaml`, defaultType **future**.
- **Set leverage** per symbol on connect; prefer **isolated** margin.
- On execute(signal):
  - Compute **SL/TP1/TP2/TP3** via `risk.sl_tp_levels`.
  - Compute **qty** via `risk.position_size` with leverage and steps.
  - Place **market entry** (buy).  
  - Place **reduceOnly** TP1/TP2 **limit** orders and **stop-market SL** (reduceOnly).  
  - Print/Log a concise summary (entry/qty/sl/tps).
- Handle errors and run in **dry mode** if ccxt not installed or API missing.

---

## 6) Data Feed – Acceptance Criteria
**File:** `core/data_feed.py`

- Provide `get_latest_candle()` returning dict with keys: `timestamp, open, high, low, close, volume` (ints/floats).
- If ccxt is available: fetch last 2 OHLCV candles each second (or websocket in future).
- If not: run dummy generator (dry mode) so the loop can run.

---

## 7) Vibe Layer – Acceptance Criteria
**Files:** `vibe/vibe_controller.py`, `vibe/mode_switcher.py`

- `VibeController.score(candle)` returns 0–100 based on simple range + volume.
- `ModeSwitcher.select(score)` returns `"Calm"` or `"Hype"` with hysteresis (keep last mode).

---

## 8) Vibe Loop (Main) – Acceptance Criteria
**File:** `main.py`

Pseudocode:
```python
cfg = load_config()
data = DataFeed(cfg); strat = BreakoutStrategy(cfg)
om = OrderManager(cfg); guard = RiskGuard(cfg)
vibe = VibeController(cfg); mode = ModeSwitcher(cfg)
ui = VibeConsole(cfg)

await data.connect(); await om.connect()
while True:
    candle = await data.get_latest_candle()
    if candle is None: await sleep(0.2); continue
    vibe_score = vibe.score(candle)
    mode_name = mode.select(vibe_score)
    signal = strat.generate_signal(candle, vibe_score, mode_name)
    if signal and guard.is_safe_to_trade(candle):
        await om.execute(signal)
    ui.render(vibe_score, mode_name, signal)
    await sleep(1)
```
- Must not crash if any module returns `None`; log and continue.

---

## 9) Backtest Module (New) – Acceptance Criteria
**File:** `core/backtest.py` (new)

- Load CSV with columns: `timestamp,open,high,low,close,volume`.
- Run through strategy rules and risk sizing as if live.
- Metrics output (JSON/print): trades count, winrate, profit factor, max drawdown, total PnL, equity curve.
- Reuse indicators from `core/indicators.py` to avoid drift.

**CLI Example:**
```
python -m core.backtest --csv data/AVAX_1m.csv --config config.yaml
```

---

## 10) Config – Acceptance Criteria
**File:** `config.yaml`

Keys (examples):  
```
exchange: binanceusdm
api_key: "YOUR_API_KEY"
api_secret: "YOUR_API_SECRET"
symbol: "AVAX/USDT"
timeframe: "1m"
leverage: 5
margin_mode: "isolated"
risk_per_trade: 0.01
max_daily_loss: 0.02
atr_period: 14
ema_fast: 9
ema_slow: 21
bb_period: 20
bb_std: 2.0
rsi_period: 14
rsi_min: 50.0
rsi_max: 70.0
vol_ma_period: 20
vol_multiplier: 1.5
bb_bw_threshold: 0.06
funding_window_minutes: 10
sl_atr_mult: 1.0
slippage_limit: 0.001
liq_buffer_multiple: 3.0
session_tz: "UTC"
price_step: 0.01
qty_step: 0.1
```

---

## 11) Test Plan – Acceptance Criteria
- Run `python main.py` in **dry mode** (no API).
- With test API keys: a full cycle entry→TP/SL works (reduceOnly orders placed).
- Backtest on sample CSV → show non-zero trades, sensible metrics, no crashes on NaNs.

---

## 12) Prompts You Can Paste to Codex

### A) Order Manager (Futures)
```
Update core/order_manager.py to place Binance USD-M futures orders with ccxt:
- set leverage on connect, isolated margin if supported
- market entry for LONG signals
- reduceOnly TP1/TP2 limit orders, stop-market SL reduceOnly
- use risk.position_size & risk.sl_tp_levels
- handle dry mode when ccxt or keys missing
- print a one-line trade summary after execution
```

### B) Risk Guard (Funding + Cooldown)
```
Enhance safety/risk_guard.py:
- track daily start balance vs. current balance
- block trading if daily loss >= max_daily_loss
- implement 3-loss streak cooldown with timer
- add funding window filter: block ±N minutes around funding
- expose is_safe_to_trade(candle) -> bool
```

### C) Backtest Module
```
Create core/backtest.py:
- load CSV (timestamp,open,high,low,close,volume)
- reuse indicators & strategy to simulate trades
- compute metrics: winrate, PF, maxDD, total PnL, equity curve
- CLI: python -m core.backtest --csv data.csv --config config.yaml
```

---

## 13) Done Definition
- Code passes a dry run locally (no API).
- With test keys: a full cycle entry→TP/SL works (reduceOnly orders placed).
- Backtest prints metrics without errors.
- Daily loss hard stop works; funding window enforced.

Good luck, Codex. Build Reborn with care. 🔥
