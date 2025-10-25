## Reborn — Copilot instructions (concise)

Purpose: give an AI assistant the minimal, actionable context needed to work productively on this repo.

- Quick start (dry/dev):
  - Create a virtualenv and install deps:

    python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

  - Run the main loop in dry mode (no API keys required):

    python main.py

- Big-picture architecture (see `Reborn_Codex_Context.md`):
  - core/: indicators, strategy, order_manager, risk — the trading logic and execution.
  - vibe/: mode & score controller (Calm / Hype) that tweaks signal strictness.
  - safety/: risk guard and stop-trigger enforcing daily-loss, cooldowns, funding filters.
  - memory/: persistent runtime state (trades.json, state.json).
  - `main.py` wires data feed → strategy → risk guard → order manager → vibe UI.

- Key integration points and external deps:
  - Exchange: `ccxt.binanceusdm` (futures, defaultType: future). API keys live in `config.yaml`.
  - Libraries: see `requirements.txt` (pandas, numpy, ccxt, pyyaml, websockets, etc.).

- Project-specific conventions to follow (do not assume generic defaults):
  - Async-first: use `async/await` for I/O (exchange, websockets, data fetchers).
  - Centralized config: NO magic numbers — put defaults and overrides in `config.yaml`.
  - Rounding: always round prices/quantities to `price_step` / `qty_step` from config.
  - Exit orders must be `reduceOnly` and set as limit (TPs) or stop-market (SL).
  - Dry mode fallback: code paths should run without API keys (use dummy data feed).

- Small, concrete examples the assistant can use immediately:
  - Signal shape returned by strategy: `{"side":"long","entry":float,"atr":float,"timestamp":int}`
  - Position sizing (described in `Reborn_Codex_Context.md`):
    - qty_by_risk = (balance * risk_pct) / (entry - stop)
    - max_qty_by_margin = (balance * leverage * max_margin_frac) / entry
    - qty = min(qty_by_risk, max_qty_by_margin) rounded to `qty_step`

- Safety rules to enforce in code changes:
  - Hard stop: block new trades when daily loss ≥ `max_daily_loss` (config.yaml).
  - Loss-streak cooldown: after 3 losing trades, enter cooldown (config-driven).
  - Funding window: avoid new entries within ±N minutes of funding timestamps (configurable).

- Files to read first (high signal-to-noise):
  - `Reborn_Codex_Context.md` — explicit strategy, acceptance criteria, and module list.
  - `config.sample.yaml` / `config.yaml` — canonical runtime parameters.
  - `main.py` — top-level wiring and loop (dry-run friendly).
  - `requirements.txt` — runtime dependencies to consider when coding/running.

- Typical small tasks and how to validate them locally:
  - Add an indicator → run `python main.py` in dry mode and confirm the loop doesn't crash.
  - Update order manager → dry run: ensure it logs the intended order payload and writes to `memory/trades.json`.
  - Add backtest CLI → run `python -m core.backtest --csv <file> --config config.yaml` and check metrics output.

- What not to change without discussion:
  - Trading safety checks (daily hard stop, reduceOnly order flags, slippage limits).
  - Core signal acceptance criteria in `core/strategy.py` unless proposing tested improvements.

If any of the above is unclear or you want me to include additional examples (unit tests, a backtest harness, or a sample `core/` function), tell me which area to expand and I'll iterate.
