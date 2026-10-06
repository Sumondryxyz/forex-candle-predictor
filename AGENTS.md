# AGENTS.md

Repo-specific notes for future sessions.

## What this is
Next-candle UP/DOWN signal bot for FX, exposed as a Flask web app. It only
prints/predicts signals with a confidence and a WIN/LOSS check — it never
places trades. Live data comes from an unofficial TradingView chart websocket
(default) or Quotex (blocked by Cloudflare from datacenter IPs).

## Environment
- Reinstalling does not persist: the runtime env resets and wipes pip packages.
  If the app dies with `ModuleNotFoundError: No module named 'flask'`, run
  `pip install -r requirements.txt` and start again.
- Tests: `python -m pytest -q` (needs `pyquotex` from
  `git+https://github.com/cleitonleonel/pyquotex.git` for one test).
- Lint: `python -m ruff check .`
- Run the app: `python -m predictor.webapp` (env `PORT`, `HOST`; default 5000).

## Model
- `predictor/engine.py` has two backends behind one API (`features`, `train`,
  `walk_forward`, model `.prob`):
  - **linear** (default when pandas + scikit-learn import): 35 causal features +
    regularised logistic regression (`MODEL_C`, default 0.1).
  - **numpy** (Termux fallback): 22 causal features + bagged logistic ensemble.
- Measured finding: gradient-boosted trees overfit 1-minute FX and sit at ~51%;
  the linear model is stronger once filtered on confidence. Don't "fix" the
  engine by swapping in LightGBM/HGB — it makes honest accuracy worse.
- `walk_forward` uses a rolling window (not expanding) and returns a `sweep`
  table across confidence floors; the dashboard renders it.
- 1-minute direction is near coin-flip; the edge lives at confidence >= 0.60 on
  5m/15m. Typical honest numbers: ~55-58%.

## Sources
- `predictor/sources/tradingview.py` is an unofficial chart websocket
  (`wss://data.tradingview.com/...`), not a documented data API. Be honest about
  this when asked; it can change without notice and is outside TradingView ToS.
- The live feed caches history in `predictor/feed.py` (`MAX_CANDLES = 6000`).

## Keeping it running
- No init/supervisor here (PID 1 is `openhands-agent`, `systemctl` offline,
  no cron). Processes die on every environment recycle.
- Sandbox: `./scripts/serve_tmux.sh` runs `scripts/run_server.sh` (a watchdog
  that reinstalls deps and restarts the app) inside tmux.
- Real host: `deploy/forex-predictor.service` (systemd), `deploy/Dockerfile` +
  `deploy/docker-compose.yml` (restart: unless-stopped).
- Keep exactly one worker in any WSGI setup: feed state is in-process.
