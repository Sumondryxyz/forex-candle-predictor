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

## Render deploy
- `render.yaml` (Blueprint) + `deploy/Dockerfile.render` (slim image:
  pandas/scikit-learn/flask/websockets/gunicorn only). The live app does NOT
  need lightgbm or pyquotex, so the image stays small; `requirements-web.txt`
  is that slim set. `pyquotex` imports are lazy (inside functions), so the
  webapp imports fine without it.
- Free plan: 750 shared instance-hours/month; spins down after 15 min without
  INBOUND traffic. This app is outbound-only, so it sleeps unless pinged.
  `.github/workflows/keep-awake.yml` pings `/api/health` every 5 min; it needs
  a repo secret `RENDER_URL`.
- Build verified with `sudo docker build -f deploy/Dockerfile.render .`; the
  container boots gunicorn on `$PORT` (default 10000) and serves live signals.

## Model tuning (measured 2026-10)
- `research_sweep.py` is the reproducible harness: `python research_sweep.py
  --pairs EURUSD,GBPUSD,USDJPY --intervals 5m,15m --days 60`. Compare a
  candidate change on the SAME data before shipping.
- Heterogeneous edge is the norm: EURUSD/GBPUSD 5m-15m carry a small edge,
  USDJPY 15m was a coin-flip. Never trust one pair/interval.
- Default `MIN_CONF` is 0.65, not 0.60: 5 of 6 combos are net-positive at 0.65
  vs 3 of 6 at 0.60. Fewer signals, better ROI.
- `engine._train_linear` isotonic-calibrates on a held-out tail (80/20) when
  history >= `MODEL_CALIB_MIN` (1500). Set `MODEL_CALIBRATE=0` to disable.
  Calibration helps overall but hurt GBPUSD 15m - expected, not a bug.
- scikit-learn >= 1.6 removed `CalibratedClassifierCV(cv="prefit")`; use
  `FrozenEstimator(base)`. `predictor/model.py` handles both.
- LightGBM (`train_backtest.py`) and the linear engine are close; the live app
  keeps linear because it is lighter and has no compile step.

## Paper-trading ledger
- `predictor/ledger.py` + a module-level `LEDGER` in `feed.py`. When a signal
  clears `MIN_CONF` and the next candle closes, `BaseFeed._ingest` appends one
  JSONL row (direction, confidence, entry/exit, win, pnl) to `LEDGER_PATH`.
- `GET /api/ledger` and the dashboard panel show win rate, ROI and a per-combo
  split. This is the only honest forward measure; walk-forward is a simulation.
- `Ledger.rows()` tolerates a truncated trailing line. Keep `LEDGER_PATH` on a
  mounted disk in production so redeploys do not wipe the record.

## Upgrade attempts that did NOT hold up (do not repeat blindly)
Measured out-of-sample on 60 days of EURUSD/GBPUSD/USDJPY 5m+15m. Every idea
below looked promising in-sample and then failed the honest test:

- **Ensemble (linear + calibrated GBM, `predictor/advanced.py`)**: total ROI at
  0.65 was ~0 vs ~+21% for plain linear. Keep the module for experiments
  (`research_sweep.py --backend ensemble`) but do NOT make it the live default.
- **Per-pair auto-threshold**: choosing a threshold on the first 60% of signals
  and testing on the last 40% gave +49% vs +63% for a fixed 0.65. It overfits.
- **Train-window size**: 1200/2550/4000 candles; the huge value at 4000 came from
  1-4 signals on one combo. Noise, not edge.
- **Feature pruning by |correlation|**: all-features and every top-k subset were
  negative on the evaluation tail. The full 35-feature set is fine.
- **Time-bagging** (average probs over several rolling windows): the apparent
  gain came from 8-12 USDJPY signals; EURUSD 5m got worse. Noise.

Lesson: FX next-candle edge is tiny. Guard every change by selecting on one time
slice and measuring on a LATER, untouched slice; small signal counts lie.
`engine.walk_forward_raw()` exists exactly for that split-and-measure workflow.

