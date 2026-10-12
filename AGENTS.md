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
- Default `MIN_CONF` is **0.70** (raised from 0.65 in 2026-10): on 30/45/90-day
  windows the average hit rate at 0.70 beat 0.65 in *every* window (58.1% vs
  52.9% on 30 days; 59.3% vs 57.0% on 90 days) with net-positive ROI throughout.
  Fewer signals, better accuracy. Re-measure with `research_sweep.py --min-conf`.
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
- **Longer prediction horizon** (target `close[t+3]` instead of `close[t+1]`):
  trained AND scored on the same k it averaged 57.0% vs 57.9% for k=1, with
  USDJPY 5m collapsing to 47.5%. An ad-hoc harness showed a fake gain because it
  trained via `engine.labels()` (1-step) but scored with a hand-written 3-step
  label — trainer and evaluator MUST share the label definition. Rejected.

Lesson: FX next-candle edge is tiny. Guard every change by selecting on one time
slice and measuring on a LATER, untouched slice; small signal counts lie.
`engine.walk_forward_raw()` exists exactly for that split-and-measure workflow.

### "Human-like" analysis — measured, rejected as a model change
Prompted by a request to teach the model to read charts like a human. Measured
pooled over all out-of-sample folds, 90 days, 6 pairs, 5m+15m (honest):

- **Human feature block** (pin bar, engulfing, liquidity sweep, inside/outside
  bar, slope, position-in-range, round-number distance): at floor 0.70 it scored
  56.4% vs 57.1% base, and at 0.75 58.2% vs 58.6%. A first-half/second-half
  split made it look like +2-3 points, but the pooled all-fold number is flat to
  negative -> in-sample fluke. NOT shipped as features.
- **Human filters on the base model** (trend agreement, HTF agreement, active
  session, RSI not extended): none beat plain `all`. `trend_agree` HURT at 0.70
  (56.9% vs 57.1%); the others were within noise. "Trade with the trend" is not
  an edge on 1m/5m FX next-candle.
- **Cross-asset training** (one global model on all pairs + pair one-hot vs a
  per-pair model, 80/20 time split): contradictory — 5m favoured per-pair, 15m
  favoured global. Not a stable, shippable gain.

The high thresholds were the real lever: base35 pooled hit rises 57.1% (0.70)
-> 58.6% (0.75) -> 59-60% (~0.85), so this became a live user control instead.

## Live confidence floor (`POST /api/min-conf`)
- `engine.summarise(raw, min_conf, payout)` re-thresholds cached raw walk-forward
  arrays; `BaseFeed.set_min_conf()` calls it and stores the new floor. No
  retrain, no redeploy — instant.
- Dashboard has a "Min confidence" selector (0.65-0.90); changing it re-renders
  the signal at the new floor. The 0.85+ range is where the honest hit rate is
  ~59-60%, at the cost of fewer signals.

## Paper-trade ledger integrity (the fake-97% bug)
- `BaseFeed._ingest` resolves a pending signal only when the resolving candle
  is (a) exactly `period` after the signal candle AND (b) within
  `10*period` of wall-clock now. Quotex once served a *contiguous* 2023 series
  as "history"; step == 60s passed the gap check, so the whole replay got
  recorded as resolved signals and the ledger showed a fake ~97% win rate
  (EURUSD 1m: 32 trades, 31 wins). Historical/replayed candles must never count.
- `Ledger` de-duplicates by `(pair, interval, candle_time)` on both write and
  read, so a re-ingested candle cannot inflate the count. It uses an `RLock`
  because `record()` re-enters `rows()` via `_load_keys()`.
- A clean forward ledger (only genuinely live candles) is the ONE number that
  proves or disproves the edge. Expect it to sit near the honest ~55-60%
  walk-forward, not 97%.
- Regression tests: `tests/test_quotex_live.py::test_stale_contiguous_series_does_not_resolve`,
  `::test_gapped_candle_does_not_resolve_the_signal`,
  `tests/test_ledger.py::test_same_candle_is_recorded_only_once`
  `::test_duplicate_rows_already_in_the_file_are_collapsed`.

## TradingView volume — wired but off by default (`USE_VOLUME`)
- TradingView DOES report per-bar volume (nonzero "tick volume", ~600-1600 on
  EURUSD 5m). `predictor/sources/tradingview.py` already parsed it.
- It was silently dropped: `engine.arrays()` never returned volume and
  `engine._features_pandas` hard-coded `volume=zeros`. Now `engine.volumes()`
  extracts it and `engine.features(o,h,low,c,t,v)` threads it through.
  `BaseFeed.use_volume = _env_flag("USE_VOLUME")` gates it; default OFF.
- Honest A/B (90d, 6 pairs, 5m+15m, pooled out-of-sample): rich volume block
  (vol_z, vol_trend, vol_pressure buy/sell, vol_climax, vol_efficiency) looked
  +2.6pt at 0.70, but the early/late split disagreed (early HIGH floors lost,
  late won) and per-combo on the late half across 5 intervals it was
  **12 wins / 14 losses / 4 ties** at 0.75. Small-n combos (n=1-10) swung the
  pooled number. Verdict: tick volume is NOT a reliable edge here. Shipped as
  opt-in only; do not flip the default without a fresh, broad confirmation.


## Phone login → session capture (no PC)
- Quotex's login page is what Cloudflare blocks from datacenter IPs. The phone
  (residential IP) does the login; a userscript forwards the session.
- `scripts/session_capture.py` serves a loopback-only page (`127.0.0.1:8787`):
  `/` (instructions), `/capture.user.js`, `POST /save`, `GET /status`.
- `tools/quotex-session-capture.user.js` (Kiwi + Tampermonkey) reads the token
  from `/api/v1/cabinets/digest` and cookies with `GM_cookie` — `document.cookie`
  cannot see the HttpOnly `laravel_session` / `cf_clearance`, so GM_cookie is
  required, not optional. It polls because Quotex is a SPA (login has no reload).
- Pure logic lives in `predictor/session_capture.py` (normalize/build/save), so it
  is unit-tested without a browser. The written layout matches what
  `predictor/quotex_session.load_session` already reads (flat or email-keyed).
- Never commit `session.json`; it holds a bearer token. It is gitignored.

## Using it for real money (read this first)
- The engine NEVER places a trade and there is no broker-execution path. That is
  deliberate: it is a signal + risk tool, not an autotrader. Do not add one.
- The only number that decides real-money sizing is whether the HONEST
  walk-forward win rate clears the broker's break-even `1 / (1 + payout)`
  (55.6% at 80% payout). Confidence alone is not an edge; a 54% signal loses
  money over time no matter how confident it feels.
- `predictor/risk.py` is that arithmetic: `break_even_rate`, `expected_value`,
  `kelly_fraction`, `suggest_stake`, `assess`, and `simulate_bankroll` (replays
  the paper ledger as a compounding what-if). Stake is quarter-Kelly, hard-capped
  at 2% of bankroll, and 0 whenever the edge is <= 0.
- Surfaces: `/risk [PAIR] [INTERVAL] [BANKROLL]` (Telegram), `/api/risk` (web,
  includes the bankroll replay), and the "Risk & stake" card.
- Discipline for live use: forward-test on the paper ledger for weeks first;
  only act when `verdict == positive-edge`; never chase a losing combo; the FX
  edge (~55-58%) is thin, so treat any single result as noise.

## Reading the signal (how it is meant to be used)
- The honest edge is small (~55-58%). Do not advertise more. `format_signal`
  and `/signal` print the confidence floor next to every call for this reason.
- Trust a call only when `tradeable` is true (confidence >= `MIN_CONF`, 0.70).
  Most of the time every pair is below the floor; that is the correct behaviour,
  not a bug.
- `/smart` (Telegram) ranks pairs by their honest walk-forward rate at the
  current floor and names the best one. This is the intended "intelligence":
  stop chasing pairs that are noise, follow the one with a measured edge.

## Rejected ideas (measured, do not re-try blind)
- **Extra context features on the linear backend** (volatility regime, band
  width, efficiency ratio, longer-horizon returns, return autocorrelation):
  A/B via `research_context_features.py` was 5 wins / 3 losses at floor 0.55 and
  3/3 at 0.70 across 8 pair/intervals. Not a win — leave the feature set alone.
- **Per-pair adaptive confidence floor** (pick the best floor on an early
  out-of-sample slice, apply it to a later slice): `research_adaptive_floor.py`
  was 6 wins / 10 losses / 1 tie vs the fixed 0.70 floor. The chosen floor does
  not transfer; keep the fixed floor.
- Lesson: on FX the honest edge is fragile. Always A/B with the harness on the
  SAME data (and both halves of the walk-forward) before shipping "smarter".

## Getting work onto GitHub (the sandbox token is read-only)
- The injected GITHUB_TOKEN is a `ghu_` GitHub App user token with read-only
  access: `git push` returns 403 and the API returns "Resource not accessible by
  integration", even though the repo object reports `push: true`. Do not rely on
  it to push.
- Distribution path for a phone-only user: export the branch as a `.patch`
  (`git format-patch --stdout origin/main..HEAD`) and a `.bundle`
  (`git bundle create x.bundle refs/heads/<branch> --not refs/remotes/origin/main`),
  serve them, and let the user push with their own credentials.
