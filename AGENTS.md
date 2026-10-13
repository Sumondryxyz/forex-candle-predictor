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
- **429 on pair bursts (fixed 2026-10).** Each feed opens one websocket per
  poll, so every pair adds to the handshake rate and TradingView starts
  answering HTTP 429 — with ten pairs about half the feeds failed to connect.
  `tradingview.py` now paces handshakes process-wide (`_pace_connect`, default
  0.6s gap, `TV_CONNECT_GAP`) and retries a 429 with exponential backoff
  (`TV_RATE_RETRIES`, default 6). Ten concurrent pairs all connect now. If more
  pairs are added, raise `TV_CONNECT_GAP` rather than removing the pacing.

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
- **Pair screen (measured 2026-10, 30/45/~53-day windows, payout 0.80).** The
  default set is the six majors: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD.
  EURJPY/GBPJPY/CADJPY/NZDCAD were tried and then **reverted** — they connect but
  add load without a proven edge, and on a free tier the extra feeds risk the
  memory limit. TradingView history caps at ~5100 candles, so "90 days" is
  really ~53. Break-even at 0.80 is 55.56%. **No pair cleared it across
  windows:** CADJPY/NZDCAD/AUDUSD/GBPJPY 15m looked good on one window and fell
  to 47-53% on another; USDCHF 15m was below break-even in every window
  (51.9/47.2/53.5%). The apparent winners do not survive `readiness.assess`
  either (Wilson lower bound below break-even, p 0.12-0.36). Treat all of these
  as forward-test candidates, not proven edges — and remember payout, not pair
  choice, moves break-even the most (0.90 -> 52.6%).
- Default `MIN_CONF` is **0.70** (raised from 0.65 in 2026-10): on 30/45/90-day
  windows the average hit rate at 0.70 beat 0.65 in *every* window (58.1% vs
  52.9% on 30 days; 59.3% vs 57.0% on 90 days) with net-positive ROI throughout.
  Fewer signals, better accuracy. Re-measure with `research_sweep.py --min-conf`.
- **Cross-pair confirmation (`CONSENSUS=1`, off by default).** A signal only
  counts as tradeable when a second model, trained on *every* live pair of the
  same interval, predicts the same direction. Measured 15m, 6 majors, ~53 days,
  rolling walk-forward: hit rate at 0.70 rose **55.2% -> 57.8%** and improved
  *every* pair; 348 signals became 277. On 5m the same filter changed nothing
  (there is no 5m edge to confirm — 46.9% alone), which is the honest reason it
  is opt-in and documented as 15m-only. The pooled model is cached
  (`CONSENSUS_RETRAIN`, default 300s) and only trains when >= 2 pairs are live,
  so a single-pair deploy is unaffected. See `predictor/consensus.py`.
- **Tried and rejected: the ensemble backend (`advanced.py`, linear+GBM blend).**
  On the same 15m walk-forward it *lost* to the plain linear model (52.4% vs
  56.1% at 0.70) — the GBM overfits FX noise, as the engine docstring warns.
  Do not switch the live model to it without a fresh A/B.
- **Honest A/B rule (learned the hard way).** Several upgrades look great on the
  full out-of-sample stream and vanish under a split-half test: pick the rule on
  the early half, measure on the late half. Measured on 15m, 6 majors, ~53 days:
  | candidate | full-sample | split-half (late) | verdict |
  |---|---|---|---|
  | regime gate (htf_align / ema_cross) | separates 48% vs 58% | -0.4 to 0.0 pp | noise |
  | meta-labeling (confidence + features) | +0.0 pp | +0.0 pp | noise |
  | session/hour gate | London 65.9% vs Asia 51.5% | no hour survives | noise |
  | training window < default | 53.0% vs 59.9% | default already best | no change |
  | consensus blend 50/50 | 58.3% | 55.6% | no better than binary |
  Always run the split-half check before shipping a filter; only cross-pair
  consensus survived it. `research_consensus.py` reproduces the consensus A/B.
- **Consensus is real but not yet *significant*.** 15m, 6 majors: baseline
  55.2% -> confirmed 57.8% (n=277), and it reproduced on a fresh window
  (55.5% -> 58.2%, n=273). But the Wilson 95% lower bound is ~51% and the
  binomial p-value vs break-even 55.56% is ~0.32, so the edge is *not* proven —
  keep it on the paper ledger, do not treat it as a green light.
- **Timeframe is the biggest lever, and it was hiding a real edge.** Same model,
  same 6 majors, walk-forward, payout 0.80 (break-even 55.56%):
  | tf | baseline rate | Wilson lo | p | verdict |
  |---|---|---|---|---|
  | 5m | 47.1% | — | — | no edge at all |
  | 15m | 56.1% | 51.4% | 0.32 | not significant |
  | 30m | 59.1% | — | — | significant |
  | 1h | 63.3% | 59.8% | <0.001 | **proven edge** |
  1h holds up under the split-half test (late half: 65.7% at 0.70, Wilson lo
  59.8%, p<0.001) and 5 of 6 pairs clear break-even individually (USDCHF 72.3%,
  GBPUSD 65.4%, USDCAD 67.0%, AUDUSD 62.7%, USDJPY 58.4%; EURUSD n=17 too
  small). The confidence floor is flat from 0.65 to 0.85 (~64%), so the exact
  floor matters far less on 1h than on 5m. **Prefer 30m/1h for "only good
  signals"; the 5m signal has no measurable edge and should not be trusted.**
  Defaults now offer 1m,5m,15m,30m,1h, and `MIN_HISTORY_CANDLES` (3000) keeps
  30m/1h from being starved by a 72h `HISTORY_HOURS`.
- **The 1h edge is confirmed on a large sample.** `research_accuracy.py` (the
  scoreboard: hit rate + Wilson lower bound + binomial p-value per pair) over
  250 days, all six majors, floor 0.70: **ALL n=963, 66.8%, Wilson lo 63.7%,
  p<0.001 -> EDGE.** Four pairs individually significant (GBPUSD 68.7%,
  USDCHF 73.6%, USDCAD 75.0%, USDJPY 63.0%). The same report over 120 days shows
  5m 53.9% (Wilson lo 50.3%, p=0.84) and 15m 55.2% (Wilson lo 50.3%, p=0.58):
  both below break-even's lower bound, i.e. no usable edge. Run
  `python research_accuracy.py --intervals 1h --days 250` to reproduce.
- **1m and 5m are not tradeable, at any confidence floor.** 120 days, six
  majors, every floor from 0.60 to 0.80: 1m peaks at 55.5% (Wilson lo 52.2%,
  p=0.53) and 5m at 53.3% (Wilson lo 50.2%, p=0.93). Neither lower bound ever
  clears break-even 55.56%, and the rates barely move with the floor -- the
  model has no repeatable edge at these horizons, so a signal there is a coin
  flip minus the spread. If asked "can I trade 1m/5m?", the honest answer is no:
  steer to 30m/1h. (The payout break-even of 55.56% is also highest exactly
  where the edge is weakest.)
- **Confidence only means something on 1h.** Splitting the walk-forward into
  confidence bands (250d for 1h, 120d for the rest, six majors) shows the
  confidence number is *not* calibrated on the fast timeframes -- a "55%+" 1m
  signal is a coin flip:
  | tf | conf 50-60% | conf 60-70% | conf 70%+ |
  |---|---|---|---|
  | 1m | 50.6% | 51.7% | 51.2% |
  | 5m | 49.7% | 53.5% | 51.3% |
  | 15m | 51.5% | 50.0% | 55.7% |
  | 1h | 52.4% | 57.8% | **67.4% (Wilson lo 64.3%, p<0.001)** |
  So "trust the high-confidence signals" is good advice, but only once the
  timeframe is 1h -- there the 70%+ band is the proven edge. On 1m/5m a high
  confidence is just overconfidence in noise; do not read a demo balance going
  up as proof (a day of coin-flip trades is variance, not edge).
- **Nothing rescues 1m/5m — and one attempt was a leakage trap.** Tried, all
  honest split-half/walk-forward, six majors: higher confidence (already shown
  flat), multi-timeframe context, volatility filter, hour-of-day filter, and
  pooling all six pairs' 1m history into one model. None cleared break-even;
  best honest results were 54.0% (1m low-vol, Wilson lo 48.1%) and 51.9%
  (5m + 30m context, Wilson lo 48.2%). Base rates agree: on 1m a candle closes
  above its open only 45.7% of the time, so even "always UP" loses.
  The trap: an MTF run *without* a close-time guard scored 68% on 1m -- a 17pp
  jump that was pure lookahead, because a 15m candle's close is a future price
  while the 1m candle is still forming. `research_mtf.py` aligns to the last
  *fully closed* slower candle and the number collapses to ~49%. Rule of thumb:
  a suspiciously large jump from adding a feature is leakage until proven
  otherwise.
- The hard limit on 1m is data: TradingView caps history at ~5100 candles, so
  1m sees only ~5.5 days (5m ~26 days, 15m ~75 days). A model cannot learn a
  durable edge from a few days of one regime. This is why the edge only appears
  once the timeframe is slow enough (30m/1h) to get months of history.
  `predictor/sources/dukascopy.py` lifts the cap: free per-day 1m files going
  back years (LZMA, 24-byte big-endian records), reachable from datacenter IPs
  though throttled to ~0.5-1 request/s. `research_fetch_duka.py` caches months
  to a pickle and `research_duka_accuracy.py` scores 1m/5m/15m/1h on the same
  resampled prices, so the fast timeframes can finally be tested with real
  history instead of a 5.5-day sample.
- **The "5100 candle cap" was ours, not TradingView's.** `MAX_BARS` (5000) was
  truncating every request; TradingView actually serves ~9.5k 1m bars (8.7 days)
  and far more on slower timeframes. Raised `MAX_BARS` to 10000 (env
  `TV_MAX_BARS`) and `CHUNK` from 300 to 1000 (env `TV_CHUNK`): a 9k-bar pull
  went from ~2 minutes to ~2 seconds, and 1h now fetches 10000 bars (~587 days)
  instead of 5000. The paging loop also stops after 3 rounds with no new bars,
  so it no longer burns 15s recv-timeouts after history runs out.
- **More 1h history made the edge stronger, not weaker.** With 10000 bars
  (~580 days), six majors, floor 0.70: ALL n=923, **69.8% hit, Wilson lo 66.7%,
  p<0.001** (up from 66.8% on 250 days), and four pairs are now each
  individually significant (GBPUSD 68.5%, AUDUSD 64.1%, USDCHF 78.5%,
  USDCAD 69.4%).
- **Even near-double 1m history does not help.** Pulling the full ~9.5k 1m bars
  (8.7 days, vs 5.5 before) still scores ALL n=1364, 49.9%, Wilson lo 47.2%,
  p=1.000 at floor 0.70 -- no edge. The fast-timeframe problem is not only
  sample size; 1m has no repeatable structure worth the spread.
- `readiness.p_value` computes the binomial tail in log space (`lgamma`): the
  old `comb()` sum raised `OverflowError` once n reached a few hundred, which a
  1h backtest hits routinely. Guarded by a test.
- **Test ledger pollution:** `predictor.feed.LEDGER` (and the web app's import of
  it) is a module-level singleton that writes `var/ledger.jsonl`. Tests that
  exercised the resolve path without stubbing it appended synthetic rows with a
  hard-coded fixture price (entry_close 1.1000), which the dashboard then showed
  as a real ~98% win rate. `tests/conftest.py` disables the singleton for the
  whole session; a test that builds its own `Ledger(tmp_path/...)` is unaffected.
- **Render deploy:** `render.yaml` builds `deploy/Dockerfile.render` with
  `autoDeploy: true`, so pushing to `main` redeploys automatically. The free
  plan has no disk, so `LEDGER_PATH` points at the writable `/app/var/ledger.jsonl`
  (resets on deploy). `Ledger.record` catches `OSError`, logs once and keeps
  scoring in memory, so a bad path can never take the feed down. Attach a disk at
  `/var/data` and point `LEDGER_PATH` there to keep the record across deploys.
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

## Go-live readiness gate (`predictor/readiness.py`) — the real-money checklist
`risk.py` sizes a single trade; this gate asks the harder question: is the edge
statistically real, or a hot streak? A 58% rate on 20 trades is noise; the same
on 500 trades can be a real edge. The gate makes that call:

- Compares the forward (paper) win rate against break-even `1/(1+payout)`.
- Wilson 95% lower bound: the pessimistic end of the interval must itself clear
  break-even — the point estimate alone is not enough.
- One-sided binomial p-value (pure-Python tail sum via `math.comb`, so no scipy
  is needed on the deploy box; verified identical to scipy to ~1e-16): how
  likely this run is if the true rate were exactly break-even.
- Sample floor `MIN_TRADES = 100`: below it the verdict is NEVER `go`, however
  good the rate looks.
- Verdicts: `go` (real, survivable edge), `forward-only` (keep testing),
  `no-go` (measurably below break-even — loses money over time).
- Surfaces: `/readiness` (Telegram), `/api/readiness` (web, per-combo too), and
  the "Go-live readiness gate" card. Verified live: 51 forward trades at 94%
  still returns `forward-only` because it is under the sample floor — correct.
- Before any real money: (1) `no-go` combos are off the table; (2) only `go`
  combos are eligible; (3) size each with `risk.suggest_stake` at <=2% bankroll;
  (4) re-check weekly — an edge can decay.

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
- **Smart Money Concepts (SMC) as model features** (order blocks, fair-value
  gaps, liquidity sweeps, break of structure, premium/discount):
  `research_smc_features.py` A/B was 4/4 at floor 0.55, 3/5 at 0.60, 4/4 at 0.65
  across 8 pair/intervals — a coin-flip, and the floors that "won" had only
  18-39 signals. NOT wired into the model. Shipped instead as the read-only
  analysis layer `predictor/smc.py` (see below).
- Lesson: on FX the honest edge is fragile. Always A/B with the harness on the
  SAME data (and both halves of the walk-forward) before shipping "smarter".

## Smart Money analysis (`predictor/smc.py`) — analysis, not a signal
SMC is deliberately NOT a model feature (measured above, did not win). It is a
human-readable overlay that explains the structure behind a call:
- Reads confirmed swings (k=3 either side, so no look-ahead), market structure
  (HH/HL vs LH/LL), break of structure, the last unfilled fair-value gap, liquidity
  sweeps in the last 5 bars, and premium/discount position in the range.
- Surfaces: `/smc [PAIR] [INTERVAL]` (Telegram), `/api/smc` (web), the "Smart
  money" card, and a one-line structure note appended to each push alert.
- `smc.recommendation()` only says how structure and the premium/discount zone
  line up (e.g. "bullish structure in discount"). It is NOT a direction call and
  NOT a probability — the model's own probability remains the actual signal.
- Do not "promote" SMC to a feature without a fresh A/B that wins on both halves
  of the walk-forward.

## Getting work onto GitHub (the sandbox token is read-only)
- The injected GITHUB_TOKEN is a `ghu_` GitHub App user token with read-only
  access: `git push` returns 403 and the API returns "Resource not accessible by
  integration", even though the repo object reports `push: true`. Do not rely on
  it to push.
- Distribution path for a phone-only user: export the branch as a `.patch`
  (`git format-patch --stdout origin/main..HEAD`) and a `.bundle`
  (`git bundle create x.bundle refs/heads/<branch> --not refs/remotes/origin/main`),
  serve them, and let the user push with their own credentials.
