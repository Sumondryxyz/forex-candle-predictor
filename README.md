# Forex next-candle predictor

Reads **live candles** and prints an UP/DOWN prediction for the next candle, with
a confidence figure and a WIN/LOSS check once that candle closes. It never places
a trade.

Two data sources are supported: **TradingView** (default — the public chart
websocket, no login, works from any host) and **Quotex** (your own account via the
unofficial `pyquotex` client). There is also a Flask web app.

The Quotex path runs on a phone (Termux) — a home/mobile connection is needed
because Cloudflare blocks Quotex logins from datacenter IPs. The TradingView path
has no such restriction.

## Quick start (Termux)

```bash
pkg update -y && pkg install python git python-numpy -y
pip install "git+https://github.com/cleitonleonel/pyquotex.git"
git clone https://github.com/Sumondryxyz/forex-candle-predictor.git
cd forex-candle-predictor

export QUOTEX_EMAIL='you@example.com'
export QUOTEX_PASSWORD='yourpassword'      # single quotes: $ and # break in double quotes

python quotex_candles.py    # live broker candles, one line each
python quotex_signals.py    # UP/DOWN + confidence, then WIN/LOSS per candle
```

Put the two `export` lines in `~/.bashrc` so they survive a Termux restart.

Settings, all optional:

| Variable | Default | Meaning |
|---|---|---|
| `PAIR` | `EURUSD` | any Quotex asset, e.g. `EURUSD_otc` |
| `PERIOD` | `60` | candle seconds: 60 = 1m, 300 = 5m |
| `MIN_CONF` | `0.60` | below this the signal prints as `skip` |
| `HISTORY_HOURS` | `72` | how much history to train on |
| `RETRAIN_EVERY` | `60` | retrain after this many new candles |
| `PAYOUT` | `0.80` | broker payout, used for the ROI figure |
| `QUOTEX_HOST` | `market-qx.trade` | change if your account uses another domain |

Sample output:

```
17:12:00  SIGNAL UP (CALL)  conf 63.4%
    -> WIN (close 1.15195)
17:13:00  skip  (DOWN 54.1%)
```

## Web app

A small Flask dashboard wraps the same pipeline in a browser UI, so you do not
have to run the terminal scripts:

```bash
pip install -r requirements.txt
python -m predictor.webapp            # http://127.0.0.1:5000
PORT=12000 python -m predictor.webapp # custom port
```

Open the page, pick a source, pair and interval, and it shows the latest UP/DOWN
signal (direction, confidence, tradeable flag, a sparkline of the last 60 candles
and the next candle open time) plus a one-click walk-forward backtest. Nothing is
ever traded.

## Telegram bot

The same feed and engine drive a Telegram bot, so signals arrive on your phone.

```bash
TELEGRAM_BOT_TOKEN=123:abc python bot.py
```

1. In Telegram, talk to **@BotFather** → `/newbot` → copy the token.
2. Run the command above (locally, or as a Botkeep **worker**).
3. Send `/start` to your bot.

| Command | What it does |
|---|---|
| `/signal [PAIR] [INTERVAL]` | one prediction, e.g. `/signal EURUSD 5m` |
| `/scan` | every configured pair at your interval |
| `/session` | paper WIN/LOSS record for the current pair |
| `/pairs` | available pairs and intervals |
| `/setpair EURUSD`, `/setinterval 5m` | remember defaults |
| `/subscribe` / `/unsubscribe` | auto-push tradeable signals |
| `/status` | your current settings |

The bot **never places a trade**. Auto-push sends at most one message per candle
(`PUSH_INTERVAL` seconds between checks, default 30) and announces each result
once, as ✅ WIN or ❌ LOSS.

> ⚠️ Quotex blocks logins from datacenter IPs. If you host the bot, keep
> `DATA_SOURCE=tradingview` (no login needed) or run it from a home/mobile
> connection.

### Phone-only setup (Termux)

No PC needed. From the repository root:

```bash
bash scripts/install-termux.sh   # installs everything, writes .env
# edit .env: set TELEGRAM_BOT_TOKEN (and Quotex creds if you use them)
set -a; . ./.env; set +a
python bot.py
```

To keep it running after a reboot, install the **Termux:Boot** app, then
`mkdir -p ~/.termux/boot && cp scripts/termux-boot.sh ~/.termux/boot/` and
disable battery optimisation for Termux.

### Quotex session without a PC

Cloudflare blocks the Quotex *login page* from datacenter IPs, so the phone logs
in and hands the session to the bot. The one-tap way:

```bash
python scripts/session_capture.py     # serves http://127.0.0.1:8787
```

Open that URL on the phone and follow the page:

1. Install **Tampermonkey** in Kiwi Browser, open `/capture.user.js`, Install.
2. Log in to Quotex. The script reads the token from `/api/v1/cabinets/digest`
   and the cookies with Tampermonkey's `GM_cookie` API (needed for the HttpOnly
   `laravel_session` / `cf_clearance`), then saves `session.json` automatically.
   A “Session saved ✓” badge confirms it. No password is ever read.
3. Run the bot: `QUOTEX_SESSION_FILE=session.json DATA_SOURCE=quotex python bot.py`.

The server binds to `127.0.0.1` only, so nothing leaves the phone. Set
`CAPTURE_KEY=...` on both the server and the userscript to require a shared key.

Prefer no extension? `tools/quotex-session-helper.html` does the same by hand
(bookmarklet + Cookie-Editor). Validate either file with:

```bash
python scripts/check_quotex_session.py session.json
```

The token is like a password: never commit it or share it.

### Data sources

Pick the source from the dropdown (or `DATA_SOURCE=...`):

- **tradingview** (default) — reads candles from TradingView's public chart
  websocket. No login, no account, and **not blocked from datacenter IPs**, so it
  works out of the box. Unofficial/undocumented and outside TradingView's terms;
  it only reads candles.
- **quotex** — the broker's own feed through `pyquotex`. Same pairs and payouts
  the bot trades, but Cloudflare challenges the login page from datacenter IPs, so
  it needs a home/mobile connection, a residential proxy (`QUOTEX_PROXY`), or an
  exported browser session (`QUOTEX_SESSION_FILE`). See the Quotex section below.

Each source+pair+interval gets one background poller (`TradingViewFeed` /
`QuotexFeed`, both built on `predictor/feed.py`) that loads history, trains the
numpy engine, then retrains as candles close and keeps a WIN/LOSS session count.
If a source fails it shows the error and retries with a backoff instead of
crashing.

Useful environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `PORT` / `HOST` | `5000` / `0.0.0.0` | bind address |
| `DATA_SOURCE` | tradingview | default source (`tradingview` or `quotex`) |
| `WEB_PAIRS` | EURUSD,GBPUSD,USDJPY,AUDUSD,USDCHF,USDCAD | pairs offered in the dropdown |
| `WEB_INTERVALS` | 1m,5m,15m | intervals offered |
| `MIN_CONF` | 0.70 | skip signals below this confidence |
| `PAYOUT` | 0.80 | payout used for the walk-forward ROI |
| `HISTORY_HOURS` | 72 | history to train on |
| `RETRAIN_EVERY` | 60 | retrain after this many new candles |
| `LEDGER_ENABLED` | 1 | log resolved signals to the paper-trading ledger |
| `LEDGER_PATH` | var/ledger.jsonl | where the ledger is written |
| `QUOTEX_ACCOUNT` | PRACTICE | `PRACTICE` or `REAL` |
| `QUOTEX_SESSION_FILE` | — | optional browser cookies JSON to skip the login |
| `QUOTEX_PROXY` | — | optional residential proxy (`http://user:pass@host:port`) for datacenter hosts |

HTTP API, for scripts or a frontend of your own:

- `GET /api/signals?pair=EURUSD&interval=5m&source=tradingview` (live feed)
- `GET /api/live?...` (alias of `/api/signals`)
- `GET /api/candles?pair=EURUSD&interval=5m&limit=500` (raw OHLCV for your own
  analysis, e.g. feeding an LLM)
- `GET /api/backtest?pair=EURUSD&interval=5m` (the feed's walk-forward check)
- `GET /api/ledger` (honest forward WIN/LOSS record of resolved signals)
- `GET /api/config`, `GET /api/health`

### Paper-trading ledger (the honest scoreboard)

Walk-forward is a simulation. The ledger is the real thing: every signal that
clears `MIN_CONF` is written to `var/ledger.jsonl` (JSON Lines) the moment the
next candle closes, scored WIN or LOSS against the actual close. The dashboard's
"Paper-trading ledger" panel and `GET /api/ledger` show the running win rate,
ROI per trade, a per-pair/interval breakdown and the latest results. Nothing is
traded. Leave a pair connected for a few days and this becomes the only number
that matters — if the win rate stays under the 55.6% break-even of an 80% payout,
the edge is not real yet. On Render, mount a disk and point `LEDGER_PATH` at it
so redeploys do not wipe the record.


The default TradingView source needs no connection setup. The Quotex source needs
a residential connection: Cloudflare blocks the Quotex login from datacenter IPs.

### Keeping it running 24/7

A plain `python -m predictor.webapp` dies when the terminal closes, when the
host recycles, or when a dependency disappears. Pick the option that matches
where you run it.

**This sandbox / a dev box (no init system).** Use the supervisor inside tmux so
it outlives the shell and restarts the app if it crashes or the environment
wipes the packages:

```bash
./scripts/serve_tmux.sh          # start (idempotent), attach: tmux attach -t predictor
./scripts/serve_tmux.sh stop     # stop
```

`scripts/run_server.sh` is the supervisor: it re-checks dependencies, restarts
the app on any exit, and backs off up to 60s. Note the sandbox itself still
recycles its VM periodically — nothing inside it can survive a full recycle.

**A real server (recommended).** Install the systemd unit in `deploy/`, which
restarts on crash and starts on boot:

```bash
sudo cp deploy/forex-predictor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now forex-predictor
sudo systemctl status forex-predictor
```

Edit `WorkingDirectory` and the `ExecStart` path in the unit to match your
checkout, and create a `.venv` with `pip install -r requirements.txt` there.

**Docker (portable).** Build and run with a restart policy:

```bash
docker compose -f deploy/docker-compose.yml up -d --build
```

The container has `restart: unless-stopped` and a healthcheck on `/api/health`.

**Render (free plan, no server of your own).** The repo ships a Blueprint
(`render.yaml`) and a slim image (`deploy/Dockerfile.render`) that installs only
what the live app needs. Steps:

1. Push this repo to GitHub (already done if you cloned it from there).
2. In Render: **New + → Blueprint**, connect the repo, and apply. Render reads
   `render.yaml`, builds the Docker image, and gives you
   `https://<name>.onrender.com`. The health check is `/api/health`.
3. **Keep it awake.** A Render free service spins down after 15 minutes without
   *inbound* traffic. This app only makes *outbound* TradingView connections, so
   without a ping the feed sleeps and stops predicting. Add a GitHub repository
   secret named `RENDER_URL` set to your service URL
   (e.g. `https://forex-candle-predictor.onrender.com`); the included
   `.github/workflows/keep-awake.yml` then pings `/api/health` every 5 minutes.

Know the free-plan limits: 750 shared instance-hours per month (one always-on
service uses ~730, so a second free service would be suspended mid-month), and
a 30-60s cold start if it ever does sleep. For a service that truly never
sleeps, use the systemd or Docker route above on a VPS, or Oracle Cloud's
Always Free VM.

Other platforms (Railway / Fly.io / Koyeb) work the same way: run the image in
`deploy/Dockerfile.render`, bind `0.0.0.0:$PORT`, and keep one worker.

Two things are worth knowing whichever route you take:

- The Flask dev server is fine for personal use; for anything public put a real
  WSGI server in front (`gunicorn -w 1 -k gthread -b 0.0.0.0:$PORT predictor.webapp:create_app()`).
  Keep one worker: the candle feed lives in process memory.
- The TradingView feed is unofficial and can be rate-limited; the app already
  reconnects with backoff, so leave it running rather than restarting often.

## Honest accuracy statement

Nobody achieves 80–90% accuracy on next-candle direction in a live market. Any
system claiming that is either curve-fitted, tested with look-ahead bias, or
lying. On one-minute FX the next candle is close to a coin flip, and gradient
boosted trees actually do *worse* here than a plain regularised linear model:
trees memorise the noise and sit at ~51%, while the linear model keeps a small,
real edge once you filter on confidence.

Strict rolling walk-forward, 60 days of 5m/15m candles per pair, the linear
engine used by the live feed, isotonic calibration on. Hit rate and ROI per
trade at an 80% payout (break-even is 55.6%):

| pair + interval | ≥0.60 signals | hit% | ROI% | ≥0.65 signals | hit% | ROI% |
|---|---|---|---|---|---|---|
| EURUSD 5m  | 306 (12%) | 56.9 | +2.4  | 140 (6%) | 61.4 | +10.6 |
| EURUSD 15m | 377 (15%) | 53.9 | −3.1  | 167 (7%) | 59.9 | +7.8  |
| GBPUSD 5m  | 209 (8%)  | 57.9 | +4.2  | 131 (5%) | 56.5 | +1.7  |
| GBPUSD 15m | 663 (26%) | 56.1 | +1.0  | 146 (6%) | 58.2 | +4.8  |
| USDJPY 5m  | 66 (3%)   | 62.1 | +11.8 | 45 (2%)  | 57.8 | +4.0  |
| USDJPY 15m | 212 (8%)  | 51.4 | −7.5  | 78 (3%)  | 51.3 | −7.7  |

Read this honestly, not as a promise:

- The edge lives in **selectivity**. Raising the floor from 0.65 to **0.70** lifts
  the average hit rate on every window measured (30/45/90 days): e.g. 58.1% vs
  52.9% on 30 days, 59.3% vs 57.0% on 90 days, with net-positive ROI throughout.
  Fewer signals, but the ones it prints are more often right — which is what you
  want if the goal is accuracy. Default is now 0.70.
- A one-candle **horizon** (predict `close[t+3]` vs `close[t]` instead of
  `close[t+1]`) does *not* help: on the real training path the 3-candle target
  averaged 57.0% vs 57.9% for the next candle, and USDJPY 5m collapsed to 47.5%.
  An early harness suggested a gain, but it trained on 1-step labels and scored
  on 3-step ones — a bug, not an edge. Rejected.
- It is **not uniformly good**. USDJPY 15m has no edge here, and a single pair
  can be a coin-flip; never act on one combo. Judge the aggregate.
- The linear engine and a calibrated LightGBM (`train_backtest.py`) are close and
  trade places per combo (e.g. EURUSD 5m favours the GBM at ≥0.75). Neither is
  clearly better, so the lightweight linear model stays the default.
- Confidence now means something: with isotonic calibration a printed 0.65 is
  close to a 65% hit rate on the calibration tail, so the ≥0.65/0.70 buckets
  really do score higher. The dashboard prints the whole walk-forward sweep, so
  judge it on your own data. Trade a demo account until your WIN/LOSS count says
  otherwise. Expect roughly 56–61% on the filtered signals.

`research_sweep.py` reproduces the table above:

```bash
python research_sweep.py --pairs EURUSD,GBPUSD,USDJPY --intervals 5m,15m --days 60
```


The engine picks its backend automatically: **linear** (35 causal features, a
regularised logistic regression) when pandas + scikit-learn are importable, and
a **numpy** fallback (22 causal features, a bagged logistic ensemble calibrated
on a held-out tail) so nothing has to compile on a phone. Either way it retrains
every `RETRAIN_EVERY` candles and prints a walk-forward hit rate measured on
candles it never trained on — that number, not the in-sample one, is what to
trust. Expect roughly 55–58% live. Trade a demo account until your own WIN/LOSS
count says otherwise.

## How the Quotex connection works

The community `pyquotex` client does not connect to current accounts as-is;
`predictor/quotex_session.py` patches two things, without which the socket
always answers `authorization/reject`:

- Quotex sets `laravel_session` twice during login — guest first, then
  authenticated — and pyquotex forwards both, so the server sees a guest. Only
  the last value of each cookie is kept.
- The socket session is no longer in `window.settings`; it comes from the
  authenticated `/api/v1/cabinets/digest` endpoint.

Caveats, all verified the hard way:

- This is **not an official API** and using it breaks Quotex's terms; accounts
  can be banned. This project only reads candles — it never places a trade.
- Cloudflare blocks the login from datacenter IPs (HTTP 403 / endless "verify
  you are human"), so it only works from a home or mobile connection. It is
  therefore untested in CI; it was verified by hand from a phone on Termux.
- Passwords containing `$` or `#` need single quotes in the shell, otherwise
  the shell eats part of them and Quotex answers "Invalid email or password".
- A stale cached session makes login silently reuse a dead token: delete
  `session.json` and rerun.
- The HTML login page is behind Cloudflare, but the **websocket endpoint itself
  is reachable** from datacenter IPs — only the login page is blocked. So if you
  can get a valid session, the feed connects from anywhere.
- If your browser (or phone) is already logged in, export its session to a JSON
  file and point `QUOTEX_SESSION_FILE` at it to skip the password login. Two
  layouts are accepted: pyquotex's own `session.json`, keyed by email
  (`{"you@example.com": {"cookies": "...", "token": "...", "user_agent": "..."}}`),
  or a flat `{"cookies": ..., "token": ..., "user_agent": ...}` object. The
  `token` is what the websocket authenticates with; `cookies` and `user_agent`
  must match the browser that created the session (Cloudflare binds the
  handshake to them). Grab the token from `/api/v1/cabinets/digest` while logged
  in, or from the page's `window.settings` object.

## Bigger model (needs a PC or a Termux build toolchain)

The `predictor/` package holds the full research pipeline: 35 causal features,
regularised logistic regression, walk-forward backtesting and confidence sweeps.
It needs pandas/scikit-learn, so it is meant for a PC. The live web app uses the
same engine, so it automatically picks up this stronger model when sklearn is
installed.

```bash
pip install -r requirements.txt
python train_backtest.py --pair EURUSD --interval 5m --context-pairs GBPUSD,AUDUSD,USDCHF
python train_backtest.py --pair EURUSD --interval 1m                 # tradingview (default)
python train_backtest.py --pair EURUSD --interval 1m --source quotex  # broker feed
```

Reports land in `reports/`, models in `models/`.

## Layout

- `quotex_candles.py` — print live broker candles
- `quotex_signals.py` — live UP/DOWN signals (numpy only)
- `predictor/engine.py` — next-candle features/training/walk-forward (linear + numpy backends), shared by the CLI and web app
- `predictor/feed.py` — `BaseFeed`: shared background poller/trainer/retrainer
- `predictor/tradingview_live.py` — background TradingView feed (default)
- `predictor/quotex_live.py` — background Quotex feed
- `predictor/webapp.py` — Flask web app + JSON API
- `predictor/templates/index.html` — dashboard page
- `predictor/quotex_session.py` — login/cookie fixes for the unofficial client
- `predictor/sources/` — TradingView (default) and Quotex candle feeds for `train_backtest.py`
- `predictor/features.py` — 35 indicator features, all causal (no look-ahead)
- `predictor/model.py` — LightGBM (or scikit-learn) classifier, walk-forward
- `predictor/backtest.py` — accuracy vs confidence reporting
- `predictor/scheduler.py` — candle-close alignment and forex market hours
- `train_backtest.py` — CLI: train + backtest + save model

## Methodology

- Labels: 1 if next close > current close; unchanged closes are dropped.
- Walk-forward: retrain every 250 candles on a rolling 3000-candle window and
  predict only forward, so reported accuracy is out-of-sample.
- Calibration is fitted on the last 20% of each training window, never on the
  candles being predicted.
- Context pairs are clipped to the same time span as the training window, so no
  future information leaks in through pooling.
- ROI assumes an 80% payout on wins and a full stake loss; spread and slippage
  are not modelled.

Educational use. Not financial advice.
