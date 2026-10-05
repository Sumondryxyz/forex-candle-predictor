# Quotex next-candle predictor

Terminal tool that reads **live candles from your own Quotex account** and
prints an UP/DOWN prediction for the next candle, with a confidence figure and
a WIN/LOSS check once that candle closes. It never places a trade.

Runs on a phone (Termux) — which is also the only way it works, because
Cloudflare blocks Quotex logins from datacenter IPs.

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
pip install "git+https://github.com/cleitonleonel/pyquotex.git"
export QUOTEX_EMAIL='you@example.com' QUOTEX_PASSWORD='yourpassword'
python -m predictor.webapp            # http://127.0.0.1:5000
PORT=12000 python -m predictor.webapp # custom port
```

Open the page, pick a pair and interval, and it shows the latest UP/DOWN signal
(direction, confidence, tradeable flag, a sparkline of the last 60 candles and
the next candle open time) plus a one-click walk-forward backtest. Nothing is
ever traded.

The web app uses **only the live Quotex feed** — there is no yfinance fallback.
Each pair/interval gets one background poller (`QuotexFeed`) that logs in,
loads history, trains the numpy engine, then retrains as candles close and keeps
a WIN/LOSS session count. If the login is blocked it shows the error and retries
with a backoff instead of crashing.

Useful environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `PORT` / `HOST` | `5000` / `0.0.0.0` | bind address |
| `WEB_PAIRS` | EURUSD,GBPUSD,USDJPY,AUDUSD,USDCHF,USDCAD | pairs offered in the dropdown |
| `WEB_INTERVALS` | 1m,5m,15m | intervals offered |
| `MIN_CONF` | 0.60 | skip signals below this confidence |
| `PAYOUT` | 0.80 | payout used for the walk-forward ROI |
| `HISTORY_HOURS` | 72 | history to train on |
| `RETRAIN_EVERY` | 60 | retrain after this many new candles |
| `QUOTEX_ACCOUNT` | PRACTICE | `PRACTICE` or `REAL` |
| `QUOTEX_SESSION_FILE` | — | optional browser cookies JSON to skip the login |
| `QUOTEX_PROXY` | — | optional residential proxy (`http://user:pass@host:port`) for datacenter hosts |

HTTP API, for scripts or a frontend of your own:

- `GET /api/signals?pair=EURUSD&interval=5m` (live Quotex feed)
- `GET /api/live?pair=EURUSD&interval=1m` (alias of `/api/signals`)
- `GET /api/backtest?pair=EURUSD&interval=5m` (the feed's walk-forward check)
- `GET /api/config`, `GET /api/health`

Like the CLI, the web app needs a residential connection: Cloudflare blocks the
Quotex login from datacenter IPs.

## Honest accuracy statement

Nobody achieves 80–90% accuracy on next-candle direction in a live market. Any
system claiming that is either curve-fitted, tested with look-ahead bias, or
lying. With strict walk-forward validation (the model never sees the candle it
predicts) on EURUSD 5m, pooled over GBPUSD/AUDUSD/USDCHF and calibrated:

| Confidence filter | Signals | Accuracy | ROI per trade @80% payout |
|---|---|---|---|
| all candles | 8430 | 52.7% | −5.1% |
| ≥ 0.55 | 1368 (16%) | 58.7% | +5.7% |
| ≥ 0.60 | 384 (4.6%) | 60.4% | +8.8% |
| ≥ 0.65 | 119 (1.4%) | 65.6% | +18.0% |
| ≥ 0.70 | 59 (0.7%) | 71.2% | +28.1% |

The realistic path to a high hit rate is **selectivity**: skip most candles and
trade only the small subset where the model is confident. Everything above
≥0.65 rests on 50–120 trades, so treat it as noisy; the robust result is the
~58–60% band at ≥0.55/0.60, which already clears the 55.6% break-even of an
80%-payout binary option.

`quotex_signals.py` is numpy-only so nothing has to compile on a phone: 22
causal features, a bagged ensemble of logistic regressions, probabilities
calibrated on a held-out tail of the history, and a retrain every
`RETRAIN_EVERY` candles. On startup it prints a walk-forward hit rate measured
on candles the model never trained on — that number, not the in-sample one, is
what to trust. Expect roughly 55–60% live. Trade a demo account until your own
WIN/LOSS count says otherwise.

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
- If your browser is already logged in, dump its cookies to a JSON file with
  `cookies`, `token` and `user_agent` keys and point `QUOTEX_SESSION_FILE` at
  it to skip the password login.

## Bigger model (needs a PC or a Termux build toolchain)

The `predictor/` package holds the full research pipeline: 35 causal features,
pooled multi-pair training, isotonic calibration and walk-forward backtesting.
It needs pandas/scikit-learn, so it is meant for a PC.

```bash
pip install -r requirements.txt
python train_backtest.py --pair EURUSD --interval 5m --context-pairs GBPUSD,AUDUSD,USDCHF
python train_backtest.py --pair EURUSD --interval 1m --source quotex
```

Reports land in `reports/`, models in `models/`.

## Layout

- `quotex_candles.py` — print live broker candles
- `quotex_signals.py` — live UP/DOWN signals (numpy only)
- `predictor/engine.py` — numpy-only features/training/walk-forward, shared by the CLI and web app
- `predictor/quotex_live.py` — background Quotex feed for the web app
- `predictor/webapp.py` — Flask web app + JSON API
- `predictor/templates/index.html` — dashboard page
- `predictor/quotex_session.py` — login/cookie fixes for the unofficial client
- `predictor/sources/` — the Quotex candle feed used by `train_backtest.py`
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
