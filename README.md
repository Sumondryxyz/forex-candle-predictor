# Forex next-candle predictor

Telegram bot that predicts the direction of the **next candle** for forex pairs
(binary/Quotex-style UP/DOWN), with a walk-forward backtest that reports the
accuracy it actually achieves — not a marketing number.

## Honest accuracy statement

Nobody achieves 80–90% accuracy on next-candle direction in a live market. Any
system claiming that is either curve-fitted, tested with look-ahead bias, or
lying. Measured here on EURUSD with strict walk-forward validation (the model
never sees the candle it predicts):

EURUSD 5m, model pooled over GBPUSD/AUDUSD/USDCHF, isotonic-calibrated:

| Confidence filter | Signals | Accuracy | ROI per trade @80% payout |
|---|---|---|---|
| all candles | 8430 | 52.7% | −5.1% |
| ≥ 0.55 | 1368 (16%) | 58.7% | +5.7% |
| ≥ 0.60 | 384 (4.6%) | 60.4% | +8.8% |
| ≥ 0.65 | 119 (1.4%) | 65.6% | +18.0% |
| ≥ 0.70 | 59 (0.7%) | 71.2% | +28.1% |
| ≥ 0.75 | 48 (0.6%) | 75.0% | +35.0% |

The realistic path to a high hit rate is **selectivity**: skip most candles and
only trade the small subset where the model is confident. Everything above
≥0.65 rests on 50–120 trades, so treat those numbers as noisy — the robust,
large-sample result is the ~58–60% band at ≥0.55/0.60, which already clears the
55.6% break-even of an 80%-payout binary option. Re-validate on your own
broker's candles before risking money.

What moved the numbers versus the first version (53% / 61% at ≥0.70):

- **Probability calibration** (isotonic on a held-out slice) — a reported 0.70
  now means roughly a 70% hit rate, so the filter actually selects good trades.
- **Pooled multi-pair training** — 4x the training rows per window.
- **Higher-timeframe context**, 20-candle range position, mean-reversion z-score,
  return autocorrelation, and London/NY session flags (35 features total).

### 1-minute timeframe (Quotex-style trading)

EURUSD 1m, same pooled + calibrated setup, 1773 out-of-sample candles:

| Confidence filter | Signals | Accuracy | ROI per trade @80% payout |
|---|---|---|---|
| all candles | 1773 | 55.0% | −1.0% |
| ≥ 0.55 | 453 (26%) | 58.5% | +5.3% |
| ≥ 0.60 | 185 (10%) | 59.5% | +7.0% |
| ≥ 0.70 | 63 (3.6%) | 63.5% | +14.3% |

yfinance only keeps 7 days of 1m history, so the 1m sample is small — re-run
training regularly to keep the model on recent data.

## Quotex usage

Quotex has no official API, and unofficial trading clients violate its terms and
risk account bans, so this project **does not place trades**. It sends a signal
a few seconds before each candle closes and you enter manually:

```bash
python train_backtest.py --pair EURUSD --interval 1m --context-pairs GBPUSD,AUDUSD,USDCHF
MIN_CONFIDENCE=0.60 TELEGRAM_BOT_TOKEN=... python bot.py
```

### Quotex candles instead of Yahoo (optional, unofficial)

By default candles come from Yahoo Finance, which lags real time by 15–60s and
does not carry Quotex's OTC pairs. `DATA_SOURCE=quotex` switches to the broker's
own WebSocket via the community `pyquotex` client:

```bash
pip install "git+https://github.com/cleitonleonel/pyquotex.git"   # needs Python >= 3.12
export DATA_SOURCE=quotex QUOTEX_EMAIL=... QUOTEX_PASSWORD=... QUOTEX_ACCOUNT=PRACTICE
```

Caveats, all verified the hard way:

- This is **not an official API** and using it breaks Quotex's terms; accounts
  can be banned. Read-only candle access is all this project does — it never
  places a trade.
- Cloudflare blocks the login from datacenter IPs (HTTP 403 / endless "verify
  you are human"), so it only works from a home connection. It is therefore
  untested in CI and in this repo's development environment.
- If your browser is already logged in, dump its cookies to a JSON file with
  `cookies`, `token` and `user_agent` keys and point `QUOTEX_SESSION_FILE` at
  it to skip the password login.

Then in Telegram: `/auto EURUSD 1m`. The bot only messages when confidence
clears `MIN_CONFIDENCE`, states the entry time in UTC, and stays silent while
the forex market is closed (weekends). `/stop` cancels it.

## Layout

- `predictor/data.py` — OHLC download (yfinance spot forex)
- `predictor/sources/` — pluggable candle feeds (`DATA_SOURCE=yfinance|quotex`)
- `predictor/features.py` — 35 indicator features, all causal (no look-ahead)
- `predictor/model.py` — LightGBM classifier, walk-forward evaluation
- `predictor/backtest.py` — accuracy vs confidence reporting
- `train_backtest.py` — CLI: train + backtest + save model
- `predictor/scheduler.py` — candle-close alignment and forex market hours
- `bot.py` — Telegram interface

## Usage

```bash
pip install -r requirements.txt
python train_backtest.py --pair EURUSD --interval 5m --context-pairs GBPUSD,AUDUSD,USDCHF
export TELEGRAM_BOT_TOKEN=...          # from @BotFather
export MIN_CONFIDENCE=0.65             # below this the bot says NO TRADE
export ALLOWED_USER_IDS=123456789      # optional access control
python bot.py
```

Bot commands: `/signal EURUSD 1m`, `/auto EURUSD 1m`, `/stop`,
`/stats EURUSD 1m`, `/pairs`.

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
