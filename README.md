# Forex next-candle predictor

Telegram bot that predicts the direction of the **next candle** for forex pairs
(binary/Quotex-style UP/DOWN), with a walk-forward backtest that reports the
accuracy it actually achieves — not a marketing number.

## Honest accuracy statement

Nobody achieves 80–90% accuracy on next-candle direction in a live market. Any
system claiming that is either curve-fitted, tested with look-ahead bias, or
lying. Measured here on EURUSD with strict walk-forward validation (the model
never sees the candle it predicts):

| Interval | Signals taken | Accuracy | Baseline |
|---|---|---|---|
| 5m, all signals | 8316 | 53.3% | 51.0% |
| 5m, confidence ≥ 0.70 | 544 (6.5% of candles) | 61.6% | — |
| 5m, confidence ≥ 0.75 | 152 (1.8% of candles) | 68.4% | — |
| 15m, all signals | 1444 | 52.5% | 50.0% |
| 15m, confidence ≥ 0.70 | 68 | 61.8% | — |

The realistic path to a high hit rate is **selectivity**: skip most candles and
only trade the small subset where the model is confident. Accuracy above ~65%
in the table comes from samples of only 15–150 trades, so treat it as noisy and
re-validate on your own broker's data before risking money. Binary options
typically pay ~80%, so break-even is ~55.6% accuracy.

## Layout

- `predictor/data.py` — OHLC download (yfinance spot forex)
- `predictor/features.py` — 23 indicator features, all causal (no look-ahead)
- `predictor/model.py` — LightGBM classifier, walk-forward evaluation
- `predictor/backtest.py` — accuracy vs confidence reporting
- `train_backtest.py` — CLI: train + backtest + save model
- `bot.py` — Telegram interface

## Usage

```bash
pip install -r requirements.txt
python train_backtest.py --pair EURUSD --interval 5m   # writes models/ and reports/
export TELEGRAM_BOT_TOKEN=...          # from @BotFather
export MIN_CONFIDENCE=0.65             # below this the bot says NO TRADE
export ALLOWED_USER_IDS=123456789      # optional access control
python bot.py
```

Bot commands: `/signal EURUSD 5m`, `/stats EURUSD 5m`, `/pairs`.

## Methodology

- Labels: 1 if next close > current close; unchanged closes are dropped.
- Walk-forward: retrain every 250 candles on a rolling 3000-candle window and
  predict only forward, so reported accuracy is out-of-sample.
- No spread, slippage or payout is modelled — accuracy only.

Educational use. Not financial advice.
