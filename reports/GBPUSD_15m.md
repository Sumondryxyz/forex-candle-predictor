# Walk-forward backtest: GBPUSD 15m

Out-of-sample samples: 2002
Overall accuracy: 54.0%
Majority-class baseline: 52.05%
Up-candle rate: 47.95%
Break-even accuracy at 80% payout: 55.56%

## Accuracy vs confidence filter
```
 min_confidence  signals  coverage_pct  accuracy_pct  roi_pct_per_trade
           0.50     2002        100.00         54.00              -2.81
           0.55      834         41.66         54.92              -1.15
           0.60      458         22.88         55.24              -0.57
           0.65       83          4.15         57.83               4.10
           0.70       46          2.30         47.83             -13.91
           0.75       44          2.20         47.73             -14.09
           0.80       41          2.05         48.78             -12.20
```
