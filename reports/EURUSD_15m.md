# Walk-forward backtest: EURUSD 15m

Out-of-sample samples: 2486
Overall accuracy: 51.09%
Majority-class baseline: 51.73%
Up-candle rate: 48.27%
Break-even accuracy at 80% payout: 55.56%

## Accuracy vs confidence filter
```
 min_confidence  signals  coverage_pct  accuracy_pct  roi_pct_per_trade
           0.50     2486        100.00         51.09              -8.05
           0.55      361         14.52         54.85              -1.27
           0.60      100          4.02         49.00             -11.80
           0.65       36          1.45         47.22             -15.00
           0.70       20          0.80         60.00               8.00
           0.75       12          0.48         83.33              50.00
           0.80        7          0.28         85.71              54.29
```
