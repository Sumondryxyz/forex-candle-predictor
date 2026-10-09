# Walk-forward backtest: EURUSD 5m

Out-of-sample samples: 1911
Overall accuracy: 50.92%
Majority-class baseline: 52.12%
Up-candle rate: 47.88%
Break-even accuracy at 80% payout: 55.56%

## Accuracy vs confidence filter
```
 min_confidence  signals  coverage_pct  accuracy_pct  roi_pct_per_trade
           0.50     1911        100.00         50.92              -8.35
           0.55      449         23.50         53.90              -2.98
           0.60      190          9.94         58.95               6.11
           0.65       91          4.76         50.55              -9.01
           0.70       49          2.56         57.14               2.86
           0.75       34          1.78         64.71              16.47
           0.80       28          1.47         75.00              35.00
```
