# Walk-forward backtest: EURUSD 5m

Out-of-sample samples: 8430
Overall accuracy: 52.7%
Majority-class baseline: 51.07%
Up-candle rate: 48.93%
Break-even accuracy at 80% payout: 55.56%

## Accuracy vs confidence filter
```
 min_confidence  signals  coverage_pct  accuracy_pct  roi_pct_per_trade
           0.50     8430        100.00         52.70              -5.13
           0.55     1368         16.23         58.70               5.66
           0.60      384          4.56         60.42               8.75
           0.65      119          1.41         65.55              17.98
           0.70       59          0.70         71.19              28.14
           0.75       48          0.57         75.00              35.00
           0.80       30          0.36         70.00              26.00
```
