# Walk-forward backtest: EURUSD 1m

Out-of-sample samples: 1773
Overall accuracy: 54.99%
Majority-class baseline: 50.31%
Up-candle rate: 50.31%
Break-even accuracy at 80% payout: 55.56%

## Accuracy vs confidence filter
```
 min_confidence  signals  coverage_pct  accuracy_pct  roi_pct_per_trade
           0.50     1773        100.00         54.99              -1.02
           0.55      453         25.55         58.50               5.30
           0.60      185         10.43         59.46               7.03
           0.65       96          5.41         59.38               6.88
           0.70       63          3.55         63.49              14.29
           0.75       15          0.85         66.67              20.00
           0.80        6          0.34         66.67              20.00
```
