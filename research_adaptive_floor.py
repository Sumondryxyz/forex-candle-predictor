"""A/B: does a per-pair adaptive confidence floor beat a fixed floor?

Honest protocol: run the same rolling walk-forward, split the out-of-sample
stream in half by time, choose the floor on the EARLY half (as a live user
could), then measure that floor on the LATE half the model has never been
tuned on. Compare against the fixed 0.70 floor on the same late half.
"""
from __future__ import annotations

from predictor import engine
from predictor.sources.tradingview import fetch_ohlc

FLOORS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
FIXED = 0.70


def _rate(conf, correct, floor):
    picked = conf >= floor
    n = int(picked.sum())
    if n == 0:
        return 0, None
    return n, 100 * float(correct[picked].mean())


def _pick_floor(conf, correct, min_signals=40):
    """Best floor on the early half, by hit rate; require enough signals."""
    best, best_rate = None, -1
    for f in FLOORS:
        n, r = _rate(conf, correct, f)
        if r is not None and n >= min_signals and r > best_rate:
            best, best_rate = f, r
    return best


def run(pairs, intervals, days):
    print(f"{days}d | pick floor on early half, test on late half | fixed floor {FIXED}")
    print(f"{'pair':<8}{'tf':<5}{'chosen':>7}{'late@chosen':>14}{'late@0.70':>12}{'delta':>8}")
    wins = losses = ties = 0
    for p in pairs:
        for tf in intervals:
            try:
                df = fetch_ohlc(p, tf, days=days)
            except Exception as exc:  # noqa: BLE001
                print(f"{p:<8}{tf:<5}FETCH-FAIL {exc}")
                continue
            o, h = df["open"].to_numpy(float), df["high"].to_numpy(float)
            low, c = df["low"].to_numpy(float), df["close"].to_numpy(float)
            t = df.index.astype("int64").to_numpy() / 1e9
            raw = engine.walk_forward_raw(engine.features(o, h, low, c, t), c)
            if raw is None:
                continue
            conf, correct = raw["conf"], raw["correct"]
            half = len(conf) // 2
            chosen = _pick_floor(conf[:half], correct[:half])
            if chosen is None:
                continue
            _, late_chosen = _rate(conf[half:], correct[half:], chosen)
            _, late_fixed = _rate(conf[half:], correct[half:], FIXED)
            if late_chosen is None or late_fixed is None:
                continue
            delta = late_chosen - late_fixed
            if delta > 0.5:
                wins += 1
            elif delta < -0.5:
                losses += 1
            else:
                ties += 1
            print(f"{p:<8}{tf:<5}{chosen:>7}{late_chosen:>13.2f}%{late_fixed:>11.2f}%{delta:>+7.2f}")
    print(f"\nadaptive wins={wins} loses={losses} ties={ties}")


if __name__ == "__main__":
    import sys

    pairs = (sys.argv[1] if len(sys.argv) > 1 else "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCHF,USDCAD").split(",")
    intervals = (sys.argv[2] if len(sys.argv) > 2 else "5m,15m,1m").split(",")
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 90
    run(pairs, intervals, days)
