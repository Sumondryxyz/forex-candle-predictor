"""Tests for the SMC analysis layer. Pure arithmetic, no network."""

from predictor import smc


def _candle(o, h, low, c):
    return {"open": o, "high": h, "low": low, "close": c}


def _zigzag(n=48, leg=4, step=0.0006):
    """Clear zigzag: `leg` bars up, then `leg` bars down, repeated."""
    out = []
    px = 1.1000
    direction = 1
    for i in range(n):
        if i % leg == 0 and i > 0:
            direction = -direction
        o = px
        px = round(px + direction * step, 6)
        out.append(_candle(o, max(o, px) + 0.0001, min(o, px) - 0.0001, px))
    return out


def _uptrend(n=48):
    """Net upward zigzag: up legs grow faster than down legs retrace."""
    out = []
    px = 1.1000
    for i in range(n):
        direction = 0.0006 if (i // 4) % 2 == 0 else -0.0004
        o = px
        px = round(px + direction, 6)
        out.append(_candle(o, max(o, px) + 0.0001, min(o, px) - 0.0001, px))
    return out


def test_not_enough_candles():
    assert smc.analyze([_candle(1, 1, 1, 1)] * 3)["ok"] is False


def test_uptrend_reads_bullish():
    out = smc.analyze(_uptrend())
    assert out["ok"] is True
    assert out["bias"] == "up"
    assert "bullish" in out["structure"]


def test_downtrend_reads_bearish():
    down = []
    px = 1.1000
    for i in range(48):
        direction = -0.0006 if (i // 4) % 2 == 0 else 0.0004
        o = px
        px = round(px + direction, 6)
        down.append(_candle(o, max(o, px) + 0.0001, min(o, px) - 0.0001, px))
    out = smc.analyze(down)
    assert out["ok"] is True
    assert out["bias"] == "down"


def test_zone_and_position_are_bounded():
    out = smc.analyze(_uptrend())
    assert 0.0 <= out["position"] <= 1.5
    assert out["zone"] in (
        "premium (expensive) — sellers' area",
        "discount (cheap) — buyers' area",
        "equilibrium (fair value)",
    )


def test_sweep_detected_on_rejection_wick():
    candles = _uptrend(24)
    last_high = max(float(c["high"]) for c in candles)
    # Drift down a few bars, then a final spike that pierces the range high and
    # closes back below it — a textbook buy-side liquidity sweep.
    below = last_high - 0.0010
    for _ in range(4):
        candles.append(_candle(below, below + 0.0001, below - 0.0001, below))
    candles.append(_candle(below, last_high + 0.0020, below - 0.0002, below + 0.0002))
    out = smc.analyze(candles)
    assert out["ok"] is True
    assert "sweep" in out["liquidity_sweep"]
