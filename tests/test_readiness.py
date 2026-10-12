"""Tests for the go-live readiness gate. Pure arithmetic, no network."""

from predictor import readiness


def test_wilson_lower_bound_is_below_point_estimate():
    lower = readiness.wilson_lower_bound(60, 100)
    assert lower is not None and 0.50 < lower < 0.60


def test_wilson_zero_trades_is_none():
    assert readiness.wilson_lower_bound(0, 0) is None


def test_no_trades_is_forward_only():
    out = readiness.assess(forward_wins=0, forward_trades=0, payout=0.80)
    assert out["verdict"] == "forward-only"
    assert out["reason"].startswith("no resolved forward")


def test_below_break_even_is_no_go():
    out = readiness.assess(forward_wins=50, forward_trades=100, payout=0.80)
    assert out["verdict"] == "no-go"


def test_small_positive_sample_is_forward_only():
    # 12/20 = 60% beats break-even but is far too small to trust.
    out = readiness.assess(forward_wins=12, forward_trades=20, payout=0.80)
    assert out["verdict"] == "forward-only"
    assert "only 20 trades" in out["reason"]


def test_large_strong_sample_is_go():
    # 60% over 500 trades: lower bound ~55.7% > 55.6% break-even, p tiny.
    out = readiness.assess(forward_wins=300, forward_trades=500, payout=0.80)
    assert out["verdict"] == "go"
    assert out["wilson_lower"] > out["break_even"]
    assert out["p_value"] < 0.05


def test_large_marginal_sample_is_forward_only_not_go():
    # 56% over 200 trades: positive but the lower bound dips to ~49%, not real.
    out = readiness.assess(forward_wins=112, forward_trades=200, payout=0.80)
    assert out["verdict"] == "forward-only"


def test_passes_through_walk_forward_context():
    out = readiness.assess(
        forward_wins=300, forward_trades=500, payout=0.80,
        walk_forward_rate=58.0, walk_forward_signals=90,
    )
    assert out["walk_forward_rate"] == 58.0
    assert out["walk_forward_signals"] == 90
