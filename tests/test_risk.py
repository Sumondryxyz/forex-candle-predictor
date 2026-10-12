"""Tests for risk and position sizing. Pure arithmetic, no network."""

import pytest

from predictor import risk


def test_break_even_matches_the_stated_80_payout_number():
    # The app has always said 55.6% at 80% payout; lock that in.
    assert risk.break_even_rate(0.80) == pytest.approx(0.5556, abs=1e-4)
    assert risk.break_even_rate(0.90) == pytest.approx(0.5263, abs=1e-4)


def test_expected_value_sign_flips_at_break_even():
    assert risk.expected_value(0.58, 0.80) > 0
    assert risk.expected_value(0.50, 0.80) < 0
    assert risk.expected_value(risk.break_even_rate(0.80), 0.80) == pytest.approx(0.0, abs=1e-9)


def test_kelly_is_zero_without_an_edge():
    assert risk.kelly_fraction(0.50, 0.80) == 0.0
    assert risk.kelly_fraction(0.58, 0.80) == pytest.approx(0.055, abs=1e-3)


def test_suggest_stake_caps_at_the_max_fraction():
    # A suspiciously high win rate must still be capped for safety.
    out = risk.suggest_stake(0.95, 0.80, 1000)
    assert out["stake_fraction"] == risk.MAX_STAKE_FRACTION
    assert out["stake"] == 20.0


def test_suggest_stake_is_zero_below_break_even():
    out = risk.suggest_stake(0.54, 0.80, 1000)
    assert out["stake"] == 0.0
    assert out["tradeable"] is False
    assert out["expected_value"] < 0


def test_assess_unmeasured_blocks_sizing():
    out = risk.assess(None, 0.80, 1000)
    assert out["verdict"] == "unmeasured"
    assert out["sizing"] is None


def test_assess_verdicts():
    assert risk.assess(0.50, 0.80, 1000)["verdict"] == "no-edge"
    assert risk.assess(0.56, 0.80, 1000)["verdict"] == "marginal"
    assert risk.assess(0.58, 0.80, 1000)["verdict"] == "positive-edge"


def _row(win, payout=0.8):
    return {"win": win, "payout": payout}


def test_simulate_bankroll_compounds_wins_and_tracks_drawdown():
    rows = [_row(True)] * 3
    out = risk.simulate_bankroll(rows, 1000.0, stake_fraction=0.01)
    assert out["trades"] == 3
    assert out["end"] > 1000.0
    assert out["max_drawdown_pct"] == 0.0
    assert len(out["curve"]) == 4


def test_simulate_bankroll_losing_run_draws_down_and_stops_at_zero():
    out = risk.simulate_bankroll([_row(False)] * 5, 1000.0, stake_fraction=0.10)
    assert out["end"] < 1000.0
    assert out["max_drawdown_pct"] > 20
    busted = risk.simulate_bankroll([_row(False)] * 200, 1000.0, stake_fraction=0.10)
    assert busted["end"] == 0.0
    assert busted["curve"][-1] == 0.0
