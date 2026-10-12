from datetime import UTC, datetime

import pytest

from predictor.scheduler import (
    LEAD_SECONDS,
    is_market_open,
    next_candle_open,
    seconds_until_next_close,
)


def _utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


def test_seconds_until_next_close_fires_before_the_candle_ends():
    now = _utc(2024, 1, 3, 10, 0, 20)
    assert seconds_until_next_close("1m", now) == 60 - 20 - LEAD_SECONDS


def test_close_within_lead_time_skips_to_the_following_candle():
    now = _utc(2024, 1, 3, 10, 0, 58)
    assert seconds_until_next_close("1m", now) == pytest.approx(60 - 58 + 60 - LEAD_SECONDS)


def test_next_candle_open_is_aligned_to_the_interval():
    assert next_candle_open("5m", _utc(2024, 1, 3, 10, 2, 30)) == _utc(2024, 1, 3, 10, 5)


def test_unsupported_interval_is_rejected():
    with pytest.raises(ValueError, match="unsupported interval"):
        seconds_until_next_close("3m")


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (_utc(2024, 1, 3, 12, 0), True),      # Wednesday
        (_utc(2024, 1, 6, 12, 0), False),     # Saturday
        (_utc(2024, 1, 7, 20, 0), False),     # Sunday before the open
        (_utc(2024, 1, 7, 22, 0), True),      # Sunday after the open
        (_utc(2024, 1, 5, 22, 0), False),     # Friday after the close
    ],
)
def test_market_hours(moment, expected):
    assert is_market_open(moment) is expected
