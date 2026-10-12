"""Timing helpers for candle-close aligned signals."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

INTERVAL_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}

# How early before the next candle opens the signal should be delivered.
LEAD_SECONDS = 5


def interval_seconds(interval: str) -> int:
    if interval not in INTERVAL_SECONDS:
        raise ValueError(f"unsupported interval {interval}; use one of {sorted(INTERVAL_SECONDS)}")
    return INTERVAL_SECONDS[interval]


def seconds_until_next_close(interval: str, now: datetime | None = None) -> float:
    """Seconds until the current candle closes, minus the delivery lead time."""
    step = interval_seconds(interval)
    now = now or datetime.now(UTC)
    epoch = now.timestamp()
    remaining = step - (epoch % step)
    if remaining <= LEAD_SECONDS:
        remaining += step
    return remaining - LEAD_SECONDS


def next_candle_open(interval: str, now: datetime | None = None) -> datetime:
    step = interval_seconds(interval)
    now = now or datetime.now(UTC)
    return datetime.fromtimestamp((now.timestamp() // step + 1) * step, tz=UTC)


def is_market_open(now: datetime | None = None) -> bool:
    """Spot forex trades from Sunday 21:00 UTC to Friday 21:00 UTC."""
    now = now or datetime.now(UTC)
    weekday = now.weekday()
    if weekday == 5:
        return False
    if weekday == 6:
        return now.hour >= 21
    if weekday == 4:
        return now.hour < 21
    return True


def market_reopen(now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    ahead = now
    while not is_market_open(ahead):
        ahead += timedelta(minutes=15)
    return ahead
