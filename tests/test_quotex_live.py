import asyncio
import time

import numpy as np

from predictor.quotex_live import QuotexFeed


def _candles(n: int, start: int | None = None, seed: int = 0):
    # Anchor history to wall-clock now so the live-freshness guard treats the
    # final candles as current (see BaseFeed._ingest).
    if start is None:
        start = int(time.time()) - n * 60
    rng = np.random.default_rng(seed)
    close = 1.1 + np.cumsum(rng.normal(0, 1e-4, n))
    high = close + rng.uniform(0, 2e-4, n)
    low = close - rng.uniform(0, 2e-4, n)
    open_ = np.r_[close[0], close[:-1]]
    return [
        {
            "time": start + i * 60,
            "open": float(open_[i]),
            "high": float(high[i]),
            "low": float(low[i]),
            "close": float(close[i]),
        }
        for i in range(n)
    ]


class FakeClient:
    """Stands in for pyquotex's Quotex: same async surface the feed uses."""

    def __init__(self, rows):
        self.rows = rows
        self.closed = False
        self.connected = False

    async def connect(self):
        self.connected = True
        return True, None

    async def get_candles(self, pair, end, count, period):
        return self.rows

    def close(self):
        self.closed = True


def _bootstrap(feed, rows):
    client = FakeClient(rows)
    asyncio.run(feed._bootstrap(client))
    return client


def test_snapshot_is_empty_before_bootstrap():
    feed = QuotexFeed("EURUSD", "1m")
    snap = feed.snapshot()
    assert snap["status"] == "idle"
    assert snap["direction"] is None
    assert snap["candles"] == 0


def test_bootstrap_trains_and_reports_a_signal():
    feed = QuotexFeed("EURUSD", "1m")
    _bootstrap(feed, _candles(1200))

    snap = feed.snapshot()
    assert snap["status"] == "live"
    assert snap["error"] is None
    assert snap["direction"] in {"UP", "DOWN"}
    assert snap["side"] in {"CALL", "PUT"}
    assert 0.5 <= snap["confidence"] <= 1.0
    assert snap["candles"] == 1200
    assert snap["walk_forward"] is not None
    assert len(snap["recent"]) == 60


def test_signal_candle_is_the_next_candle_after_the_last_closed_one():
    # The call is for the next candle: it opens exactly one interval after the
    # last closed candle and closes one interval later. This is what removes the
    # "running or next candle?" ambiguity in the UI.
    from datetime import datetime

    feed = QuotexFeed("EURUSD", "1m")
    _bootstrap(feed, _candles(1200, start=1_700_000_000))
    snap = feed.snapshot()
    last = datetime.fromisoformat(snap["last_candle_time"]).timestamp()
    open_t = datetime.fromisoformat(snap["signal_candle_open"]).timestamp()
    close_t = datetime.fromisoformat(snap["signal_candle_close"]).timestamp()
    assert open_t == last + 60  # 1m interval
    assert close_t == open_t + 60


def test_ingest_resolves_pending_signal():
    feed = QuotexFeed("EURUSD", "1m")
    feed.min_conf = 0.0  # force every candle to become a pending signal
    history = _candles(1200)
    _bootstrap(feed, history)
    t0 = history[-1]["time"]

    first = _candles(1, start=t0 + 60, seed=1)[0]
    feed._ingest([first])
    assert feed.snapshot()["pending"]["ref_close"] == first["close"]

    second = _candles(1, start=t0 + 120, seed=2)[0]
    feed._ingest([second])
    session = feed.snapshot()["session"]
    assert session["signals"] == 1
    assert session["wins"] in {0, 1}
    # The just-closed candle immediately opens a new pending signal.
    assert feed.snapshot()["pending"]["ref_close"] == second["close"]

def test_gapped_candle_does_not_resolve_the_signal(tmp_path, monkeypatch):
    # Regression: a candle that is NOT the immediate next bar (e.g. a Quotex
    # history-backfill artifact from another era) used to be recorded as a
    # resolved signal, inflating the ledger to a fake ~97% win rate.
    from predictor import feed as feed_mod

    led = feed_mod.Ledger(path=str(tmp_path / "l.jsonl"))
    monkeypatch.setattr(feed_mod, "LEDGER", led)
    f = feed_mod.BaseFeed("EURUSD", "1m")
    f.min_conf = 0.0
    _bootstrap(f, _candles(1200))

    # A candle from an entirely different era: 1m period expected, gap is huge.
    stale = _candles(1, start=1_600_000_000, seed=3)[0]
    f._ingest([stale])
    assert led.rows() == []


def test_stale_contiguous_series_does_not_resolve(tmp_path, monkeypatch):
    # The actual reported bug: Quotex bootstrapped a *contiguous* 2023 series
    # (step == period) as "history", so the gap check passed and the replay got
    # recorded as resolved signals -> fake ~97% win rate. Freshness must reject
    # a series from the past even when candles are perfectly spaced.
    from predictor import feed as feed_mod

    led = feed_mod.Ledger(path=str(tmp_path / "l.jsonl"))
    monkeypatch.setattr(feed_mod, "LEDGER", led)
    f = feed_mod.BaseFeed("EURUSD", "1m")
    f.min_conf = 0.0
    old = 1_700_000_000  # ~Nov 2023
    _bootstrap(f, _candles(1200, start=old))
    t0 = f._candles[-1]["time"]
    f._ingest(_candles(1, start=t0 + 60, seed=7))  # next stale candle, step == 60
    assert led.rows() == []


def test_snapshot_is_json_serialisable_after_resolving_signals():
    """Regression: numpy bools/int64 in the session must not break jsonify.

    With a strong model most candles clear min_conf, so the pending signal
    resolves into ``_results``; a raw numpy comparison there used to make
    ``sum(results)`` an int64 and /api/live returned 500.
    """
    import json

    feed = QuotexFeed("EURUSD", "1m")
    feed.min_conf = 0.0  # force every candle to become a pending signal
    history = _candles(1200)
    _bootstrap(feed, history)
    t0 = history[-1]["time"]

    for i in range(3):
        feed._ingest(_candles(1, start=t0 + (i + 1) * 60, seed=i + 1))

    snap = feed.snapshot()
    assert snap["session"]["signals"] == 2
    json.dumps(snap)  # would raise TypeError with a numpy scalar


def test_ingest_ignores_duplicate_and_incomplete_candles():
    feed = QuotexFeed("EURUSD", "1m")
    _bootstrap(feed, _candles(1200))
    dup = feed.snapshot()["candles"]
    feed._ingest([{"time": 1_700_000_000, "open": 1, "high": 1, "low": 1, "close": None}])
    assert feed.snapshot()["candles"] == dup



def test_use_volume_flag_defaults_off_and_can_be_enabled(monkeypatch):
    monkeypatch.delenv("USE_VOLUME", raising=False)
    assert QuotexFeed("EURUSD", "5m").use_volume is False
    monkeypatch.setenv("USE_VOLUME", "true")
    assert QuotexFeed("EURUSD", "5m").use_volume is True


def test_short_history_raises():
    feed = QuotexFeed("EURUSD", "1m")
    try:
        asyncio.run(feed._bootstrap(FakeClient(_candles(100))))
    except RuntimeError as exc:
        assert "history candles" in str(exc)
    else:  # pragma: no cover - would mean the guard is missing
        raise AssertionError("expected a RuntimeError for short history")


def test_background_thread_reaches_live_then_stops(monkeypatch):
    import time

    rows = _candles(1200)
    clients = []

    def factory():
        client = FakeClient(rows)
        clients.append(client)
        return client

    monkeypatch.setenv("QUOTEX_EMAIL", "demo@example.com")
    monkeypatch.setenv("QUOTEX_PASSWORD", "secret")
    feed = QuotexFeed("EURUSD", "1m", client_factory=factory)
    feed.start()
    try:
        for _ in range(50):
            if feed.snapshot()["status"] == "live":
                break
            time.sleep(0.1)
        snap = feed.snapshot()
        assert snap["status"] == "live"
        assert snap["direction"] in {"UP", "DOWN"}
        assert clients and clients[0].connected
    finally:
        feed.stop()
    assert not feed._thread.is_alive()
