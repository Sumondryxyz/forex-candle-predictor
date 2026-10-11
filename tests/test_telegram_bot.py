"""Tests for the Telegram front end.

No network: the Telegram API and the live feed are stubbed, but the real
formatting and command routing run.
"""
import pytest

from predictor import telegram_bot as tb


class FakeTelegram:
    def __init__(self):
        self.sent = []

    def send(self, chat_id, text, **extra):
        self.sent.append((chat_id, text))

    def call(self, method, **params):
        return {"username": "testbot"}


@pytest.fixture()
def state(tmp_path):
    return tb.State(tmp_path / "state.json")


def _snap(**kw):
    base = {
        "pair": "EURUSD", "interval": "5m", "source": "tradingview", "status": "live",
        "direction": "UP", "confidence": 0.81, "tradeable": True, "min_conf": 0.70,
        "candles": 240, "payout": 0.85,
        "signal_candle_open": "2026-10-05T17:45:00+00:00",
        "session": {"signals": 4, "wins": 3, "rate": 75.0},
    }
    base.update(kw)
    return base


def _msg(text, chat_id=42):
    return {"chat": {"id": chat_id}, "text": text}


# ---------------------------------------------------------------- formatting
def test_format_signal_shows_direction_confidence_floor():
    out = tb.format_signal(_snap())
    assert "EURUSD" in out and "5m" in out
    assert "UP (CALL)" in out
    assert "81.0%" in out
    assert "70.0%" in out          # confidence floor
    assert "no trades" in out.lower()


def test_format_signal_below_floor_is_marked_not_tradeable():
    out = tb.format_signal(_snap(confidence=0.55, tradeable=False))
    assert "below floor" in out


def test_format_signal_error_and_waiting():
    assert "boom" in tb.format_signal({"pair": "EURUSD", "interval": "1m", "error": "boom"})
    waiting = tb.format_signal({"pair": "EURUSD", "interval": "1m", "direction": None,
                                "status": "warming up", "source": "tradingview"})
    assert "No signal yet" in waiting


def test_format_scan_lists_pairs():
    out = tb.format_scan([_snap(), _snap(pair="GBPUSD", direction="DOWN", tradeable=False)])
    assert "EURUSD" in out and "GBPUSD" in out
    assert "🟢UP" in out and "🔴DOWN" in out


def test_split_chunks_long_messages():
    chunks = list(tb._split("x" * 9000, 3800))
    assert all(len(c) <= 3800 for c in chunks)
    assert sum(len(c) for c in chunks) == 9000


# ---------------------------------------------------------------- commands
def test_help_and_start(state):
    tg = FakeTelegram()
    tb.handle(tg, state, _msg("/help"))
    assert "never places a trade" in tg.sent[-1][1]


def test_setpair_and_setinterval_validate(state):
    tg = FakeTelegram()
    tb.handle(tg, state, _msg("/setpair GBPUSD"))
    tb.handle(tg, state, _msg("/setinterval 15m"))
    prefs = state.get(42)
    assert prefs["pair"] == "GBPUSD" and prefs["interval"] == "15m"

    tb.handle(tg, state, _msg("/setpair NOPE"))
    assert state.get(42)["pair"] == "GBPUSD"  # unchanged
    assert "Usage" in tg.sent[-1][1]


def test_subscribe_toggle(state):
    tg = FakeTelegram()
    tb.handle(tg, state, _msg("/subscribe"))
    assert state.get(42)["subscribed"] is True
    tb.handle(tg, state, _msg("/unsubscribe"))
    assert state.get(42)["subscribed"] is False


def test_signal_uses_stub_feed(state, monkeypatch):
    monkeypatch.setattr(tb, "_safe_snapshot", lambda p, i: _snap(pair=p, interval=i))
    tg = FakeTelegram()
    tb.handle(tg, state, _msg("/signal EURUSD 5m"))
    assert "UP (CALL)" in tg.sent[-1][1]


def test_signal_rejects_unknown_pair(state):
    tg = FakeTelegram()
    tb.handle(tg, state, _msg("/signal ZZZZZZ 5m"))
    assert "Unknown pair" in tg.sent[-1][1]


def test_scan_uses_stub_feed(state, monkeypatch):
    monkeypatch.setattr(tb, "_safe_snapshot", lambda p, i: _snap(pair=p, interval=i))
    tg = FakeTelegram()
    tb.handle(tg, state, _msg("/scan"))
    assert "Scan" in tg.sent[-1][1]


def test_session_report(state, monkeypatch):
    monkeypatch.setattr(tb, "_safe_snapshot", lambda p, i: _snap())
    tg = FakeTelegram()
    tb.handle(tg, state, _msg("/session"))
    body = tg.sent[-1][1]
    assert "Wins: 3" in body and "Paper only" in body


def test_unknown_command(state):
    tg = FakeTelegram()
    tb.handle(tg, state, _msg("/wat"))
    assert "Unknown command" in tg.sent[-1][1]


def test_non_command_text(state):
    tg = FakeTelegram()
    tb.handle(tg, state, _msg("hello"))
    assert "/help" in tg.sent[-1][1]


# ---------------------------------------------------------------- push
def test_push_sends_once_per_candle(state, monkeypatch):
    monkeypatch.setattr(tb, "PUSH_INTERVAL", 0.01)
    monkeypatch.setattr(tb, "_pairs", lambda: ["EURUSD"])
    monkeypatch.setattr(tb, "_intervals", lambda: ["5m"])
    monkeypatch.setattr(tb, "_safe_snapshot", lambda p, i: _snap())
    state.update(42, subscribed=True, pair="EURUSD", interval="5m")

    tg = FakeTelegram()
    import threading
    stop = threading.Event()
    t = threading.Thread(target=tb.push_loop, args=(tg, state, stop), daemon=True)
    t.start()
    import time
    time.sleep(0.2)
    stop.set()
    t.join(timeout=2)

    pushes = [s for s in tg.sent if s[1].startswith("🔔")]
    assert len(pushes) == 1, "same candle must be pushed only once"
    assert state.get(42)["last_signal"] == "2026-10-05T17:45:00+00:00"


def test_push_skips_untradeable(state, monkeypatch):
    monkeypatch.setattr(tb, "PUSH_INTERVAL", 0.01)
    monkeypatch.setattr(tb, "_pairs", lambda: ["EURUSD"])
    monkeypatch.setattr(tb, "_intervals", lambda: ["5m"])
    monkeypatch.setattr(tb, "_safe_snapshot", lambda p, i: _snap(tradeable=False))
    state.update(42, subscribed=True)

    tg = FakeTelegram()
    import threading
    stop = threading.Event()
    t = threading.Thread(target=tb.push_loop, args=(tg, state, stop), daemon=True)
    t.start()
    import time
    time.sleep(0.15)
    stop.set()
    t.join(timeout=2)
    assert not [s for s in tg.sent if s[1].startswith("🔔")]
