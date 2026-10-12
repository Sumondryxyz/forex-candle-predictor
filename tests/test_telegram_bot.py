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
        "last_result": {
            "direction": "UP", "confidence": 0.81, "entry_close": 1.0851,
            "exit_close": 1.0855, "win": True, "resolved_at": "2026-10-05T17:45:00+00:00",
        },
    }
    base.update(kw)
    return base


def _msg(text, chat_id=42):
    return {"chat": {"id": chat_id}, "text": text}


class _StubFeed:
    """Feed stub that serves a synthetic zigzag so SMC has structure to read."""

    def candles(self, limit=300):
        out = []
        px = 1.1000
        for i in range(60):
            step = 0.0006 if (i // 4) % 2 == 0 else -0.0004
            o = px
            px = round(px + step, 6)
            out.append({"open": o, "high": max(o, px) + 0.0001,
                        "low": min(o, px) - 0.0001, "close": px})
        return out


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


def test_format_signal_includes_last_result():
    out = tb.format_signal(_snap())
    assert "Last call: ✅WIN" in out


def test_format_result_win_and_loss():
    win = tb.format_result(_snap())
    assert "WIN" in win and "1.0851" in win and "1.0855" in win
    loss = tb.format_result(_snap(last_result={
        "direction": "DOWN", "confidence": 0.72, "entry_close": 1.1,
        "exit_close": 1.2, "win": False, "resolved_at": "2026-10-05T17:50:00+00:00"}))
    assert "LOSS" in loss and "🔴 DOWN" in loss


def test_format_result_none_yet():
    out = tb.format_result(_snap(last_result=None))
    assert "No resolved signal yet" in out


def test_result_command(state, monkeypatch):
    monkeypatch.setattr(tb, "_safe_snapshot", lambda p, i: _snap())
    tg = FakeTelegram()
    tb.handle(tg, state, _msg("/result EURUSD 5m"))
    assert "WIN" in tg.sent[-1][1]


def test_push_announces_result_once(state, monkeypatch):
    monkeypatch.setattr(tb, "PUSH_INTERVAL", 0.01)
    monkeypatch.setattr(tb, "_pairs", lambda: ["EURUSD"])
    monkeypatch.setattr(tb, "_intervals", lambda: ["5m"])
    monkeypatch.setattr(tb, "_safe_snapshot", lambda p, i: _snap(tradeable=False))
    state.update(42, subscribed=True)

    tg = FakeTelegram()
    import threading
    import time
    stop = threading.Event()
    t = threading.Thread(target=tb.push_loop, args=(tg, state, stop), daemon=True)
    t.start()
    time.sleep(0.2)
    stop.set()
    t.join(timeout=2)
    results = [s for s in tg.sent if "WIN" in s[1] or "LOSS" in s[1]]
    assert len(results) == 1, "a resolution must be announced only once"
    assert state.get(42)["last_result"] == "2026-10-05T17:45:00+00:00"


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

    pushes = [s for s in tg.sent if "Signal:" in s[1]]
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
    assert not [s for s in tg.sent if "Signal:" in s[1]]


# ---------------------------------------------------------------- smart scan
def _wf(rate, signals=50, roi=1.0):
    return {"rate": rate, "signals": signals, "roi": roi, "coverage": 10.0}


def test_format_smart_ranks_by_rate_with_medals():
    snaps = [
        _snap(pair="EURUSD", walk_forward=_wf(57.0)),
        _snap(pair="GBPUSD", walk_forward=_wf(48.0)),
        _snap(pair="USDJPY", walk_forward=_wf(60.0)),
    ]
    out = tb.format_smart(snaps)
    assert "Smart scan" in out
    assert "USDJPY" in out and "EURUSD" in out and "GBPUSD" in out
    assert "Best right now:" in out and "USDJPY" in out.split("Best right now:")[1]
    assert "🚫" in out and "✅" in out


def test_format_smart_handles_still_training():
    out = tb.format_smart([_snap(walk_forward={})])
    assert "Still training" in out


def test_smart_command_uses_stub_feed(state, monkeypatch):
    monkeypatch.setattr(tb, "_safe_snapshot",
                        lambda p, i: _snap(pair=p, interval=i, walk_forward=_wf(56.0)))
    tg = FakeTelegram()
    tb.handle(tg, state, _msg("/smart"))
    assert tg.sent and "Smart scan" in tg.sent[0][1]


# ---------------------------------------------------------------- risk
def _wf_frac(rate_pct):
    return {"rate": rate_pct, "signals": 80, "roi": 1.0, "coverage": 10.0}


def test_format_risk_positive_edge_suggests_a_stake():
    out = tb.format_risk(_snap(walk_forward=_wf_frac(58.0), payout=0.80), 1000.0)
    assert "positive-edge" in out
    assert "Suggested stake" in out
    assert "55.6%" in out  # break-even at 80% payout


def test_format_risk_no_edge_suggests_nothing():
    out = tb.format_risk(_snap(walk_forward=_wf_frac(50.0), payout=0.80), 1000.0)
    assert "no-edge" in out
    assert "No stake suggested" in out


def test_format_risk_unmeasured_when_no_walk_forward():
    out = tb.format_risk(_snap(walk_forward={}), 1000.0)
    assert "unmeasured" in out


def test_risk_command_parses_bankroll_and_remembers_it(state, monkeypatch):
    monkeypatch.setattr(tb, "_safe_snapshot",
                        lambda p, i: _snap(pair=p, interval=i, walk_forward=_wf_frac(58.0)))
    tg = FakeTelegram()
    tb.handle(tg, state, _msg("/risk EURUSD 5m 500"))
    assert tg.sent and "Risk" in tg.sent[0][1]
    assert "$500" in tg.sent[0][1]
    assert state.get(42)["bankroll"] == 500.0


# ---------------------------------------------------------------- smc
def test_format_smc_renders_structure(monkeypatch):
    from predictor import smc

    monkeypatch.setattr(tb, "get_feed", lambda p, i: _StubFeed())
    out = tb.format_smc("EURUSD", "5m")
    assert "SMC" in out
    assert "Structure" in out
    assert smc  # module import guard


def test_smc_command_routes(state, monkeypatch):
    monkeypatch.setattr(tb, "format_smc", lambda p, i: f"SMC {p} {i}")
    tg = FakeTelegram()
    tb.handle(tg, state, _msg("/smc GBPUSD 15m"))
    assert tg.sent and tg.sent[0][1] == "SMC GBPUSD 15m"


# ---------------------------------------------------------------- readiness
def test_format_readiness_no_trades(monkeypatch):
    out = tb.format_readiness({"trades": 0, "wins": 0}, 0.80)
    assert "READINESS" in out.upper()
    assert "FORWARD-ONLY" in out.upper()
    assert "no trades yet" in out


def test_format_readiness_go_on_strong_record():
    out = tb.format_readiness({"trades": 500, "wins": 300}, 0.80)
    assert "GO" in out
    assert "55.6%" in out  # break-even at 0.80 payout


class _StubLedger:
    def summary(self, recent=0):
        return {"trades": 0, "wins": 0}


def test_readiness_command_routes(state, monkeypatch):
    monkeypatch.setattr(tb, "LEDGER", _StubLedger())
    monkeypatch.setattr(tb, "_safe_snapshot", lambda p, i: _snap())
    tg = FakeTelegram()
    tb.handle(tg, state, _msg("/readiness"))
    assert tg.sent and "readiness" in tg.sent[0][1].lower()
