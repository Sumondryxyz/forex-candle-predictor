import json

from predictor.ledger import Ledger


def test_record_appends_and_summarises(tmp_path):
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path=str(path))

    led.record(
        pair="EURUSD", interval="5m", source="tradingview", direction="UP",
        confidence=0.7, entry_close=1.1000, exit_close=1.1005, payout=0.8,
        candle_time=1_700_000_000,
    )
    led.record(
        pair="EURUSD", interval="5m", source="tradingview", direction="UP",
        confidence=0.66, entry_close=1.1000, exit_close=1.0995, payout=0.8,
        candle_time=1_700_000_300,
    )

    assert path.exists()
    assert len(led.rows()) == 2

    s = led.summary()
    assert s["trades"] == 2
    assert s["wins"] == 1 and s["losses"] == 1
    assert s["win_rate"] == 50.0
    # one +0.8 and one -1.0 -> mean = -0.1 -> -10%
    assert s["roi"] == -10.0
    assert s["by_combo"][0]["combo"] == "EURUSD 5m"
    assert s["recent"][0]["win"] is False  # newest first


def test_win_and_loss_flags_follow_the_direction(tmp_path):
    led = Ledger(path=str(tmp_path / "l.jsonl"))
    down = led.record(
        pair="GBPUSD", interval="15m", source="tradingview", direction="DOWN",
        confidence=0.7, entry_close=1.2000, exit_close=1.1990, payout=0.8,
    )
    assert down["win"] is True and down["pnl"] == 0.8
    up = led.record(
        pair="GBPUSD", interval="15m", source="tradingview", direction="UP",
        confidence=0.7, entry_close=1.2000, exit_close=1.1990, payout=0.8,
    )
    assert up["win"] is False and up["pnl"] == -1.0


def test_disabled_ledger_writes_nothing_but_still_reports(tmp_path):
    path = tmp_path / "off.jsonl"
    led = Ledger(path=str(path), enabled=False)
    row = led.record(
        pair="USDJPY", interval="5m", source="tradingview", direction="UP",
        confidence=0.7, entry_close=150.0, exit_close=150.1, payout=0.8,
    )
    assert row["win"] is True
    assert not path.exists()
    assert led.summary()["trades"] == 0


def test_partial_trailing_line_is_tolerated(tmp_path):
    path = tmp_path / "broken.jsonl"
    led = Ledger(path=str(path))
    led.record(
        pair="EURUSD", interval="5m", source="tradingview", direction="UP",
        confidence=0.7, entry_close=1.1, exit_close=1.1, payout=0.8,
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"pair": "EURUSD", "interval": "5m"')  # truncated, no newline
    rows = led.rows()
    assert len(rows) == 1
    assert json.loads(json.dumps(rows[0]))["pair"] == "EURUSD"


def test_same_candle_is_recorded_only_once(tmp_path):
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path=str(path))
    kw = {
        "pair": "EURUSD", "interval": "1m", "source": "tradingview",
        "direction": "UP", "confidence": 0.9, "entry_close": 1.1,
        "exit_close": 1.1005, "payout": 0.8, "candle_time": 1_700_000_000,
    }
    led.record(**kw)
    led.record(**kw)  # re-ingested candle, same resolution
    assert len(led.rows()) == 1
    assert led.summary()["trades"] == 1


def test_duplicate_rows_already_in_the_file_are_collapsed(tmp_path):
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path=str(path))
    row = led.record(pair="EURUSD", interval="1m", source="tradingview",
                     direction="UP", confidence=0.9, entry_close=1.1,
                     exit_close=1.1005, payout=0.8, candle_time=1_700_000_000)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")  # a legacy duplicate
    assert len(led.rows()) == 1
    assert led.summary()["trades"] == 1


def test_global_ledger_is_disabled_in_tests():
    """The suite must never write to the real var/ledger.jsonl (see conftest)."""
    from predictor import feed, webapp

    assert feed.LEDGER.enabled is False
    assert webapp.LEDGER.enabled is False


def test_unwritable_path_is_ignored_not_fatal(tmp_path, monkeypatch):
    """Render's free plan has no disk at /var/data; a bad LEDGER_PATH must not
    raise on every resolved candle -- the row is still returned, in memory."""
    ledger = Ledger(path=str(tmp_path / "nope" / "ledger.jsonl"))

    def boom(*a, **k):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("pathlib.Path.mkdir", boom)
    row = ledger.record(pair="EURUSD", interval="1m", source="tradingview",
                        direction="UP", confidence=0.8, entry_close=1.1,
                        exit_close=1.11, payout=0.8, candle_time=123.0)
    assert row["win"] is True
    assert ledger._write_failed is True
    assert ledger.last_error
