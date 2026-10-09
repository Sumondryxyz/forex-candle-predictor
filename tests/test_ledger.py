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
