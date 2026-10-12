import asyncio

import pandas as pd
import pytest

from predictor.sources import DEFAULT_SOURCE, get_fetcher
from predictor.sources import quotex as quotex_source
from predictor.sources import tradingview as tv_source


def test_tradingview_is_the_default():
    assert DEFAULT_SOURCE == "tradingview"
    assert get_fetcher() is tv_source.fetch_ohlc
    assert get_fetcher("tradingview") is tv_source.fetch_ohlc


def test_quotex_is_still_selectable():
    assert get_fetcher("quotex") is quotex_source.fetch_ohlc


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="unknown source"):
        get_fetcher("binance")


def test_symbol_and_resolution_mapping():
    assert tv_source.symbol_for("eurusd") == "FX:EURUSD"
    assert tv_source.symbol_for("EURUSD_OTC") == "FX:EURUSD"
    assert tv_source.symbol_for("OANDA:EURUSD") == "OANDA:EURUSD"
    assert tv_source.resolution_for("1m") == "1"
    assert tv_source.resolution_for("15m") == "15"


def test_frame_and_parse_round_trip():
    raw = tv_source._frame("chart_create_session", ["cs_1", ""])
    assert raw.startswith("~m~")
    parsed = list(tv_source._parse(raw))
    assert parsed == [("json", {"m": "chart_create_session", "p": ["cs_1", ""]})]


def test_parse_handles_heartbeats():
    assert list(tv_source._parse("~m~3~m~~h~7")) == [("hb", "7")]


def test_tradingview_payload_is_normalised(monkeypatch):
    rows = [
        {"time": 1704067200, "open": 1.10, "high": 1.11, "low": 1.09, "close": 1.105, "volume": 7.0},
        {"time": 1704067260, "open": 1.105, "high": 1.12, "low": 1.10, "close": 1.115, "volume": 9.0},
    ]

    async def fake_get_candles(self, pair, end, count, period):
        return rows

    monkeypatch.setattr(tv_source.TradingViewClient, "get_candles", fake_get_candles)

    df = tv_source.fetch_ohlc("EURUSD", "1m", days=1)

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index[0] == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    assert df["volume"].tolist() == [7.0, 9.0]


def test_quotex_payload_is_normalised(monkeypatch):
    rows = [
        {"time": 1704067200, "open": 1.10, "high": 1.11, "low": 1.09, "close": 1.105, "ticks": 7},
        {"time": 1704067260, "open": 1.105, "high": 1.12, "low": 1.10, "close": 1.115, "ticks": 9},
    ]

    async def fake_download(pair, interval, candles):
        return rows

    monkeypatch.setattr(quotex_source, "_download", fake_download)

    df = quotex_source.fetch_ohlc("EURUSD", "1m", days=1)

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index[0] == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    assert df["volume"].tolist() == [7, 9]


class _FakeWS:
    """Scripted websocket that replays frames and answers request_more_data."""

    def __init__(self, script):
        self._script = list(script)
        self.sent = []

    async def send(self, data):
        self.sent.append(data)
        if "request_more_data" in data:
            # push one more history chunk, then complete
            self._script.append(tv_source._frame("timescale_update", ["cs_1", {"sds_1": {"s": [
                {"i": 2, "v": [1704067320, 1.115, 1.13, 1.11, 1.125, 3.0]},
            ]}}]))
            self._script.append(tv_source._frame("series_completed", ["cs_1", "sds_1"]))

    async def recv(self):
        if not self._script:
            raise TimeoutError  # client treats this as "no more data"
        return self._script.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def test_get_candles_assembles_bars_and_pages_history(monkeypatch):
    import websockets

    first_chunk = tv_source._frame("timescale_update", ["cs_1", {"sds_1": {"s": [
        {"i": 0, "v": [1704067200, 1.10, 1.11, 1.09, 1.105, 7.0]},
        {"i": 1, "v": [1704067260, 1.105, 1.12, 1.10, 1.115, 9.0]},
    ]}}])
    script = [
        tv_source._frame("symbol_resolved", ["cs_1", "sds_sym_1", {}]),
        "~m~3~m~~h~7",  # heartbeat must be echoed, not parsed as JSON
        first_chunk,
        tv_source._frame("series_completed", ["cs_1", "sds_1"]),
    ]
    ws = _FakeWS(script)
    monkeypatch.setattr(websockets, "connect", lambda *a, **k: ws)

    async def run():
        return await tv_source.TradingViewClient().get_candles("EURUSD", 0, 600, 60)

    rows = asyncio.run(run())

    assert [r["time"] for r in rows] == [1704067200, 1704067260, 1704067320]
    assert rows[-1]["volume"] == 3.0
    assert any("request_more_data" in s for s in ws.sent)
    assert any("~h~7" in s for s in ws.sent)  # heartbeat echoed back

