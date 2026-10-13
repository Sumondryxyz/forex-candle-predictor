import asyncio

import pandas as pd
import pytest

from predictor.sources import DEFAULT_SOURCE, get_fetcher
from predictor.sources import dukascopy as duka_source
from predictor.sources import quotex as quotex_source
from predictor.sources import tradingview as tv_source


def test_tradingview_is_the_default():
    assert DEFAULT_SOURCE == "tradingview"
    assert get_fetcher() is tv_source.fetch_ohlc
    assert get_fetcher("tradingview") is tv_source.fetch_ohlc


def test_quotex_is_still_selectable():
    assert get_fetcher("quotex") is quotex_source.fetch_ohlc


def test_dukascopy_is_selectable():
    assert get_fetcher("dukascopy") is duka_source.fetch_ohlc


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

    monkeypatch.setattr(tv_source, "CHUNK", 300)  # script chunks are 2 bars each
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


def test_get_candles_retries_through_a_429(monkeypatch):
    """A burst of pairs makes TradingView answer 429; the client must back off
    and retry instead of failing the feed."""
    import websockets

    monkeypatch.setattr(tv_source, "CHUNK", 300)
    good = [
        tv_source._frame("timescale_update", ["cs_1", {"sds_1": {"s": [
            {"i": 0, "v": [1704067200, 1.10, 1.11, 1.09, 1.105, 7.0]},
        ]}}]),
        tv_source._frame("series_completed", ["cs_1", "sds_1"]),
    ]
    attempts = {"n": 0}

    def connect(*a, **k):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError(
                "InvalidStatus: server rejected WebSocket connection: HTTP 429"
            )
        return _FakeWS(list(good))

    monkeypatch.setattr(websockets, "connect", connect)
    monkeypatch.setattr(tv_source, "_MIN_CONNECT_GAP", 0.0)

    async def _no_sleep(*_):
        return None

    monkeypatch.setattr(tv_source.asyncio, "sleep", _no_sleep)

    async def run():
        return await tv_source.TradingViewClient().get_candles("EURUSD", 0, 300, 60)

    rows = asyncio.run(run())
    assert attempts["n"] == 2  # first handshake 429'd, second succeeded
    assert len(rows) == 1




def _make_bi5(day, records):
    """Build an LZMA-compressed Dukascopy day from (ms, o, c, low, high, vol)."""
    import lzma
    import struct
    payload = b"".join(struct.pack(">5if", *r) for r in records)
    return lzma.compress(payload)


def test_dukascopy_decode_reads_records():
    import datetime as dt
    day = dt.date(2024, 1, 3)
    raw = _make_bi5(day, [(0, 108039, 108034, 108032, 108040, 86.15),
                          (60000, 108034, 108040, 108030, 108045, 12.5)])
    rows = duka_source._decode(raw, day)
    assert len(rows) == 2
    ts, o, h, low, c, v = rows[1]
    assert ts == dt.datetime(2024, 1, 3, 0, 1, tzinfo=dt.UTC)
    assert (o, h, low, c) == (1.08034, 1.08045, 1.08030, 1.08040)
    assert v == 12.5


def test_dukascopy_404_is_an_empty_day(monkeypatch):
    import datetime as dt
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.HTTPError("u", 404, "missing", {}, None)

    monkeypatch.setattr(duka_source.urllib.request, "urlopen", boom)
    assert duka_source._fetch_day("EURUSD", dt.date(2024, 1, 6)) == []


def test_dukascopy_resamples_1m_to_slower(monkeypatch):
    import datetime as dt

    idx = pd.date_range("2024-01-03", periods=5, freq="1min", tz="UTC")
    cols = {"open": [1.0, 1.1, 1.2, 1.3, 1.4],
            "high": [1.5, 1.1, 1.2, 1.9, 1.4],
            "low": [0.9, 1.0, 1.1, 1.2, 1.3],
            "close": [1.1, 1.2, 1.25, 1.4, 1.45],
            "volume": [1.0, 1.0, 1.0, 1.0, 1.0]}

    def fake_day(pair, day, **k):
        return [(t.to_pydatetime(), o, h, l, c, v) for t, o, h, l, c, v in
                zip(idx, cols["open"], cols["high"], cols["low"],
                    cols["close"], cols["volume"])]

    monkeypatch.setattr(duka_source, "_fetch_day", fake_day)
    monkeypatch.setattr(duka_source.time, "sleep", lambda *_: None)
    df = duka_source.fetch_ohlc("EURUSD", "5m", 1, end=dt.date(2024, 1, 3))
    assert len(df) == 1
    assert df.iloc[0]["open"] == 1.0
    assert df.iloc[0]["close"] == 1.45
    assert df.iloc[0]["high"] == 1.9


def test_rate_limit_detection():
    assert tv_source._is_rate_limited(RuntimeError("HTTP 429"))
    assert tv_source._is_rate_limited(RuntimeError("Too Many Requests"))
    assert not tv_source._is_rate_limited(RuntimeError("some other failure"))

