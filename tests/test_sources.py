import pandas as pd

from predictor.sources import get_fetcher
from predictor.sources import quotex as quotex_source


def test_quotex_is_the_only_source():
    assert get_fetcher() is quotex_source.fetch_ohlc
    assert get_fetcher("quotex") is quotex_source.fetch_ohlc
    assert get_fetcher("whatever") is quotex_source.fetch_ohlc


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
