import pandas as pd
import pytest

from predictor import data
from predictor.sources import get_fetcher
from predictor.sources import quotex as quotex_source


def test_default_source_is_yfinance():
    assert get_fetcher() is data.fetch_ohlc
    assert get_fetcher("yfinance") is data.fetch_ohlc


def test_quotex_source_is_selectable():
    assert get_fetcher("quotex") is quotex_source.fetch_ohlc


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="unknown DATA_SOURCE"):
        get_fetcher("mt5")


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
