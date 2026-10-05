import numpy as np
import pandas as pd
import pytest

from predictor import webapp


@pytest.fixture
def candles() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 3200
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = 1.1 + np.cumsum(rng.normal(0, 1e-4, n))
    high = close + rng.uniform(0, 2e-4, n)
    low = close - rng.uniform(0, 2e-4, n)
    return pd.DataFrame(
        {"open": np.r_[close[0], close[:-1]], "high": high, "low": low, "close": close, "volume": 0},
        index=idx,
    )


@pytest.fixture
def client(monkeypatch, candles):
    # Keep the test offline: the real fetcher would hit yfinance.
    monkeypatch.setattr(webapp, "get_fetcher", lambda: (lambda pair, interval, days: candles))
    webapp._cache.entries.clear()
    app = webapp.create_app()
    app.testing = True
    return app.test_client()


def test_index_renders_dashboard(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Forex Candle Predictor" in res.data


def test_health_and_config(client):
    assert client.get("/api/health").get_json()["status"] == "ok"
    body = client.get("/api/config").get_json()
    assert "EURUSD" in body["pairs"]
    assert "5m" in body["intervals"]


def test_signals_endpoint_returns_prediction(client):
    body = client.get("/api/signals?pair=EURUSD&interval=5m").get_json()
    assert body["direction"] in {"UP", "DOWN"}
    assert body["side"] in {"CALL", "PUT"}
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["candles"] > 300
    assert len(body["recent"]) == 60


def test_backtest_endpoint_returns_confidence_table(client):
    body = client.get("/api/backtest?pair=EURUSD&interval=5m").get_json()
    assert body["summary"]["samples"] > 0
    assert body["by_confidence"][0]["coverage_pct"] == 100.0
    assert body["payout"] == webapp.backtest.DEFAULT_PAYOUT


def test_unsupported_interval_is_rejected(client):
    res = client.get("/api/signals?pair=EURUSD&interval=3m")
    assert res.status_code == 400
    assert "unsupported interval" in res.get_json()["error"]
