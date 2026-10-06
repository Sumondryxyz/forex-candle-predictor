import pytest

from predictor import webapp

SNAPSHOT = {
    "pair": "EURUSD",
    "interval": "5m",
    "source": "tradingview",
    "status": "live",
    "error": None,
    "account": "-",
    "market_open": True,
    "next_candle_open": "2024-01-01T00:05:00+00:00",
    "updated": "2024-01-01T00:00:00+00:00",
    "candles": 1200,
    "walk_forward": {"signals": 40, "rate": 60.0, "coverage": 5.0, "roi": 8.0},
    "session": {"signals": 0, "wins": 0, "rate": None},
    "direction": "UP",
    "side": "CALL",
    "probability": 0.62,
    "confidence": 0.62,
    "tradeable": True,
    "min_conf": 0.6,
    "payout": 0.8,
    "last_close": 1.1,
    "last_candle_time": "2024-01-01T00:00:00+00:00",
    "recent": [{"time": "t", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 1.0}] * 60,
}


class FakeFeed:
    """Stands in for a real feed so the routes are tested without a network."""

    source = "tradingview"

    def __init__(self, pair, interval):
        self.pair, self.interval = pair, interval

    def start(self):
        pass

    def snapshot(self):
        return dict(SNAPSHOT, pair=self.pair, interval=self.interval, source=self.source)

    def candles(self, limit=500):
        row = {"time": 1, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 1.0}
        return [row] * min(limit, 60)


def _fake_feed(source):
    class _Fake(FakeFeed):
        pass

    _Fake.source = source
    return _Fake


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        webapp,
        "SOURCES",
        {"tradingview": _fake_feed("tradingview"), "quotex": _fake_feed("quotex")},
    )
    monkeypatch.setattr(webapp, "DEFAULT_SOURCE", "tradingview")
    webapp._feeds.clear()
    app = webapp.create_app()
    app.testing = True
    return app.test_client()


def test_index_renders_dashboard(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Forex Candle Predictor" in res.data
    assert b"yfinance" not in res.data


def test_health_and_config_expose_sources(client):
    health = client.get("/api/health").get_json()
    assert health["status"] == "ok"
    assert health["source"] == "tradingview"
    assert health["model"] in {"linear", "numpy"}
    body = client.get("/api/config").get_json()
    assert body["default_source"] == "tradingview"
    assert body["sources"] == ["quotex", "tradingview"]
    assert "EURUSD" in body["pairs"]
    assert "5m" in body["intervals"]


def test_live_endpoint_returns_feed_snapshot(client):
    body = client.get("/api/live?pair=EURUSD&interval=5m").get_json()
    assert body["source"] == "tradingview"
    assert body["direction"] == "UP"
    assert body["confidence"] == 0.62
    assert len(body["recent"]) == 60


def test_source_can_be_selected_per_request(client):
    body = client.get("/api/live?pair=EURUSD&interval=5m&source=quotex").get_json()
    assert body["source"] == "quotex"


def test_unknown_source_is_rejected(client):
    res = client.get("/api/live?pair=EURUSD&interval=5m&source=binance")
    assert res.status_code == 400
    assert "unknown source" in res.get_json()["error"]


def test_signals_is_an_alias_of_live(client):
    live = client.get("/api/live?pair=GBPUSD&interval=1m").get_json()
    signals = client.get("/api/signals?pair=GBPUSD&interval=1m").get_json()
    assert signals == live


def test_candles_endpoint_returns_raw_bars(client):
    body = client.get("/api/candles?pair=EURUSD&interval=5m&limit=100").get_json()
    assert body["count"] == 60
    assert set(body["candles"][0]) == {"time", "open", "high", "low", "close", "volume"}


def test_backtest_returns_the_feed_walk_forward(client):
    body = client.get("/api/backtest?pair=EURUSD&interval=5m").get_json()
    assert body["walk_forward"]["rate"] == 60.0
    assert body["payout"] == 0.8
    assert body["candles"] == 1200


def test_unsupported_interval_is_rejected(client):
    res = client.get("/api/live?pair=EURUSD&interval=3m")
    assert res.status_code == 400
    assert "unsupported interval" in res.get_json()["error"]
