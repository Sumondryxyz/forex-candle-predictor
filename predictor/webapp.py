#!/usr/bin/env python3
"""Web app for the next-candle predictor.

Wraps the existing training/backtest pipeline in a small Flask server:

    python -m predictor.webapp            # http://127.0.0.1:5000
    PORT=12000 python -m predictor.webapp

Endpoints
---------
``GET /``               single-page dashboard
``GET /api/health``     liveness probe
``GET /api/config``     pairs/intervals the server accepts
``GET /api/signals``    latest UP/DOWN prediction for one pair+interval
``GET /api/backtest``   walk-forward accuracy vs confidence filter

Data comes from the same pluggable source as the CLI (``DATA_SOURCE``, i.e.
yfinance by default). Results are cached briefly so repeated page loads do not
re-fetch and re-train on every request.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

from flask import Flask, jsonify, render_template, request

from . import backtest, model
from .features import build_dataset
from .quotex_live import QuotexFeed
from .scheduler import is_market_open, next_candle_open
from .sources import get_fetcher

DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"]
DEFAULT_INTERVALS = ["1m", "5m", "15m"]

SIGNAL_TTL = int(os.getenv("WEB_SIGNAL_TTL", "120"))
BACKTEST_TTL = int(os.getenv("WEB_BACKTEST_TTL", "1800"))

# Walk-forward retrains a model per step, so cap the work a single request can
# trigger. The values mirror the CLI defaults but stay on the small side.
MAX_CANDLES = int(os.getenv("WEB_MAX_CANDLES", "5000"))
BACKTEST_TRAIN_SIZE = int(os.getenv("WEB_TRAIN_SIZE", "2000"))
BACKTEST_STEP = int(os.getenv("WEB_STEP", "500"))


class InputError(ValueError):
    """A request parameter the server refuses to act on."""


@dataclass
class _Cache:
    entries: dict[tuple, tuple[float, dict]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, key: tuple, ttl: int):
        with self.lock:
            hit = self.entries.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
        return None

    def put(self, key: tuple, value: dict) -> None:
        with self.lock:
            self.entries[key] = (time.time(), value)


_cache = _Cache()

# One background Quotex poller per (pair, interval), created on first request.
_feeds: dict[tuple[str, str], QuotexFeed] = {}
_feeds_lock = threading.Lock()


def get_feed(pair: str, interval: str) -> QuotexFeed:
    pair, interval = _validate(pair, interval)
    key = (pair, interval)
    with _feeds_lock:
        feed = _feeds.get(key)
        if feed is None:
            feed = QuotexFeed(pair, interval)
            feed.start()
            _feeds[key] = feed
        return feed


def _pairs() -> list[str]:
    raw = os.getenv("WEB_PAIRS")
    return [p.strip().upper() for p in raw.split(",") if p.strip()] if raw else DEFAULT_PAIRS


def _intervals() -> list[str]:
    raw = os.getenv("WEB_INTERVALS")
    return [i.strip() for i in raw.split(",") if i.strip()] if raw else DEFAULT_INTERVALS


def _days_for(interval: str) -> int:
    """Enough calendar days to clear MAX_CANDLES for an interval."""
    per_day = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24}.get(interval, 288)
    return max(1, -(-MAX_CANDLES // per_day))


def _validate(pair: str, interval: str) -> tuple[str, str]:
    pair = (pair or "EURUSD").upper()
    interval = interval or "5m"
    if interval not in _intervals():
        raise InputError(f"unsupported interval {interval!r}; use one of {_intervals()}")
    return pair, interval


def compute_signals(pair: str, interval: str) -> dict:
    pair, interval = _validate(pair, interval)
    key = ("signals", os.getenv("DATA_SOURCE", "yfinance"), pair, interval, MAX_CANDLES)
    cached = _cache.get(key, SIGNAL_TTL)
    if cached is not None:
        return cached

    df = get_fetcher()(pair, interval, _days_for(interval))
    df = df.tail(MAX_CANDLES)
    if len(df) < 300:
        raise InputError(f"only {len(df)} candles available for {pair} {interval}; try a longer interval")

    X, y = build_dataset(df)
    if len(X) < 200 or y.nunique() < 2:
        raise InputError("not enough history to train a model; try a longer interval")

    clf = model.train(X, y, calibrate=False)
    pred = model.predict_latest({"model": clf, "features": list(X.columns)}, df)

    recent = df.tail(60)
    payload = {
        "pair": pair,
        "interval": interval,
        "direction": pred.direction,
        "side": "CALL" if pred.direction == "UP" else "PUT",
        "probability": round(pred.probability, 4),
        "confidence": round(pred.confidence, 4),
        "tradeable": pred.is_tradeable,
        "last_close": float(df["close"].iloc[-1]),
        "last_candle_time": pred.timestamp.isoformat(),
        "next_candle_open": next_candle_open(interval).isoformat(),
        "market_open": is_market_open(),
        "candles": len(df),
        "recent": [
            {
                "time": ts.isoformat(),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
            }
            for ts, row in recent.iterrows()
        ],
    }
    _cache.put(key, payload)
    return payload


def compute_backtest(pair: str, interval: str) -> dict:
    pair, interval = _validate(pair, interval)
    key = ("backtest", os.getenv("DATA_SOURCE", "yfinance"), pair, interval, MAX_CANDLES)
    cached = _cache.get(key, BACKTEST_TTL)
    if cached is not None:
        return cached

    df = get_fetcher()(pair, interval, _days_for(interval))
    df = df.tail(MAX_CANDLES)
    X, _ = build_dataset(df)
    if len(X) <= BACKTEST_TRAIN_SIZE + BACKTEST_STEP:
        raise InputError("not enough history for a walk-forward backtest; try a longer interval")

    results = model.walk_forward(
        df,
        train_size=BACKTEST_TRAIN_SIZE,
        step=BACKTEST_STEP,
        calibrate=False,
    )
    payload = {
        "pair": pair,
        "interval": interval,
        "summary": backtest.summary(results),
        "by_confidence": backtest.accuracy_by_confidence(results).to_dict("records"),
        "payout": backtest.DEFAULT_PAYOUT,
    }
    _cache.put(key, payload)
    return payload


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            pairs=_pairs(),
            intervals=_intervals(),
            default_source=os.getenv("DATA_SOURCE", "yfinance"),
        )

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "source": os.getenv("DATA_SOURCE", "yfinance")})

    @app.get("/api/config")
    def config():
        return jsonify(
            {
                "pairs": _pairs(),
                "intervals": _intervals(),
                "sources": ["yfinance", "quotex"],
                "default_source": os.getenv("DATA_SOURCE", "yfinance"),
            }
        )

    @app.get("/api/signals")
    def signals():
        pair = request.args.get("pair", "EURUSD")
        interval = request.args.get("interval", "5m")
        source = request.args.get("source", "yfinance").lower()
        if source == "quotex":
            # Quotex is served by the background feed; /api/live is the same data.
            try:
                return jsonify(get_feed(pair, interval).snapshot())
            except InputError as exc:
                return jsonify({"error": str(exc)}), 400
        try:
            return jsonify(compute_signals(pair, interval))
        except InputError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001 - surface upstream failures as JSON
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 502

    @app.get("/api/live")
    def live():
        """Live Quotex feed: starts a background poller on first request."""
        pair = request.args.get("pair", "EURUSD")
        interval = request.args.get("interval", "5m")
        try:
            return jsonify(get_feed(pair, interval).snapshot())
        except InputError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/backtest")
    def backtest_view():
        pair = request.args.get("pair", "EURUSD")
        interval = request.args.get("interval", "5m")
        try:
            return jsonify(compute_backtest(pair, interval))
        except InputError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 502

    return app


app = create_app()


def main() -> None:
    port = int(os.getenv("PORT", "5000"))
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=port, debug=os.getenv("FLASK_DEBUG") == "1")


if __name__ == "__main__":
    main()
