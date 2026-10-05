#!/usr/bin/env python3
"""Web app for the next-candle predictor, backed by the live Quotex feed.

    python -m predictor.webapp            # http://127.0.0.1:5000
    PORT=12000 python -m predictor.webapp

Endpoints
---------
``GET /``               single-page dashboard
``GET /api/health``     liveness probe
``GET /api/config``     pairs/intervals the server accepts
``GET /api/signals``    latest UP/DOWN prediction for one pair+interval
``GET /api/live``       same as ``/api/signals`` (the background Quotex feed)
``GET /api/backtest``   walk-forward accuracy the feed computed on its history

Every pair/interval is served by one background ``QuotexFeed`` that logs into
the broker, trains the numpy engine and keeps the latest signal in memory. The
bot never places a trade. Quotex blocks logins from datacenter IPs, so run this
from a home or mobile connection.
"""
from __future__ import annotations

import os
import threading

from flask import Flask, jsonify, render_template, request

from .quotex_live import QuotexFeed

DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"]
DEFAULT_INTERVALS = ["1m", "5m", "15m"]


class InputError(ValueError):
    """A request parameter the server refuses to act on."""


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


def _validate(pair: str, interval: str) -> tuple[str, str]:
    pair = (pair or "EURUSD").upper()
    interval = interval or "5m"
    if interval not in _intervals():
        raise InputError(f"unsupported interval {interval!r}; use one of {_intervals()}")
    return pair, interval


def compute_signals(pair: str, interval: str) -> dict:
    """Latest signal from the background Quotex feed (starts it if needed)."""
    return get_feed(pair, interval).snapshot()


def compute_backtest(pair: str, interval: str) -> dict:
    """The feed's own walk-forward check on the history it trained on."""
    pair, interval = _validate(pair, interval)
    snap = get_feed(pair, interval).snapshot()
    return {
        "pair": pair,
        "interval": interval,
        "status": snap["status"],
        "error": snap["error"],
        "candles": snap["candles"],
        "payout": snap.get("payout"),
        "min_conf": snap.get("min_conf"),
        "walk_forward": snap.get("walk_forward"),
    }


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("index.html", pairs=_pairs(), intervals=_intervals())

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "source": "quotex"})

    @app.get("/api/config")
    def config():
        return jsonify({"pairs": _pairs(), "intervals": _intervals(), "source": "quotex"})

    @app.get("/api/signals")
    @app.get("/api/live")
    def signals():
        """Live Quotex feed: starts a background poller on first request."""
        pair = request.args.get("pair", "EURUSD")
        interval = request.args.get("interval", "5m")
        try:
            return jsonify(compute_signals(pair, interval))
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

    return app


app = create_app()


def main() -> None:
    port = int(os.getenv("PORT", "5000"))
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=port, debug=os.getenv("FLASK_DEBUG") == "1")


if __name__ == "__main__":
    main()
