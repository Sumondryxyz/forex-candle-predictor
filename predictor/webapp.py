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

from . import engine
from .feed import LEDGER
from .quotex_live import QuotexFeed
from .tradingview_live import TradingViewFeed

DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"]
DEFAULT_INTERVALS = ["1m", "5m", "15m"]

SOURCES = {"tradingview": TradingViewFeed, "quotex": QuotexFeed}
DEFAULT_SOURCE = os.getenv("DATA_SOURCE", "tradingview").lower()


class InputError(ValueError):
    """A request parameter the server refuses to act on."""


# One background poller per (source, pair, interval), created on first request.
_feeds: dict[tuple[str, str, str], object] = {}
_feeds_lock = threading.Lock()


def _resolve_source(name: str | None = None) -> str:
    source = (name or DEFAULT_SOURCE).lower()
    if source not in SOURCES:
        raise InputError(f"unknown source {source!r}; use one of {sorted(SOURCES)}")
    return source


def get_feed(pair: str, interval: str, source: str | None = None):
    source = _resolve_source(source)
    pair, interval = _validate(pair, interval)
    key = (source, pair, interval)
    with _feeds_lock:
        feed = _feeds.get(key)
        if feed is None:
            feed = SOURCES[source](pair, interval)
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


def compute_signals(pair: str, interval: str, source: str | None = None) -> dict:
    """Latest signal from the background feed (starts it if needed)."""
    return get_feed(pair, interval, source).snapshot()


def compute_backtest(pair: str, interval: str, source: str | None = None) -> dict:
    """The feed's own walk-forward check on the history it trained on."""
    pair, interval = _validate(pair, interval)
    snap = get_feed(pair, interval, source).snapshot()
    return {
        "pair": pair,
        "interval": interval,
        "source": snap["source"],
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
        return render_template(
            "index.html", pairs=_pairs(), intervals=_intervals(), sources=sorted(SOURCES),
            default_source=DEFAULT_SOURCE,
        )

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "source": DEFAULT_SOURCE, "model": engine.backend_name()})

    @app.get("/api/config")
    def config():
        return jsonify(
            {
                "pairs": _pairs(),
                "intervals": _intervals(),
                "sources": sorted(SOURCES),
                "default_source": DEFAULT_SOURCE,
            }
        )

    @app.get("/api/signals")
    @app.get("/api/live")
    def signals():
        """Live feed: starts a background poller on first request."""
        pair = request.args.get("pair", "EURUSD")
        interval = request.args.get("interval", "5m")
        source = request.args.get("source")
        try:
            return jsonify(compute_signals(pair, interval, source))
        except InputError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/candles")
    def candles_view():
        """Raw closed candles for external (e.g. AI) analysis."""
        pair = request.args.get("pair", "EURUSD")
        interval = request.args.get("interval", "5m")
        source = request.args.get("source")
        try:
            limit = max(1, min(int(request.args.get("limit", 500)), 5000))
        except ValueError:
            return jsonify({"error": "limit must be an integer"}), 400
        try:
            feed = get_feed(pair, interval, source)
            rows = feed.candles(limit)
            snap = feed.snapshot()
            return jsonify(
                {
                    "pair": pair.upper(),
                    "interval": interval,
                    "source": snap["source"],
                    "status": snap["status"],
                    "error": snap["error"],
                    "count": len(rows),
                    "candles": rows,
                }
            )
        except InputError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/backtest")
    def backtest_view():
        pair = request.args.get("pair", "EURUSD")
        interval = request.args.get("interval", "5m")
        source = request.args.get("source")
        try:
            return jsonify(compute_backtest(pair, interval, source))
        except InputError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/ledger")
    def ledger_view():
        """Honest forward record: every resolved paper signal and its WIN/LOSS."""
        try:
            recent = max(1, min(int(request.args.get("recent", 50)), 500))
        except ValueError:
            return jsonify({"error": "recent must be an integer"}), 400
        return jsonify(LEDGER.summary(recent=recent))

    return app


app = create_app()


def main() -> None:
    port = int(os.getenv("PORT", "5000"))
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=port, debug=os.getenv("FLASK_DEBUG") == "1")


if __name__ == "__main__":
    main()
