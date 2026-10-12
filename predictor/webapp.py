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

    @app.post("/api/min-conf")
    def set_min_conf_view():
        """Raise or lower the confidence floor live, no retrain, no redeploy."""
        payload = request.get_json(silent=True) or {}
        pair = payload.get("pair") or request.args.get("pair", "EURUSD")
        interval = payload.get("interval") or request.args.get("interval", "5m")
        source = payload.get("source") or request.args.get("source")
        try:
            value = float(payload.get("min_conf", request.args.get("min_conf")))
        except (TypeError, ValueError):
            return jsonify({"error": "min_conf must be a number"}), 400
        try:
            feed = get_feed(pair, interval, source)
            feed.set_min_conf(value)
            return jsonify(feed.snapshot())
        except (InputError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/ledger")
    def ledger_view():
        """Honest forward record: every resolved paper signal and its WIN/LOSS."""
        try:
            recent = max(1, min(int(request.args.get("recent", 50)), 500))
        except ValueError:
            return jsonify({"error": "recent must be an integer"}), 400
        return jsonify(LEDGER.summary(recent=recent))

    @app.get("/api/risk")
    def risk_view():
        """Position sizing for one signal, plus a what-if bankroll replay.

        Never places a trade: given the honest walk-forward rate it returns the
        break-even rate, the edge, a capped fractional-Kelly stake, and how the
        paper ledger would have compounded at a fixed stake fraction.
        """
        pair = request.args.get("pair", "EURUSD")
        interval = request.args.get("interval", "5m")
        source = request.args.get("source")
        try:
            bankroll = float(request.args.get("bankroll", 1000))
            stake_fraction = float(request.args.get("stake_fraction", 0.01))
        except (TypeError, ValueError):
            return jsonify({"error": "bankroll and stake_fraction must be numbers"}), 400
        try:
            snap = compute_signals(pair, interval, source)
        except InputError as exc:
            return jsonify({"error": str(exc)}), 400
        from . import risk

        rate = (snap.get("walk_forward") or {}).get("rate")
        rate_frac = (float(rate) / 100.0) if rate is not None else None
        payout = float(snap.get("payout") or 0.80)
        verdict = risk.assess(rate_frac, payout, bankroll)
        rows = [
            r for r in LEDGER.rows()
            if r.get("pair") == pair.upper() and r.get("interval") == interval
        ]
        sim = risk.simulate_bankroll(rows, bankroll, stake_fraction=stake_fraction)
        return jsonify({
            "pair": pair.upper(),
            "interval": interval,
            "source": snap.get("source"),
            "payout": payout,
            "break_even": risk.break_even_rate(payout),
            **verdict,
            "simulation": sim,
            "note": "Position sizing only. No trade is placed; this is not advice.",
        })

    @app.get("/api/smc")
    def smc_view():
        """Smart Money Concepts read of market structure for one combo.

        Analysis only — order blocks, fair-value gaps, liquidity sweeps and
        premium/discount. It does NOT predict a direction and places no trade;
        we A/B-tested SMC as model features and it did not win out-of-sample, so
        it stays a human-readable overlay next to the signal.
        """
        pair = request.args.get("pair", "EURUSD")
        interval = request.args.get("interval", "5m")
        source = request.args.get("source")
        try:
            limit = max(20, min(int(request.args.get("limit", 300)), 1000))
        except ValueError:
            return jsonify({"error": "limit must be an integer"}), 400
        from . import smc

        try:
            feed = get_feed(pair, interval, source)
            candles = feed.candles(limit)
        except InputError as exc:
            return jsonify({"error": str(exc)}), 400
        snap = feed.snapshot()
        analysis = smc.analyze(candles)
        return jsonify({
            "pair": pair.upper(),
            "interval": interval,
            "source": snap.get("source"),
            "candles": len(candles),
            **analysis,
            "recommendation": smc.recommendation(analysis),
        })

    @app.get("/api/readiness")
    def readiness_view():
        """Go-live gate: is the forward-tested edge statistically real?

        Combines the honest paper ledger with the walk-forward rate and returns a
        go / forward-only / no-go verdict. Never places a trade; this is the
        checklist that decides whether real money is justified.
        """
        try:
            bankroll = float(request.args.get("bankroll", 1000))
        except (TypeError, ValueError):
            return jsonify({"error": "bankroll must be a number"}), 400
        from . import readiness

        payout = float(os.getenv("PAYOUT", "0.80"))
        summary = LEDGER.summary(recent=0)
        overall = readiness.assess(
            forward_wins=summary.get("wins", 0),
            forward_trades=summary.get("trades", 0),
            payout=payout,
        )
        combos = []
        for row in summary.get("by_combo", []):
            wf = None
            try:
                feed = get_feed(*row["combo"].split())
                wf = (feed.snapshot().get("walk_forward") or {}).get("rate")
            except (InputError, ValueError):
                wf = None
            combos.append({
                "combo": row["combo"],
                **readiness.assess(
                    forward_wins=row.get("wins", 0),
                    forward_trades=row.get("trades", 0),
                    payout=payout,
                    walk_forward_rate=wf,
                ),
            })
        return jsonify({
            "payout": payout,
            "bankroll": bankroll,
            "overall": overall,
            "by_combo": combos,
            "note": "Readiness gate only. No trade is placed; this is not advice.",
        })

    return app


app = create_app()


def main() -> None:
    # SERVER_PORT first: Botkeep (and other PaaS panels) publish the HTTP port
    # there, while Render/Heroku use PORT.
    port = int(os.getenv("SERVER_PORT") or os.getenv("PORT") or "5000")
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=port, debug=os.getenv("FLASK_DEBUG") == "1")


if __name__ == "__main__":
    main()
