#!/usr/bin/env python3
"""Background candle feed shared by the Quotex and TradingView backends.

A feed owns one pair/interval, runs a daemon thread that connects a client,
loads history, trains the numpy engine, then polls for newly closed candles and
keeps the latest UP/DOWN signal in memory for the web app. It never places a
trade. Subclasses provide the client (``_default_client``) and, optionally, a
source-specific ``_friendly_error``.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone

from . import engine
from .scheduler import interval_seconds, is_market_open, next_candle_open

# How many closed candles to keep in memory (matches the CLI's rolling window).
MAX_CANDLES = 6000


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class BaseFeed:
    """Background poller holding the latest signal for one pair/interval."""

    source = "quotex"

    def __init__(self, pair: str, interval: str, client_factory=None):
        self.pair = pair.upper()
        self.interval = interval
        self.period = interval_seconds(interval)
        self.min_conf = _env_float("MIN_CONF", 0.60)
        self.payout = _env_float("PAYOUT", 0.80)
        self.history_hours = _env_int("HISTORY_HOURS", 72)
        self.retrain_every = _env_int("RETRAIN_EVERY", 60)
        self.account = "-"
        self._client_factory = client_factory or self._default_client

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self._candles: list[dict] = []
        self._seen: set[int] = set()
        self._model = None
        self._pending: dict | None = None
        self._results: deque[bool] = deque(maxlen=200)
        self._since_retrain = 0
        self._walk_forward: dict | None = None
        self._status = "idle"
        self._error: str | None = None
        self._updated: float | None = None

    # ------------------------------------------------------------- client
    def _default_client(self):  # pragma: no cover - overridden by subclasses
        raise NotImplementedError

    def _friendly_error(self, exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc}"

    # -------------------------------------------------------------- state
    def _set_status(self, status: str, error: str | None = None) -> None:
        with self._lock:
            self._status = status
            self._error = error

    def snapshot(self) -> dict:
        with self._lock:
            candles = list(self._candles)
            model = self._model
            status, error = self._status, self._error
            pending = self._pending
            results = list(self._results)
            walk = self._walk_forward
            updated = self._updated

        out = {
            "pair": self.pair,
            "interval": self.interval,
            "source": self.source,
            "status": status,
            "error": error,
            "account": self.account,
            "market_open": is_market_open(),
            "next_candle_open": next_candle_open(self.interval).isoformat(),
            "updated": datetime.fromtimestamp(updated, tz=timezone.utc).isoformat() if updated else None,
            "candles": len(candles),
            "walk_forward": walk,
            "payout": self.payout,
            "min_conf": self.min_conf,
            "session": {
                "signals": len(results),
                "wins": sum(results),
                "rate": round(100 * sum(results) / len(results), 1) if results else None,
            },
        }
        if not candles or model is None:
            out.update({"direction": None, "confidence": None, "tradeable": False})
            return out

        o, h, low, c, t = engine.arrays(candles)
        X = engine.features(o, h, low, c, t)
        prob = float(model.prob(X[-1:])[0])
        direction = "UP" if prob > 0.5 else "DOWN"
        conf = max(prob, 1 - prob)
        last = candles[-1]
        out.update(
            {
                "direction": direction,
                "side": "CALL" if direction == "UP" else "PUT",
                "probability": round(prob, 4),
                "confidence": round(conf, 4),
                "tradeable": conf >= self.min_conf,
                "last_close": float(last["close"]),
                "last_candle_time": datetime.fromtimestamp(
                    last["time"], tz=timezone.utc
                ).isoformat(),
                "recent": [
                    {
                        "time": datetime.fromtimestamp(x["time"], tz=timezone.utc).isoformat(),
                        "open": float(x["open"]),
                        "high": float(x["high"]),
                        "low": float(x["low"]),
                        "close": float(x["close"]),
                        "volume": float(x.get("volume") or 0.0),
                    }
                    for x in candles[-60:]
                ],
            }
        )
        if pending:
            out["pending"] = {
                "direction": pending["dir"],
                "ref_close": pending["ref_close"],
            }
        return out

    def candles(self, limit: int = 500) -> list[dict]:
        """Raw closed candles, oldest first, for external analysis."""
        with self._lock:
            rows = list(self._candles[-limit:])
        return [
            {
                "time": int(x["time"]),
                "open": float(x["open"]),
                "high": float(x["high"]),
                "low": float(x["low"]),
                "close": float(x["close"]),
                "volume": float(x.get("volume") or 0.0),
            }
            for x in rows
        ]

    # ------------------------------------------------------------ ingest
    def _ingest(self, fresh: list[dict]) -> list[dict]:
        """Append newly closed candles; resolve the pending signal; maybe retrain."""
        added = []
        with self._lock:
            for candle in sorted(fresh or [], key=lambda x: x["time"]):
                if candle["time"] in self._seen or candle.get("close") is None:
                    continue
                self._seen.add(candle["time"])
                self._candles.append(candle)
                self._candles = self._candles[-MAX_CANDLES:]
                self._since_retrain += 1
                added.append(candle)

                if self._pending:
                    went_up = candle["close"] > self._pending["ref_close"]
                    self._results.append(went_up == (self._pending["dir"] == "UP"))
                    self._pending = None

            if added and self._model is not None:
                o, h, low, c, t = engine.arrays(self._candles)
                X = engine.features(o, h, low, c, t)
                if self._since_retrain >= self.retrain_every:
                    refreshed = engine.train(X, c)
                    if refreshed is not None:
                        self._model = refreshed
                    self._since_retrain = 0
                prob = float(self._model.prob(X[-1:])[0])
                direction = "UP" if prob > 0.5 else "DOWN"
                conf = max(prob, 1 - prob)
                if conf >= self.min_conf:
                    self._pending = {"dir": direction, "ref_close": c[-1]}
                self._updated = time.time()
        return added

    # ------------------------------------------------------------- runner
    async def _sleep(self, seconds: float) -> None:
        """Sleep that wakes promptly when stop() is called."""
        deadline = time.monotonic() + seconds
        while not self._stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(0.2, remaining))

    async def _bootstrap(self, client) -> None:
        self._set_status("loading")
        candles = await client.get_candles(
            self.pair, time.time(), self.history_hours * 3600, self.period
        )
        candles = [c for c in (candles or []) if c.get("close") is not None]
        candles.sort(key=lambda c: c["time"])
        if len(candles) < 500:
            raise RuntimeError(
                f"only {len(candles)} history candles from {self.source}; raise HISTORY_HOURS"
            )

        o, h, low, c, t = engine.arrays(candles)
        X = engine.features(o, h, low, c, t)
        model = engine.train(X, c)
        if model is None:
            raise RuntimeError("not enough usable candles to train the model")

        with self._lock:
            self._candles = candles[-MAX_CANDLES:]
            self._seen = {x["time"] for x in self._candles}
            self._model = model
            self._walk_forward = engine.walk_forward(
                X, c, min_conf=self.min_conf, payout=self.payout
            )
            self._updated = time.time()
        self._set_status("live")

    async def _run(self) -> None:
        backoff = 15
        client = None
        while not self._stop.is_set():
            try:
                if client is None:
                    self._set_status("connecting")
                    client = self._client_factory()
                    ok, reason = await client.connect()
                    if not ok:
                        raise RuntimeError(f"{self.source} connect failed: {reason}")
                    await self._bootstrap(client)
                    backoff = 15

                await self._sleep(max(5, self.period // 6))
                if self._stop.is_set():
                    break
                fresh = await client.get_candles(
                    self.pair, time.time(), 10 * self.period, self.period
                )
                self._ingest(fresh)
                self._set_status("live")
            except Exception as exc:  # noqa: BLE001 - keep retrying, surface the error
                self._set_status("error", self._friendly_error(exc))
                if client is not None:
                    with contextlib.suppress(Exception):
                        await client.close()
                    client = None
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 300)

    # --------------------------------------------------------------- api
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()

    def _thread_main(self) -> None:
        asyncio.run(self._run())

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
