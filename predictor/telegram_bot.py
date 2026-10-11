#!/usr/bin/env python3
"""Telegram front end for the next-candle predictor.

Reuses the web app's live feeds (``predictor.webapp.get_feed``) so there is one
code path for data, features and model. The bot predicts the next candle's
direction with a confidence and reports a WIN/LOSS paper check. It never places
a trade and holds no broker credentials beyond what the feed needs.

    TELEGRAM_BOT_TOKEN=123:abc python -m predictor.telegram_bot
"""
from __future__ import annotations

import html
import json
import logging
import os
import threading
import time
from pathlib import Path

import requests

from .webapp import DEFAULT_SOURCE, InputError, _intervals, _pairs, get_feed

log = logging.getLogger("telegram_bot")

API = "https://api.telegram.org/bot{token}/{method}"
POLL_TIMEOUT = 30
STATE_PATH = Path(os.getenv("TELEGRAM_STATE", "var/telegram.json"))
PUSH_ENABLED = os.getenv("PUSH_ENABLED", "1") != "0"
PUSH_INTERVAL = int(os.getenv("PUSH_INTERVAL", "30"))


# ---------------------------------------------------------------- telegram io
class Telegram:
    def __init__(self, token: str):
        self.base = API.format(token=token)

    def call(self, method: str, **params):
        resp = requests.post(self.base.format(method=method), json=params,
                             timeout=POLL_TIMEOUT + 10)
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"{method}: {data.get('description')}")
        return data["result"]

    def send(self, chat_id: int, text: str, **extra) -> None:
        for chunk in _split(text, 3800):
            try:
                self.call("sendMessage", chat_id=chat_id, text=chunk,
                          parse_mode="HTML", disable_web_page_preview=True, **extra)
            except Exception as exc:  # noqa: BLE001 - never let one chat kill the loop
                log.warning("send to %s failed: %s", chat_id, exc)

    def updates(self, offset: int | None):
        return self.call("getUpdates", offset=offset, timeout=POLL_TIMEOUT,
                         allowed_updates=["message"])


def _split(text: str, size: int):
    while len(text) > size:
        cut = text.rfind("\n", 0, size)
        if cut <= 0:
            cut = size
        yield text[:cut]
        text = text[cut:].lstrip("\n")
    if text:
        yield text


# ---------------------------------------------------------------- state
class State:
    """Subscriptions and per-chat preferences, persisted as JSON."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.chats: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            self.chats = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.chats = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.chats, indent=1))
        except OSError as exc:
            log.warning("could not persist state: %s", exc)

    def get(self, chat_id: int) -> dict:
        with self.lock:
            return dict(self.chats.get(str(chat_id), {}))

    def update(self, chat_id: int, **kw) -> dict:
        with self.lock:
            entry = self.chats.setdefault(str(chat_id), {})
            entry.update(kw)
            self._save()
            return dict(entry)

    def subscribed(self) -> list[tuple[int, dict]]:
        with self.lock:
            return [(int(k), dict(v)) for k, v in self.chats.items() if v.get("subscribed")]

    def last_sent(self, chat_id: int) -> str | None:
        with self.lock:
            return self.chats.get(str(chat_id), {}).get("last_signal")


# ---------------------------------------------------------------- formatting
def _pct(x) -> str:
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "—"


def format_signal(snap: dict) -> str:
    pair = html.escape(snap.get("pair", "?"))
    interval = html.escape(snap.get("interval", "?"))
    status = snap.get("status")
    if snap.get("error"):
        return f"⚠️ <b>{pair} {interval}</b>\n{html.escape(str(snap['error']))}"

    if not snap.get("direction"):
        return (f"⏳ <b>{pair} {interval}</b>\nNo signal yet ({html.escape(str(status))}).\n"
                f"<i>source: {html.escape(str(snap.get('source')))}</i>")

    direction = snap["direction"]
    arrow = "🟢 UP (CALL)" if direction == "UP" else "🔴 DOWN (PUT)"
    conf = snap.get("confidence")
    tradeable = snap.get("tradeable")
    head = f"<b>{pair} · {interval}</b>"
    body = [
        head,
        f"Signal: <b>{arrow}</b>",
        f"Confidence: <b>{_pct(conf)}</b>  (floor {_pct(snap.get('min_conf'))})",
        f"Tradeable: {'✅ yes' if tradeable else '🚫 below floor'}",
    ]
    if snap.get("payout"):
        body.append(f"Payout: {_pct(snap['payout'])}")
    body.append(f"Candles: {snap.get('candles')}")
    if snap.get("signal_candle_open"):
        body.append(f"Next candle opens: {html.escape(str(snap['signal_candle_open'])[11:19])} UTC")
    sess = snap.get("session") or {}
    if sess.get("signals"):
        body.append(f"Paper record: {sess.get('wins')}/{sess.get('signals')} "
                    f"({sess.get('rate')}%)")
    res = snap.get("last_result") or {}
    if res:
        mark = "✅WIN" if res.get("win") else "❌LOSS"
        body.append(f"Last call: {mark} ({res.get('direction')} "
                    f"{_pct(res.get('confidence'))})")
    body.append(f"<i>{html.escape(str(snap.get('source')))} · no trades are placed</i>")
    return "\n".join(body)


def format_result(snap: dict) -> str:
    """How the previous call actually closed — the WIN/LOSS check."""
    pair = html.escape(snap.get("pair", "?"))
    interval = html.escape(snap.get("interval", "?"))
    res = snap.get("last_result")
    if not res:
        return f"<b>{pair} {interval}</b>\nNo resolved signal yet."
    direction = res.get("direction")
    arrow = "🟢 UP" if direction == "UP" else "🔴 DOWN"
    mark = "✅ <b>WIN</b>" if res.get("win") else "❌ <b>LOSS</b>"
    return "\n".join([
        f"<b>{pair} · {interval}</b>",
        f"Last call: {arrow} at {_pct(res.get('confidence'))}",
        f"Result: {mark}",
        f"Entry {res.get('entry_close')} → exit {res.get('exit_close')}",
        f"<i>resolved {html.escape(str(res.get('resolved_at'))[11:19])} UTC · paper only</i>",
    ])


def format_scan(snaps: list[dict]) -> str:
    rows = ["<b>Scan</b>"]
    for s in snaps:
        pair, interval = s.get("pair"), s.get("interval")
        if s.get("error"):
            rows.append(f"{pair} {interval}: ⚠️ {html.escape(str(s['error']))[:60]}")
            continue
        if not s.get("direction"):
            rows.append(f"{pair} {interval}: ⏳ no signal")
            continue
        arrow = "🟢UP" if s["direction"] == "UP" else "🔴DOWN"
        mark = "✅" if s.get("tradeable") else "🚫"
        rows.append(f"{mark} {pair} {interval}: {arrow} {_pct(s.get('confidence'))}")
    rows.append("<i>✅ = above confidence floor · no trades are placed</i>")
    return "\n".join(rows)


HELP = (
    "<b>Next-candle predictor</b>\n"
    "Predicts the next candle's direction with a confidence. "
    "It never places a trade.\n\n"
    "<b>Commands</b>\n"
    "/signal [PAIR] [INTERVAL] — one pair, e.g. <code>/signal EURUSD 5m</code>\n"
    "/result [PAIR] [INTERVAL] — how the previous call closed (WIN/LOSS)\n"
    "/scan — every configured pair at your interval\n"
    "/session — paper WIN/LOSS record\n"
    "/pairs — available pairs and intervals\n"
    "/setpair EURUSD — remember a default pair\n"
    "/setinterval 5m — remember a default interval\n"
    "/subscribe — auto-push tradeable signals here\n"
    "/unsubscribe — stop auto-push\n"
    "/status — your current settings\n"
)


# ---------------------------------------------------------------- commands
def handle(tg: Telegram, state: State, msg: dict) -> None:
    chat_id = msg["chat"]["id"]
    text = (msg.get("text") or "").strip()
    if not text.startswith("/"):
        tg.send(chat_id, "Send /help to see what I can do.")
        return
    parts = text.split()
    cmd = parts[0].split("@")[0].lower()
    args = parts[1:]

    prefs = state.get(chat_id)
    takes_pair = cmd in ("/signal", "/result")
    pair = (args[0].upper() if args and takes_pair else None) or prefs.get("pair") or _pairs()[0]
    interval = (args[1] if len(args) > 1 and takes_pair else None) or prefs.get("interval") or _intervals()[0]

    if cmd in ("/start", "/help"):
        tg.send(chat_id, HELP)
        return

    if cmd == "/pairs":
        tg.send(chat_id, "<b>Pairs</b>\n" + ", ".join(_pairs()) +
                "\n<b>Intervals</b>\n" + ", ".join(_intervals()) +
                f"\n<i>source: {html.escape(DEFAULT_SOURCE)}</i>")
        return

    if cmd == "/setpair":
        if not args or args[0].upper() not in _pairs():
            tg.send(chat_id, "Usage: <code>/setpair EURUSD</code>\nPairs: " + ", ".join(_pairs()))
            return
        state.update(chat_id, pair=args[0].upper())
        tg.send(chat_id, f"Default pair set to <b>{args[0].upper()}</b>.")
        return

    if cmd == "/setinterval":
        if not args or args[0] not in _intervals():
            tg.send(chat_id, "Usage: <code>/setinterval 5m</code>\nIntervals: " + ", ".join(_intervals()))
            return
        state.update(chat_id, interval=args[0])
        tg.send(chat_id, f"Default interval set to <b>{args[0]}</b>.")
        return

    if cmd == "/subscribe":
        state.update(chat_id, subscribed=True)
        tg.send(chat_id, "✅ Subscribed. I will push tradeable signals "
                         f"for <b>{html.escape(pair)} {html.escape(interval)}</b>.")
        return

    if cmd == "/unsubscribe":
        state.update(chat_id, subscribed=False)
        tg.send(chat_id, "🛑 Unsubscribed.")
        return

    if cmd == "/status":
        s = state.get(chat_id)
        tg.send(chat_id, "<b>Your settings</b>\n"
                         f"Pair: {html.escape(str(s.get('pair') or _pairs()[0]))}\n"
                         f"Interval: {html.escape(str(s.get('interval') or _intervals()[0]))}\n"
                         f"Auto-push: {'on' if s.get('subscribed') else 'off'}\n"
                         f"Chat id: <code>{chat_id}</code>")
        return

    if cmd == "/session":
        snap = _safe_snapshot(pair, interval)
        sess = snap.get("session") or {}
        tg.send(chat_id, f"<b>{html.escape(pair)} {html.escape(interval)} — paper record</b>\n"
                         f"Resolved signals: {sess.get('signals', 0)}\n"
                         f"Wins: {sess.get('wins', 0)}\n"
                         f"Rate: {sess.get('rate') if sess.get('rate') is not None else '—'}%\n"
                         "<i>Paper only — no trades are placed.</i>")
        return

    if cmd == "/signal":
        if pair not in _pairs():
            tg.send(chat_id, f"Unknown pair {html.escape(pair)}. Pairs: " + ", ".join(_pairs()))
            return
        if interval not in _intervals():
            tg.send(chat_id, f"Unknown interval {html.escape(interval)}. Use: " + ", ".join(_intervals()))
            return
        tg.send(chat_id, format_signal(_safe_snapshot(pair, interval)))
        return

    if cmd == "/result":
        if pair not in _pairs():
            tg.send(chat_id, f"Unknown pair {html.escape(pair)}. Pairs: " + ", ".join(_pairs()))
            return
        tg.send(chat_id, format_result(_safe_snapshot(pair, interval)))
        return

    if cmd == "/scan":
        snaps = [_safe_snapshot(p, interval) for p in _pairs()]
        tg.send(chat_id, format_scan(snaps))
        return

    tg.send(chat_id, "Unknown command. Send /help.")


def _safe_snapshot(pair: str, interval: str) -> dict:
    try:
        return get_feed(pair, interval).snapshot()
    except InputError as exc:
        return {"pair": pair, "interval": interval, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"pair": pair, "interval": interval, "error": f"feed error: {exc}"}


# ---------------------------------------------------------------- auto-push
def push_loop(tg: Telegram, state: State, stop: threading.Event) -> None:
    while not stop.wait(PUSH_INTERVAL):
        for chat_id, prefs in state.subscribed():
            pair = prefs.get("pair") or _pairs()[0]
            interval = prefs.get("interval") or _intervals()[0]
            snap = _safe_snapshot(pair, interval)

            # Announce how the previous call closed, once per resolution.
            res = snap.get("last_result") or {}
            resolved_at = res.get("resolved_at")
            if resolved_at and state.get(chat_id).get("last_result") != resolved_at:
                mark = "✅ WIN" if res.get("win") else "❌ LOSS"
                tg.send(chat_id, f"🔔 {mark} — {html.escape(str(pair))} "
                                 f"{html.escape(str(res.get('direction') or ''))} call resolved.")
                state.update(chat_id, last_result=resolved_at)

            marker = snap.get("signal_candle_open")
            if not snap.get("tradeable") or not marker:
                continue
            if state.last_sent(chat_id) == marker:
                continue  # already pushed this candle
            tg.send(chat_id, "🔔 " + format_signal(snap))
            state.update(chat_id, last_signal=marker)


# ---------------------------------------------------------------- main
def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is required")
    tg = Telegram(token)
    state = State(STATE_PATH)

    me = tg.call("getMe")
    log.info("connected as @%s", me.get("username"))

    stop = threading.Event()
    if PUSH_ENABLED:
        threading.Thread(target=push_loop, args=(tg, state, stop), daemon=True).start()
        log.info("auto-push every %ss", PUSH_INTERVAL)

    offset = None
    while not stop.is_set():
        try:
            for update in tg.updates(offset):
                offset = update["update_id"] + 1
                msg = update.get("message")
                if msg and msg.get("text"):
                    try:
                        handle(tg, state, msg)
                    except Exception as exc:  # noqa: BLE001
                        log.exception("handler failed: %s", exc)
        except requests.RequestException as exc:
            log.warning("poll failed: %s", exc)
            time.sleep(3)
        except Exception as exc:  # noqa: BLE001
            log.exception("loop error: %s", exc)
            time.sleep(3)


if __name__ == "__main__":
    main()
