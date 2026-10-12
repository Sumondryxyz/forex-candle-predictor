#!/usr/bin/env python3
"""Live Quotex candle feed for the web app.

A thin ``BaseFeed`` subclass that logs into Quotex with ``pyquotex`` (using the
login patches in ``quotex_session``). Quotex has no official API and Cloudflare
challenges the HTML login page from datacenter IPs, so this works best from a
home or mobile connection, through a residential proxy, or with an exported
browser session (``QUOTEX_SESSION_FILE``). It never places a trade.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import quotex_session
from .feed import BaseFeed

DEFAULT_HOST = os.getenv("QUOTEX_HOST", "market-qx.trade")
ROOT_PATH = os.getenv("QUOTEX_ROOT_PATH", str(Path.home() / ".quotex"))


def _proxies() -> dict[str, str] | None:
    """Optional residential proxy, e.g. QUOTEX_PROXY=http://user:pass@host:port."""
    url = os.getenv("QUOTEX_PROXY", "").strip()
    return {"http": url, "https": url} if url else None


class QuotexFeed(BaseFeed):
    """Background Quotex poller holding the latest signal for one pair/interval."""

    source = "quotex"

    def __init__(self, pair: str, interval: str, client_factory=None):
        super().__init__(pair, interval, client_factory)
        self.account = os.getenv("QUOTEX_ACCOUNT", "PRACTICE")

    def _default_client(self):
        try:
            from pyquotex.stable_api import Quotex
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "pyquotex is not installed; "
                "pip install git+https://github.com/cleitonleonel/pyquotex.git"
            ) from exc

        quotex_session.apply(DEFAULT_HOST)
        client = Quotex(
            email=os.getenv("QUOTEX_EMAIL", ""),
            password=os.getenv("QUOTEX_PASSWORD", ""),
            lang="en",
            host=DEFAULT_HOST,
            root_path=ROOT_PATH,
            proxies=_proxies(),
        )
        client.set_account_mode(self.account)
        session_file = os.getenv("QUOTEX_SESSION_FILE", "")
        if session_file:
            session = quotex_session.load_session(session_file, os.getenv("QUOTEX_EMAIL"))
            client.set_session(
                user_agent=session["user_agent"],
                cookies=session.get("cookies"),
                ssid=session.get("token"),
            )
        return client

    def _friendly_error(self, exc: Exception) -> str:
        """Translate pyquotex/Cloudflare failures into something actionable."""
        text = f"{type(exc).__name__}: {exc}"
        low = text.lower()
        if "403" in text or "reason_phrase" in low or "cloudflare" in low or "just a moment" in low:
            return (
                "Cloudflare blocked the Quotex login page (HTTP 403). Use DATA_SOURCE=tradingview "
                "to read the same pairs without a Quotex login, or log in from a home/mobile "
                "connection, export the session to QUOTEX_SESSION_FILE, or set QUOTEX_PROXY to a "
                f"residential proxy ({type(exc).__name__})."
            )
        if "invalid email or password" in low:
            return "Quotex rejected the credentials; check QUOTEX_EMAIL / QUOTEX_PASSWORD."
        return text
