"""Patches that make pyquotex work with current Quotex accounts.

Two things break the stock client:

* Quotex sets ``laravel_session`` twice during login (guest, then
  authenticated). pyquotex forwards both and the server keeps the first, so
  every authenticated request — including the websocket handshake — is treated
  as a guest and rejected.
* ``window.settings`` no longer carries the socket session; it now comes from
  the authenticated ``/api/v1/cabinets/digest`` endpoint.

Accounts are also served from ``market-qx.trade`` rather than ``qxbroker.com``.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

DEFAULT_HOST = os.getenv("QUOTEX_HOST", "market-qx.trade")
_TOKEN_RE = re.compile(r'"?token"?\s*[:=]\s*"([A-Za-z0-9]{32,})"')
_applied = False


def load_session(path: str, email: str | None = None) -> dict[str, Any]:
    """Read a Quotex browser session so the bot can skip the blocked login page.

    Accepts either pyquotex's own ``session.json`` layout, which is keyed by
    email (``{"you@example.com": {"cookies": ..., "token": ..., "user_agent": ...}}``),
    or a flat ``{"cookies": ..., "token": ..., "user_agent": ...}`` dict. When
    ``email`` is given and present as a key, that entry is used.

    The websocket authenticates with the token, so a token is required. The
    cookies (in particular ``cf_clearance``) and the user agent must match the
    browser that created the session, otherwise Cloudflare rejects the handshake.
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise TypeError(f"session file {path!r} is not a JSON object")
    if email and isinstance(data.get(email), dict):
        data = data[email]
    if not data.get("token"):
        raise ValueError(
            f"session file {path!r} has no 'token'; export cookies, token and user_agent"
        )
    if not data.get("user_agent"):
        from pyquotex.config import USER_AGENT

        data = {**data, "user_agent": USER_AGENT}
    return data


def _get_cookies(self) -> str:
    jar: dict[str, str] = {}
    for cookie in self._client.cookies.jar:
        jar[cookie.name] = cookie.value
    return "; ".join(f"{name}={value}" for name, value in jar.items())


def _make_get_settings(host: str):
    from pyquotex.config import update_session

    async def get_settings(self) -> tuple[Any, dict[str, Any] | None]:
        cookies = self.get_cookies()
        token = None

        response = await self.send_request(
            method="GET",
            url=f"https://{host}/api/v1/cabinets/digest",
            headers={
                "Cookie": cookies,
                "Referer": f"https://{host}/en/trade",
                "Accept": "application/json",
                "X-Requested-With": "XMLHttpRequest",
            },
        )
        if response.status_code == 200:
            token = response.json().get("data", {}).get("token")
        if not token:
            match = _TOKEN_RE.search(str(self.get_soup()))
            token = match.group(1) if match else None

        self.ssid = token
        self.api.session_data["cookies"] = cookies
        self.api.session_data["token"] = token
        self.api.session_data["user_agent"] = self.headers["User-Agent"]
        update_session(self.api.username, self.api.session_data)
        return self.response, {"token": token}

    return get_settings


def apply(host: str = DEFAULT_HOST) -> None:
    """Monkey-patch pyquotex. Safe to call repeatedly."""
    global _applied
    if _applied:
        return

    from pyquotex.network import navigator
    from pyquotex.network.login import Login

    navigator.Browser.get_cookies = _get_cookies
    Login.get_settings = _make_get_settings(host)
    _applied = True
