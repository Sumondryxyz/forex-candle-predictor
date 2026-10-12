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
# Matches pyquotex's default UA closely enough for the Cloudflare handshake; the
# optional pyquotex import below is used only when it is actually installed.
_FALLBACK_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
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
        ua = _FALLBACK_UA
        try:  # prefer pyquotex's own UA when the optional dependency is present
            from pyquotex.config import USER_AGENT as ua
        except ImportError:
            pass
        data = {**data, "user_agent": ua}
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
    """Monkey-patch pyquotex. Safe to call repeatedly.

    Patches three things the stock client gets wrong against the current site:

    * ``Login.base_url`` is hard-coded to ``qxbroker.com``; accounts now live on
      ``market-qx.trade``, so the sign-in page must be fetched from ``host``.
    * ``Response.reason_phrase`` does not exist on curl_cffi's response, which
      crashes the error path (an AttributeError instead of the real 403).
    * the login/settings flow as described at the top of this module.
    """
    global _applied
    if _applied:
        return

    from pyquotex.network import navigator
    from pyquotex.network.login import Login

    navigator.Browser.get_cookies = _get_cookies
    Login.get_settings = _make_get_settings(host)
    Login.base_url = host
    Login.https_base_url = f"https://{host}"
    if not hasattr(navigator.Response, "reason_phrase"):
        navigator.Response.reason_phrase = property(
            lambda self: getattr(self, "reason", "") or ""
        )
    _applied = True
