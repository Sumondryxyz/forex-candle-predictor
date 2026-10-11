"""Turn a captured Quotex login into a ``session.json`` the bot can use.

Cloudflare blocks the Quotex login page from datacenter IPs, so the bot is fed a
session exported from a browser that is already logged in. This module holds the
pure logic behind the phone-only capture flow (``tools/quotex-session-capture.user.js``
plus ``scripts/session_capture.py``):

* ``normalize_cookies`` accepts the several shapes a browser hands you — a
  Cookie-Editor JSON export, a raw ``a=1; b=2`` header, or a ``{"cookies": [...]}``
  object — and returns the flat ``a=1; b=2`` string pyquotex expects.
* ``build_session`` validates the token (the websocket authenticates with it) and
  assembles one session entry.
* ``save_session`` writes ``session.json``, keyed by email when one is known so a
  file can hold several accounts.

No network and no Quotex code here, so it is safe to unit-test.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"^[A-Za-z0-9]{32,}$")


def normalize_cookies(raw: Any) -> str:
    """Coerce a browser cookie export into a ``name=value; ...`` header string.

    Accepts a Cookie-Editor JSON export (a list of ``{name, value}``), a dict
    with a ``cookies`` list, a plain ``a=1; b=2`` string, or a dict of
    name->value. Returns ``""`` when nothing usable is found.
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return ""
        try:
            return normalize_cookies(json.loads(text))
        except (ValueError, TypeError):
            return re.sub(r"\s+", " ", text)
    if isinstance(raw, dict):
        if isinstance(raw.get("cookies"), (list, dict)):
            return normalize_cookies(raw["cookies"])
        pairs = [f"{k}={v}" for k, v in raw.items() if v is not None]
        return "; ".join(pairs)
    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, dict) and item.get("name"):
                parts.append(f"{item['name']}={item.get('value', '')}")
            elif isinstance(item, str):
                parts.append(item)
        return "; ".join(parts)
    return ""


def build_session(
    *,
    token: str | None,
    cookies: Any,
    user_agent: str | None = None,
    email: str | None = None,
) -> dict[str, Any]:
    """Validate the captured pieces and return one session entry.

    Raises ``ValueError`` when the token or cookies are missing, so a bad capture
    fails loudly instead of writing a session the bot cannot use.
    """
    token = (token or "").strip()
    if not _TOKEN_RE.match(token):
        raise ValueError(
            "token looks invalid: expected 32+ alphanumeric characters from the "
            "digest endpoint (run the capture while logged in)"
        )
    cookie_str = normalize_cookies(cookies)
    if not cookie_str:
        raise ValueError("no cookies captured; export them with Cookie-Editor")
    entry: dict[str, Any] = {"token": token, "cookies": cookie_str}
    if user_agent:
        entry["user_agent"] = user_agent
    if email:
        entry["email"] = email
    return entry


def save_session(
    path: str | Path,
    entry: dict[str, Any],
    email: str | None = None,
) -> dict[str, Any]:
    """Write ``entry`` to ``path``, merging into an existing file when possible.

    With an ``email`` the file is keyed by email (``{"you@x": {...}}``); without
    one it is a flat object. An existing file is preserved so several accounts
    can live side by side. Returns the object that was written.
    """
    path = Path(path)
    data: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            if isinstance(existing, dict):
                data = existing
        except (ValueError, TypeError):
            data = {}

    if email:
        data[email] = entry
    else:
        # Flat layout: drop any stray email key we added for reference.
        data = {k: v for k, v in entry.items() if k != "email"}

    path.write_text(json.dumps(data, indent=2))
    return data
