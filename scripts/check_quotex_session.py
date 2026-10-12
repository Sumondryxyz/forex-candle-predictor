#!/usr/bin/env python3
"""Validate a Quotex browser-session export and optionally test the connection.

Cloudflare blocks the Quotex login page from datacenter IPs, so the reliable way
in is a session exported from a browser you are already logged into. This script
checks that the file has what the websocket needs, then (if pyquotex is
installed) tries to fetch a real candle.

    python scripts/check_quotex_session.py session.json

The session file must contain ``token`` (the websocket authenticates with it)
and should contain ``cookies`` (including ``cf_clearance``) and ``user_agent``
matching the browser that created it. Two layouts are accepted: a flat object,
or one keyed by email.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

OK = "\033[32m✓\033[0m"
BAD = "\033[31m✗\033[0m"


def load(path: str) -> dict:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"{BAD} {path} is not a JSON object")
    email = os.getenv("QUOTEX_EMAIL")
    if email and isinstance(data.get(email), dict):
        print(f"  using entry for {email}")
        data = data[email]
    return data


def check(data: dict) -> bool:
    ok = True
    for key, required in (("token", True), ("cookies", False), ("user_agent", False)):
        present = bool(data.get(key))
        mark = OK if present else (BAD if required else "\033[33m!\033[0m")
        print(f"  {mark} {key}: {'present' if present else 'MISSING'}")
        if required and not present:
            ok = False
    cookies = data.get("cookies") or ""
    if cookies:
        has_cf = "cf_clearance" in cookies
        print(f"  {OK if has_cf else BAD} cf_clearance cookie: {'present' if has_cf else 'MISSING'}")
        ok = ok and has_cf
    return ok


def live_test() -> None:
    try:
        from pyquotex.stable_api import Quotex

        from predictor.quotex_session import apply
    except ImportError as exc:
        print(f"\n  ! pyquotex not installed ({exc}); skipping the live test.")
        print("    pip install git+https://github.com/cleitonleonel/pyquotex.git")
        return

    email = os.getenv("QUOTEX_EMAIL", "")
    path = os.getenv("QUOTEX_SESSION_FILE", "")
    host = os.getenv("QUOTEX_HOST", "market-qx.trade")
    apply(host)
    session = json.loads(Path(path).read_text())
    if email and isinstance(session.get(email), dict):
        session = session[email]

    client = Quotex(email=email, password="", lang="en", host=host,
                    root_path=os.getenv("QUOTEX_ROOT_PATH", str(Path.home() / ".quotex")))
    client.set_session(user_agent=session.get("user_agent"), cookies=session.get("cookies"))
    client.set_token(session["token"])
    client.set_account_mode(os.getenv("QUOTEX_ACCOUNT", "PRACTICE"))
    print("\n  connecting to the Quotex websocket ...")
    if not client.connect():
        print(f"  {BAD} websocket refused the session (stale token or IP mismatch)")
        return
    print(f"  {OK} connected; balance {client.get_balance()}")
    candles = client.get_candles("EURUSD", 300, 5, 60)
    print(f"  {OK} fetched {len(candles) if candles else 0} EURUSD 1m candles")
    client.close()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    print(f"Checking {path}")
    data = load(path)
    ok = check(data)
    if not ok:
        print(f"\n{BAD} session file is incomplete; re-export it.")
        raise SystemExit(1)
    print(f"\n{OK} structure looks good.")
    if os.getenv("QUOTEX_SESSION_FILE"):
        live_test()
    else:
        print("  set QUOTEX_SESSION_FILE to run the live test.")


if __name__ == "__main__":
    main()
