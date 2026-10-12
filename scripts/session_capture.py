#!/usr/bin/env python3
"""Local capture server: log in to Quotex once and save session.json.

Cloudflare blocks the Quotex login page from datacenter IPs, so the phone (with
its residential IP) is the reliable place to log in. This little server turns
that one login into a ``session.json`` the bot can reuse:

    python scripts/session_capture.py

It listens on ``127.0.0.1:8787`` only (never your network) and serves:

    GET  /            a phone-friendly page with the userscript + bookmarklet
    POST /save        the userscript posts {token, cookies, user_agent, email}
    GET  /status      what was captured last (token masked)

The userscript ``tools/quotex-session-capture.user.js`` (Kiwi Browser +
Tampermonkey) reads the token from ``/api/v1/cabinets/digest`` and the cookies
from ``document.cookie`` while you are logged in, then posts them here. No
password is ever seen; only the session the browser already holds.
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predictor.session_capture import build_session, save_session

HOST = os.getenv("CAPTURE_HOST", "127.0.0.1")
PORT = int(os.getenv("CAPTURE_PORT", "8787"))
SESSION_PATH = Path(os.getenv("QUOTEX_SESSION_FILE", "session.json"))
CAPTURE_KEY = os.getenv("CAPTURE_KEY", "").strip()

PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quotex session capture</title>
<style>
 :root{--bg:#0b0f17;--panel:#121826;--line:#232c40;--text:#e6ebf5;--muted:#93a0b8;
   --accent:#4f8cff;--ok:#22c55e;--radius:12px}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--text);padding:18px;
   font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
 .wrap{max-width:680px;margin:0 auto}
 h1{font-size:21px;margin:0 0 4px} p.sub{color:var(--muted);margin:0 0 18px}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);
   padding:16px;margin-bottom:14px} .card h2{font-size:15px;margin:0 0 10px;color:var(--accent)}
 ol{padding-left:20px;margin:0} li{margin:5px 0}
 a.btn,button{display:inline-block;padding:11px 16px;border:0;border-radius:9px;
   background:var(--accent);color:#fff;font-weight:600;text-decoration:none;margin-top:8px;font-size:15px}
 button{width:100%;background:var(--ok);color:#04220f}
 code{background:#0d1220;padding:1px 5px;border-radius:5px;font-size:13px}
 pre{background:#0d1220;border:1px solid var(--line);border-radius:9px;padding:11px;
   overflow:auto;font-size:12px;white-space:pre-wrap;word-break:break-all}
 .ok{color:var(--ok)} .bad{color:#ff8a8a} .muted{color:var(--muted);font-size:13px}
 .warn{background:#2a1a14;border:1px solid #5b3a22;color:#ffcf9e;border-radius:9px;
   padding:12px;font-size:13px;margin-bottom:14px}
</style></head><body><div class="wrap">
 <h1>Quotex session capture</h1>
 <p class="sub">Log in once on your phone; the session saves itself for the bot.</p>
 <div class="warn">The token is like a password. It stays on this phone in
   <code>session.json</code>. Never share it or commit it to Git.</div>

 <div class="card"><h2>1 — install the userscript (best)</h2>
  <ol>
   <li>Kiwi Browser → install the <b>Tampermonkey</b> extension.</li>
   <li>Tap to open the script, then Tampermonkey → <b>Install</b>:
     <br><a class="btn" href="/capture.user.js">Open quotex-session-capture.user.js</a></li>
   <li>Open <b>Quotex</b> and log in. The script saves the session automatically
       and shows a small “Session saved” badge.</li>
  </ol>
 </div>

 <div class="card"><h2>2 — or the bookmarklet (no extension)</h2>
  <ol>
   <li>Open Quotex in Kiwi and log in.</li>
   <li>Save this as a bookmark, then tap it on the Quotex tab:
     <br><a class="btn" href="javascript:__CAPTURE_BM__">Capture session</a></li>
  </ol>
 </div>

 <div class="card"><h2>3 — status</h2>
  <pre id="st">loading…</pre>
  <button onclick="load()">Refresh</button>
  <p class="muted">Saved to <code>__SESSION_PATH__</code>. Then run the bot with
    <code>QUOTEX_SESSION_FILE=session.json DATA_SOURCE=quotex python bot.py</code></p>
 </div>
</div>
<script>
async function load(){
  const el=document.getElementById('st');
  try{
    const r=await fetch('/status'); const d=await r.json();
    if(d.ok){ el.innerHTML='<span class="ok">session saved</span>\\n'+d.summary; }
    else { el.innerHTML='<span class="bad">no session yet</span>\\n'+d.summary; }
  }catch(e){ el.textContent='server error: '+e; }
}
load();
</script></body></html>"""

# The bookmarklet mirrors the userscript: read the digest token + document.cookie,
# then POST to this local server.
BOOKMARKLET = (
    "javascript:(async()=>{try{"
    "const r=await fetch('/api/v1/cabinets/digest',{credentials:'include'});"
    "const j=await r.json();const d=j&&j.data?j.data:j;"
    "const token=d.token||d.ssid||'';"
    "if(!token){alert('Not logged in: no token. Log in to Quotex first.');return;}"
    "await fetch('http://" + HOST + ":" + str(PORT) + "/save',{method:'POST',"
    "headers:{'Content-Type':'application/json'},"
    "body:JSON.stringify({token:token,cookies:document.cookie,"
    "user_agent:navigator.userAgent,key:'" + CAPTURE_KEY + "'})});"
    "alert('Session saved to your phone. You can close this tab.');"
    "}catch(e){alert('Capture failed: '+e)}})()"
)


def _mask(token: str) -> str:
    return token[:6] + "…" + token[-4:] if len(token) > 12 else "…"


def _summary() -> tuple[bool, str]:
    if not SESSION_PATH.exists():
        return False, f"{SESSION_PATH} does not exist yet."
    try:
        data = json.loads(SESSION_PATH.read_text())
    except (ValueError, TypeError):
        return False, f"{SESSION_PATH} is not valid JSON."
    entries = data if isinstance(data, dict) else {}
    # A flat session has token at the top level; an email-keyed one nests it.
    flat = "token" in entries
    count = 1 if flat else len(entries)
    lines = [f"accounts: {count}", f"path: {SESSION_PATH}"]
    if flat:
        lines.append(f"token: {_mask(str(entries.get('token','')))}")
        ck = str(entries.get("cookies", ""))
        lines.append("cookies: " + (", ".join(sorted({c.split('=')[0] for c in ck.split('; ') if '=' in c})) or "none"))
    else:
        for email, entry in list(entries.items())[:5]:
            if isinstance(entry, dict):
                lines.append(f"{email}: token {_mask(str(entry.get('token','')))}")
    return True, "\n".join(lines)


class Handler(BaseHTTPRequestHandler):
    server_version = "QuotexCapture/1.0"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # The userscript posts from the Quotex origin; allow it. The server binds
        # to loopback, so only this phone can reach it.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send(204, b"", "text/plain")

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            html = (PAGE.replace("__CAPTURE_BM__", BOOKMARKLET)
                        .replace("__SESSION_PATH__", str(SESSION_PATH)))
            self._send(200, html.encode(), "text/html; charset=utf-8")
        elif path == "/capture.user.js":
            script = Path(__file__).resolve().parent.parent / "tools" / "quotex-session-capture.user.js"
            if script.exists():
                self._send(200, script.read_bytes(), "application/javascript; charset=utf-8")
            else:
                self._send(404, b"userscript not found", "text/plain")
        elif path == "/status":
            ok, summary = _summary()
            self._send(200, json.dumps({"ok": ok, "summary": summary}).encode(),
                       "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/save":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except ValueError:
            self._send(400, json.dumps({"ok": False, "error": "invalid JSON"}).encode(),
                       "application/json")
            return

        if CAPTURE_KEY and payload.get("key") != CAPTURE_KEY:
            self._send(403, json.dumps({"ok": False, "error": "bad capture key"}).encode(),
                       "application/json")
            return

        try:
            entry = build_session(
                token=payload.get("token"),
                cookies=payload.get("cookies"),
                user_agent=payload.get("user_agent"),
                email=payload.get("email"),
            )
        except ValueError as exc:
            self._send(400, json.dumps({"ok": False, "error": str(exc)}).encode(),
                       "application/json")
            return

        save_session(SESSION_PATH, entry, email=payload.get("email") or None)
        print(f"  ✓ saved session for {payload.get('email') or 'flat'} -> {SESSION_PATH}")
        self._send(200, json.dumps({"ok": True, "path": str(SESSION_PATH)}).encode(),
                   "application/json")

    def log_message(self, *args) -> None:  # keep the console quiet
        pass


def main() -> None:
    print(f"Quotex session capture on http://{HOST}:{PORT}")
    print(f"  saving to {SESSION_PATH}")
    print("  open that URL in your phone browser, then follow the steps.")
    print("  (Ctrl+C to stop)")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
