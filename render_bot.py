"""Render free-plan launcher for the Telegram bot.

Opens the HTTP port FIRST (so Render's port scan passes), then imports the
heavy bot code and runs it.
"""
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer


class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


def serve():
    port = int(os.getenv("PORT", "10000"))
    HTTPServer(("0.0.0.0", port), Health).serve_forever()


if __name__ == "__main__":
    threading.Thread(target=serve, daemon=True).start()
    print("health server started", flush=True)
    try:
        from predictor.telegram_bot import main
        main()
    except BaseException:
        traceback.print_exc()
        raise
