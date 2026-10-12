"""Render free-plan launcher for the Telegram bot.

Render web services must listen on $PORT, but the bot only polls Telegram.
This starts a tiny HTTP server (for Render's health check and the keep-awake
ping) in a thread, then runs the bot in the main thread.
"""
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from predictor.telegram_bot import main


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
    main()
