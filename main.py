#!/usr/bin/env python3
"""Generic process entry point for the predictor web app.

Botkeep (and most PaaS panels) publish the HTTP port in ``SERVER_PORT``; Render
and Heroku use ``PORT``. We serve through a single-worker gunicorn because the
candle feed and the model live in process memory and must not be forked across
workers.

    python main.py
"""
import os


def _port() -> int:
    return int(os.getenv("SERVER_PORT") or os.getenv("PORT") or "10000")


def main() -> None:
    from gunicorn.app.base import BaseApplication

    from predictor.webapp import app

    class _Server(BaseApplication):
        def load_config(self) -> None:
            self.cfg.set("bind", f"0.0.0.0:{_port()}")
            self.cfg.set("workers", 1)
            self.cfg.set("worker_class", "gthread")
            self.cfg.set("threads", 8)
            self.cfg.set("timeout", 120)

        def load(self):
            return app

    _Server().run()


if __name__ == "__main__":
    main()
