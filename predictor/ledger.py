"""Paper-trading ledger: an honest, append-only record of every signal the feed
acted on and how the next candle actually closed.

Nothing here places a trade. Each time a signal that cleared ``MIN_CONF`` is
resolved by the next closed candle, the feed appends one row. The rows are
written as JSON Lines so the file survives restarts and can be tailed, grepped
or loaded into pandas. Reading back gives the true forward WIN/LOSS count, the
only number that proves (or disproves) an edge.

Enable with ``LEDGER_ENABLED`` (default on); the file lives at ``LEDGER_PATH``
(default ``var/ledger.jsonl``). Set it to shared storage (a Render disk) if you
want the record to persist across deploys.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_PATH = "var/ledger.jsonl"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


class Ledger:
    """Thread-safe append-only log of resolved paper trades."""

    def __init__(self, path: str | None = None, enabled: bool | None = None):
        self.path = Path(path or os.getenv("LEDGER_PATH", _DEFAULT_PATH))
        self.enabled = _env_bool("LEDGER_ENABLED", True) if enabled is None else enabled
        self._lock = threading.Lock()

    def record(
        self,
        *,
        pair: str,
        interval: str,
        source: str,
        direction: str,
        confidence: float,
        entry_close: float,
        exit_close: float,
        payout: float,
        candle_time: float | None = None,
    ) -> dict:
        """Append one resolved signal and return the stored row."""
        went_up = exit_close > entry_close
        win = went_up == (direction == "UP")
        row = {
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "candle_time": (
                datetime.fromtimestamp(candle_time, tz=timezone.utc).isoformat()
                if candle_time is not None
                else None
            ),
            "pair": pair,
            "interval": interval,
            "source": source,
            "direction": direction,
            "confidence": round(float(confidence), 4),
            "entry_close": float(entry_close),
            "exit_close": float(exit_close),
            "payout": float(payout),
            "win": bool(win),
            "pnl": float(payout) if win else -1.0,
        }
        if self.enabled:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
        return row

    def rows(self) -> list[dict]:
        """All stored rows, oldest first. Tolerates a partial trailing line."""
        if not self.path.exists():
            return []
        out: list[dict] = []
        with self._lock, self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def summary(self, recent: int = 50) -> dict:
        """Win rate and ROI overall, per pair/interval, and the latest rows."""
        rows = self.rows()
        if not rows:
            return {
                "trades": 0, "wins": 0, "losses": 0,
                "win_rate": None, "roi": None, "by_combo": [], "recent": [],
                "path": str(self.path),
            }

        def stats(subset: list[dict]) -> dict:
            n = len(subset)
            wins = sum(1 for r in subset if r["win"])
            rate = wins / n
            return {
                "trades": n,
                "wins": wins,
                "losses": n - wins,
                "win_rate": round(100 * rate, 2),
                "roi": round(100 * sum(r["pnl"] for r in subset) / n, 2),
            }

        combos: dict[str, list[dict]] = {}
        for r in rows:
            combos.setdefault(f"{r['pair']} {r['interval']}", []).append(r)
        by_combo = [
            {"combo": k, **stats(v)} for k, v in sorted(combos.items())
        ]
        overall = stats(rows)
        return {
            **overall,
            "by_combo": by_combo,
            "recent": rows[-recent:][::-1],
            "path": str(self.path),
        }
