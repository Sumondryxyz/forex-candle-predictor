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
from datetime import UTC, datetime
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
        # RLock: record() holds the lock and calls _load_keys() -> rows(), which
        # re-acquires it. A plain Lock deadlocks here.
        self._lock = threading.RLock()
        self._seen: set[tuple] | None = None

    @staticmethod
    def _key(*, pair: str, interval: str, candle_time: float | None) -> tuple:
        return (pair, interval, candle_time)

    def _load_keys(self) -> set[tuple]:
        keys = set()
        for row in self.rows():
            keys.add((row.get("pair"), row.get("interval"), row.get("candle_time")))
        return keys

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
        """Append one resolved signal and return the stored row.

        A given (pair, interval, candle) is recorded at most once, so a signal
        resolved twice (e.g. a re-ingested candle) cannot inflate the win rate.
        """
        went_up = exit_close > entry_close
        win = went_up == (direction == "UP")
        candle_iso = (
            datetime.fromtimestamp(candle_time, tz=UTC).isoformat()
            if candle_time is not None
            else None
        )
        row = {
            "resolved_at": datetime.now(UTC).isoformat(),
            "candle_time": candle_iso,
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
        if not self.enabled:
            return row
        with self._lock:
            if candle_iso is not None:
                if self._seen is None:
                    self._seen = self._load_keys()
                key = self._key(pair=pair, interval=interval, candle_time=candle_iso)
                if key in self._seen:
                    return row
                self._seen.add(key)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        return row

    def rows(self) -> list[dict]:
        """All stored rows, oldest first, de-duplicated by (pair, interval, candle).

        Tolerates a partial trailing line. Duplicate rows can exist in files
        written before de-duplication was added, so collapse them here too --
        one resolved signal should count once.
        """
        if not self.path.exists():
            return []
        out: list[dict] = []
        seen: set[tuple] = set()
        with self._lock, self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (row.get("pair"), row.get("interval"), row.get("candle_time"))
                if key[2] is not None:
                    if key in seen:
                        continue
                    seen.add(key)
                out.append(row)
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
