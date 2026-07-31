#!/usr/bin/env python3
"""Telegram bot that serves next-candle direction signals.

Commands:
  /signal <PAIR> [interval]  e.g. /signal EURUSD 5m
  /stats <PAIR> [interval]   backtested accuracy of the loaded model
  /pairs                     supported pairs
"""
from __future__ import annotations

import logging
import os

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from predictor import backtest, model
from predictor.data import PAIRS, fetch_ohlc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_INTERVAL = os.getenv("DEFAULT_INTERVAL", "5m")
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.60"))
PAYOUT = float(os.getenv("PAYOUT", "0.80"))
ALLOWED_USER_IDS = {int(x) for x in os.getenv("ALLOWED_USER_IDS", "").replace(" ", "").split(",") if x}

DISCLAIMER = (
    "\u26a0\ufe0f Signals are statistical estimates, not guarantees. "
    "Backtested edge is a few percent above coin-flip \u2014 trade small and manage risk."
)


def _authorised(update: Update) -> bool:
    return not ALLOWED_USER_IDS or (update.effective_user and update.effective_user.id in ALLOWED_USER_IDS)


def _args(context: ContextTypes.DEFAULT_TYPE) -> tuple[str, str]:
    pair = (context.args[0] if context.args else "EURUSD").upper()
    interval = context.args[1] if len(context.args) > 1 else DEFAULT_INTERVAL
    return pair, interval


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Next-candle direction bot.\n"
        "/signal EURUSD 5m \u2014 prediction for the upcoming candle\n"
        "/stats EURUSD 5m \u2014 walk-forward accuracy\n"
        "/pairs \u2014 supported pairs\n\n" + DISCLAIMER
    )


async def pairs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Supported: " + ", ".join(sorted(PAIRS)))


async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorised(update):
        return
    pair, interval = _args(context)
    try:
        bundle = model.load(pair, interval)
        df = fetch_ohlc(pair, interval)
        pred = model.predict_latest(bundle, df)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        await update.message.reply_text(f"\u274c {exc}")
        return

    verdict = "TRADE" if pred.confidence >= MIN_CONFIDENCE else "NO TRADE (low confidence)"
    arrow = "\U0001f7e2 UP" if pred.direction == "UP" else "\U0001f534 DOWN"
    edge = backtest.roi_pct(pred.confidence, PAYOUT)
    await update.message.reply_text(
        f"*{pair} {interval}*\n"
        f"Next candle: {arrow}\n"
        f"Confidence: {pred.confidence * 100:.1f}%\n"
        f"Expected value at {PAYOUT * 100:.0f}% payout: {edge:+.1f}% per trade\n"
        f"Decision: {verdict}\n"
        f"Last closed candle: {pred.timestamp}\n\n{DISCLAIMER}",
        parse_mode="Markdown",
    )


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pair, interval = _args(context)
    try:
        metrics = model.load(pair, interval)["metrics"]
    except FileNotFoundError as exc:
        await update.message.reply_text(f"\u274c {exc}")
        return
    lines = [
        f"*{pair} {interval}* walk-forward results",
        f"Samples: {metrics['samples']}",
        f"Overall accuracy: {metrics['overall_accuracy_pct']}%",
        f"Baseline: {metrics['always_majority_baseline_pct']}%",
        "",
        (
            f"Break-even at {PAYOUT * 100:.0f}% payout: "
            f"{100 * backtest.breakeven_accuracy(PAYOUT):.1f}%"
        ),
        "",
        "min conf | signals | accuracy | roi/trade",
    ]
    for row in metrics.get("by_confidence", []):
        lines.append(
            f"{row['min_confidence']:.2f} | {row['signals']} | "
            f"{row['accuracy_pct']}% | {row.get('roi_pct_per_trade')}%"
        )
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CommandHandler("pairs", pairs))
    app.add_handler(CommandHandler("signal", signal))
    app.add_handler(CommandHandler("stats", stats))
    log.info("bot polling")
    app.run_polling()


if __name__ == "__main__":
    main()
