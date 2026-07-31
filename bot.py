#!/usr/bin/env python3
"""Telegram bot that serves next-candle direction signals.

Commands:
  /signal <PAIR> [interval]  e.g. /signal EURUSD 5m
  /auto <PAIR> [interval]    push a signal before every candle close
  /stop                      cancel automatic signals
  /stats <PAIR> [interval]   backtested accuracy of the loaded model
  /pairs                     supported pairs
"""
from __future__ import annotations

import logging
import os

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from predictor import backtest, model
from predictor.data import PAIRS
from predictor.scheduler import (
    is_market_open,
    next_candle_open,
    seconds_until_next_close,
)
from predictor.sources import get_fetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_INTERVAL = os.getenv("DEFAULT_INTERVAL", "5m")
fetch_ohlc = get_fetcher()
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
        "/signal EURUSD 1m \u2014 prediction for the upcoming candle\n"
        "/auto EURUSD 1m \u2014 push high-confidence signals every candle\n"
        "/stop \u2014 cancel automatic signals\n"
        "/stats EURUSD 1m \u2014 walk-forward accuracy\n"
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

    await update.message.reply_text(_format_signal(pair, interval, pred), parse_mode="Markdown")


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


def _format_signal(pair: str, interval: str, pred: model.Prediction) -> str:
    arrow = "\U0001f7e2 UP (CALL)" if pred.direction == "UP" else "\U0001f534 DOWN (PUT)"
    verdict = "TRADE" if pred.confidence >= MIN_CONFIDENCE else "NO TRADE (low confidence)"
    return (
        f"*{pair} {interval}*\n"
        f"Next candle: {arrow}\n"
        f"Confidence: {pred.confidence * 100:.1f}%\n"
        f"Expected value at {PAYOUT * 100:.0f}% payout: "
        f"{backtest.roi_pct(pred.confidence, PAYOUT):+.1f}% per trade\n"
        f"Decision: {verdict}\n"
        f"Entry at: {next_candle_open(interval):%H:%M:%S} UTC\n\n{DISCLAIMER}"
    )


async def _auto_tick(context: ContextTypes.DEFAULT_TYPE) -> None:
    job = context.job
    pair, interval = job.data["pair"], job.data["interval"]
    job.job_queue.run_once(
        _auto_tick, seconds_until_next_close(interval), data=job.data, chat_id=job.chat_id,
        name=job.name,
    )
    if not is_market_open():
        return
    try:
        pred = model.predict_latest(model.load(pair, interval), fetch_ohlc(pair, interval))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        log.warning("auto signal failed for %s %s: %s", pair, interval, exc)
        return
    if pred.confidence < MIN_CONFIDENCE:
        return
    await context.bot.send_message(
        job.chat_id, _format_signal(pair, interval, pred), parse_mode="Markdown"
    )


def _job_name(chat_id: int) -> str:
    return f"auto-{chat_id}"


async def auto(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorised(update):
        return
    pair, interval = _args(context)
    chat_id = update.effective_chat.id
    for job in context.job_queue.get_jobs_by_name(_job_name(chat_id)):
        job.schedule_removal()
    try:
        model.load(pair, interval)
        delay = seconds_until_next_close(interval)
    except (FileNotFoundError, ValueError) as exc:
        await update.message.reply_text(f"\u274c {exc}")
        return
    context.job_queue.run_once(
        _auto_tick,
        delay,
        data={"pair": pair, "interval": interval},
        chat_id=chat_id,
        name=_job_name(chat_id),
    )
    await update.message.reply_text(
        f"\u2705 Auto signals on for {pair} {interval}. "
        f"Only confidence \u2265 {MIN_CONFIDENCE * 100:.0f}% is sent; /stop to cancel."
    )


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    jobs = context.job_queue.get_jobs_by_name(_job_name(update.effective_chat.id))
    for job in jobs:
        job.schedule_removal()
    await update.message.reply_text("Auto signals off." if jobs else "No auto signals running.")


def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CommandHandler("pairs", pairs))
    app.add_handler(CommandHandler("signal", signal))
    app.add_handler(CommandHandler("auto", auto))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(CommandHandler("stats", stats))
    log.info("bot polling")
    app.run_polling()


if __name__ == "__main__":
    main()
