#!/usr/bin/env python3
"""Telegram bot for the next-candle predictor.

It reuses the same live feeds and numpy engine as the web app: the bot predicts
the next candle's direction and reports a WIN/LOSS paper check. It never places
a trade.

    TELEGRAM_BOT_TOKEN=123:abc python bot.py
"""
from predictor.telegram_bot import main

if __name__ == "__main__":
    main()
