#!/usr/bin/env python3
"""Live TradingView candle feed for the web app.

A ``BaseFeed`` subclass using the public TradingView chart websocket. Unlike
Quotex it needs no login and is not blocked from datacenter IPs, so it is the
default. It only reads candles; the bot never places a trade.
"""
from __future__ import annotations

from .feed import BaseFeed
from .sources.tradingview import TradingViewClient


class TradingViewFeed(BaseFeed):
    """Background TradingView poller holding the latest signal for one pair/interval."""

    source = "tradingview"

    def __init__(self, pair: str, interval: str, client_factory=None):
        super().__init__(pair, interval, client_factory)
        self.account = "-"

    def _default_client(self):
        return TradingViewClient()
