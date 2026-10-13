"""Test-wide guard against writing to the real paper-trading ledger.

``predictor.feed`` builds a module-level ``Ledger`` from ``LEDGER_PATH`` (default
``var/ledger.jsonl``). A test that exercises the resolve path without stubbing
that singleton appends synthetic rows -- carrying a hard-coded fixture price --
straight into the live scoreboard. Those rows showed a fake high win rate
(e.g. 98% on 1m) and made the dashboard look like it had a real edge.
"""
import pytest


@pytest.fixture(autouse=True, scope="session")
def _disable_global_ledger():
    from predictor import feed, webapp

    feed.LEDGER.enabled = False
    webapp.LEDGER.enabled = False
    yield
