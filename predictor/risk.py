"""Risk and position sizing for the signal engine.

This module never places a trade. It turns a signal's probability estimate into
the numbers a real account actually needs: the break-even win rate for the
broker's payout, whether the measured edge clears it, and a conservative stake.

The honest arithmetic matters. Binary options pay ``payout`` on a win and lose
the whole stake otherwise, so the break-even win rate is ``1 / (1 + payout)``.
At an 80% payout that is 55.6%: a 54% signal loses money over time no matter how
confident it feels. ``expected_value`` and ``kelly_fraction`` below make that
explicit instead of hiding it.
"""

from __future__ import annotations

# A real account should never bet a meaningful fraction on a fragile FX edge.
# Quarter-Kelly with a hard cap keeps a bad run from ending the account.
KELLY_MULT = 0.25
MAX_STAKE_FRACTION = 0.02


def break_even_rate(payout: float) -> float:
    """Win rate at which a binary option breaks even, e.g. 0.5 payout -> 0.667."""
    if payout <= 0:
        return 1.0
    return 1.0 / (1.0 + payout)


def expected_value(win_rate: float, payout: float) -> float:
    """Expected profit per unit staked. Positive only above break-even."""
    return win_rate * payout - (1.0 - win_rate)


def kelly_fraction(win_rate: float, payout: float) -> float:
    """Full-Kelly stake fraction for a binary bet; 0 when there is no edge.

    ``f* = (p*b - q) / b`` with ``b = payout``, ``p = win_rate``, ``q = 1 - p``.
    """
    if payout <= 0:
        return 0.0
    p = win_rate
    q = 1.0 - p
    f = (p * payout - q) / payout
    return max(0.0, f)


def suggest_stake(
    win_rate: float,
    payout: float,
    bankroll: float,
    *,
    kelly_mult: float = KELLY_MULT,
    max_fraction: float = MAX_STAKE_FRACTION,
) -> dict:
    """Conservative stake suggestion, with the reasoning attached.

    Returns a fixed fraction of ``bankroll`` (fractional Kelly, capped), plus
    the edge, break-even rate and expected value so a caller can show WHY the
    suggestion is what it is — including "0, the edge is below break-even".
    """
    be = break_even_rate(payout)
    ev = expected_value(win_rate, payout)
    raw_kelly = kelly_fraction(win_rate, payout)
    frac = min(raw_kelly * kelly_mult, max_fraction)
    if ev <= 0:
        frac = 0.0
    stake = round(max(0.0, frac) * max(0.0, bankroll), 2)
    return {
        "win_rate": round(win_rate, 4),
        "break_even": round(be, 4),
        "edge": round(win_rate - be, 4),
        "expected_value": round(ev, 4),
        "kelly_fraction": round(raw_kelly, 4),
        "stake_fraction": round(frac, 4),
        "stake": stake,
        "tradeable": ev > 0,
    }


def assess(win_rate: float | None, payout: float, bankroll: float) -> dict:
    """Verdict for a signal: no-edge / marginal / positive-edge, plus stake.

    ``win_rate`` is the honest out-of-sample rate (e.g. the walk-forward figure).
    ``None`` means the engine has not measured an edge yet, which is a hard stop.
    """
    if win_rate is None:
        return {
            "verdict": "unmeasured",
            "reason": "no out-of-sample win rate yet — do not size a bet",
            "sizing": None,
        }
    sizing = suggest_stake(win_rate, payout, bankroll)
    be = sizing["break_even"]
    edge = sizing["edge"]
    if edge <= 0:
        verdict, reason = "no-edge", f"win rate {win_rate:.1%} is below break-even {be:.1%}"
    elif edge < 0.01:
        verdict, reason = "marginal", f"edge {edge:+.1%} is too thin to trust with real money"
    else:
        verdict, reason = "positive-edge", f"edge {edge:+.1%} above break-even {be:.1%}"
    return {"verdict": verdict, "reason": reason, "sizing": sizing}


def simulate_bankroll(
    rows: list[dict],
    bankroll: float,
    *,
    stake_fraction: float,
    payout_fallback: float = 0.8,
) -> dict:
    """Replay the honest paper ledger as if real money followed every signal.

    A fixed fraction of the CURRENT bankroll is staked each time (compounding).
    This is a what-if, not a promise: it uses the same WIN/LOSS the feed already
    recorded, so it cannot invent an edge the signals did not have.
    """
    equity = float(bankroll)
    peak = equity
    max_drawdown = 0.0
    curve = [equity]
    for r in rows:
        payout = float(r.get("payout") or payout_fallback)
        stake = stake_fraction * equity
        equity += stake * payout if r.get("win") else -stake
        equity = max(0.0, equity)
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
        curve.append(round(equity, 2))
        if equity <= 0:
            break
    n = len(rows)
    return {
        "start": round(float(bankroll), 2),
        "end": round(equity, 2),
        "return_pct": round(100 * (equity - bankroll) / bankroll, 2) if bankroll else 0.0,
        "trades": n,
        "max_drawdown_pct": round(100 * max_drawdown, 2),
        "curve": curve,
    }
