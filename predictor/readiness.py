"""Go-live readiness gate.

The question this answers is the only one that matters before risking money:
is the measured edge statistically real, or is it noise that happens to look
positive? A 58% win rate on 20 trades is meaningless; on 500 trades it may be a
real edge. This module does the arithmetic honestly and returns a verdict.

It never places a trade and it never says "ready" from one lucky run:

- Wilson lower bound: the pessimistic end of the win-rate confidence interval,
  compared against the broker's break-even ``1 / (1 + payout)``.
- Binomial p-value (one-sided): how likely a run this good (or better) is if the
  true win rate were exactly break-even.
- Sample-size floor: below MIN_TRADES a verdict is FORWARD_ONLY regardless, so
  nobody reads a hot streak as an edge.

Verdicts: ``go`` (real, survivable edge over enough trades), ``forward-only``
(keep paper/forward testing), ``no-go`` (measurably below break-even).
"""
from __future__ import annotations

from math import exp, lgamma, log, log1p

from . import risk

MIN_TRADES = 100  # below this, a verdict is never "go"
Z = 1.96  # 95% confidence


def wilson_lower_bound(wins: int, n: int, z: float = Z) -> float | None:
    """95% Wilson lower bound on the win rate (pessimistic estimate)."""
    if n <= 0:
        return None
    phat = wins / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5)
    return max(0.0, (centre - margin) / denom)


def p_value(wins: int, n: int, null_rate: float) -> float | None:
    """One-sided binomial p-value: P(this many wins or more | true rate = null).

    Computed in log space with ``lgamma`` so it stays exact for any n: the raw
    binomial coefficients overflow once n reaches a few hundred, which a long
    forward record (or a backtest reusing this helper) can easily reach.
    """
    if n <= 0:
        return None
    if null_rate <= 0.0:
        return 0.0
    if null_rate >= 1.0:
        return 1.0
    log_p = log(null_rate)
    log_q = log1p(-null_rate)
    total = 0.0
    for k in range(wins, n + 1):
        log_c = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
        total += exp(log_c + k * log_p + (n - k) * log_q)
    return min(1.0, total)


def assess(
    *,
    forward_wins: int,
    forward_trades: int,
    payout: float,
    walk_forward_rate: float | None = None,
    walk_forward_signals: int | None = None,
    alpha: float = 0.05,
    min_trades: int = MIN_TRADES,
) -> dict:
    """Combine forward results and walk-forward context into a go-live verdict.

    ``forward_wins``/``forward_trades`` are the honest paper ledger (resolved
    signals). ``walk_forward_rate`` is the % from the out-of-sample backtest,
    used only as supporting context — the forward record decides.
    """
    be = risk.break_even_rate(payout)
    out: dict = {
        "forward_trades": forward_trades,
        "forward_wins": forward_wins,
        "payout": payout,
        "break_even": round(be, 4),
        "min_trades": min_trades,
        "alpha": alpha,
    }
    if walk_forward_rate is not None:
        out["walk_forward_rate"] = walk_forward_rate
        out["walk_forward_signals"] = walk_forward_signals

    if forward_trades <= 0:
        out.update({
            "verdict": "forward-only",
            "rate": None,
            "wilson_lower": None,
            "p_value": None,
            "reason": "no resolved forward trades yet — keep forward-testing",
        })
        return out

    rate = forward_wins / forward_trades
    lower = wilson_lower_bound(forward_wins, forward_trades)
    p = p_value(forward_wins, forward_trades, be)
    out.update({"rate": round(rate, 4), "wilson_lower": round(lower or 0.0, 4),
                "p_value": round(p, 5) if p is not None else None})

    if rate < be:
        verdict = "no-go"
        reason = (f"forward win rate {rate:.1%} is below break-even {be:.1%} — "
                  f"this loses money over time")
    elif forward_trades < min_trades:
        verdict = "forward-only"
        reason = (f"looks positive ({rate:.1%}) but only {forward_trades} trades — "
                  f"need {min_trades} before a real verdict")
    elif lower is not None and lower <= be:
        verdict = "forward-only"
        reason = (f"win rate {rate:.1%} beats break-even, but the 95% lower bound "
                  f"{lower:.1%} does not — the edge is not yet statistically real")
    elif p is not None and p > alpha:
        verdict = "forward-only"
        reason = (f"win rate {rate:.1%} is not significant at p<{alpha} "
                  f"(p={p:.3f}) — could be chance")
    else:
        verdict = "go"
        reason = (f"win rate {rate:.1%} over {forward_trades} trades clears "
                  f"break-even {be:.1%} with a 95% lower bound of {lower:.1%} "
                  f"(p={p:.4f})")
    out.update({"verdict": verdict, "reason": reason})
    return out
