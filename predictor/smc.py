"""Smart Money Concepts (SMC) as an ANALYSIS layer, not a model feature.

We A/B-tested SMC as extra model features (see research_smc_features.py): across
8 pair/timeframe rows it won some floors and lost others, a coin-flip, so it is
NOT wired into the model. It is still useful as a human-readable read of market
structure that sits next to the signal.

Everything here is causal: a swing is only "known" k bars after it forms, so the
analysis at the last candle never peeks at the future. Nothing here places a
trade or predicts a direction on its own.
"""
from __future__ import annotations

K = 3  # bars each side that confirm a swing


def _swings(h: list[float], low: list[float], k: int = K) -> list[tuple[int, str, float]]:
    """Confirmed swing points as (index, 'high'|'low', price), oldest first.

    Consecutive swings of the same kind (a flat top/bottom) are collapsed to the
    most extreme one, so HH/HL comparisons are not defeated by duplicates.
    """
    raw: list[tuple[int, str, float]] = []
    for i in range(k, len(h) - k):
        if h[i] == max(h[i - k : i + k + 1]):
            raw.append((i, "high", h[i]))
        if low[i] == min(low[i - k : i + k + 1]):
            raw.append((i, "low", low[i]))
    out: list[tuple[int, str, float]] = []
    for item in raw:
        if out and out[-1][1] == item[1]:
            prev = out[-1]
            more_extreme = item[2] > prev[2] if item[1] == "high" else item[2] < prev[2]
            if more_extreme:
                out[-1] = item
        else:
            out.append(item)
    return out


def analyze(candles: list[dict], k: int = K) -> dict:
    """Read structure, premium/discount, fair-value gaps and liquidity.

    Returns a dict of plain numbers/strings so both the API and the Telegram bot
    can render it. Rows with < 2*k+1 candles give a 'not enough data' result.
    """
    if len(candles) < 2 * k + 5:
        return {"ok": False, "reason": "not enough candles"}

    h = [float(c["high"]) for c in candles]
    low = [float(c["low"]) for c in candles]
    c = [float(c["close"]) for c in candles]
    n = len(c)
    price = c[-1]

    sw = _swings(h, low, k)
    highs = [s for s in sw if s[1] == "high"]
    lows = [s for s in sw if s[1] == "low"]
    if not highs or not lows:
        return {"ok": False, "reason": "no confirmed swings yet"}

    last_high = highs[-1][2]
    last_low = lows[-1][2]

    # Market structure from the last two swings of each kind.
    hh = len(highs) >= 2 and highs[-1][2] > highs[-2][2]
    hl = len(lows) >= 2 and lows[-1][2] > lows[-2][2]
    lh = len(highs) >= 2 and highs[-1][2] < highs[-2][2]
    ll = len(lows) >= 2 and lows[-1][2] < lows[-2][2]
    if hh and hl:
        structure, bias = "bullish (HH/HL)", "up"
    elif lh and ll:
        structure, bias = "bearish (LH/LL)", "down"
    else:
        structure, bias = "ranging / mixed", "neutral"

    # Break of structure vs the last confirmed swing.
    bos = "none"
    if price > last_high:
        bos = "bullish BOS"
    elif price < last_low:
        bos = "bearish BOS"

    # Premium / discount: 0 at the range low, 1 at the range high.
    rng = last_high - last_low
    pos = (price - last_low) / rng if rng else 0.5
    if pos >= 0.66:
        zone = "premium (expensive) — sellers' area"
    elif pos <= 0.34:
        zone = "discount (cheap) — buyers' area"
    else:
        zone = "equilibrium (fair value)"

    # Nearest fair-value gap that still has not been filled.
    gap = None
    for i in range(n - 2, 1, -1):
        if low[i] > h[i - 2] and not (low[i] <= c[-1] <= h[i - 2]):
            gap = {"side": "bullish", "low": round(h[i - 2], 6), "high": round(low[i], 6)}
            break
        if h[i] < low[i - 2] and not (h[i] <= c[-1] <= low[i - 2]):
            gap = {"side": "bearish", "low": round(h[i], 6), "high": round(low[i - 2], 6)}
            break

    # Liquidity sweep in the last 5 bars: wick pierced a swing, closed back in.
    sweep = "none"
    for i in range(max(1, n - 5), n):
        if h[i] > last_high and c[i] < last_high:
            sweep = f"buy-side sweep above {round(last_high, 6)} (rejection)"
        if low[i] < last_low and c[i] > last_low:
            sweep = f"sell-side sweep below {round(last_low, 6)} (rejection)"

    return {
        "ok": True,
        "price": round(price, 6),
        "structure": structure,
        "bias": bias,
        "bos": bos,
        "range": {"low": round(last_low, 6), "high": round(last_high, 6)},
        "position": round(pos, 3),
        "zone": zone,
        "fair_value_gap": gap,
        "liquidity_sweep": sweep,
        "note": "Structure analysis only — not a prediction, no trade placed.",
    }


def recommendation(a: dict) -> str:
    """Plain-language read of the structure: where a setup, if any, would be.

    This is NOT a direction call and NOT a probability. It only says how the
    structure and the premium/discount zone line up, so a human can decide.
    """
    if not a.get("ok"):
        return "Not enough structure to read yet."
    bias = a["bias"]
    pos = a["position"]
    parts: list[str] = []
    if bias == "up" and pos <= 0.5:
        parts.append("Structure is bullish and price is in discount — pullback-buy area.")
    elif bias == "up" and pos > 0.5:
        parts.append("Structure is bullish but price is in premium — wait for a discount pullback.")
    elif bias == "down" and pos >= 0.5:
        parts.append("Structure is bearish and price is in premium — pullback-sell area.")
    elif bias == "down" and pos < 0.5:
        parts.append("Structure is bearish but price is in discount — wait for a premium bounce.")
    else:
        parts.append("Structure is ranging — no structural edge; the model's own probability decides.")
    if a.get("fair_value_gap"):
        g = a["fair_value_gap"]
        parts.append(f"An unfilled {g['side']} gap sits {g['low']}-{g['high']}.")
    if a.get("liquidity_sweep") != "none":
        parts.append(f"Liquidity: {a['liquidity_sweep']}.")
    parts.append("Structure read only — the model's probability is the actual signal.")
    return " ".join(parts)
