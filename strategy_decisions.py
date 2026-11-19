"""Deterministic trade decisions shared by live and historical execution."""

from dataclasses import dataclass
import math

import pandas as pd


DECISION_VERSION = "shared_entry_exit_v1"


def symbol_priority(symbol, position):
    """Existing positions are evaluated before new entries; ties are stable."""
    return (0 if position else 1), symbol


def jst_timestamp(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Decision time is missing")
    return (
        stamp.tz_localize("Asia/Tokyo")
        if stamp.tzinfo is None
        else stamp.tz_convert("Asia/Tokyo")
    )


def next_entry_time(value):
    return jst_timestamp(value).floor("15min") + pd.Timedelta(minutes=15)


def _positive(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _flag(value):
    return False if pd.isna(value) else bool(value)


def entry_side(latest, previous, observed_at, blocked_until=None):
    """Two consecutive closed signals; Short retains the live loop's priority."""
    observed_at = jst_timestamp(observed_at)
    if blocked_until is not None and observed_at < jst_timestamp(blocked_until):
        return None
    latest_at = jst_timestamp(latest["timestamp"])
    previous_at = jst_timestamp(previous["timestamp"])
    bar = pd.Timedelta(minutes=15)
    if latest_at - previous_at != bar or latest_at + bar != observed_at.floor("15min"):
        return None
    if _flag(latest.get("buy_signal", False)) and _flag(
        previous.get("buy_signal", False)
    ):
        return "short"
    if _flag(latest.get("sell_signal", False)) and _flag(
        previous.get("sell_signal", False)
    ):
        return "long"
    return None


def short_profit_stop(
    side, entry_price, current_price, stop, armed, trigger=0.75, level=0.25
):
    """No persistence or clock access: arming is part of the decision state."""
    entry = _positive(entry_price)
    current = _positive(current_price)
    if side != "short" or entry is None or current is None:
        return float(stop), bool(armed)
    armed = bool(armed) or (entry - current) / entry * 100 >= trigger
    if armed:
        stop = min(float(stop), entry * (1 - level / 100))
    return float(stop), armed


@dataclass(frozen=True)
class ExitDecision:
    reason: str | None
    take_profit: float
    stop_loss: float
    holding_hours: float
    base_stop: float
    ambiguous_touches: bool = False


def exit_decision(
    *,
    side,
    take_profit,
    base_stop,
    stop_loss,
    current_price,
    high,
    low,
    holding_hours,
    profit_lock_armed=False,
):
    """Touch means a market-close intent, never an assumed fill at the threshold."""
    if side not in ("long", "short"):
        raise ValueError("Unknown position direction")
    tp = _positive(take_profit)
    base_sl = _positive(base_stop)
    sl = _positive(stop_loss)
    if tp is None or base_sl is None or sl is None:
        raise ValueError("Exit levels must be finite positive prices")
    cp, high, low = map(_positive, (current_price, high, low))
    if side == "long":
        tp_touch = (high is not None and high >= tp) or (cp is not None and cp >= tp)
        base_touch = (low is not None and low <= base_sl) or (
            cp is not None and cp <= base_sl
        )
        lock_touch = False
    else:
        tp_touch = (low is not None and low <= tp) or (cp is not None and cp <= tp)
        base_touch = (high is not None and high >= base_sl) or (
            cp is not None and cp >= base_sl
        )
        # A bar high observed before arming must never trip the newly armed stop.
        lock_touch = bool(
            profit_lock_armed and sl < base_sl and cp is not None and cp >= sl
        )
    stop_touch = base_touch or lock_touch
    reason = None
    if holding_hours >= 72:
        reason = "長時間保有による強制決済"
    elif stop_touch:
        # OHLC does not establish which threshold was reached first.
        reason = "利益保護" if lock_touch and not base_touch else "損切り"
    elif tp_touch:
        reason = "利益確定"
    return ExitDecision(
        reason, tp, sl, float(holding_hours), base_sl, bool(tp_touch and stop_touch)
    )
