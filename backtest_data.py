"""Continuous candle history and isolated state for offline backtests."""

from contextlib import contextmanager

import numpy as np
import pandas as pd


def merge_candle_chunks(chunks, start, end):
    """Normalize daily/bulk data to a single [start, end) timeline in JST."""
    parts = []
    for chunk in chunks:
        if chunk.empty:
            continue
        frame = chunk.copy()
        if "timestamp" not in frame:
            raise ValueError("Backtest candles require timestamps")
        timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
        if timestamps.isna().any():
            raise ValueError("Backtest candle timestamp is missing")
        # get_cached_data returns naive timestamps already shifted to JST.
        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize("Asia/Tokyo")
        else:
            timestamps = timestamps.dt.tz_convert("Asia/Tokyo")
        frame["timestamp"] = timestamps
        frame = frame.loc[(timestamps >= start) & (timestamps < end)]
        if not frame.empty:
            parts.append(frame)
    if not parts:
        return pd.DataFrame()
    merged = pd.concat(parts, ignore_index=True).drop_duplicates()
    if merged["timestamp"].duplicated().any():
        raise ValueError("Conflicting backtest candles at the same timestamp")
    for column in ("open", "close"):
        values = pd.to_numeric(merged[column], errors="coerce")
        if not (np.isfinite(values) & values.gt(0)).all():
            raise ValueError(f"Invalid backtest {column} price")
        merged[column] = values
    return merged.sort_values("timestamp").reset_index(drop=True)


def load_backtest_candles(loader, symbol, start, end):
    """Read each date once, retaining warmup history and disabling date fallback."""
    intraday = []
    hourly = []
    for day in pd.date_range(start - pd.Timedelta(days=3), end, inclusive="left"):
        date_str = day.strftime("%Y%m%d")
        intraday.append(loader(symbol, "15min", date_str, fallback_days=0))
        hourly.append(loader(symbol, "1hour", date_str, fallback_days=0))
    return (
        merge_candle_chunks(intraday, start - pd.Timedelta(days=2), end),
        merge_candle_chunks(hourly, start - pd.Timedelta(days=2), end),
    )


def excel_backtest_inputs(trades, signal_history):
    """Excel requires naive datetimes; keep replay/JSON timestamps untouched."""

    def local_time(value):
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert("Asia/Tokyo").tz_localize(None)
        return stamp

    exported_trades = []
    for trade in trades:
        exported = dict(trade)
        for key in ("entry_time", "exit_time"):
            if exported.get(key) is not None:
                exported[key] = local_time(exported[key])
        exported_trades.append(exported)
    exported_history = {}
    for symbol, frame in signal_history.items():
        exported = frame.copy()
        if exported.index.tz is not None:
            exported.index = exported.index.tz_convert("Asia/Tokyo").tz_localize(None)
        exported_history[symbol] = exported
    return exported_trades, exported_history


@contextmanager
def backtest_state(bot):
    """Keep replay positions/protection and accounting out of runtime state."""
    missing = object()
    replacements = {
        "is_backtest": True,
        "entry_sizes": {},
        "opposite_narrow_state": {},
        "_last_valid_exit_levels": {},
        "short_profit_lock_armed": {},
        "total_profit": bot.total_profit,
        "_backtest_period": None,
        "_backtest_summary": None,
    }
    original = {key: getattr(bot, key, missing) for key in replacements}
    try:
        for key, value in replacements.items():
            setattr(bot, key, value)
        yield
    finally:
        for key, value in original.items():
            if value is missing:
                if hasattr(bot, key):
                    delattr(bot, key)
            else:
                setattr(bot, key, value)
