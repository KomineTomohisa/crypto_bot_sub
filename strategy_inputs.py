"""One timestamp-bounded feature pipeline for live observations and replay."""

import pandas as pd
import numpy as np

from strategy_decisions import jst_timestamp


FEATURE_HISTORY_VERSION = "jst_prior2_calendar_days_v1"
RAW_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def candle_api_dates(observed_at):
    """UTC API days covering the strategy's previous two JST calendar days."""
    at = jst_timestamp(observed_at)
    start = (at.normalize() - pd.Timedelta(days=2)).tz_convert('UTC').normalize()
    return [day.strftime('%Y%m%d') for day in pd.date_range(
        start, at.tz_convert('UTC').normalize(), freq='1D')]


def candle_cache_valid(timeframe, date_str, observed_at, modified_at):
    """Refresh mutable daily files every 30s; only seal caches written after day end."""
    at = jst_timestamp(observed_at).tz_convert('UTC')
    if timeframe not in ('1min', '5min', '15min', '30min', '1hour'):
        return False
    day = pd.to_datetime(date_str, format='%Y%m%d', utc=True)
    if day > at.normalize():
        return False
    end = day + pd.Timedelta(days=1, minutes=1)
    modified = pd.Timestamp(modified_at, unit='s', tz='UTC')
    age = (at - modified).total_seconds()
    if age < 0:
        return False
    minutes = {'1min':1, '5min':5, '15min':15, '30min':30, '1hour':60}[timeframe]
    # A formerly forming row must be fetched again when it becomes closed.
    boundary = min(at.floor(f'{minutes}min'), day + pd.Timedelta(days=1))
    if modified < boundary:
        return False
    ttl = 30 * 86400 if at >= end and modified >= end else 30
    return age < ttl


def entry_input_quality(intraday, hourly, observed_at):
    """Require complete, valid history and the current 15m row before entering."""
    try:
        if intraday is None or hourly is None:
            return {'ready': False, 'reason': 'unavailable'}
        raw15 = feature_input(intraday, observed_at, 15)
        raw60 = feature_input(hourly, observed_at, 60, closed_only=True)
        for frame in (raw15, raw60):
            if frame.empty:
                return {'ready': False, 'reason': 'empty'}
            values = frame[['open','high','low','close','volume']].apply(pd.to_numeric, errors='raise')
            if not np.isfinite(values.to_numpy()).all() or not values[['open','high','low','close']].gt(0).all().all() or not values.volume.ge(0).all():
                return {'ready': False, 'reason': 'invalid_ohlcv'}
            if not (values.high.ge(values[['open','close','low']].max(axis=1)) & values.low.le(values[['open','close','high']].min(axis=1))).all():
                return {'ready': False, 'reason': 'invalid_ohlcv'}
        gaps15 = history_quality(raw15, observed_at, 15)['missing_closed_intervals']
        gaps60 = history_quality(raw60, observed_at, 60)['missing_closed_intervals']
        current = jst_timestamp(observed_at).floor('15min')
        ready = not gaps15 and not gaps60 and raw15.timestamp.iloc[-1] == current
        return {'ready': bool(ready), 'reason': 'ready' if ready else 'missing_or_stale',
                'missing_15min': gaps15, 'missing_1hour': gaps60}
    except (ValueError, TypeError, KeyError):
        return {'ready': False, 'reason': 'invalid_input'}


def candle_partition_complete(data, timeframe, date_str):
    """An old file is immutable only after every expected UTC bar exists."""
    minutes = {'1min':1, '5min':5, '15min':15, '30min':30, '1hour':60}
    try:
        rows = data['data']['candlestick'][0]['ohlcv']
        times = pd.to_datetime([row[5] for row in rows], unit='ms', utc=True)
        start = pd.to_datetime(date_str, format='%Y%m%d', utc=True)
        expected = pd.date_range(start, start + pd.Timedelta(days=1),
                                 freq=f'{minutes[timeframe]}min', inclusive='left')
        return len(times) == len(expected) and times.equals(expected)
    except (KeyError, ValueError, TypeError, IndexError):
        return False


def feature_history_spec():
    return {
        "version": FEATURE_HISTORY_VERSION,
        "timezone": "Asia/Tokyo",
        "previous_calendar_days": 2,
        "forming_bar": "observed_values_only",
        "hourly": "closed_only",
    }


def feature_input(frame, observed_at, minutes, *, closed_only=False):
    """Bound raw input before computing EMA, normalization, or any other feature."""
    if frame.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)
    missing = set(RAW_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature input columns: {sorted(missing)}")
    result = frame[RAW_COLUMNS].copy()
    times = pd.to_datetime(result["timestamp"], errors="raise")
    if times.isna().any():
        raise ValueError("Missing candle timestamp")
    times = (
        times.dt.tz_localize("Asia/Tokyo")
        if times.dt.tz is None
        else times.dt.tz_convert("Asia/Tokyo")
    )
    result["timestamp"] = times
    at = jst_timestamp(observed_at)
    start = at.normalize() - pd.Timedelta(days=2)
    cutoff = at.floor(f"{minutes}min")
    available = (
        times + pd.Timedelta(minutes=minutes) <= at if closed_only else times.le(cutoff)
    )
    result = result.loc[times.ge(start) & available].copy()
    result = result.drop_duplicates().sort_values("timestamp").reset_index(drop=True)
    if result["timestamp"].duplicated().any():
        raise ValueError("Conflicting candle snapshots")
    return result


def prepare_strategy_inputs(bot, symbol, intraday, hourly, observed_at):
    """The caller supplies only values already observed in the forming candle."""
    raw15 = feature_input(intraday, observed_at, 15)
    raw60 = feature_input(hourly, observed_at, 60, closed_only=True)
    features15 = bot.build_features(raw15.copy())
    features60 = bot.build_features(raw60.copy())
    signals = bot.generate_signals_with_sentiment(symbol, features15, features60)
    # The lazy import keeps the data policy usable without a filesystem journal.
    from strategy_recording import record_feature_snapshot

    record_feature_snapshot(bot, symbol, observed_at, raw15, raw60, signals, features60)
    return signals, features60


def history_quality(frame, observed_at, minutes):
    at = jst_timestamp(observed_at)
    start = at.normalize() - pd.Timedelta(days=2)
    expected = pd.date_range(
        start, at.floor(f"{minutes}min"), freq=f"{minutes}min", inclusive="left"
    )
    closed = pd.DatetimeIndex(
        frame.loc[frame["timestamp"] + pd.Timedelta(minutes=minutes) <= at, "timestamp"]
    )
    return {
        "missing_closed_intervals": len(expected.difference(closed)),
        "latest_closed_bar": closed.max().isoformat() if len(closed) else None,
    }


def historical_observation(frame, timestamp, *, at_open):
    """Do not let a completed historical candle masquerade as its opening state."""
    timestamp = jst_timestamp(timestamp)
    result = frame.loc[frame["timestamp"].le(timestamp)].copy()
    if at_open and not result.empty:
        mask = result["timestamp"].eq(timestamp)
        for column in ("high", "low", "close"):
            result.loc[mask, column] = result.loc[mask, "open"]
        result.loc[mask, "volume"] = 0.0
    return result
