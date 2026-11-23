"""Opt-in, entry-only research models. Never imported by production execution.

The legacy model delegates unchanged. New flags are generated only on opening
observations, using the two consecutive closed bars. Exit observations always
receive the original strategy signals. No score weights are silently enabled.
"""
from numbers import Real
import math

import pandas as pd

from strategy_decisions import jst_timestamp


LEGACY = 'legacy_filters'
TREND = 'trend_follow_v1'
MODELS = (LEGACY, TREND)
VERSION = 'research_entry_models_v1'


def validate_model(model):
    if not isinstance(model, str) or model not in MODELS:
        raise ValueError('Unknown research entry model')
    return model


def model_spec(model=LEGACY):
    validate_model(model)
    return dict(version=VERSION, name=model, research_only=True,
                confirmation='two_consecutive_closed_15min_bars',
                mapping=dict(long='sell_signal', short='buy_signal'),
                exits='unchanged_legacy_signals_and_exit_rules',
                weighted_score_enabled=False,
                conditions=(dict(adx_min=25., adx_max=100.,
                    long='close > EMA_short > EMA_long; plus_di14 > minus_di14; MACD_histogram > 0',
                    short='close < EMA_short < EMA_long; minus_di14 > plus_di14; MACD_histogram < 0',
                    invalid='block_both_sides', equality='no_direction') if model == TREND else
                    dict(delegate='original_prepare_without_wrapping')))


def finite(value, *, positive=False, bounded=False):
    if isinstance(value, bool) or type(value).__name__ == 'bool_' or not isinstance(value, Real):
        return None
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0) or (bounded and not 0 <= value <= 100):
        return None
    return value


def trend_flags(row):
    """Return long, short; strict finite validation precedes direction checks."""
    close, fast, slow = (finite(row.get(c), positive=True) for c in ('close','EMA_short','EMA_long'))
    adx, plus, minus = (finite(row.get(c), bounded=True) for c in ('ADX','plus_di14','minus_di14'))
    histogram = finite(row.get('MACD_histogram'))
    if any(v is None for v in (close,fast,slow,adx,plus,minus,histogram)) or adx < 25:
        return False, False
    return (bool(close > fast > slow and plus > minus and histogram > 0),
            bool(close < fast < slow and minus > plus and histogram < 0))


def closed_pair(latest, previous, observed_at):
    try:
        at = jst_timestamp(observed_at)
        latest_at, previous_at = (jst_timestamp(r['timestamp']) for r in (latest,previous))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    bar = pd.Timedelta(minutes=15)
    return latest_at-previous_at == bar and latest_at+bar == at.floor('15min')


def entry_rows(latest, previous, observed_at, model=LEGACY):
    """Copy inputs; no mutation of shared caches, source signals or forming rows."""
    validate_model(model)
    rows = [latest.copy(), previous.copy()]
    if model == LEGACY:
        return rows
    ready = closed_pair(latest,previous,observed_at)
    for row in rows:
        long, short = trend_flags(row) if ready else (False, False)
        row['buy_signal'], row['sell_signal'] = short, long
    return rows


def prepare_model(cache, model=LEGACY):
    """Adapter for the existing extracted source backtest; opt-in only."""
    validate_model(model)
    if model == LEGACY:
        return cache.prepare

    def prepare(bot, symbol, intraday, hourly, observed_at):
        result, hours = cache.prepare(bot,symbol,intraday,hourly,observed_at)
        # Existing positions are observed at close-minus-1ns by the source
        # engine. Their original signal history must not be replaced.
        if jst_timestamp(observed_at) != jst_timestamp(intraday.timestamp.iloc[-1]):
            return result, hours
        result = result.copy(deep=True)
        if len(result) < 3:
            result['buy_signal'] = False
            result['sell_signal'] = False
            return result, hours
        rows = entry_rows(result.iloc[-2],result.iloc[-3],observed_at,model)
        for position, row in zip((-2,-3),rows):
            for column in ('buy_signal','sell_signal'):
                result.iloc[position,result.columns.get_loc(column)] = bool(row[column])
        # The current bar remains ineligible even if an upstream adapter later
        # changes how it represents an unfinished bar.
        for column in ('buy_signal','sell_signal'):
            result.iloc[-1,result.columns.get_loc(column)] = False
        return result, hours

    return prepare
