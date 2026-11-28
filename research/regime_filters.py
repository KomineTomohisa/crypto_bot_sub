"""Entry-only research gates; two closed rows, no future prices or live I/O."""
import math
from numbers import Real

from strategy_decisions import jst_timestamp


CONTEXTS = ('none', 'ema_follow', 'ema_fade', 'di_follow', 'di_fade', 'adx_rising')
BANDS = {'all': None, 'quiet': (0., .5), 'normal': (.25, 1.)}
SESSIONS = {'all': None, 'jst_01_09': (1, 9), 'jst_09_17': (9, 17), 'jst_17_01': (17, 1)}


def number(value, *, positive=False):
    if isinstance(value, bool) or type(value).__name__ == 'bool_' or not isinstance(value, Real):
        return None
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        return None
    return value


def allowed(latest, previous, at, side, case):
    context = case.get('entry_context', 'none')
    band = case.get('atr_band', 'all')
    session = case.get('session', 'all')
    if context not in CONTEXTS or band not in BANDS or session not in SESSIONS or side not in ('long', 'short'):
        raise ValueError('Unknown research entry gate')
    hours = SESSIONS[session]
    if hours:
        hour = jst_timestamp(at).hour
        start, end = hours
        if not (start <= hour < end if start < end else hour >= start or hour < end):
            return False
    if BANDS[band]:
        lower, upper = BANDS[band]
        for row in (latest, previous):
            atr, close = number(row.get('ATR'), positive=True), number(row.get('close'), positive=True)
            if atr is None or close is None or not lower <= 100 * atr / close < upper:
                return False
    if context == 'none':
        return True
    if context == 'adx_rising':
        current, prior = number(latest.get('ADX')), number(previous.get('ADX'))
        return current is not None and prior is not None and 0 <= prior <= current <= 100
    columns = ('EMA_short', 'EMA_long') if context.startswith('ema_') else ('plus_di14', 'minus_di14')
    follow = context.endswith('follow')
    for row in (latest, previous):
        first, second = number(row.get(columns[0])), number(row.get(columns[1]))
        if first is None or second is None or min(first, second) < 0:
            return False
        # DI+ / fast EMA being larger denotes the upward direction.
        upward = side == 'long' if follow else side == 'short'
        if not (first >= second if upward else first <= second):
            return False
    return True


def gate_pair(rows, case, at):
    for side, column in (('short', 'buy_signal'), ('long', 'sell_signal')):
        if not allowed(rows[0], rows[1], at, side, case):
            for row in rows:
                row[column] = False
    return rows
