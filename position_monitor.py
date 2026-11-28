"""Cooperative protection checks during live-loop waits; no worker/order race.

Only the existing main thread executes callbacks. The cadence is a target,
not a hard deadline: synchronous I/O and close reconciliation can overrun it.
"""
import math
import time

import pandas as pd

from trading_execution import ExchangeMaintenance


POLL_INTERVAL_SECONDS = 2.0
SLOW_CHECK_SECONDS = 5.0


def wait_with_position_monitoring(bot, seconds, stats, trade_logs, *, clock=None, sleep=None):
    """Preserve the analysis wait deadline while checking current exposure.

    No candle fetch, position reconciliation, new entry or stats refresh is
    scheduled here. A triggered exit uses the existing durable close path.
    """
    clock = clock or time.monotonic
    sleep = sleep or time.sleep
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid monitoring wait')
    if seconds == 0:
        return
    live = (getattr(bot, 'exchange_settings_gmo', {}) or {}).get('live_trade', False)
    if not live or getattr(bot, 'is_backtest', False):
        sleep(seconds)
        return
    deadline = clock() + seconds
    while clock() < deadline:
        held = sorted(s for s in bot.symbols if bot.positions.get(s))
        if not held:
            # Positions cannot be opened in this single-threaded wait path.
            sleep(max(0.0, deadline-clock()))
            return
        sleep(min(POLL_INTERVAL_SECONDS, max(0.0, deadline-clock())))
        for symbol in held:
            if not bot.positions.get(symbol):
                continue  # A preceding close may have synchronized the account.
            started = clock()
            try:
                bot._check_exit_conditions(symbol, stats, trade_logs, pd.DataFrame(), monitor_only=True)
            except ExchangeMaintenance:
                # The caller may itself be in an exception backoff. Escaping
                # from there would terminate run_live instead of entering its
                # shared availability gate on the next iteration.
                return
            except Exception as exc:
                bot.logger.warning('%s: interval exit check failed (%s); continuing other positions', symbol, type(exc).__name__)
            finally:
                elapsed = max(0.0, clock()-started)
                if elapsed > SLOW_CHECK_SECONDS:
                    bot.logger.warning('%s: interval exit check took %.3fs; polling target %.1fs exceeded',
                                       symbol, elapsed, POLL_INTERVAL_SECONDS)
