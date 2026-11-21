"""Missing indicator inputs must preserve protection without an exception storm."""
from copy import deepcopy
from decimal import Decimal
import logging
import sys
from types import MethodType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest_data import backtest_state
from live_execution import LiveExecutionMixin
from test_bot_execution_integration import BOT, method
from test_execution import service, records
from test_execution_recovery import bot_for, two_positions
from trading_execution import fallback_exit


def normal_frame():
    return pd.DataFrame({
        'ATR': [120.] * 5, 'ADX': [30.] * 5,
        'buy_signal': [False] * 5, 'sell_signal': [False] * 5,
        'plus_di14': [25.] * 5, 'minus_di14': [25.] * 5,
    })


def bot_for_exit():
    bot = SimpleNamespace(logger=logging.getLogger('atr-fallback-test'))
    for name in ('calculate_dynamic_exit_levels', '_get_dynamic_profit_loss_settings'):
        fn, _ = method(BOT, name)
        setattr(bot, name, MethodType(fn, bot))
    bot.sr_calls = []
    def sr(**kw):
        bot.sr_calls.append(kw)
        return {'take_profit_price': kw['base_take_profit'], 'stop_loss_price': kw['base_stop_loss']}
    bot._apply_sr_exit_levels = sr
    return bot


def unavailable(case):
    frame = normal_frame()
    if case == 'none': return None
    if case == 'empty': return pd.DataFrame()
    if case == 'empty_columns': return frame.iloc[:0]
    if case == 'missing_atr': return frame.drop(columns='ATR')
    if case == 'missing_adx': return frame.drop(columns='ADX')
    if case == 'only_forming': return frame.iloc[-1:]
    if case == 'all_nan':
        frame['ATR'] = np.nan
        return frame
    value = {'nan': np.nan, 'positive_infinity': np.inf, 'negative_infinity': -np.inf,
             'zero': 0., 'negative': -1., 'string': 'unavailable', 'null': None,
             'bool': True, 'pd_na': pd.NA, 'overflow': 10**1000}[case]
    frame['ATR'] = frame['ATR'].astype(object)
    frame.loc[frame.index[-2], 'ATR'] = value
    # Both the older and forming ATR remain valid; neither may replace this row.
    return frame


CASES = ['none', 'empty', 'empty_columns', 'missing_atr', 'missing_adx', 'only_forming',
         'all_nan', 'nan', 'positive_infinity', 'negative_infinity', 'zero', 'negative',
         'string', 'null', 'bool', 'pd_na', 'overflow']


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('side', ['long', 'short'])
def test_unavailable_atr_uses_base_protection_without_errors_or_state_changes(case, side, caplog):
    bot = bot_for_exit()
    bot.opposite_narrow_state = {'sol_jpy': {'long': True, 'short': True}}
    before = deepcopy(bot.opposite_narrow_state)
    frame = unavailable(case)
    original = frame.copy(deep=True) if frame is not None else None
    expected = fallback_exit(bot._get_dynamic_profit_loss_settings('sol_jpy', frame), side, 1000)
    result = bot.calculate_dynamic_exit_levels('sol_jpy', frame, side, 1000)
    assert result == expected
    assert bot.opposite_narrow_state == before and not bot.sr_calls
    assert not hasattr(bot, '_last_valid_exit_levels')  # Defaults cannot replace a valid cache.
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)
    if frame is not None: pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize('side', ['long', 'short'])
def test_cached_protection_is_preserved_before_any_settings_or_sr_lookup(side, caplog):
    bot = bot_for_exit()
    good = bot.calculate_dynamic_exit_levels('sol_jpy', normal_frame(), side, 1000)
    key = ('sol_jpy', side, 1000)
    state = deepcopy(bot.opposite_narrow_state)
    sr_count = len(bot.sr_calls)
    def forbidden(*a): raise AssertionError('cached protection must not need settings')
    bot._get_dynamic_profit_loss_settings = forbidden
    for case in CASES:
        result = bot.calculate_dynamic_exit_levels('sol_jpy', unavailable(case), side, 1000)
        assert result == good and result is not bot._last_valid_exit_levels[key]
        result['stop_loss_price'] = -1  # The returned copy cannot corrupt the cache.
        assert bot._last_valid_exit_levels[key] == good
    assert bot.opposite_narrow_state == state and len(bot.sr_calls) == sr_count
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)


@pytest.mark.parametrize('side,expected_tp,expected_sl', [('long', 1030., 976.25), ('short', 970., 1025.)])
def test_normal_calculation_is_unchanged_and_uses_closed_atr(side, expected_tp, expected_sl):
    bot = bot_for_exit(); frame = normal_frame()
    frame.loc[frame.index[-1], 'ATR'] = np.nan  # A forming indicator is allowed to be missing.
    result = bot.calculate_dynamic_exit_levels('sol_jpy', frame, side, 1000)
    assert result['take_profit_price'] == pytest.approx(expected_tp)
    assert result['stop_loss_price'] == pytest.approx(expected_sl)
    assert result['atr_value'] == 120.


def test_warning_is_emitted_on_transition_and_success_rearms_it(caplog):
    bot = bot_for_exit(); caplog.set_level(logging.INFO, logger='atr-fallback-test')
    for _ in range(8): bot.calculate_dynamic_exit_levels('sol_jpy', pd.DataFrame(), 'long', 1000)
    assert sum('exit levels fallback' in r.message for r in caplog.records) == 1
    bot.calculate_dynamic_exit_levels('sol_jpy', normal_frame(), 'long', 1000)
    assert not bot._exit_level_fallback_state
    assert sum('exit level calculation recovered' in r.message for r in caplog.records) == 1
    bot.calculate_dynamic_exit_levels('sol_jpy', pd.DataFrame(), 'long', 1000)
    assert sum('exit levels fallback' in r.message for r in caplog.records) == 2


@pytest.mark.parametrize('symbol,side,entry', [('sol_jpy', 'short', 1000), ('sol_jpy', 'long', 2000), ('ltc_jpy', 'long', 1000)])
def test_other_symbol_side_or_entry_cannot_borrow_cached_protection(symbol, side, entry):
    bot = bot_for_exit()
    bot.calculate_dynamic_exit_levels('sol_jpy', normal_frame(), 'long', 1000)
    frame = pd.DataFrame()
    expected = fallback_exit(bot._get_dynamic_profit_loss_settings(symbol, frame), side, entry)
    assert bot.calculate_dynamic_exit_levels(symbol, frame, side, entry) == expected


def test_position_reset_clears_only_its_fallback_status():
    bot = bot_for_exit()
    for symbol in ('sol_jpy', 'ltc_jpy'):
        bot.calculate_dynamic_exit_levels(symbol, pd.DataFrame(), 'long', 1000)
    LiveExecutionMixin._reset_exit_adjustments(bot, 'sol_jpy')
    assert set(bot._exit_level_fallback_state) == {('ltc_jpy', 'long', 1000)}


@pytest.mark.parametrize('existing', [False, True])
def test_backtest_does_not_leak_fallback_status_into_live_object(existing):
    bot = bot_for_exit(); bot.total_profit = 0
    if existing: bot.calculate_dynamic_exit_levels('sol_jpy', pd.DataFrame(), 'long', 1000)
    before = getattr(bot, '_exit_level_fallback_state', None)
    with pytest.raises(RuntimeError, match='stop replay'):
        with backtest_state(bot):
            assert not bot._exit_level_fallback_state
            bot.calculate_dynamic_exit_levels('ltc_jpy', pd.DataFrame(), 'short', 1000)
            raise RuntimeError('stop replay')
    if existing: assert bot._exit_level_fallback_state is before
    else: assert not hasattr(bot, '_exit_level_fallback_state')


@pytest.mark.parametrize('side,price,reason', [
    ('long', 900., '損切り'), ('short', 1100., '損切り'),
    ('long', 1100., '利益確定'), ('short', 900., '利益確定'),
    ('long', 1000., '長時間保有による強制決済'),
])
@pytest.mark.parametrize('cached', [False, True])
def test_quote_protection_still_closes_with_no_candles(side, price, reason, cached, caplog):
    bot = bot_for_exit(); at = pd.Timestamp('2026-11-21T13:00:00+09:00')
    if cached: bot.calculate_dynamic_exit_levels('sol_jpy', normal_frame(), side, 1000.)
    bot.positions = {'sol_jpy': side}; bot.entry_prices = {'sol_jpy': 1000.}
    bot.entry_sizes = {'sol_jpy': 1.}; bot.entry_times = {'sol_jpy': at-pd.Timedelta(hours=73 if price==1000 else 1)}
    bot.short_profit_lock_armed = {}; bot.exchange_settings_gmo = {'live_trade': True}
    bot.get_current_price = lambda _: price
    bot._get_market_data = lambda _: (None, None)
    closes = []; bot._close_live_position = lambda *a: closes.append(a)
    bot.save_positions = lambda: None
    for name in ('_prepare_live_inputs', '_check_exit_conditions', '_evaluate_exit_event', '_apply_short_profit_lock'):
        fn, ns = method(BOT, name); ns['now_jst'] = lambda: at
        setattr(bot, name, MethodType(fn, bot))
    assert bot._prepare_live_inputs('sol_jpy', {}, []) == (None, None)
    assert len(closes) == 1 and closes[0][1] == reason
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)


def test_short_profit_lock_remains_active_during_missing_atr():
    bot = bot_for_exit(); at = pd.Timestamp('2026-11-21T13:00:00+09:00')
    bot.short_profit_lock_armed = {}; bot.save_positions = lambda: None
    for name in ('_evaluate_exit_event', '_apply_short_profit_lock'):
        fn, _ = method(BOT, name)
        setattr(bot, name, MethodType(fn, bot))
    entry_at = at-pd.Timedelta(hours=1)
    armed = bot._evaluate_exit_event('sol_jpy', normal_frame(), 'short', 1000., entry_at, at, 990.)
    assert armed.reason is None and bot.short_profit_lock_armed['sol_jpy']
    missing = bot._evaluate_exit_event('sol_jpy', pd.DataFrame(), 'short', 1000., entry_at, at, 998.)
    assert missing.reason == '利益保護' and missing.stop_loss == pytest.approx(997.5)


def test_unexpected_calculation_failure_remains_visible_and_keeps_cached_protection(caplog):
    bot = bot_for_exit()
    good = bot.calculate_dynamic_exit_levels('sol_jpy', normal_frame(), 'long', 1000)
    def failure(*a): raise RuntimeError('controlled calculation failure')
    bot._get_dynamic_profit_loss_settings = failure
    assert bot.calculate_dynamic_exit_levels('sol_jpy', normal_frame(), 'long', 1000) == good
    assert any(r.levelno == logging.ERROR and r.exc_info for r in caplog.records)


@pytest.mark.parametrize('side', ['BUY', 'SELL'])
@pytest.mark.parametrize('cached', [False, True])
def test_missing_candles_closes_all_targets_once_and_records_fills(store, side, cached, caplog, monkeypatch):
    monkeypatch.setitem(sys.modules, 'db', SimpleNamespace(engine=store.engine))
    svc, api = service(store); two_positions(api)
    for row in api.position_rows: row['side'] = side
    bot = bot_for(svc); bot._sync_live_positions()
    bot.exchange_settings_gmo = {'live_trade': True}
    at = pd.Timestamp(api.now).tz_convert('Asia/Tokyo')
    bot.entry_times['ltc_jpy'] = at-pd.Timedelta(hours=1)
    for name in ('calculate_dynamic_exit_levels', '_get_dynamic_profit_loss_settings',
                 '_prepare_live_inputs', '_check_exit_conditions', '_evaluate_exit_event', '_apply_short_profit_lock'):
        fn, ns = method(BOT, name); ns['now_jst'] = lambda: at
        setattr(bot, name, MethodType(fn, bot))
    bot._apply_sr_exit_levels = lambda **kw: {'take_profit_price': kw['base_take_profit'], 'stop_loss_price': kw['base_stop_loss']}
    if cached:
        bot.calculate_dynamic_exit_levels('ltc_jpy', normal_frame(), bot.positions['ltc_jpy'], bot.entry_prices['ltc_jpy'])
    price = 72. if side == 'BUY' else 108.
    bot.get_current_price = lambda _: price
    api.fill_price = Decimal(str(price))
    bot._get_market_data = lambda _: (None, None)
    assert bot._prepare_live_inputs('ltc_jpy', {}, []) == (None, None)
    bot._recover_live_orders()
    bot._prepare_live_inputs('ltc_jpy', {}, [])
    assert len(api.posts) == 2 and not api.position_rows
    assert all(p['side'] == ('SELL' if side == 'BUY' else 'BUY') for p in api.posts)
    assert len(records(store, 'trades')) == len(records(store, 'bot_execution_fills')) == 2
    assert all(row['raw']['exit_reason'] == '損切り' for row in records(store, 'trades'))
    assert not store.pending('test') and not store.close_batches('test')
    assert bot.positions['ltc_jpy'] is None and not bot._exit_level_fallback_state
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)
