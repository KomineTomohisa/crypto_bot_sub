"""Hand-calculated cost examples plus the actual historical execution loop."""
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from backtest_costs import BacktestAccount
from test_backtest_continuity import candles, replay, START, END


def config(**changes):
    return dict(name='test scenario', fee_bps={'ltc_jpy': 10, 'xrp_jpy': 10},
                spread_bps=0, slippage_bps=0, carry_bps_per_day=0, **changes)


@pytest.mark.parametrize('side,exit_ref,expected', [('long', 105, 47.95), ('short', 95, 48.05)])
def test_entry_and_exit_fees_are_charged_once_on_actual_leg_notional(side, exit_ref, expected):
    ledger = BacktestAccount(2000, ['ltc_jpy'], config())
    opened = ledger.open('ltc_jpy', side, 100, 10, START)
    assert opened['entry_fee'] == 1 and ledger.cash == 999
    result = ledger.close('ltc_jpy', exit_ref, START + pd.Timedelta(hours=1))
    assert result['profit'] == pytest.approx(expected)
    assert result['gross_profit'] == 50
    assert ledger.cash == pytest.approx(2000 + expected)
    assert ledger.summary()['realized_profit'] == pytest.approx(expected)


@pytest.mark.parametrize('side', ['long', 'short'])
def test_spread_and_slippage_have_adverse_sign_and_are_not_deducted_twice(side):
    costs = config()
    costs.update(spread_bps=20, slippage_bps=10, fee_bps={'ltc_jpy': 0})
    ledger = BacktestAccount(2000, ['ltc_jpy'], costs)
    opened = ledger.open('ltc_jpy', side, 100, 10, START)
    result = ledger.close('ltc_jpy', 100, START + pd.Timedelta(hours=1))
    assert opened['entry_price'] == pytest.approx(100.2 if side == 'long' else 99.8)
    assert result['exit_price'] == pytest.approx(99.8 if side == 'long' else 100.2)
    assert result['execution_cost'] == pytest.approx(4)
    assert result['profit'] == pytest.approx(-4)
    assert ledger.cash == pytest.approx(1996)


@pytest.mark.parametrize('entry,exit_at,charges', [
    ('05:00', '05:59', 0), ('05:00', '06:00', 1), ('06:00', '06:01', 0),
])
def test_carry_cutoff_entry_and_exit_boundary(entry, exit_at, charges):
    costs = config()
    costs['carry_bps_per_day'] = 4
    ledger = BacktestAccount(2000, ['ltc_jpy'], costs)
    base = '2026-11-14T'
    ledger.open('ltc_jpy', 'long', 100, 10, base + entry + ':00+09:00')
    result = ledger.close('ltc_jpy', 100, base + exit_at + ':00+09:00')
    assert result['carry_fee'] == pytest.approx(.4 * charges)
    assert len(result['carry_events']) == charges


def test_carry_is_charged_before_other_symbol_entry_and_only_once_across_gap():
    costs = config()
    costs.update(carry_bps_per_day=10, fee_bps={'ltc_jpy': 0, 'xrp_jpy': 0})
    ledger = BacktestAccount(2000, ['ltc_jpy', 'xrp_jpy'], costs)
    ledger.open('ltc_jpy', 'short', 100, 10, START)
    ledger.advance(START + pd.Timedelta(days=2, hours=6))
    assert ledger.cash == 997  # Three 06:00 cutoffs at 1 yen each.
    assert ledger.open('xrp_jpy', 'long', 100, 10, START + pd.Timedelta(days=2, hours=6)) is None
    ledger.advance(START + pd.Timedelta(days=2, hours=6))
    assert ledger.carry_paid == 3
    result = ledger.close('ltc_jpy', 100, START + pd.Timedelta(days=2, hours=7))
    assert result['profit'] == -3 and ledger.cash == 1997


def test_carry_uses_past_observation_not_future_close():
    costs = config()
    costs.update(carry_bps_per_day=4, fee_bps={'ltc_jpy': 0})
    ledger = BacktestAccount(2000, ['ltc_jpy'], costs)
    ledger.open('ltc_jpy', 'long', 100, 10, START)
    ledger.mark('ltc_jpy', 110, START + pd.Timedelta(hours=5))
    ledger.mark('ltc_jpy', 200, START + pd.Timedelta(hours=7))
    assert ledger.carry_paid == pytest.approx(.44)
    event = ledger.positions['ltc_jpy']['carry_events'][0]
    assert event['reference_price'] == 110
    assert pd.Timestamp(event['reference_at']) < pd.Timestamp(event['at'])


@pytest.mark.parametrize('side,expected', [('long', 47.95), ('short', 48.05)])
def test_actual_backtest_reports_net_profit_and_restores_runtime_state(side, expected):
    bot, run = replay(candles(side), costs=config())
    result = run()['ltc_jpy']
    trade = bot.trades[0]
    assert result['symbol_profit'] == pytest.approx(expected)
    assert trade['gross_profit'] == 50 and trade['profit'] == pytest.approx(expected)
    assert bot.valuation['ending_available_cash'] == pytest.approx(10000 + expected)
    assert bot.valuation['costs_included'] and bot.valuation['cost_status'] == 'scenario_estimate'
    for key, value in bot.before.items():
        assert getattr(bot, key) == value


def test_actual_backtest_insufficient_fee_cash_prevents_entry():
    bot, run = replay(candles(), costs=config(), capital=1000)
    assert run()['ltc_jpy']['trades'] == 0
    assert bot.valuation['ending_available_cash'] == 1000


@pytest.mark.parametrize('side,expected', [('long', 47.55), ('short', 47.65)])
def test_actual_closed_position_carry_reduces_profit_cash_and_statistics(side, expected):
    data = candles(side)
    data[['buy_signal', 'sell_signal']] = False
    data['close'] = 100.
    flag = 'sell_signal' if side == 'long' else 'buy_signal'
    data.loc[21:22, flag] = True  # Enter 05:45; first later observation crosses 06:00.
    data.loc[25:, 'close'] = 105. if side == 'long' else 95.
    costs = config()
    costs['carry_bps_per_day'] = 4
    bot, run = replay(data, costs=costs)
    result = run()['ltc_jpy']
    assert len(bot.trades) == 1
    assert bot.trades[0]['carry_fee'] == pytest.approx(.4)
    assert result['symbol_profit'] == pytest.approx(expected)
    assert bot.valuation['ending_available_cash'] == pytest.approx(10000 + expected)


def test_actual_incomplete_tail_still_reconciles_open_cost_metadata():
    data = candles()
    data.loc[97:, 'close'] = 103.
    costs = config()
    costs['carry_bps_per_day'] = 4
    bot, run = replay(data.iloc[:106], costs=costs)
    position = run()['ltc_jpy']['open_position']
    assert position['carry_fee'] == pytest.approx(.412)
    assert len(position['carry_events']) == 1
    assert bot.valuation['open_costs_paid'] == pytest.approx(1.412)


def test_negative_cash_from_carry_is_flagged_without_inventing_liquidation():
    costs = config()
    costs.update(fee_bps={'ltc_jpy': 0}, carry_bps_per_day=4)
    ledger = BacktestAccount(1000, ['ltc_jpy'], costs)
    ledger.open('ltc_jpy', 'long', 100, 10, START)
    ledger.advance(START + pd.Timedelta(hours=6))
    assert ledger.cash == pytest.approx(-.4)
    ledger.close('ltc_jpy', 100, START + pd.Timedelta(hours=7))
    assert ledger.summary()['negative_cash_observed']
    assert ledger.cash == pytest.approx(999.6)


def test_actual_multisymbol_account_reserves_entry_fee_before_next_entry():
    bot, run = replay(candles(), costs=config(), capital=2000, symbols=['ltc_jpy', 'xrp_jpy'])
    run()
    assert len(bot.trades) == 1 and bot.trades[0]['symbol'] == 'ltc_jpy'
    assert bot.valuation['ending_available_cash'] == pytest.approx(2047.95)


@pytest.mark.parametrize('side', ['long', 'short'])
def test_actual_open_position_costs_are_paid_without_fabricating_closed_profit(side):
    data = candles(side)
    data.loc[97:, 'close'] = 103 if side == 'long' else 97
    costs = config()
    costs['carry_bps_per_day'] = 4
    bot, run = replay(data, costs=costs)
    result = run()['ltc_jpy']
    expected_carry = (103 if side == 'long' else 97) * 10 * .0004
    assert result['trades'] == 0 and result['symbol_profit'] == 0
    assert bot.valuation['realized_profit'] == 0
    assert bot.valuation['open_costs_paid'] == pytest.approx(1 + expected_carry)
    assert bot.valuation['ending_equity'] == pytest.approx(10000 + 30 - 1 - expected_carry)
    assert bot.valuation['unrealized_profit'] == 30


@pytest.mark.parametrize('side', ['long', 'short'])
def test_actual_fill_friction_changes_trade_prices_without_extra_cash_deduction(side):
    costs = config()
    costs.update(spread_bps=20, slippage_bps=10, fee_bps={'ltc_jpy': 0})
    bot, run = replay(candles(side), costs=costs)
    run()
    trade = bot.trades[0]
    direction = 1 if side == 'long' else -1
    expected = direction * trade['size'] * (trade['exit_price'] - trade['entry_price'])
    assert trade['profit'] == pytest.approx(expected)
    assert trade['execution_cost'] > 0
    assert bot.valuation['ending_available_cash'] == pytest.approx(10000 + expected)


@pytest.mark.parametrize('field,value', [
    ('spread_bps', True), ('slippage_bps', -1), ('carry_bps_per_day', float('nan')),
    ('spread_bps', float('inf')), ('slippage_bps', None), ('carry_bps_per_day', 10000),
])
def test_invalid_assumptions_are_rejected(field, value):
    costs = config()
    costs[field] = value
    with pytest.raises(ValueError):
        BacktestAccount(2000, ['ltc_jpy'], costs)


def test_missing_symbol_fee_and_unknown_config_keys_are_not_silently_zeroed():
    with pytest.raises(ValueError):
        BacktestAccount(2000, ['unknown'], config())
    costs = config()
    costs['slipage_bps'] = 10
    with pytest.raises(ValueError):
        BacktestAccount(2000, ['ltc_jpy'], costs)


def test_naive_and_reversed_time_are_rejected():
    ledger = BacktestAccount(2000, ['ltc_jpy'])
    with pytest.raises(ValueError):
        ledger.advance('2026-11-14')
    ledger.advance(START)
    with pytest.raises(ValueError):
        ledger.advance(START - pd.Timedelta(seconds=1))


def test_reference_profile_is_an_explicit_scenario_and_is_not_mutated():
    profile = json.loads(Path('configs/backtest_costs_reference.json').read_text())
    original = deepcopy(profile)
    ledger = BacktestAccount(2000, list(profile['fee_bps']), profile)
    ledger.open('sol_jpy', 'long', 100, 10, START)
    ledger.close('sol_jpy', 105, START + pd.Timedelta(hours=1))
    assert profile == original
    assert ledger.summary()['cost_assumptions'] == original
    assert ledger.summary()['cost_status'] == 'scenario_estimate'
