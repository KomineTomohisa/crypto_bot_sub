from copy import deepcopy
from types import MethodType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from strategy_decisions import entry_side, short_profit_stop
from strategy_policy import BASELINE, CANDIDATE, policy, position_policy
from test_bot_execution_integration import BOT, method
from test_execution import FakeBot, service, records
from test_exit_protection_persistence import file_bot, restart
from test_exit_signals import position
from test_backtest_continuity import replay, candles


AT = pd.Timestamp('2026-11-22T12:00:00+09:00')
SYMBOL = 'ltc_jpy'


def rows(adx=30, side='short'):
    flag = 'buy_signal' if side == 'short' else 'sell_signal'
    return (pd.Series(dict(timestamp=AT - pd.Timedelta(minutes=15), ADX=adx, **{flag: True})),
            pd.Series(dict(timestamp=AT - pd.Timedelta(minutes=30), ADX=adx, **{flag: True})))


@pytest.mark.parametrize('side', ['long', 'short'])
@pytest.mark.parametrize('adx', [30, 30.001, 100])
def test_both_closed_bars_accept_threshold(side, adx):
    latest, previous = rows(adx, side)
    assert entry_side(latest, previous, AT, policy=CANDIDATE) == side


@pytest.mark.parametrize('bad', [None, np.nan, np.inf, -np.inf, True, np.bool_(True), -1, 29.999, 101, 'bad'])
@pytest.mark.parametrize('index', [0, 1])
def test_each_closed_bar_must_have_valid_raw_adx(bad, index):
    pair = rows()
    pair[index]['ADX'] = bad
    assert entry_side(*pair, AT, policy=CANDIDATE) is None
    assert entry_side(*pair, AT, policy=BASELINE) == 'short'


def test_live_entry_and_shared_gate_match_and_keep_confirmation():
    fn, _ = method(BOT, '_entry_direction')
    bot = SimpleNamespace(strategy_policy=CANDIDATE, reentry_block_until={}, _record_inputs=False)
    assert fn(bot, SYMBOL, *rows(), AT) == 'short'
    assert bot._entry_notification_context[SYMBOL]['strategy_policy'] == CANDIDATE.snapshot()
    assert fn(bot, SYMBOL, *rows(29), AT) is None
    assert SYMBOL not in bot._entry_notification_context
    bot.reentry_block_until[SYMBOL] = AT + pd.Timedelta(minutes=15)
    assert fn(bot, SYMBOL, *rows(), AT) is None


def test_version_and_snapshot_are_strict():
    assert policy(CANDIDATE.snapshot()) == CANDIDATE
    assert policy() == BASELINE
    for invalid in ({'version': BASELINE.version}, {**CANDIDATE.snapshot(), 'short_profit_lock': True},
                    {**BASELINE.snapshot(), 'lock_trigger': float('nan')}, 'unknown'):
        with pytest.raises(ValueError):
            policy(invalid)


def test_disabled_lock_does_not_disarm_an_already_armed_position():
    assert short_profit_stop('short', 100, 99, 110, False, enabled=False) == (110, False)
    assert short_profit_stop('short', 100, 100, 110, True, enabled=False) == (99.75, True)


@pytest.mark.parametrize('selected', [BASELINE, CANDIDATE])
def test_real_ledger_pins_policy_across_restart_and_partial_close(store, tmp_path, selected):
    svc, api = service(store)
    svc.submit('LTC_JPY', 'SELL', '10', decision_key='versioned-entry',
               context={'bot_symbol': SYMBOL, 'strategy_policy': selected.snapshot()})
    bot, _ = file_bot(tmp_path, svc)
    bot.strategy_policy = CANDIDATE if selected == BASELINE else BASELINE
    bot._sync_live_positions()
    assert position_policy(bot, SYMBOL) == selected
    bot._apply_short_profit_lock(SYMBOL, 'short', 100, 99, 110)
    assert bot.short_profit_lock_armed[SYMBOL] == selected.short_profit_lock
    api.position_rows[0]['size'] = '4'
    other = restart(bot, tmp_path)
    assert position_policy(other, SYMBOL) == selected
    assert other.short_profit_lock_armed[SYMBOL] == selected.short_profit_lock


@pytest.mark.parametrize('saved', [None, {'version': 'future'}])
def test_legacy_or_invalid_policy_preserves_legacy_protection(store, tmp_path, saved):
    svc, api = service(store)
    svc.submit('LTC_JPY', 'SELL', '10', decision_key='old-entry',
               context={'bot_symbol': SYMBOL, 'strategy_policy': saved})
    bot, _ = file_bot(tmp_path, svc)
    bot.strategy_policy = CANDIDATE
    bot._sync_live_positions()
    assert position_policy(bot, SYMBOL) == BASELINE
    assert bot._apply_short_profit_lock(SYMBOL, 'short', 100, 99, 110) == 99.75


def test_policy_is_saved_before_post_and_recovered_from_lost_response(store):
    svc, api = service(store)
    bot = FakeBot(svc)
    bot.strategy_policy = CANDIDATE
    latest, previous = rows()
    entry, _ = method(BOT, '_entry_direction')
    entry(bot, SYMBOL, latest, previous, AT)
    api.lose_response = True
    bot._execute_live_entry(SYMBOL, 'sell', 10, signal_raw={'decision_bar': str(latest.timestamp), 'decision_price': '100'})
    assert len(api.posts) == 1
    assert records(store, 'bot_order_intents')[0]['context']['strategy_policy'] == CANDIDATE.snapshot()
    assert position_policy(bot, SYMBOL) == CANDIDATE


def test_candidate_direct_entry_without_closed_bar_evidence_never_posts(store):
    svc, api = service(store)
    bot = FakeBot(svc)
    bot.strategy_policy = CANDIDATE
    result = bot._execute_live_entry(SYMBOL, 'sell', 10, signal_raw={'decision_bar': str(AT), 'decision_price': '100'})
    assert not result['success'] and not api.posts


@pytest.mark.parametrize('side', ['long', 'short'])
def test_backtest_candidate_uses_gate_and_keeps_runtime_policy(side):
    frame = candles(side)
    frame['ADX'] = 29
    bot, _ = replay(frame)
    bot.strategy_policy = BASELINE
    bot.position_policies = {SYMBOL: BASELINE.snapshot()}
    original = deepcopy(bot.position_policies)
    fn, _ = method(BOT, 'backtest')
    fn(bot, 2, policy=CANDIDATE)
    assert not bot.trades
    assert bot.valuation['strategy_policy'] == CANDIDATE.snapshot()
    assert bot.strategy_policy == BASELINE and bot.position_policies == original


def test_candidate_still_takes_profit_and_records_entry_policy():
    bot, run = replay(candles('short'))
    bot.strategy_policy = CANDIDATE
    run()
    assert len(bot.trades) == 1
    assert bot.trades[0]['strategy_policy'] == CANDIDATE.snapshot()
    assert bot.trades[0]['exit_reason'] != '\u5229\u76ca\u4fdd\u8b77'


@pytest.mark.parametrize('first', [BASELINE, CANDIDATE])
def test_remaining_id_restores_its_own_policy_when_representative_disappears(store, tmp_path, first):
    svc, api = service(store)
    for key, selected in [('first', first), ('second', CANDIDATE)]:
        svc.submit('LTC_JPY', 'SELL', '10', decision_key=key,
                   context={'bot_symbol': SYMBOL, 'strategy_policy': selected.snapshot()})
    bot, _ = file_bot(tmp_path, svc)
    bot.strategy_policy = CANDIDATE
    bot._sync_live_positions()
    assert position_policy(bot, SYMBOL) == first
    remaining = deepcopy(api.position_rows[1])
    remaining['size'] = '4'
    api.position_rows = [remaining]
    other = restart(bot, tmp_path)
    assert other.position_ids[SYMBOL] == remaining['positionId']
    assert position_policy(other, SYMBOL) == CANDIDATE


def test_old_armed_position_keeps_its_protection_after_candidate_becomes_active(tmp_path):
    bot, api = file_bot(tmp_path)
    api.position_rows = [position(api, 'legacy')]
    bot._sync_live_positions()
    bot._apply_short_profit_lock(SYMBOL, 'short', 100, 99, 110)
    other = restart(bot, tmp_path)
    other.strategy_policy = CANDIDATE
    assert position_policy(other, SYMBOL) == BASELINE
    assert other._apply_short_profit_lock(SYMBOL, 'short', 100, 100, 110) == 99.75
