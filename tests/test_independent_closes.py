"""Uncertain exposure never prevents verified closes on another symbol."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest
from sqlalchemy import text

from trading_execution import ExecutionError, ExecutionService
from test_execution import service, records
from test_execution_recovery import bot_for, two_positions
from test_close_decision_queue import waiting_other_order, plan
from types import MethodType, SimpleNamespace
import sys
from datetime import timedelta
from test_bot_execution_integration import BOT, method
from test_display_monitoring_isolation import AT, wire_live_loop


def test_unknown_entry_still_blocks_all_new_exposure_but_not_verified_close(store):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    waiting_other_order(svc, api, symbol='OTHER_JPY')
    for symbol in ('OTHER_JPY', 'THIRD_JPY', 'LTC_JPY'):
        with pytest.raises(ExecutionError):
            svc.submit(symbol, 'BUY', 1, decision_key='new-' + symbol)
    bot._close_live_position('ltc_jpy', 'stop-triggered', None, None)
    assert len(api.posts) == 3 and len(records(store, 'trades')) == 2
    assert len(store.pending('test')) == 1 and not store.close_batches('test')
    restarted = bot_for(ExecutionService(svc.gateway, store, 'test'))
    restarted._recover_live_orders()
    assert len(api.posts) == 3  # Completed targets must never be sent again.


def test_same_symbol_unknown_close_blocks_duplicate_but_other_symbol_can_close(store):
    svc, api = service(store)
    two_positions(api)
    api.position_rows.append({'positionId': '700', 'symbol': 'OTHER_JPY', 'side': 'BUY',
                              'size': '3', 'price': '90', 'timestamp': api.now})
    bot = bot_for(svc)
    bot.symbol_mapping['other_jpy'] = 'OTHER_JPY'
    bot.symbols.append('other_jpy')
    bot._sync_live_positions()
    original = api._request
    hidden = []
    def request(method, path, params=None, data=None):
        if method == 'POST' and data['symbol'] == 'LTC_JPY' and not hidden:
            hidden.append(data['clientOrderId'])
            api.lose_response = True
        if method == 'GET' and path == '/v1/orders' and params.get('clientOrderId') in hidden:
            return {'status': -1}
        return original(method, path, params, data)
    api._request = request
    with pytest.raises(ExecutionError):
        bot._close_live_position('ltc_jpy', 'stop-triggered', None, None)
    for _ in range(3):
        bot._close_live_position('ltc_jpy', 'stop-triggered', None, None)
    bot._close_live_position('other_jpy', 'stop-triggered', None, None)
    assert [p['settlePosition'][0]['positionId'] for p in api.posts] == [600, 700]
    assert len(store.pending('test')) == 1
    api._request = original
    bot._recover_live_orders()
    assert [p['settlePosition'][0]['positionId'] for p in api.posts] == [600, 700, 601]
    assert not store.pending('test') and not store.close_batches('test')


def test_unregistered_close_cannot_bypass_account_pending(store):
    svc, api = service(store)
    two_positions(api)
    waiting_other_order(svc, api, symbol='OTHER_JPY')
    with pytest.raises(ExecutionError):
        svc.submit('LTC_JPY', 'SELL', 4, position_id='600', context={'baseline': api.position_rows[0]})
    assert len(api.posts) == 1


@pytest.mark.parametrize('side', ['BUY', 'SELL'])
def test_actual_monitoring_loop_stops_out_while_other_symbol_order_is_unknown(store, monkeypatch, side):
    monkeypatch.setitem(sys.modules, 'db', SimpleNamespace(engine=store.engine))
    svc, api = service(store)
    api.now = (AT - timedelta(hours=1)).isoformat()
    svc.submit('LTC_JPY', side, 5, context={'bot_symbol': 'ltc_jpy'})
    api.now = AT.isoformat()
    waiting_other_order(svc, api, symbol='OTHER_JPY')
    bot = bot_for(svc)
    run, actions, waits = wire_live_loop(bot)
    verify, _ = method(BOT, 'verify_positions')
    bot.verify_positions = MethodType(verify, bot)
    bot._legacy_short_lock_loaded = True
    bot.get_current_price = lambda _: 85. if side == 'BUY' else 115.
    run()
    assert len(api.posts) == 3 and not api.position_rows
    assert len(store.pending('test')) == 1 and not store.close_batches('test')
    assert len(records(store, 'trades')) == 1
    assert records(store, 'trades')[0]['raw']['exit_reason'] == '損切り'
    assert len(waits) == 2 and all(value <= 28 for value in waits)


def test_two_process_equivalent_preparations_share_same_close_intent(store):
    svc, api = service(store)
    two_positions(api)
    waiting_other_order(svc, api, symbol='OTHER_JPY')
    batch = store.save_close_batch('test', 'LTC_JPY', plan(api))
    pos = deepcopy(api.position_rows[0])
    body = {'symbol': 'LTC_JPY', 'side': 'SELL', 'executionType': 'MARKET',
            'settlePosition': [{'positionId': 600, 'size': '4'}]}
    context = {'close_batch': batch, 'baseline': pos}
    def prepare(_):
        return store.prepare('test', 'LTC_JPY', 'SELL', 'CLOSE', 4, '600', None, body, context)
    with ThreadPoolExecutor(max_workers=3) as pool:
        intents = list(pool.map(prepare, range(3)))
    assert len({item['client_order_id'] for item in intents}) == 1
    assert len(store.pending('test')) == 2
    assert sum(store.claim(item['client_order_id']) for item in intents) == 1
    assert len(api.posts) == 1  # Preparing or claiming never POSTs on its own.


def test_migration_is_repeatable_and_preserves_pending_orders(store):
    svc, api = service(store)
    waiting_other_order(svc, api)
    before = records(store, 'bot_order_intents')
    sql = Path('migrations/005_independent_protective_closes.sql').read_text()
    for _ in range(2):
        with store.engine.connect().execution_options(isolation_level='AUTOCOMMIT') as conn:
            conn.exec_driver_sql(sql)
    assert records(store, 'bot_order_intents') == before
    store.check_schema()
    with store.engine.begin() as conn:
        conn.execute(text('DROP INDEX bot_one_pending_open_account'))
    with pytest.raises(ExecutionError, match='005'):
        store.check_schema()
