"""A close decision must outlive a busy account slot before the first POST."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import re
import uuid

import pytest
from sqlalchemy import text

from execution_store import json_value
from trading_execution import ExecutionError, ExecutionService
from test_execution import records, service
from test_execution_recovery import bot_for, two_positions


MIGRATION = Path('migrations/002_durable_close_batches.sql')


def plan(api, reason='stop-triggered'):
    return {'id': uuid.uuid4().hex, 'symbol': 'ltc_jpy', 'side': 'long',
            'reason': reason, 'positions': deepcopy([p for p in api.position_rows if p['symbol'] == 'LTC_JPY']),
            'scores': {}, 'user_id': None, 'feature_snapshot': None}


def waiting_other_order(svc, api, symbol='OTHER_JPY'):
    """The exchange accepted a different order but its response and reads are lost."""
    original = api._request
    blocked = {'value': True}
    api.lose_response = True

    def request(method, path, params=None, data=None):
        if method == 'GET' and path == '/v1/orders' and blocked['value']:
            return {'status': -1}
        return original(method, path, params, data)

    api._request = request
    with pytest.raises(ExecutionError):
        svc.submit(symbol, 'BUY', 1, decision_key='other-order', context={'bot_symbol': symbol.lower()})
    assert len(api.posts) == 1 and svc.store.pending(svc.account)[0]['status'] == 'UNKNOWN'
    return blocked


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    monkeypatch.setattr('trading_execution.time.sleep', lambda *a: None)


def test_first_close_is_saved_while_another_order_is_unknown(store):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    waiting_other_order(svc, api)
    bot._close_live_position('ltc_jpy', 'stop-triggered', None, None)
    assert len(api.posts) == 1  # Only the earlier, unrelated order was sent.
    assert len(records(store, 'bot_order_intents')) == 1
    batches = records(store, 'bot_close_batches')
    assert len(batches) == 1 and not batches[0]['completed']
    assert batches[0]['payload']['reason'] == 'stop-triggered'
    assert batches[0]['payload']['positions'] is None
    assert set(batches[0]['payload']['target_ids']) == {'600', '601'}


def test_restart_and_cleared_pending_resumes_original_close_without_new_signal(store):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    blocked = waiting_other_order(svc, api)
    bot._close_live_position('ltc_jpy', 'stop-triggered', None, None)
    restarted = bot_for(ExecutionService(svc.gateway, store, 'test'))
    restarted._recover_live_orders()
    assert len(api.posts) == 1 and store.close_batches('test')
    # The original stop condition is no longer true; recovery must use the saved decision.
    api.fill_price = Decimal('120')
    blocked['value'] = False
    restarted._recover_live_orders()
    assert [p['settlePosition'][0]['positionId'] for p in api.posts if 'settlePosition' in p] == [600, 601]
    assert not store.pending('test') and not store.close_batches('test')
    assert [p['symbol'] for p in api.position_rows] == ['OTHER_JPY']
    assert all(t['raw']['exit_reason'] == 'stop-triggered' for t in records(store, 'trades'))
    completed = records(store, 'bot_close_batches')[0]
    assert completed['completed'] and completed['completed_at'] is not None


def test_crash_after_queue_commit_before_order_preparation_does_not_lose_close(store, monkeypatch):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    original = store.prepare
    monkeypatch.setattr(store, 'prepare', lambda *a: (_ for _ in ()).throw(RuntimeError('process stopped before prepare')))
    with pytest.raises(RuntimeError):
        bot._close_live_position('ltc_jpy', 'stop-triggered', None, None)
    assert not api.posts and not records(store, 'bot_order_intents')
    assert len(store.close_batches('test')) == 1
    monkeypatch.setattr(store, 'prepare', original)
    bot_for(ExecutionService(svc.gateway, store, 'test'))._recover_live_orders()
    assert len(api.posts) == 2 and not api.position_rows and not store.close_batches('test')


def test_repeated_trigger_keeps_first_decision_and_target_set(store):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    waiting_other_order(svc, api)
    bot._close_live_position('ltc_jpy', 'first-stop', None, None)
    before = deepcopy(store.close_batches('test')[0])
    api.position_rows.append({**deepcopy(api.position_rows[0]), 'positionId': '999'})
    bot._close_live_position('ltc_jpy', 'later-take-profit', None, None)
    assert store.close_batches('test') == [before]
    assert len(records(store, 'bot_close_batches')) == 1 and len(api.posts) == 1


def test_new_position_while_waiting_is_not_added_to_queued_targets(store):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    blocked = waiting_other_order(svc, api)
    bot._close_live_position('ltc_jpy', 'stop-triggered', None, None)
    api.position_rows.append({**deepcopy(api.position_rows[0]), 'positionId': '999'})
    blocked['value'] = False
    bot_for(ExecutionService(svc.gateway, store, 'test'))._recover_live_orders()
    assert {p['positionId'] for p in api.position_rows if p['symbol'] == 'LTC_JPY'} == {'999'}
    assert [p['settlePosition'][0]['positionId'] for p in api.posts if 'settlePosition' in p] == [600, 601]


def test_queue_blocks_new_exposure_even_without_an_order_intent(store):
    svc, api = service(store)
    two_positions(api)
    batch = store.save_close_batch('test', 'LTC_JPY', plan(api))
    assert not store.pending('test')
    with pytest.raises(ExecutionError, match='close batch'):
        svc.submit('OTHER_JPY', 'BUY', 1)
    assert not api.posts
    assert store.close_batches('test') == [batch]


def test_concurrent_close_decisions_share_one_durable_reservation(store):
    svc, api = service(store)
    two_positions(api)
    candidates = [plan(api, reason=f'stop-{i}') for i in range(3)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        saved = list(pool.map(lambda batch: store.save_close_batch('test', 'LTC_JPY', batch), candidates))
    assert len({batch['id'] for batch in saved}) == 1
    assert len(records(store, 'bot_close_batches')) == 1
    assert not api.posts


def test_failed_decision_commit_means_no_post(store, monkeypatch):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    monkeypatch.setattr(store, 'save_close_batch', lambda *a: (_ for _ in ()).throw(RuntimeError('DB unavailable')))
    with pytest.raises(RuntimeError):
        bot._close_live_position('ltc_jpy', 'stop-triggered', None, None)
    assert not api.posts and not records(store, 'bot_order_intents')


def run_migration(store):
    with store.engine.connect().execution_options(isolation_level='AUTOCOMMIT') as conn:
        conn.exec_driver_sql(MIGRATION.read_text())


@pytest.mark.parametrize('first_order_prepared', [False, True])
@pytest.mark.parametrize('migrate_after_old_write', [False, True])
def test_old_format_decision_is_preserved_during_migration_and_handover(store, first_order_prepared, migrate_after_old_write):
    svc, api = service(store)
    two_positions(api)
    batch = plan(api)
    pos = deepcopy(batch['positions'][0])
    context = {'bot_symbol': 'ltc_jpy', 'baseline': pos, 'exit_reason': batch['reason']}
    if first_order_prepared:
        body = {'symbol': 'LTC_JPY', 'side': 'SELL', 'executionType': 'MARKET',
                'settlePosition': [{'positionId': 600, 'size': '4'}]}
        intent = store.prepare('test', 'LTC_JPY', 'SELL', 'CLOSE', Decimal(4), '600', None, body, context)
        cid = intent['client_order_id']
    else:
        cid = svc.submit('LTC_JPY', 'SELL', 4, position_id='600', context=context)['client_order_id']
    # Simulate the previous process, which only knew context.close_batch.
    with store.engine.begin() as conn:
        conn.execute(text('UPDATE bot_order_intents SET context=CAST(:ctx AS jsonb) WHERE client_order_id=:id'),
                     {'ctx': json_value({**context, 'close_batch': batch}), 'id': cid})
    if migrate_after_old_write:
        old_intents = records(store, 'bot_order_intents')
        old_fills = records(store, 'bot_execution_fills')
        old_trades = records(store, 'trades')
        run_migration(store)
        before = records(store, 'bot_close_batches')
        run_migration(store)
        assert records(store, 'bot_close_batches') == before
        assert records(store, 'bot_order_intents') == old_intents
        assert records(store, 'bot_execution_fills') == old_fills
        assert records(store, 'trades') == old_trades
    else:
        assert not records(store, 'bot_close_batches')
    assert store.close_batches('test') == [batch]
    bot_for(svc)._recover_live_orders()
    assert len(api.posts) == 2 and not api.position_rows
    assert not store.close_batches('test') and records(store, 'bot_close_batches')[0]['completed']
    assert all(r['context']['close_batch_done'] for r in records(store, 'bot_order_intents'))
    run_migration(store)
    assert not store.close_batches('test')  # Re-running must never resurrect a completed close.


def test_changed_payload_cannot_redirect_registered_close(store):
    svc, api = service(store)
    two_positions(api)
    batch = store.save_close_batch('test', 'LTC_JPY', plan(api))
    changed = deepcopy(batch)
    changed['reason'] = 'tampered'
    with pytest.raises(ExecutionError, match='changed after registration'):
        svc.submit('LTC_JPY', 'SELL', 4, position_id='600',
                   context={'close_batch': changed, 'bot_symbol': 'ltc_jpy', 'baseline': api.position_rows[0]})
    assert not api.posts and store.close_batches('test') == [batch]


def test_close_reservation_is_scoped_to_its_account(store):
    svc, api = service(store)
    two_positions(api)
    batch = store.save_close_batch('test', 'LTC_JPY', plan(api))
    assert not store.close_batches('other-account')
    other = ExecutionService(svc.gateway, store, 'other-account')
    assert other.submit('OTHER_JPY', 'BUY', 1)['finalized']
    assert store.close_batches('test') == [batch]


def test_missing_target_before_first_post_stays_unresolved(store):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    blocked = waiting_other_order(svc, api)
    bot._close_live_position('ltc_jpy', 'stop-triggered', None, None)
    api.position_rows = [p for p in api.position_rows if p['positionId'] != '600']
    blocked['value'] = False
    bot_for(ExecutionService(svc.gateway, store, 'test'))._recover_live_orders()
    assert len(api.posts) == 1 and store.close_batches('test')
    assert not records(store, 'trades')


def test_migration_can_use_app_role_without_altering_old_table_ownership(store):
    role = 'close_app_' + uuid.uuid4().hex
    with store.engine.connect() as conn:
        schema = conn.execute(text('SELECT current_schema()')).scalar_one()
    assert re.fullmatch(r't_[a-f0-9]{32}', schema)
    assert re.fullmatch(r'close_app_[a-f0-9]{32}', role)
    with store.engine.begin() as conn:
        conn.exec_driver_sql('DROP TABLE bot_close_batches')
        conn.exec_driver_sql('CREATE ROLE ' + role)
        conn.exec_driver_sql(f'GRANT USAGE,CREATE ON SCHEMA {schema} TO {role}')
        conn.exec_driver_sql('GRANT SELECT ON bot_order_intents TO ' + role)
    try:
        script = '\n'.join(path.read_text().removeprefix('BEGIN;').removesuffix('COMMIT;\n')
                           for path in (MIGRATION, Path('migrations/003_close_stall_notifications.sql')))
        with store.engine.connect().execution_options(isolation_level='AUTOCOMMIT') as conn:
            conn.exec_driver_sql(f'BEGIN; SET LOCAL ROLE {role};\n' + script + '\nCOMMIT;')
        with store.engine.connect() as conn:
            rows = dict(conn.execute(text("""SELECT relname,pg_get_userbyid(relowner)
                FROM pg_class WHERE relnamespace=CAST(:schema AS regnamespace)
                  AND relname IN ('bot_order_intents','bot_close_batches')"""),
                {'schema': schema}).all())
        assert rows == {'bot_order_intents': 'postgres', 'bot_close_batches': role}
    finally:
        with store.engine.begin() as conn:
            conn.exec_driver_sql('DROP TABLE IF EXISTS bot_close_batches')
            conn.exec_driver_sql('REVOKE SELECT ON bot_order_intents FROM ' + role)
            conn.exec_driver_sql(f'REVOKE USAGE,CREATE ON SCHEMA {schema} FROM {role}')
            conn.exec_driver_sql('DROP ROLE ' + role)


def test_two_symbols_can_queue_without_relaxing_single_pending_order(store):
    svc, api = service(store)
    two_positions(api)
    api.position_rows.append({'positionId': '700', 'symbol': 'OTHER_JPY', 'side': 'BUY',
                              'size': '3', 'price': '90', 'timestamp': api.now})
    bot = bot_for(svc)
    bot.symbols.append('other_jpy')
    bot.symbol_mapping['other_jpy'] = 'OTHER_JPY'
    bot._sync_live_positions()
    blocked = waiting_other_order(svc, api, symbol='WAIT_JPY')
    bot._close_live_position('ltc_jpy', 'ltc-stop', None, None)
    bot._close_live_position('other_jpy', 'other-stop', None, None)
    assert len(store.pending('test')) == 1 and len(api.posts) == 1
    assert [b['symbol'] for b in store.close_batches('test')] == ['ltc_jpy', 'other_jpy']
    blocked['value'] = False
    bot._recover_live_orders()
    assert [p['settlePosition'][0]['positionId'] for p in api.posts if 'settlePosition' in p] == [600, 601, 700]
    assert not store.pending('test') and not store.close_batches('test')
    assert [p['symbol'] for p in api.position_rows] == ['WAIT_JPY']
