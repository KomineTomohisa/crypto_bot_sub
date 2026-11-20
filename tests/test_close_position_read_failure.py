"""Persist a close before fresh position I/O, without binding later exposure."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal

import pytest

from trading_execution import ExecutionError, ExecutionService
from test_execution import records, service
from test_execution_recovery import bot_for, two_positions
from test_close_decision_queue import run_migration


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    monkeypatch.setattr('trading_execution.time.sleep', lambda *a: None)


def failing_position_reads(api):
    original = api._request
    state = {'blocked': True, 'reads': 0}

    def request(method, path, params=None, data=None):
        if path == '/v1/openPositions':
            state['reads'] += 1
            if state['blocked']:
                return {'status': -1}
        return original(method, path, params, data)

    api._request = request
    return state


@pytest.mark.parametrize('side,close_side', [('BUY', 'SELL'), ('SELL', 'BUY')])
def test_failed_position_lookup_keeps_request_and_restart_resumes_original_targets(store, side, close_side):
    svc, api = service(store)
    two_positions(api)
    for p in api.position_rows:p['side'] = side
    bot = bot_for(svc)
    bot._sync_live_positions()
    reads = failing_position_reads(api)
    with pytest.raises(ExecutionError):
        bot._close_live_position('ltc_jpy', 'original-stop', None, None)
    request = records(store, 'bot_close_batches')[0]
    assert request['payload']['positions'] is None and not request['completed']
    assert request['payload']['target_ids'] == ['600', '601']
    assert request['payload']['target_sizes'] == {'600': '4', '601': '6'}
    assert not api.posts and not store.pending('test')
    restarted = bot_for(ExecutionService(svc.gateway, store, 'test'))
    restarted._recover_live_orders()
    assert not api.posts and len(store.close_batches('test')) == 1
    reads['blocked'] = False
    api.fill_price = Decimal('120')  # The original trigger need not still be true.
    restarted._recover_live_orders()
    assert len(api.posts) == 2 and not api.position_rows
    assert all(p['side'] == close_side for p in api.posts)
    assert not store.close_batches('test') and not store.pending('test')
    assert all(t['raw']['exit_reason'] == 'original-stop' for t in records(store, 'trades'))


def test_request_is_committed_before_the_first_fresh_position_read(store):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    original = api._request
    observed = []

    def request(method, path, params=None, data=None):
        if path == '/v1/openPositions' and not observed:
            saved = records(store, 'bot_close_batches')
            assert len(saved) == 1 and saved[0]['payload']['positions'] is None
            observed.append(saved[0]['batch_id'])
            raise RuntimeError('process stopped during first fresh read')
        return original(method, path, params, data)

    api._request = request
    with pytest.raises(RuntimeError):
        bot._close_live_position('ltc_jpy', 'original-stop', None, None)
    assert observed and not api.posts
    bot_for(ExecutionService(svc.gateway, store, 'test'))._recover_live_orders()
    assert len(api.posts) == 2 and not store.close_batches('test')


def test_unresolved_request_never_adopts_a_new_position_after_lookup_failure(store):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    reads = failing_position_reads(api)
    with pytest.raises(ExecutionError):
        bot._close_live_position('ltc_jpy', 'original-stop', None, None)
    api.position_rows.append({**deepcopy(api.position_rows[0]), 'positionId': '999'})
    reads['blocked'] = False
    bot_for(ExecutionService(svc.gateway, store, 'test'))._recover_live_orders()
    assert [p['settlePosition'][0]['positionId'] for p in api.posts] == [600, 601]
    assert [p['positionId'] for p in api.position_rows] == ['999']
    assert not store.close_batches('test')


@pytest.mark.parametrize('change', ['missing', 'quantity', 'direction'])
def test_changed_original_targets_remain_unresolved_without_any_post(store, change):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    reads = failing_position_reads(api)
    with pytest.raises(ExecutionError):
        bot._close_live_position('ltc_jpy', 'original-stop', None, None)
    if change == 'missing':api.position_rows.pop(0)
    elif change == 'quantity':api.position_rows[0]['size'] = '5'
    else:api.position_rows[0]['side'] = 'SELL'
    reads['blocked'] = False
    bot_for(ExecutionService(svc.gateway, store, 'test'))._recover_live_orders()
    assert not api.posts and not store.pending('test')
    assert store.close_batches('test')[0]['positions'] is None


def test_unresolved_request_blocks_entries_and_cannot_be_sent_or_completed(store):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    failing_position_reads(api)
    with pytest.raises(ExecutionError):
        bot._close_live_position('ltc_jpy', 'original-stop', None, None)
    batch = store.close_batches('test')[0]
    with pytest.raises(ExecutionError, match='close batch'):
        svc.submit('OTHER_JPY', 'BUY', 1)
    with pytest.raises(ExecutionError, match='not yet confirmed'):
        svc.submit('LTC_JPY', 'SELL', 4, position_id='600', context={'close_batch': batch})
    with pytest.raises(ExecutionError, match='not yet confirmed'):
        store.finish_close_batch('test', batch)
    assert not api.posts and not store.pending('test') and store.close_batches('test')


def test_repeated_signal_does_not_replace_unresolved_request_identity_or_reason(store):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    reads = failing_position_reads(api)
    with pytest.raises(ExecutionError):
        bot._close_live_position('ltc_jpy', 'original-stop', None, None)
    before = deepcopy(store.close_batches('test')[0])
    with pytest.raises(ExecutionError):
        bot._close_live_position('ltc_jpy', 'later-trigger', None, None)
    assert store.close_batches('test') == [before] and len(records(store, 'bot_close_batches')) == 1
    reads['blocked'] = False
    bot._recover_live_orders()
    assert len(api.posts) == 2 and not store.close_batches('test')


def test_resolution_is_committed_before_post_and_first_snapshot_is_immutable(store, monkeypatch):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    original_prepare = store.prepare
    monkeypatch.setattr(store, 'prepare', lambda *a: (_ for _ in ()).throw(RuntimeError('crash after resolution')))
    with pytest.raises(RuntimeError):
        bot._close_live_position('ltc_jpy', 'original-stop', None, None)
    saved = store.close_batches('test')[0]
    assert len(saved['positions']) == 2 and not api.posts
    changed = [{**p, 'size': '100'} for p in saved['positions']]
    assert store.resolve_close_batch('test', 'LTC_JPY', saved['id'], changed) == saved
    monkeypatch.setattr(store, 'prepare', original_prepare)
    bot_for(ExecutionService(svc.gateway, store, 'test'))._recover_live_orders()
    assert len(api.posts) == 2 and not store.close_batches('test')


@pytest.mark.parametrize('changed_quantity', [False, True])
def test_fallback_known_identity_and_quantity_can_resolve_without_runtime_cache(store, changed_quantity):
    svc, api = service(store)
    two_positions(api)
    api.position_rows = api.position_rows[:1]
    bot = bot_for(svc)
    bot.positions['ltc_jpy'] = 'long'
    bot.position_ids['ltc_jpy'] = '600'
    bot.entry_sizes['ltc_jpy'] = 4
    reads = failing_position_reads(api)
    with pytest.raises(ExecutionError):
        bot._close_live_position('ltc_jpy', 'original-stop', None, None)
    batch = store.close_batches('test')[0]
    assert batch['target_ids'] == ['600'] and batch['target_sizes'] is None
    if changed_quantity:api.position_rows[0]['size'] = '5'
    reads['blocked'] = False
    bot._recover_live_orders()
    if changed_quantity:
        assert not api.posts and store.close_batches('test')
    else:
        assert len(api.posts) == 1 and not store.close_batches('test')


def test_unknown_identity_is_saved_but_cannot_be_guessed_from_later_positions(store):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot.positions['ltc_jpy'] = 'long'
    bot.entry_sizes['ltc_jpy'] = 10
    with pytest.raises(ExecutionError):
        bot._close_live_position('ltc_jpy', 'original-stop', None, None)
    bot._recover_live_orders()
    assert not api.posts and store.close_batches('test')[0]['target_ids'] == []


def test_failed_request_commit_does_not_start_position_lookup(store, monkeypatch):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    reads = failing_position_reads(api)
    monkeypatch.setattr(store, 'save_close_batch', lambda *a: (_ for _ in ()).throw(RuntimeError('DB down')))
    with pytest.raises(RuntimeError):
        bot._close_live_position('ltc_jpy', 'original-stop', None, None)
    assert reads['reads'] == 0 and not api.posts


def test_concurrent_resolution_binds_only_one_snapshot(store):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    reads = failing_position_reads(api)
    with pytest.raises(ExecutionError):
        bot._close_live_position('ltc_jpy', 'original-stop', None, None)
    request = store.close_batches('test')[0]
    snapshot = deepcopy(api.position_rows)
    with ThreadPoolExecutor(max_workers=2) as pool:
        resolved = list(pool.map(lambda _: store.resolve_close_batch('test', 'LTC_JPY', request['id'], snapshot), range(2)))
    assert resolved[0] == resolved[1] == store.close_batches('test')[0]
    assert not api.posts and len(records(store, 'bot_close_batches')) == 1
    reads['blocked'] = False
    bot._recover_live_orders()
    assert len(api.posts) == 2 and not store.close_batches('test')


def test_unknown_quantity_does_not_use_a_later_snapshot_as_original_exposure(store):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot.positions['ltc_jpy'] = 'long'
    bot.position_ids['ltc_jpy'] = '600'
    with pytest.raises(ExecutionError, match='no verified quantity'):
        bot._close_live_position('ltc_jpy', 'original-stop', None, None)
    assert not api.posts and store.close_batches('test')[0]['positions'] is None


def test_repeat_migration_preserves_unresolved_request_without_order_intents(store):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    failing_position_reads(api)
    with pytest.raises(ExecutionError):
        bot._close_live_position('ltc_jpy', 'original-stop', None, None)
    before = records(store, 'bot_close_batches')
    assert not records(store, 'bot_order_intents')
    run_migration(store)
    run_migration(store)
    assert records(store, 'bot_close_batches') == before
    assert not api.posts and store.close_batches('test')[0]['positions'] is None
