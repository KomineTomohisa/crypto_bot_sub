"""Reproduce production snapshot lag and split closes using PostgreSQL + fake I/O."""
from copy import deepcopy
from decimal import Decimal
import sys
from types import SimpleNamespace

import pytest

from strategy_decisions import next_entry_time
from strategy_recording import close_recorder, read_journal
from trading_execution import ExecutionError, ExecutionPending, ExecutionService
from test_execution import FakeBot, records, service


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    waits = []
    monkeypatch.setattr('trading_execution.time.sleep', waits.append)
    return waits


def bot_for(svc):
    bot = FakeBot(svc)
    bot.notification_settings = {'send_on_exit': False}
    bot._ceil_to_next_15min = next_entry_time
    bot.send_notification = lambda *a, **kw: None
    bot._next_fee_sync = float('inf')
    return bot


def two_positions(api):
    api.position_rows = [
        {'positionId': pid, 'symbol': 'LTC_JPY', 'side': 'BUY',
         'size': size, 'price': '90', 'timestamp': api.now}
        for pid, size in [('600', '4'), ('601', '6')]
    ]


def lagging_order_reads(api, *, permanent=False, only_closes=False):
    original = api._request
    reads = {}

    def request(method, path, params=None, data=None):
        response = original(method, path, params, data)
        if path == '/v1/orders':
            response = deepcopy(response)
            for row in response['data']['list']:
                oid = row['orderId']
                reads[oid] = reads.get(oid, 0) + 1
                if (not only_closes or row['settleType'] == 'CLOSE') and (permanent or reads[oid] == 1):
                    row['executedSize'] = '0'
        return response

    api._request = request
    return reads


def test_snapshot_lag_retries_gets_without_second_post(store, no_retry_wait):
    svc, api = service(store)
    reads = lagging_order_reads(api)
    result = svc.submit('LTC_JPY', 'BUY', 10, context={'bot_symbol': 'ltc_jpy'})
    assert result['finalized'] and result['executed_size'] == 10
    assert len(api.posts) == 1 and list(reads.values()) == [2]
    assert no_retry_wait == [0.2]
    assert len(records(store, 'bot_execution_fills')) == 1


def test_lag_exhaustion_stays_pending_and_restart_does_not_resend(store, no_retry_wait):
    svc, api = service(store)
    reads = lagging_order_reads(api, permanent=True)
    with pytest.raises(ExecutionPending):
        svc.submit('LTC_JPY', 'BUY', 10)
    assert list(reads.values()) == [3] and no_retry_wait == [0.2, 0.5]
    assert store.pending('test') and not records(store, 'bot_execution_fills')
    api._request = api.__class__._request.__get__(api)
    recovered = ExecutionService(svc.gateway, store, 'test').recover()
    assert recovered[0]['finalized'] and len(api.posts) == 1


@pytest.mark.parametrize('field,value', [('executedSize', '11'), ('executedSize', '-1'), ('side', 'SELL')])
def test_invariant_violations_fail_without_temporal_retries(store, no_retry_wait, field, value):
    svc, api = service(store)
    original = api._request

    def request(method, path, params=None, data=None):
        response = original(method, path, params, data)
        if path == '/v1/orders':
            response = deepcopy(response)
            response['data']['list'][0][field] = value
        return response

    api._request = request
    with pytest.raises(ExecutionError):
        svc.submit('LTC_JPY', 'BUY', 10)
    assert not no_retry_wait and len(api.posts) == 1
    assert not records(store, 'bot_execution_fills')


def test_slow_read_does_not_start_more_retry_attempts(store, monkeypatch, no_retry_wait):
    svc, api = service(store)
    api.hide_fills = True
    ticks = iter([0, 3])
    monkeypatch.setattr('trading_execution.time.monotonic', lambda: next(ticks))
    result = svc.submit('LTC_JPY', 'BUY', 10)
    assert not result['finalized'] and not no_retry_wait
    assert len(api.posts) == 1


def test_split_close_lag_finishes_both_in_same_call_and_persists_before_post(store, monkeypatch):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    monkeypatch.setitem(sys.modules, 'db', SimpleNamespace(engine=store.engine))
    lagging_order_reads(api)
    original = api._request
    persisted = []

    def request(method, path, params=None, data=None):
        if method == 'POST':
            intent = store.get(data['clientOrderId'])
            persisted.append(len(intent['context']['close_batch']['positions']))
        return original(method, path, params, data)

    api._request = request
    logs = []
    bot._close_live_position('ltc_jpy', '損切り', {}, logs)
    assert persisted == [2, 2]
    assert len(api.posts) == 2 and not api.position_rows
    assert not store.pending('test') and not store.close_batches('test')
    assert bot.positions['ltc_jpy'] is None and len(logs) == 2
    assert sum(r['size'] for r in records(store, 'trades')) == 10


def test_restart_continues_remaining_close_without_rechecking_price_signal(store):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    api.hide_fills = True
    bot._close_live_position('ltc_jpy', '損切り', None, None)
    assert len(api.posts) == 1 and len(api.position_rows) == 1
    assert store.pending('test') and store.close_batches('test')
    api.hide_fills = False
    api.fill_price = Decimal('120')  # The price no longer needs to satisfy the old trigger.
    restarted = bot_for(ExecutionService(svc.gateway, store, 'test'))
    restarted._recover_live_orders()
    assert len(api.posts) == 2 and not api.position_rows
    assert not store.pending('test') and not store.close_batches('test')
    assert [r['raw']['exit_reason'] for r in records(store, 'trades')] == ['損切り', '損切り']


def test_restart_after_first_finalization_still_finds_batch(store, monkeypatch):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    original = store.prepare

    def prepare(*args):
        if str(args[5]) == '601':
            raise RuntimeError('Simulated crash before second target is prepared')
        return original(*args)

    monkeypatch.setattr(store, 'prepare', prepare)
    with pytest.raises(RuntimeError):
        bot._close_live_position('ltc_jpy', '損切り', None, None)
    assert not store.pending('test') and len(api.posts) == 1
    assert store.close_batches('test')
    monkeypatch.setattr(store, 'prepare', original)
    restarted = bot_for(ExecutionService(svc.gateway, store, 'test'))
    restarted._recover_live_orders()
    assert len(api.posts) == 2 and not api.position_rows
    assert not store.close_batches('test')


def test_partial_cancellation_keeps_batch_and_blocks_new_exposure(store):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    api.fill_ratio = Decimal('.5')
    bot._close_live_position('ltc_jpy', '損切り', None, None)
    assert not store.pending('test') and store.close_batches('test')
    assert not bot._live_funds_check('ltc_jpy', 'buy', 10)
    with pytest.raises(ExecutionError, match='close batch'):
        svc.submit('LTC_JPY', 'BUY', 10)
    api.fill_ratio = Decimal(1)
    bot._recover_live_orders()
    assert not api.position_rows and not store.close_batches('test')
    assert sum(r['size'] for r in records(store, 'trades')) == 10


def test_known_zero_rejection_can_retry_as_new_close_after_terminal_confirmation(store):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    api.reject = True
    api.reject_code = 'ERR-5106'
    bot._close_live_position('ltc_jpy', '損切り', None, None)
    assert not store.pending('test') and store.close_batches('test')
    api.reject = False
    bot._recover_live_orders()
    assert not api.position_rows and not store.close_batches('test')


def test_missing_target_without_recorded_fills_does_not_discard_batch(store):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    api.hide_fills = True
    bot._close_live_position('ltc_jpy', '損切り', None, None)
    api.hide_fills = False
    api.position_rows = []  # The second target vanished without an execution we can verify.
    bot._recover_live_orders()
    assert len(api.posts) == 1 and store.close_batches('test')
    assert not bot._live_funds_check('ltc_jpy', 'buy', 10)


def test_recovery_never_closes_new_unrelated_position(store):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    api.hide_fills = True
    bot._close_live_position('ltc_jpy', '損切り', None, None)
    api.hide_fills = False
    api.position_rows.append({**deepcopy(api.position_rows[0]), 'positionId': '999'})
    bot._recover_live_orders()
    assert [r['positionId'] for r in api.position_rows] == ['999']
    assert [r['settlePosition'][0]['positionId'] for r in api.posts] == [600, 601]


def test_stale_worker_cannot_send_completed_target_again(store):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    bot._close_live_position('ltc_jpy', '損切り', None, None)
    first = records(store, 'bot_order_intents')[0]
    with pytest.raises(ExecutionError, match='changed'):
        svc.submit('LTC_JPY', 'SELL', first['size'], position_id=first['position_id'], context=first['context'])
    assert len(api.posts) == 2


def test_recovered_entry_outcome_is_recorded_once_with_original_snapshot(store, tmp_path):
    svc, api = service(store)
    bot = bot_for(svc)
    bot._record_inputs = True
    bot.log_dir = str(tmp_path)
    bot.entry_scores = {'ltc_jpy': {}}
    bot._feature_event_refs = {'ltc_jpy': 'original-snapshot'}
    lagging_order_reads(api, permanent=True)
    try:
        with pytest.raises(ExecutionPending):
            bot._execute_live_entry('ltc_jpy', 'buy', 10,
                signal_raw={'decision_bar': '2026-11-20T07:30:00+09:00', 'decision_price': '100'})
        bot._feature_event_refs['ltc_jpy'] = 'later-unrelated-snapshot'
        api._request = api.__class__._request.__get__(api)
        bot._recover_live_orders()
        bot._recover_live_orders()
        intent = records(store, 'bot_order_intents')[0]
        bot._record_live_order_result(store.result(intent['client_order_id']))
    finally:
        close_recorder(bot)
    outcomes = [r['payload'] for r in read_journal(bot._input_recorder.path) if r['kind'] == 'order_result']
    assert len(outcomes) == 1 and outcomes[0]['result']['finalized']
    assert outcomes[0]['purpose'] == 'entry' and outcomes[0]['feature_snapshot'] == 'original-snapshot'
    assert outcomes[0]['event_id'] and len(api.posts) == 1


def test_recovered_close_event_contains_actual_trades_once(store, tmp_path):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._record_inputs = True
    bot.log_dir = str(tmp_path)
    bot._sync_live_positions()
    api.hide_fills = True
    try:
        bot._close_live_position('ltc_jpy', '損切り', None, None)
        api.hide_fills = False
        bot._recover_live_orders()
        bot._recover_live_orders()
    finally:
        close_recorder(bot)
    outcomes = [r['payload'] for r in read_journal(bot._input_recorder.path)
                if r['kind'] == 'order_result' and r['payload']['result']['finalized']]
    assert len(outcomes) == 2 and len({r['event_id'] for r in outcomes}) == 2
    assert all(r['reason'] == '損切り' and len(r['trades']) == 1 for r in outcomes)
    assert sum(Decimal(r['trades'][0]['size']) for r in outcomes) == 10


def test_executed_status_with_stale_zero_totals_cannot_finalize(store):
    svc, api = service(store)
    original = api._request

    def request(method, path, params=None, data=None):
        response = original(method, path, params, data)
        if path == '/v1/orders':
            response = deepcopy(response)
            response['data']['list'][0]['executedSize'] = '0'
        if path == '/v1/executions':
            response = {'status': 0, 'data': {'list': []}}
        return response

    api._request = request
    with pytest.raises(ExecutionPending):
        svc.submit('LTC_JPY', 'BUY', 10)
    assert store.pending('test') and not records(store, 'bot_execution_fills')
    assert len(api.posts) == 1


def test_restart_after_final_commit_backfills_missing_event_with_stable_identity(store, tmp_path):
    svc, api = service(store)
    result = svc.submit('LTC_JPY', 'BUY', 10, context={'bot_symbol': 'ltc_jpy'})
    ids = []
    for name in ('first-start', 'second-start'):
        bot = bot_for(ExecutionService(svc.gateway, store, 'test'))
        bot._record_inputs = True
        bot.log_dir = str(tmp_path / name)
        try:
            bot._recover_live_orders()
            bot._recover_live_orders()
        finally:
            close_recorder(bot)
        outcomes = [r['payload'] for r in read_journal(bot._input_recorder.path) if r['kind'] == 'order_result']
        assert len(outcomes) == 1
        assert outcomes[0]['result']['client_order_id'] == result['client_order_id']
        ids.append(outcomes[0]['event_id'])
    assert ids[0] == ids[1] and len(api.posts) == 1


def test_event_backfill_is_bounded_and_progresses_past_first_page(store, tmp_path):
    svc, api = service(store)
    # Terminal, zero-fill rejections are cheap and have the same delivery semantics.
    api.reject = True
    api.reject_code = 'ERR-5106'
    for i in range(18):
        svc.submit('LTC_JPY', 'BUY', 10, decision_key=f'rejected-{i}')
    bot = bot_for(svc)
    bot._record_inputs = True
    bot.log_dir = str(tmp_path)
    try:
        bot._recover_live_orders()
        assert len(bot._backfilled_result_ids) == 16
        assert not getattr(bot, '_execution_events_backfilled', False)
        bot._recover_live_orders()
        assert len(bot._backfilled_result_ids) == 18 and bot._execution_events_backfilled
    finally:
        close_recorder(bot)
    outcomes = [r for r in read_journal(bot._input_recorder.path) if r['kind'] == 'order_result']
    assert len(outcomes) == 18


def test_crash_before_first_post_keeps_all_targets_in_prepared_intent(store, monkeypatch):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    original = store.claim
    monkeypatch.setattr(store, 'claim', lambda *a: (_ for _ in ()).throw(RuntimeError('crash before claim')))
    with pytest.raises(RuntimeError):
        bot._close_live_position('ltc_jpy', '損切り', None, None)
    assert not api.posts and store.pending('test') and store.close_batches('test')
    monkeypatch.setattr(store, 'claim', original)
    restarted = bot_for(ExecutionService(svc.gateway, store, 'test'))
    restarted._recover_live_orders()
    assert len(api.posts) == 2 and not api.position_rows
    assert not store.pending('test') and not store.close_batches('test')


def test_competing_batch_cannot_reuse_pending_target_then_close_other_targets(store):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    api.hide_fills = True
    bot._close_live_position('ltc_jpy', '損切り', None, None)
    first = store.pending('test')[0]
    context = deepcopy(first['context'])
    context['close_batch']['id'] = 'competing-batch'
    with pytest.raises(ExecutionError, match='Another close batch'):
        svc.submit('LTC_JPY', 'SELL', first['size'], position_id=first['position_id'], context=context)
    assert len(api.posts) == 1
    assert store.close_batches('test')[0]['id'] == first['context']['close_batch']['id']


def test_batch_lookup_failure_does_not_abort_existing_position_monitoring(store, monkeypatch):
    svc, api = service(store)
    two_positions(api)
    bot = bot_for(svc)
    bot._sync_live_positions()
    monkeypatch.setattr(store, 'close_batches', lambda *a: (_ for _ in ()).throw(RuntimeError('DB read unavailable')))
    bot._recover_live_orders()
    bot._sync_live_positions()
    assert bot.positions['ltc_jpy'] == 'long' and bot.entry_sizes['ltc_jpy'] == 10
    assert not api.posts
