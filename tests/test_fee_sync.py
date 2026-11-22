"""Slow fee reads never hold up monitoring; only complete history is committed."""
from datetime import date, datetime, timedelta, timezone
from threading import Event
from types import MethodType, SimpleNamespace
import logging
import sys
import time

import pytest

from exchange_availability import ExchangeAvailability
from fee_sync import FeeSyncInterrupted, FeeSyncWorker
from live_execution import LiveExecutionMixin
from test_bot_execution_integration import BOT, method
from test_display_monitoring_isolation import AT, wire_live_loop
from test_execution import records, service
from test_execution_recovery import bot_for
from trading_execution import ExchangeMaintenance


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('background operation did not finish')
        Event().wait(.005)


def worker_for(sync, **kwargs):
    svc = SimpleNamespace(gateway=SimpleNamespace(api=SimpleNamespace()), sync_fees=sync)
    return FeeSyncWorker(svc, logging.getLogger('fee-test'), **kwargs)


@pytest.mark.parametrize('side', ['BUY', 'SELL'])
def test_actual_live_loop_closes_while_fee_request_is_still_blocked(store, monkeypatch, side):
    monkeypatch.setitem(sys.modules, 'db', SimpleNamespace(engine=store.engine))
    svc, api = service(store)
    api.now = (AT - timedelta(hours=1)).isoformat()
    svc.submit('LTC_JPY', side, 5, context={'bot_symbol': 'ltc_jpy'})
    api.now = AT.isoformat()
    bot = bot_for(svc)
    run, actions, waits = wire_live_loop(bot)
    bot._queue_fee_sync = MethodType(LiveExecutionMixin._queue_fee_sync, bot)
    started, release, returned = Event(), Event(), Event()
    original_request = api._request
    calls = []

    def request(verb, path, params=None, data=None):
        if 'exchangeFee' in path:
            calls.append(path)
            started.set()
            assert release.wait(10), 'test did not release the fake fee request'
            returned.set()
        return original_request(verb, path, params, data)

    api._request = request
    original_sync = bot._sync_live_positions

    def positions(*args):
        assert started.wait(3)
        assert not returned.is_set()
        return original_sync(*args)

    bot._sync_live_positions = positions
    verify, _ = method(BOT, 'verify_positions')
    bot.verify_positions = MethodType(verify, bot)
    bot._legacy_short_lock_loaded = True
    bot.get_current_price = lambda _: 85. if side == 'BUY' else 115.
    try:
        run()
        assert started.is_set() and not returned.is_set()
        assert len(calls) == 1  # No second worker or queued duplicate on later cycles.
        assert not api.position_rows and len(api.posts) == 2
        assert len(records(store, 'trades')) == 1
        assert records(store, 'trades')[0]['raw']['exit_reason'] == '損切り'
        assert not store.pending('test') and not store.close_batches('test')
        assert not records(store, 'bot_account_fees') and store.fee_through('test') is None
        assert len(waits) == 2 and all(0 < value <= 28 for value in waits)
        assert any(item[0] == 'monitor' for item in actions)
        stopped = bot._fee_sync_worker
        bot._queue_fee_sync()
        assert bot._fee_sync_worker is stopped and stopped.thread.is_alive()
    finally:
        release.set()
        bot._stop_fee_sync()
        bot._fee_sync_worker.thread.join(3)
    assert not bot._fee_sync_worker.thread.is_alive()
    assert store.fee_through('test') is None  # Stopped reads never confirm a window.


def test_worker_uses_completion_time_for_hourly_schedule_and_single_flight():
    clock = [0.]
    started, release = Event(), Event()
    calls = []

    def sync(today, **kwargs):
        calls.append(today)
        started.set()
        assert release.wait(3)
        clock[0] = 10.
        kwargs['check_continue']()
        return False

    worker = worker_for(sync, clock=lambda: clock[0])
    try:
        worker.wake()
        assert started.wait(3)
        for _ in range(20):
            worker.wake()
        assert len(calls) == 1
        release.set()
        wait_for(lambda: worker._next_run == 3610.)
        worker.wake()
        wait_for(lambda: not worker.wakeup.is_set())
        assert len(calls) == 1
    finally:
        release.set()
        worker.close()


def test_failure_retries_after_one_minute_without_leaking_exception(caplog):
    clock = [0.]
    calls = []

    def sync(today, **kwargs):
        calls.append(today)
        if len(calls) == 1:
            raise RuntimeError('secret-must-not-leak')
        return False

    worker = worker_for(sync, clock=lambda: clock[0])
    try:
        worker.wake()
        wait_for(lambda: worker._next_run == 60. and 'RuntimeError' in caplog.text)
        clock[0] = 59.
        worker.wake()
        wait_for(lambda: not worker.wakeup.is_set())
        assert len(calls) == 1
        clock[0] = 60.
        worker.wake()
        wait_for(lambda: worker._next_run == 3660.)
        assert len(calls) == 2
        assert 'RuntimeError' in caplog.text and 'secret-must-not-leak' not in caplog.text
    finally:
        worker.close()


def test_fee_maintenance_blocks_shared_gate_and_resumes_only_after_explicit_open(store):
    svc, api = service(store)
    clock = [0.]
    gate = ExchangeAvailability(logging.getLogger('fee-test'), clock=lambda: clock[0])
    api.availability = gate
    original = api._request
    calls = []

    def request(verb, path, params=None, data=None):
        calls.append(path)
        if len(calls) == 1:
            return {'status': 1, 'messages': [{'message_code': 'ERR-5201'}]}
        return original(verb, path, params, data)

    api._request = request
    worker = FeeSyncWorker(svc, logging.getLogger('fee-test'), clock=lambda: clock[0])
    try:
        worker.wake()
        wait_for(lambda: gate.blocked)
        with pytest.raises(ExchangeMaintenance):
            gate.require_available()
        clock[0] = 61.
        worker.wake()
        wait_for(lambda: not worker.wakeup.is_set())
        assert len(calls) == 1 and store.fee_through('test') is None
        assert gate.poll(lambda: {'status': 0, 'data': {'status': 'OPEN'}})
        worker.wake()
        wait_for(lambda: worker._next_run == 3661.)
        assert len(calls) == 2 and store.fee_through('test') is not None
    finally:
        worker.close()


@pytest.mark.parametrize('paginated', [False, True])
def test_time_budget_discards_partial_pages_then_retry_records_complete_window(store, paginated):
    svc, api = service(store)
    clock = [0.]
    slow = [True]
    reads = []

    def request(verb, path, params=None, data=None):
        reads.append(params.copy())
        assert 'exchangeFee' in path
        row = {'symbol': 'LTC_JPY', 'feeType': 'LEVERAGE_FEE',
               'deliveryDate': params['toDate'], 'deliveryAmount': '-12',
               'deliveryDatetime': '2026-11-22T00:00:00Z'}
        if slow[0]:
            clock[0] += 31.
        return {'status': 0, 'data': {
            'list': [row] if 'cursor' not in params else [],
            'lastFinalizedDate': params['toDate'],
            'pagination': {'cursor': 'page-2'} if paginated and 'cursor' not in params else {},
        }}

    api._request = request
    worker = FeeSyncWorker(svc, logging.getLogger('fee-test'), clock=lambda: clock[0])
    try:
        worker.wake()
        wait_for(lambda: worker._next_run == 91.)
        assert len(reads) == 1
        assert not records(store, 'bot_account_fees') and store.fee_through('test') is None
        slow[0] = False
        clock[0] = 91.
        worker.wake()
        wait_for(lambda: worker._next_run == 3691.)
        assert len(reads) == (3 if paginated else 2) and 'cursor' not in reads[1]
        if paginated:
            assert reads[2]['cursor'] == 'page-2'
        assert len(records(store, 'bot_account_fees')) == 1
        assert store.fee_through('test') == datetime.strptime(reads[-1]['toDate'], '%Y%m%d').date()
    finally:
        worker.close()


def test_bounded_history_resumes_from_committed_window_after_restart(store):
    svc, api = service(store)
    store.record_fees('test', [], date(2026, 11, 1))
    today = date(2026, 11, 22)
    assert svc.sync_fees(today, max_windows=1) is True
    assert store.fee_through('test') == date(2026, 11, 9)
    restarted, _ = service(store, api)
    assert restarted.sync_fees(today, max_windows=1) is True
    assert store.fee_through('test') == date(2026, 11, 17)
    assert restarted.sync_fees(today, max_windows=1) is False
    assert store.fee_through('test') == today


def test_worker_schedules_remaining_backfill_after_thirty_seconds():
    clock = [0.]
    calls = []

    def sync(today, **kwargs):
        kwargs['check_continue']()
        assert kwargs['max_windows'] == 1
        calls.append(today)
        return len(calls) == 1

    worker = worker_for(sync, clock=lambda: clock[0])
    try:
        worker.wake()
        wait_for(lambda: worker._next_run == 30.)
        clock[0] = 30.
        worker.wake()
        wait_for(lambda: worker._next_run == 3630.)
        assert len(calls) == 2
    finally:
        worker.close()


@pytest.mark.parametrize('utc_hour,utc_minute,expected', [
    (20, 59, date(2026, 11, 21)),
    (21, 0, date(2026, 11, 22)),
])
def test_worker_preserves_six_am_jst_business_day_cutoff(monkeypatch, utc_hour, utc_minute, expected):
    frozen = datetime(2026, 11, 21, utc_hour, utc_minute, tzinfo=timezone.utc)
    monkeypatch.setattr('fee_sync.datetime', SimpleNamespace(now=lambda tz: frozen))
    calls = []

    def sync(today, **kwargs):
        calls.append(today)
        return False

    worker = worker_for(sync)
    try:
        worker.wake()
        wait_for(lambda: worker._next_run > 0)
        assert calls == [expected]
    finally:
        worker.close()


def test_unfinalized_day_does_not_advance_checkpoint_or_queue_busy_backfill(store):
    svc, api = service(store)
    today = date(2026, 11, 22)
    store.record_fees('test', [], today - timedelta(days=1))
    api._request = lambda *args, **kwargs: {'status': 0, 'data': {'list': [], 'lastFinalizedDate': '20261121'}}
    assert svc.sync_fees(today, max_windows=1) is False
    assert store.fee_through('test') == date(2026, 11, 21)


@pytest.mark.parametrize('mode', ['paper', 'backtest'])
def test_paper_and_backtest_never_start_fee_worker(mode):
    bot = SimpleNamespace(exchange_settings_gmo={'live_trade': mode == 'backtest'},
                          is_backtest=mode == 'backtest')
    LiveExecutionMixin._queue_fee_sync(bot)
    assert not hasattr(bot, '_fee_sync_worker')


def test_worker_creation_failure_does_not_prevent_actual_position_verification(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError('secret-must-not-leak')
    monkeypatch.setattr('fee_sync.FeeSyncWorker', broken)
    trace = []
    ledger = SimpleNamespace(pending=lambda _: [], close_batches=lambda _: [])
    bot = SimpleNamespace(
        exchange_settings_gmo={'live_trade': True}, _legacy_short_lock_loaded=True,
        logger=logging.getLogger('fee-test'), _execution=lambda: SimpleNamespace(store=ledger, account='test'),
        _queue_trade_notifications=lambda: None, _queue_close_alerts=lambda: None,
        _sync_live_positions=lambda: trace.append('positions'),
    )
    bot._recover_live_orders = MethodType(LiveExecutionMixin._recover_live_orders, bot)
    bot._queue_fee_sync = MethodType(LiveExecutionMixin._queue_fee_sync, bot)
    verify, _ = method(BOT, 'verify_positions')
    verify(bot)
    assert trace == ['positions']


@pytest.mark.parametrize('bad', [0, -1, True, 1.5])
def test_invalid_window_budget_is_rejected_before_db_reads(bad):
    svc = SimpleNamespace()
    from trading_execution import ExecutionService
    with pytest.raises(ValueError):
        ExecutionService.sync_fees(svc, date(2026, 11, 22), max_windows=bad)
