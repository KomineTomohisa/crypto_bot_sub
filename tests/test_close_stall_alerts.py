"""No real notifications: exercise persisted timing, leases and async failures."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import logging
from email.header import Header
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from close_alerts import CloseAlertPolicy, CloseAlertWorker
from execution_store import ExecutionStore
from test_execution import records, service
from test_execution_recovery import bot_for, two_positions
from test_close_decision_queue import plan, waiting_other_order
from test_close_position_read_failure import failing_position_reads
from trading_execution import ExecutionError
from test_bot_execution_integration import method, BOT


POLICY = CloseAlertPolicy()


def queued(store, *, symbol='LTC_JPY', created=None):
    svc, api = service(store)
    two_positions(api)
    candidate = plan(api)
    candidate['symbol'] = symbol.lower()
    for pos in candidate['positions']:pos['symbol'] = symbol
    saved = store.save_close_batch('test', symbol, candidate)
    at = created or datetime.now(timezone.utc)
    with store.engine.begin() as conn:
        conn.execute(text('UPDATE bot_close_batches SET created_at=:at WHERE batch_id=:id'), {'at':at,'id':saved['id']})
    return svc, api, saved, at


def claim(store, at, policy=POLICY, **kw):
    return store.claim_close_alerts('test', policy, now=at, **kw)


def complete(store, batch, at):
    with store.engine.begin() as conn:
        conn.execute(text('UPDATE bot_close_batches SET completed=true,completed_at=:at WHERE batch_id=:id'), {'at':at,'id':batch['id']})


def wait_for(predicate, timeout=3):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if predicate():return
        threading.Event().wait(0.01)
    raise AssertionError('background condition not reached')


def test_initial_threshold_and_close_pending_category(store):
    svc, api, batch, at = queued(store)
    assert not claim(store, at+timedelta(seconds=119))
    jobs=claim(store, at+timedelta(seconds=120))
    assert len(jobs)==1 and jobs[0]['kind']=='warning' and jobs[0]['phase']=='close_pending'
    assert not claim(store, at+timedelta(seconds=121))
    assert not api.posts


@pytest.mark.parametrize('reason', ['position_read_failure', 'pending_other_order'])
def test_persisted_unresolved_request_reports_why_it_is_waiting(store, reason):
    svc,api=service(store);two_positions(api)
    bot=bot_for(svc);bot._sync_live_positions()
    if reason=='position_read_failure':
        failing_position_reads(api)
        with pytest.raises(ExecutionError):bot._close_live_position('ltc_jpy','stop-triggered',None,None)
    else:
        waiting_other_order(svc,api)
        bot._close_live_position('ltc_jpy','stop-triggered',None,None)
    row=records(store,'bot_close_batches')[0]
    assert row['payload']['positions'] is None
    posts_before=len(api.posts)
    job=claim(store,row['created_at']+timedelta(seconds=120))[0]
    expected='position_confirmation' if reason=='position_read_failure' else 'order_reconciliation'
    assert job['phase']==expected
    assert len(api.posts)==posts_before  # Monitoring never submits any order.


def test_successful_warning_reminder_and_restart_suppression(store):
    svc, api, batch, created=queued(store)
    at=created+timedelta(seconds=120)
    first=claim(store,at)[0]
    assert store.finish_close_alert('test',first,True,now=at)
    restarted=ExecutionStore(store.engine)
    assert not restarted.claim_close_alerts('test',POLICY,now=at+timedelta(seconds=1799))
    reminder=restarted.claim_close_alerts('test',POLICY,now=at+timedelta(seconds=1800))
    assert len(reminder)==1 and reminder[0]['kind']=='warning'
    assert store.close_batches('test')==[batch]  # Targets must remain immutable.


def test_failure_retries_after_backoff_without_claiming_delivery(store):
    svc, api, batch, created=queued(store)
    at=created+timedelta(seconds=120)
    job=claim(store,at)[0]
    store.finish_close_alert('test',job,False,now=at)
    assert records(store,'bot_close_batches')[0]['alert_warning_sent_at'] is None
    assert not claim(store,at+timedelta(seconds=299))
    assert len(claim(store,at+timedelta(seconds=300)))==1


def test_recovery_is_immediate_once_warned_and_successful_recovery_is_not_repeated(store):
    svc, api, batch, created=queued(store)
    at=created+timedelta(seconds=120)
    warning=claim(store,at)[0]
    store.finish_close_alert('test',warning,True,now=at)
    complete(store,batch,at+timedelta(seconds=1))
    recovery=claim(store,at+timedelta(seconds=1))[0]
    assert recovery['kind']=='recovery'
    store.finish_close_alert('test',recovery,True,now=at+timedelta(seconds=1))
    assert not claim(store,at+timedelta(days=1))
    assert not ExecutionStore(store.engine).claim_close_alerts('test',POLICY,now=at+timedelta(days=2))


def test_recovery_delivery_failure_also_uses_backoff(store):
    svc, api, batch, created=queued(store)
    at=created+timedelta(seconds=120)
    store.finish_close_alert('test',claim(store,at)[0],True,now=at)
    complete(store,batch,at)
    recovery=claim(store,at)[0]
    store.finish_close_alert('test',recovery,False,now=at)
    assert not claim(store,at+timedelta(seconds=299))
    assert claim(store,at+timedelta(seconds=300))[0]['kind']=='recovery'


def test_no_recovery_notice_if_warning_was_never_delivered(store):
    svc, api, batch, created=queued(store)
    at=created+timedelta(seconds=120)
    job=claim(store,at)[0]
    complete(store,batch,at)
    assert not store.close_alert_is_current('test',job,now=at)
    store.finish_close_alert('test',job,False,now=at)
    assert not claim(store,at+timedelta(days=1))


def test_warning_in_flight_cannot_be_overtaken_by_recovery(store):
    svc, api, batch, created=queued(store)
    at=created+timedelta(seconds=120)
    warning=claim(store,at)[0]
    complete(store,batch,at)
    assert not claim(store,at+timedelta(seconds=1))
    store.finish_close_alert('test',warning,True,now=at+timedelta(seconds=2))
    assert claim(store,at+timedelta(seconds=3))[0]['kind']=='recovery'


def test_concurrent_monitors_claim_one_notice(store):
    svc, api, batch, created=queued(store)
    at=created+timedelta(seconds=120)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:claim(store,at),range(2)))
    assert sum(map(len,results))==1


def test_expired_lease_retries_after_crash_and_old_ack_cannot_overwrite(store):
    svc, api, batch, created=queued(store)
    at=created+timedelta(seconds=120)
    old=claim(store,at)[0]
    assert not store.close_alert_is_current('test',old,now=at+timedelta(seconds=600))
    newer=claim(store,at+timedelta(seconds=600))[0]
    assert old['claim_token']!=newer['claim_token']
    assert not store.finish_close_alert('test',old,True,now=at+timedelta(seconds=601))
    assert store.close_alert_is_current('test',newer,now=at+timedelta(seconds=601))
    assert records(store,'bot_close_batches')[0]['alert_warning_sent_at'] is None


def test_alerts_are_scoped_to_the_account(store):
    svc, api, batch, created=queued(store)
    at=created+timedelta(seconds=120)
    assert not store.claim_close_alerts('other-account',POLICY,now=at)
    job=claim(store,at)[0]
    assert not store.close_alert_is_current('other-account',job,now=at)
    assert not store.finish_close_alert('other-account',job,True,now=at)


def test_claim_limit_skips_already_leased_rows_without_starving_other_jobs(store):
    at=datetime.now(timezone.utc)
    for i in range(5):queued(store,symbol=f'S{i}_JPY',created=at)
    now=at+timedelta(seconds=120)
    first=claim(store,now,limit=2);second=claim(store,now,limit=2)
    assert len(first)==len(second)==2
    assert not {j['batch_id'] for j in first}&{j['batch_id'] for j in second}


def test_migration_repeat_preserves_payload_and_notification_history(store):
    svc, api, batch, created=queued(store)
    at=created+timedelta(seconds=120)
    store.finish_close_alert('test',claim(store,at)[0],True,now=at)
    before=records(store,'bot_close_batches')
    with store.engine.connect().execution_options(isolation_level='AUTOCOMMIT') as conn:
        conn.exec_driver_sql(Path('migrations/003_close_stall_notifications.sql').read_text())
        conn.exec_driver_sql(Path('migrations/003_close_stall_notifications.sql').read_text())
    assert records(store,'bot_close_batches')==before


def test_real_close_continues_and_recovery_is_offered_after_recorded_completion(store):
    svc, api, batch, created=queued(store)
    at=created+timedelta(seconds=120)
    store.finish_close_alert('test',claim(store,at)[0],True,now=at)
    bot=bot_for(svc)
    bot._recover_live_orders()
    assert len(api.posts)==2 and not api.position_rows
    recovery=claim(store,at+timedelta(seconds=1))
    assert len(recovery)==1 and recovery[0]['kind']=='recovery'


@pytest.mark.parametrize('value',['bad','0','-1','999999999999999999999'])
def test_bad_environment_values_use_safe_defaults(monkeypatch,value):
    monkeypatch.setenv('CLOSE_STALL_ALERT_SECONDS',value)
    assert CloseAlertPolicy.from_environment(logging.getLogger('test'))==POLICY


def test_messages_use_only_safe_status_fields_not_raw_exception_or_positions():
    at=datetime.now(timezone.utc)
    job={'batch_id':'abcdefgh1234','symbol':'LTC_JPY','created_at':at-timedelta(minutes=3),
         'observed_at':at,'kind':'warning','phase':'position_confirmation','raw':'secret-must-not-leak'}
    subject,body,kind=POLICY.message(job)
    assert kind=='error' and 'LTC_JPY' in subject and 'abcdefgh' in body
    assert 'secret-must-not-leak' not in body and '確認待ち' in body
    job['kind']='recovery'
    assert POLICY.message(job)[2]=='info'


def test_notification_send_is_async_and_failure_does_not_stop_closes(store):
    svc,api,batch,created=queued(store,created=datetime.now(timezone.utc)-timedelta(minutes=3))
    # Keep this batch unresolved while another position still needs protection.
    original=api._request
    api._request=lambda method,path,params=None,data=None: ({'status':-1} if path=='/v1/openPositions' else original(method,path,params,data))
    bot=bot_for(svc);bot.notification_settings={'enabled':True,'send_on_error':True,'send_on_exit':False}
    entered,release=threading.Event(),threading.Event()
    def slow_failure(*a):
        entered.set();release.wait(3);raise RuntimeError('secret-must-not-leak')
    bot.send_notification=slow_failure
    try:
        bot._recover_live_orders()
        assert entered.wait(2)  # Recovery already returned while sending is blocked.
        api._request=original
        bot._recover_live_orders()
        assert len(api.posts)==2 and not api.position_rows
    finally:
        release.set()
        if getattr(bot,'_close_alert_worker',None):bot._close_alert_worker.close(2)
    assert records(store,'bot_close_batches')[0]['alert_warning_sent_at'] is None


@pytest.mark.parametrize('settings', [
    {'enabled': False, 'send_on_error': True},
    {'enabled': True, 'send_on_error': False},
])
def test_disabled_notifications_do_not_claim_or_send(store, settings):
    svc,api,batch,created=queued(store,created=datetime.now(timezone.utc)-timedelta(minutes=3))
    bot=bot_for(svc);bot.notification_settings=settings
    bot.send_notification=lambda *a: (_ for _ in ()).throw(AssertionError('must not send'))
    bot._queue_close_alerts()
    assert records(store,'bot_close_batches')[0]['alert_last_attempt_at'] is None
    assert not hasattr(bot,'_close_alert_worker')


def test_notification_method_reports_actual_channel_success_and_attempts_both():
    fn,_=method(BOT,'send_notification')
    calls=[]
    bot=SimpleNamespace(notification_settings={'enabled':True,'send_on_error':True},logger=logging.getLogger('test'),
        _send_email=lambda *a:(_ for _ in ()).throw(RuntimeError('secret-must-not-leak')),
        _send_line=lambda *a:calls.append('line') or True)
    assert fn(bot,'subject','body','error') is True and calls==['line']
    bot._send_line=lambda *a:False
    assert fn(bot,'subject','body','error') is False
    bot.notification_settings['enabled']=False
    assert fn(bot,'subject','body','error') is False


@pytest.mark.parametrize('delivery', [True, False, None, 'exception'])
def test_worker_acknowledges_only_explicit_success(store, delivery, caplog):
    svc,api,batch,created=queued(store,created=datetime.now(timezone.utc)-timedelta(minutes=3))
    job=store.claim_close_alerts('test',POLICY)[0]
    def send(*args):
        if delivery=='exception':raise RuntimeError('secret-must-not-leak')
        return delivery
    worker=CloseAlertWorker(store,'test',send,logging.getLogger('test'),POLICY)
    try:
        assert worker.submit(job)
        wait_for(lambda:records(store,'bot_close_batches')[0]['alert_claim_token'] is None)
    finally:worker.close(2)
    row=records(store,'bot_close_batches')[0]
    assert (row['alert_warning_sent_at'] is not None)==(delivery is True)
    assert not store.claim_close_alerts('test',POLICY)
    assert 'secret-must-not-leak' not in caplog.text


def test_worker_skips_obsolete_warning_after_completion(store):
    svc,api,batch,created=queued(store,created=datetime.now(timezone.utc)-timedelta(minutes=3))
    job=store.claim_close_alerts('test',POLICY)[0]
    complete(store,batch,datetime.now(timezone.utc))
    calls=[]
    worker=CloseAlertWorker(store,'test',lambda *a:calls.append(a) or True,logging.getLogger('test'),POLICY)
    try:
        assert worker.submit(job)
        wait_for(lambda:records(store,'bot_close_batches')[0]['alert_claim_token'] is None)
    finally:worker.close(2)
    assert not calls and not store.claim_close_alerts('test',POLICY)


def test_full_queue_releases_claim_with_backoff_and_monitor_failure_is_contained(store):
    svc,api,batch,created=queued(store,created=datetime.now(timezone.utc)-timedelta(minutes=3))
    bot=bot_for(svc);bot.notification_settings={'enabled':True,'send_on_error':True}
    bot.send_notification=lambda *a:True
    bot._close_alert_worker=SimpleNamespace(thread=SimpleNamespace(is_alive=lambda:True),policy=POLICY,submit=lambda job:False)
    bot._queue_close_alerts()
    row=records(store,'bot_close_batches')[0]
    assert row['alert_claim_token'] is None and row['alert_last_attempt_at'] is not None
    assert row['alert_warning_sent_at'] is None and not store.claim_close_alerts('test',POLICY)
    # Monitoring cannot make recovery abandon the actual close.
    store.claim_close_alerts=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('DB unavailable'))
    bot._recover_live_orders()
    assert len(api.posts)==2 and not api.position_rows


@pytest.mark.parametrize('outcome', [True, False, 'exception'])
def test_line_helper_reports_delivery_without_external_io(monkeypatch,outcome,caplog):
    fn,ns=method(BOT,'_send_line')
    monkeypatch.setenv('LINE_DEFAULT_USER_ID','fake-recipient')
    def send(*a):
        if outcome=='exception':raise RuntimeError('secret-must-not-leak')
        return outcome
    ns['LineMessaging']=lambda:SimpleNamespace(send_text=send)
    bot=SimpleNamespace(logger=logging.getLogger('test'))
    assert fn(bot,'subject','body') is (outcome is True)
    assert 'secret-must-not-leak' not in caplog.text
    monkeypatch.delenv('LINE_DEFAULT_USER_ID')
    assert fn(bot,'subject','body') is False


@pytest.mark.parametrize('fail', [False,True])
def test_email_helper_reports_delivery_and_exhausted_retries(monkeypatch,fail,caplog):
    import smtplib
    fn,ns=method(BOT,'_send_email');ns['Header']=Header
    calls=[]
    class FakeSMTP:
        def __init__(self,*a,**kw):calls.append('connect')
        def __enter__(self):return self
        def __exit__(self,*a):return False
        def starttls(self):pass
        def login(self,*a):pass
        def sendmail(self,*a):
            if fail:raise RuntimeError('secret-must-not-leak')
            return {}
    monkeypatch.setattr(smtplib,'SMTP',FakeSMTP)
    monkeypatch.setattr(time,'sleep',lambda _:None)
    bot=SimpleNamespace(logger=logging.getLogger('test'),
        notification_settings={'enabled':True,'email':{'smtp_server':'fake','smtp_port':587,'sender':'fake','password':'dummy'}},
        _resolve_recipients_for_mode=lambda:['recipient@example.invalid'],_is_sim_mode=lambda:False,
        _resolve_smtp_password_for_mode=lambda value:value)
    assert fn(bot,'subject','body') is (not fail)
    assert len(calls)==(3 if fail else 1)
    assert 'secret-must-not-leak' not in caplog.text
    bot._resolve_recipients_for_mode=lambda:[]
    assert fn(bot,'subject','body') is False
    bot.notification_settings['enabled']=False
    assert fn(bot,'subject','body') is False
