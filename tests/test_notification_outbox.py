"""No real mail/LINE: atomic registration, blocked delivery, retries and restarts."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta,timezone
from email import message_from_string
from email.header import Header
import logging
from pathlib import Path
import threading
from types import MethodType,SimpleNamespace
import uuid

import pytest
from sqlalchemy import text

from execution_store import ExecutionStore
from notification_outbox import NotificationOutboxWorker,route_fingerprint,safe_message
from test_bot_execution_integration import BOT,method
from test_execution import records,service
from test_execution_recovery import bot_for


PREF={'enabled':True,'entry':True,'exit':True,'error':True,'info':True,'channels':['email','line']}


def until(predicate,timeout=4):
    import time
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if predicate():return
        threading.Event().wait(.01)
    raise AssertionError('background delivery did not reach expected state')


def jobs(store):return records(store,'bot_notification_outbox')


def queued(store,channels=('email',),key='fixture-event'):
    prefs={**PREF,'channels':list(channels)}
    assert store.enqueue_message('test',key,prefs,'fixture subject','fixture body','info')==len(channels)


def worker(store,deliver,route=None):
    return NotificationOutboxWorker(store,'test',lambda kind:True,
        route or (lambda channel:['fixture@example.invalid'] if channel=='email' else 'fake-line-user'),
        deliver,logging.getLogger('outbox-test'),poll_seconds=3600)


def make_bot(svc,deliver,channels=('email',)):
    bot=bot_for(svc)
    bot.notification_settings={'enabled':True,'send_on_entry':True,'send_on_exit':True,'send_on_error':True}
    bot._configured_trade_channels=lambda:list(channels)
    bot._prepare_notification_route=lambda channel:['fixture@example.invalid'] if channel=='email' else 'fake-line-user'
    bot._deliver_outbox_notification=deliver
    return bot


def open_order(bot):
    return bot._execute_live_entry('ltc_jpy','buy',10,
        signal_raw={'decision_bar':'2026-11-21T15:00:00+09:00','decision_price':'100'})


def test_slow_delivery_cannot_stop_entry_return_or_the_following_close(store):
    svc,api=service(store);entered=threading.Event();release=threading.Event();sent=[]
    def slow(*args):
        entered.set();release.wait(4);sent.append(args);return True
    bot=make_bot(svc,slow)
    try:
        assert open_order(bot)['success']
        assert entered.wait(2) and not sent
        bot._close_live_position('ltc_jpy','stop',None,None)
        assert len(api.posts)==2 and not api.position_rows
        assert len(jobs(store))==2 and all(row['delivered_at'] is None for row in jobs(store))
    finally:release.set()
    until(lambda:all(row['delivered_at'] is not None for row in jobs(store)))


def test_confirmation_commits_delivery_jobs_even_if_process_dies_before_notifier(store):
    svc,api=service(store)
    result=svc.submit('LTC_JPY','BUY',1,context={'notification_delivery':{**PREF,'channels':['email']}})
    assert result['finalized'] and len(jobs(store))==1
    assert jobs(store)[0]['delivered_at'] is None  # No bot/worker existed before this restart.
    restarted_store=ExecutionStore(store.engine);svc.store=restarted_store
    sent=[];bot=make_bot(svc,lambda *a:sent.append(a) or True)
    bot._recover_live_orders()
    until(lambda:len(sent)==1 and jobs(store)[0]['delivered_at'] is not None)
    bot._recover_live_orders()
    assert len(api.posts)==1 and len(sent)==1


def test_notification_insert_failure_rolls_back_fill_confirmation_without_resending_post(store,monkeypatch):
    svc,api=service(store);original=store._insert_notifications
    def failure(*a,**kw):raise RuntimeError('controlled outbox failure')
    monkeypatch.setattr(store,'_insert_notifications',failure)
    with pytest.raises(RuntimeError):
        svc.submit('LTC_JPY','BUY',1,context={'notification_delivery':PREF})
    assert len(api.posts)==1 and not records(store,'bot_execution_fills') and not jobs(store)
    assert store.pending('test')
    monkeypatch.setattr(store,'_insert_notifications',original)
    svc.recover()
    assert len(api.posts)==1 and len(records(store,'bot_execution_fills'))==1 and len(jobs(store))==2


@pytest.mark.parametrize('originally_disabled',[False,True])
def test_pending_upgrade_adopts_settings_without_overwriting_disabled_choice(store,originally_disabled):
    svc,api=service(store);api.hide_fills=True
    context={'notification_delivery':{**PREF,'enabled':False}} if originally_disabled else {}
    assert not svc.submit('LTC_JPY','BUY',1,context=context)['finalized']
    api.hide_fills=False;sent=[]
    bot=make_bot(svc,lambda *a:sent.append(a) or True)
    bot._recover_live_orders()
    assert not store.pending('test') and len(api.posts)==1
    if originally_disabled:
        assert not jobs(store) and not sent
    else:
        until(lambda:len(sent)==1 and jobs(store)[0]['delivered_at'] is not None)
        assert jobs(store)[0]['recovered'] is True


def test_finalization_uses_locked_settings_instead_of_stale_intent(store):
    svc,api=service(store);api.hide_fills=True
    result=svc.submit('LTC_JPY','BUY',1)
    stale=store.get(result['client_order_id'])
    store.set_notification_preferences('test',stale['client_order_id'],PREF)
    api.hide_fills=False
    assert svc.reconcile(stale,recovered=True)['finalized']
    assert len(jobs(store))==2 and len(api.posts)==1


def test_success_ack_failure_keeps_lease_and_restarts_with_frozen_payload(store,monkeypatch):
    queued(store);sent=[]
    w=worker(store,lambda *a:sent.append(a) or True)
    original=store.finish_notification
    def unavailable(*a,**kw):raise RuntimeError('controlled acknowledgement failure')
    monkeypatch.setattr(store,'finish_notification',unavailable)
    assert w.process_one() and len(sent)==1
    assert jobs(store)[0]['claim_token'] and jobs(store)[0]['delivered_at'] is None
    assert not ExecutionStore(store.engine).claim_notification('test')
    monkeypatch.setattr(store,'finish_notification',original)
    with store.engine.begin() as conn:
        conn.execute(text("UPDATE bot_notification_outbox SET claim_until=now()-interval '1 second'"))
    assert worker(ExecutionStore(store.engine),lambda *a:sent.append(a) or True).process_one()
    assert sent[0]==sent[1] and jobs(store)[0]['delivered_at'] is not None


def test_successful_channel_is_not_retried_when_other_channel_fails(store):
    queued(store,('email','line'));calls=[];attempt=[False]
    def deliver(channel,*args):
        calls.append(channel)
        if channel=='email' and not attempt[0]:attempt[0]=True;return False
        return True
    w=worker(store,deliver)
    assert w.process_one() and w.process_one() and not w.process_one()
    by_channel={row['channel']:row for row in jobs(store)}
    assert by_channel['line']['delivered_at'] is not None and by_channel['email']['delivered_at'] is None
    with store.engine.begin() as conn:
        conn.execute(text("UPDATE bot_notification_outbox SET next_attempt_at=now() WHERE channel='email'"))
    assert w.process_one()
    assert calls.count('line')==1 and calls.count('email')==2
    assert all(row['delivered_at'] is not None for row in jobs(store))


def test_failed_delivery_retries_with_same_body_and_identity_without_rereading_trade(store,monkeypatch):
    svc,api=service(store)
    svc.submit('LTC_JPY','BUY',1,context={'notification_delivery':{**PREF,'channels':['email']}})
    captured=[]
    def send(*args):captured.append(args);return len(captured)>1
    w=worker(store,send);assert w.process_one()
    def forbidden(*a):raise AssertionError('frozen body must not be regenerated')
    monkeypatch.setattr(store,'result',forbidden)
    with store.engine.begin() as conn:conn.execute(text('UPDATE bot_notification_outbox SET next_attempt_at=now()'))
    assert w.process_one() and captured[0]==captured[1]
    assert jobs(store)[0]['delivered_at'] is not None


def test_claim_lease_survives_restart_and_old_ack_cannot_overwrite(store):
    queued(store);old=store.claim_notification('test')
    frozen=store.freeze_notification('test',old,'fixed subject','fixed body','route-hash')
    assert frozen['subject']=='fixture subject'  # Generic payload was already fixed at enqueue.
    assert not ExecutionStore(store.engine).claim_notification('test')
    with store.engine.begin() as conn:conn.execute(text("UPDATE bot_notification_outbox SET claim_until=now()-interval '1 second'"))
    newer=ExecutionStore(store.engine).claim_notification('test')
    assert newer['job_id']==old['job_id'] and newer['claim_token']!=old['claim_token']
    assert not store.finish_notification('test',old,True)
    assert store.finish_notification('test',newer,True)


def test_concurrent_workers_cannot_claim_the_same_job(store):
    queued(store)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claimed=list(pool.map(lambda _:store.claim_notification('test'),range(2)))
    assert sum(row is not None for row in claimed)==1


def test_account_scope_and_duplicate_event_key_are_persistent(store):
    queued(store)
    assert store.enqueue_message('test','fixture-event',PREF,'changed','changed','info')==1  # Only new LINE channel.
    assert len(jobs(store))==2
    assert not store.claim_notification('other-account')
    job=store.claim_notification('test')
    assert store.freeze_notification('other-account',job,'a','b','route') is None
    assert not store.finish_notification('other-account',job,True)


def test_disabled_transport_defers_without_consuming_line_retry_window(store):
    queued(store,('line',));sent=[]
    w=worker(store,lambda *a:sent.append(a) or True);w.enabled=lambda kind:False
    assert w.process_one() and not sent
    assert jobs(store)[0]['first_attempt_at'] is None and jobs(store)[0]['delivered_at'] is None


@pytest.mark.parametrize('channel',['email','line'])
def test_changed_destination_requires_review_without_sending_to_new_recipient(store,channel):
    queued(store,(channel,));route=[['first@example.invalid'] if channel=='email' else 'first-line'];sent=[]
    w=worker(store,lambda *a:sent.append(a) or False,lambda _:route[0])
    w.process_one();route[0]=['other@example.invalid'] if channel=='email' else 'other-line'
    with store.engine.begin() as conn:conn.execute(text('UPDATE bot_notification_outbox SET next_attempt_at=now()'))
    w.process_one()
    assert len(sent)==1 and jobs(store)[0]['terminal_reason']=='destination_changed'


def test_expired_line_retry_window_is_not_silently_reused(store):
    queued(store,('line',));sent=[]
    w=worker(store,lambda *a:sent.append(a) or False)
    w.process_one()
    with store.engine.begin() as conn:
        conn.execute(text("UPDATE bot_notification_outbox SET next_attempt_at=now(),first_attempt_at=now()-interval '24 hours'"))
    w.process_one()
    assert len(sent)==1 and jobs(store)[0]['terminal_reason']=='line_retry_window_expired'
    assert jobs(store)[0]['delivered_at'] is None


def test_retry_backoff_and_stable_payload_survive_repeated_migration(store):
    queued(store);job=store.claim_notification('test');at=datetime.now(timezone.utc)
    store.finish_notification('test',job,False,now=at,error_type='Timeout')
    assert not store.claim_notification('test',now=at+timedelta(seconds=29))
    assert store.claim_notification('test',now=at+timedelta(seconds=30)) is not None
    before=jobs(store)
    with store.engine.connect().execution_options(isolation_level='AUTOCOMMIT') as conn:
        conn.exec_driver_sql(Path('migrations/004_notification_outbox.sql').read_text())
    assert jobs(store)==before


def test_historical_or_originally_disabled_notifications_are_not_backfilled(store):
    svc,api=service(store)
    svc.submit('LTC_JPY','BUY',1,context={'notification_delivery':{**PREF,'enabled':False}})
    assert not jobs(store)
    bot=make_bot(svc,lambda *a:(_ for _ in ()).throw(AssertionError('historical mail')))
    bot._recover_live_orders()
    assert not jobs(store)


@pytest.mark.parametrize('status,header,expected',[(200,None,True),(409,str(uuid.uuid4()),True),(409,None,False),(401,None,False),(503,None,False)])
def test_line_retry_key_and_acceptance_are_checked_without_external_io(monkeypatch,status,header,expected):
    fn,ns=method(BOT,'_send_line');calls=[];closed=[];key=str(uuid.uuid4())
    monkeypatch.setenv('LINE_CHANNEL_ACCESS_TOKEN','fixture-token')
    def post(*a,**kw):
        calls.append(kw)
        return SimpleNamespace(status_code=status,headers={'x-line-accepted-request-id':header} if header else {},close=lambda:closed.append(1))
    ns['requests']=SimpleNamespace(post=post)
    bot=SimpleNamespace(logger=logging.getLogger('outbox-test'))
    assert fn(bot,'subject','body',recipient='fixture-user',retry_key=key) is expected
    assert calls[0]['headers']['X-Line-Retry-Key']==key and calls[0]['allow_redirects'] is False
    assert calls[0]['json']['to']=='fixture-user' and closed==[1]


def test_email_uses_fixed_message_id_and_resolved_recipient(monkeypatch):
    import smtplib
    fn,ns=method(BOT,'_send_email');ns['Header']=Header;sent=[]
    class SMTP:
        def __init__(self,*a,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*a):return False
        def starttls(self):pass
        def login(self,*a):pass
        def sendmail(self,sender,recipient,message):sent.append((recipient,message))
    monkeypatch.setattr(smtplib,'SMTP',SMTP)
    bot=SimpleNamespace(notification_settings={'enabled':True,'email':{'smtp_server':'fixture','smtp_port':587,'sender':'fixture','password':'dummy'}},
        logger=logging.getLogger('outbox-test'),_is_sim_mode=lambda:False,_resolve_smtp_password_for_mode=lambda value:value)
    for _ in range(2):assert fn(bot,'subject','body',message_id='<stable@example.invalid>',recipients=['fixture@example.invalid'])
    assert all(message_from_string(payload)['Message-ID']=='<stable@example.invalid>' for _,payload in sent)


def test_generic_live_notification_is_enqueued_without_calling_transports(store):
    svc,api=service(store);sent=[];bot=make_bot(svc,lambda *a:sent.append(a) or True)
    bot.exchange_settings_gmo={'live_trade':True}
    bot._queue_trade_notifications=lambda:None  # Simulate process exit immediately after durable enqueue.
    fn,_=method(BOT,'send_notification')
    assert fn(bot,'queued subject','queued body','error')
    assert len(jobs(store))==1 and not sent
    assert not records(store,'bot_order_intents')  # Operational notices never create exchange orders.


def test_operational_messages_redact_connection_credentials_before_persisting(store):
    svc,_=service(store);bot=make_bot(svc,lambda *a:True);bot._queue_trade_notifications=lambda:None
    assert bot._enqueue_live_message('error','postgresql://user:fixture-secret@localhost/db\nAPI_SECRET=fixture-key','error')
    assert 'fixture-secret' not in jobs(store)[0]['body'] and 'fixture-key' not in jobs(store)[0]['body']
    assert '[DB接続情報省略]' in jobs(store)[0]['body']
