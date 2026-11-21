"""Maintenance is explicit, shared and recoverable without blind order retries."""
import ast
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import logging
import threading
from types import MethodType, SimpleNamespace

import pytest
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from exchange_availability import ExchangeAvailability, maintenance_reason
from trading_execution import ExchangeMaintenance, ExecutionError
from live_execution import LiveExecutionMixin
from test_bot_execution_integration import API, BOT, TREE, method
from test_execution import service, records
from test_execution_recovery import bot_for, two_positions


OPEN={'status':0,'data':{'status':'OPEN'}}
MAINT={'status':1,'messages':[{'message_code':'ERR-5201','message_string':'maintenance'}]}


class Clock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now
    def advance(self,seconds):self.now+=seconds


def availability():
    clock=Clock()
    return ExchangeAvailability(logging.getLogger('maintenance-test'),clock=clock),clock


def pause(gate,response=MAINT):
    with pytest.raises(ExchangeMaintenance):gate.observe(response)


@pytest.mark.parametrize('response,expected',[
    (MAINT,'MAINTENANCE'),
    ({'status':1,'messages':[{'message_code':'ERR-5202'}]},'MAINTENANCE'),
    ({'status':1,'messages':[{'message_code':'ERR-5203'}]},'PREOPEN'),
    ({'status':0,'data':{'status':'MAINTENANCE'}},'MAINTENANCE'),
    ({'status':0,'data':{'status':'PREOPEN'}},'PREOPEN'),
    ({'status':1,'messages':[{'message_string':'MAINTENANCE. Please wait for a while'}]},'MAINTENANCE'),
    (OPEN,None),
    ({'status':1,'messages':[{'message_code':'ERR-5012','message_string':'MAINTENANCE. Please wait for a while'}]},None),
    ({'status':1,'messages':[{'message_code':'ERR-5201'},{'message_code':'ERR-5012'}]},None),
    ({'status':1,'messages':[{'message_string':'Requests are too many.'}]},None),
    ({'status':0,'data':[]},None), ({},None), (None,None),
])
def test_only_explicit_maintenance_is_classified(response,expected):
    assert maintenance_reason(response)==expected


def test_normal_and_paused_probes_use_monotonic_intervals_and_transition_logs(caplog):
    gate,clock=availability();calls=[]
    caplog.set_level(logging.INFO,logger='maintenance-test')
    def fetch():calls.append(clock());return OPEN if clock()<30 or clock()>=150 else MAINT
    assert gate.poll(fetch)
    clock.advance(29);assert gate.poll(fetch) and calls==[0]
    clock.advance(1);assert not gate.poll(fetch) and gate.blocked
    for _ in range(50):assert not gate.poll(fetch)
    assert calls==[0,30]
    clock.advance(60);assert not gate.poll(fetch)
    assert sum('confirmed; preserving' in r.message for r in caplog.records)==1
    clock.advance(60);assert gate.poll(fetch) and not gate.blocked
    assert sum('OPEN confirmed' in r.message for r in caplog.records)==1
    assert calls==[0,30,90,150]


@pytest.mark.parametrize('had_maintenance',[False,True])
def test_status_failure_never_assumes_open_and_keeps_old_pause(had_maintenance,caplog):
    gate,clock=availability();gate.poll(lambda:OPEN)
    if had_maintenance:pause(gate)
    clock.advance(60)
    def timeout():raise requests.Timeout('secret-must-not-leak')
    assert not gate.poll(timeout) and not gate.entries_allowed
    assert gate.blocked is had_maintenance
    assert 'secret-must-not-leak' not in caplog.text
    assert any(r.levelno==logging.ERROR for r in caplog.records)


def test_public_ticker_success_cannot_clear_private_maintenance():
    gate,_=availability();pause(gate)
    gate.observe({'status':0,'data':[{'last':'100'}]})
    assert gate.blocked and not gate.entries_allowed


def test_concurrent_status_checks_send_one_probe():
    gate,_=availability();entered=threading.Event();release=threading.Event();calls=[]
    def fetch():calls.append(1);entered.set();release.wait(2);return OPEN
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(gate.poll,fetch)
        assert entered.wait(1)
        assert not gate.poll(fetch)
        release.set();assert first.result()
    assert calls==[1]


def test_newer_private_maintenance_wins_over_an_inflight_open_probe():
    gate,_=availability()
    def fetch():pause(gate);return OPEN
    assert not gate.poll(fetch) and gate.blocked


def api_with_responses(payload, *, http_error=False):
    gate,clock=availability();calls=[];closed=[]
    def response(*a,**kw):
        calls.append((a,kw))
        def status():
            if http_error:raise requests.HTTPError('503')
        return SimpleNamespace(json=lambda:deepcopy(payload),raise_for_status=status,close=lambda:closed.append(1))
    api=SimpleNamespace(base_url='https://offline.invalid/private',api_key='dummy',_sign=lambda *a:'signature',availability=gate)
    for name in ('_request','_public_request'):
        fn,ns=method(API,name);ns['HTTP_SESSION']=SimpleNamespace(get=response,post=response)
        setattr(api,name,MethodType(fn,api))
    return api,gate,clock,calls,closed


@pytest.mark.parametrize('request_method',['GET','POST'])
@pytest.mark.parametrize('http_error',[False,True])
def test_actual_api_maintenance_response_has_no_http_retry_and_no_false_unsent_claim(request_method,http_error):
    api,gate,clock,calls,closed=api_with_responses(MAINT,http_error=http_error)
    with pytest.raises(ExchangeMaintenance) as first:
        api._request(request_method,'/v1/order',data={'size':'1'})
    assert not first.value.not_sent and len(calls)==len(closed)==1
    with pytest.raises(ExchangeMaintenance) as second:api._request(request_method,'/v1/order')
    assert second.value.not_sent and len(calls)==1
    # Only the public status endpoint can still pass the local pause.
    with pytest.raises(ExchangeMaintenance):api._public_request('/v1/status')
    assert len(calls)==2
    with pytest.raises(ExchangeMaintenance):api._public_request('/v1/ticker')
    assert len(calls)==2


def test_authentication_error_is_not_maintenance_or_suppressed():
    api,gate,_,calls,_=api_with_responses({'status':1,'messages':[{'message_code':'ERR-5012'}]})
    assert api._request('GET','/v1/account/margin')['status']==1
    assert not gate.blocked and len(calls)==1


def test_read_retry_stops_if_another_thread_has_confirmed_maintenance():
    api,gate,_,calls,_=api_with_responses(OPEN)
    fn,ns=method(API,'_request')
    def timeout(*a,**kw):calls.append(1);pause(gate);raise requests.Timeout()
    ns['HTTP_SESSION']=SimpleNamespace(get=timeout)
    with pytest.raises(ExchangeMaintenance):fn(api,'GET','/v1/openPositions')
    assert len(calls)==1


def test_gmo_http_adapter_cannot_retry_before_maintenance_is_inspected():
    node=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='_make_session')
    namespace={'requests':requests,'HTTPAdapter':HTTPAdapter,'Retry':Retry}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'session-policy','exec'),namespace)
    session=namespace['_make_session']()
    try:
        policy=session.get_adapter('https://api.coin.z.com/public/v1/status').max_retries
        assert policy.total==policy.connect==policy.read==policy.status==0
        assert 'POST' not in policy.allowed_methods
        assert session.get_adapter('https://public.bitbank.cc/ltc_jpy').max_retries.total==5
    finally:session.close()


def test_background_price_refresher_does_not_repeat_requests_or_log_errors(caplog):
    gate,clock=availability();calls=[]
    bot=SimpleNamespace(gmo_api=SimpleNamespace(availability=gate),symbol_mapping={'ltc_jpy':'LTC_JPY','xrp_jpy':'XRP_JPY'},
                        last_api_call=0,rate_limit_delay=0,logger=logging.getLogger('maintenance-test'))
    fn,ns=method(BOT,'get_current_price')
    def get(*a,**kw):
        calls.append(1)
        return SimpleNamespace(json=lambda:MAINT,raise_for_status=lambda:None,close=lambda:None)
    ns['HTTP_SESSION']=SimpleNamespace(get=get)
    owner=next(n for n in TREE.body if isinstance(n,ast.ClassDef) and n.name=='PriceCacheRefresher')
    run,_=method(owner,'run_forever')
    refresher=SimpleNamespace(symbols=list(bot.symbol_mapping),get_price=lambda sym:fn(bot,sym),interval=30,logger=bot.logger)
    # This function imports time itself, so intercept its sleep for a single round.
    import time
    def stop(_):raise KeyboardInterrupt()
    from unittest.mock import patch
    with patch.object(time,'sleep',stop):
        with pytest.raises(KeyboardInterrupt):run(refresher)
    assert len(calls)==1 and not any(r.levelno>=logging.ERROR for r in caplog.records)


def attach_gate(api):
    gate,clock=availability();gate.poll(lambda:OPEN);api.availability=gate
    original=api._request;public=api._public_request
    def request(method,path,params=None,data=None):
        gate.require_available()
        result=original(method,path,params,data);gate.observe(result);return result
    api._request=request
    api._public_request=lambda path,params=None:OPEN if path=='/v1/status' else public(path,params)
    return gate,clock


def test_close_request_survives_maintenance_and_resumes_without_new_signal(store,caplog):
    svc,api=service(store);two_positions(api);bot=bot_for(svc);bot._sync_live_positions()
    gate,clock=attach_gate(api);pause(gate)
    before=deepcopy(bot.entry_sizes)
    with pytest.raises(ExchangeMaintenance):bot._close_live_position('ltc_jpy','original-stop',None,None)
    assert len(store.close_batches('test'))==1 and not api.posts
    assert bot.entry_sizes==before
    with pytest.raises(ExchangeMaintenance):bot._sync_live_positions()
    assert bot.entry_sizes==before and 'ltc_jpy' in bot._position_sync_failed
    clock.advance(60);assert bot._exchange_status_ready()
    bot._recover_live_orders()
    assert len(api.posts)==2 and not api.position_rows and not store.close_batches('test')
    assert all(row['raw']['exit_reason']=='original-stop' for row in records(store,'trades'))
    assert not any(r.levelno>=logging.ERROR for r in caplog.records)


def test_status_unknown_blocks_new_entry_but_not_quote_based_protection(store):
    svc,api=service(store);gate,clock=attach_gate(api)
    api._public_request=lambda *a,**kw: {'status':-1}
    clock.advance(30);assert not gate.poll(lambda:{'status':-1}) and not gate.blocked
    bot=bot_for(svc)
    assert not bot._live_funds_check('ltc_jpy','buy',1) and not api.posts
    # Protective read/close requests are allowed while only the status probe is unknown.
    two_positions(api);bot._sync_live_positions();bot._close_live_position('ltc_jpy','stop',None,None)
    assert len(api.posts)==2 and not api.position_rows


def test_locally_blocked_post_after_claim_is_finalized_unsent_and_close_retries_after_open(store,monkeypatch):
    svc,api=service(store);two_positions(api);bot=bot_for(svc);bot._sync_live_positions()
    gate,clock=attach_gate(api);claim=store.claim;blocked=[False]
    def claim_then_pause(client):
        result=claim(client)
        if result and not blocked[0]:blocked[0]=True;pause(gate)
        return result
    monkeypatch.setattr(store,'claim',claim_then_pause)
    with pytest.raises(ExchangeMaintenance):bot._close_live_position('ltc_jpy','stop',None,None)
    assert not api.posts and not store.pending('test')
    assert records(store,'bot_order_intents')[0]['status']=='REJECTED'
    assert store.close_batches('test')
    clock.advance(60);assert bot._exchange_status_ready();bot._recover_live_orders()
    assert len(api.posts)==2 and not store.close_batches('test')


def test_post_timeout_then_maintenance_remains_unknown_and_never_resends(store):
    svc,api=service(store);gate,clock=attach_gate(api);original=api._request;api.lose_response=True
    def request(method,path,params=None,data=None):
        try:return original(method,path,params,data)
        finally:
            if method=='POST':pause(gate)
    api._request=request
    with pytest.raises(ExchangeMaintenance):svc.submit('LTC_JPY','BUY',1)
    assert len(api.posts)==1 and store.pending('test')[0]['status']=='UNKNOWN'
    clock.advance(60);gate.poll(lambda:OPEN);api._request=original
    bot_for(svc)._recover_live_orders()
    assert len(api.posts)==1 and not store.pending('test') and len(records(store,'bot_execution_fills'))==1


@pytest.mark.parametrize('case',['maintenance','auth','open'])
def test_startup_defers_only_confirmed_maintenance_and_checks_authenticated_margin(case,caplog):
    gate,_=availability()
    response=MAINT if case=='maintenance' else OPEN
    bot=SimpleNamespace(logger=logging.getLogger('maintenance-test'),
        exchange_settings_gmo={'live_trade':True,'api_key':'dummy','api_secret':'dummy'},
        gmo_api=SimpleNamespace(availability=gate,_public_request=lambda *a:response))
    bot._exchange_status_ready=MethodType(LiveExecutionMixin._exchange_status_ready,bot)
    def margin():
        gate.require_available()
        if case=='auth':raise ExecutionError('ERR-5012')
        return {'actualProfitLoss':1000}
    bot._live_margin_available=margin
    fn,_=method(BOT,'_verify_api_connection')
    assert fn(bot) is (case!='auth')
    assert gate.blocked is (case=='maintenance')
    if case=='auth':assert any(r.levelno==logging.ERROR for r in caplog.records)


def test_missing_equity_during_maintenance_is_not_returned_as_zero():
    gate,_=availability();pause(gate)
    bot=SimpleNamespace(exchange_settings_gmo={'live_trade':True},logger=logging.getLogger('maintenance-test'),
                        gmo_api=SimpleNamespace(_request=lambda *a,**kw:gate.require_available()))
    fn,_=method(BOT,'get_total_balance')
    with pytest.raises(ExchangeMaintenance):fn(bot)


@pytest.mark.parametrize('scenario',['startup','during_cycle'])
def test_actual_live_loop_waits_and_reconciles_before_resuming_analysis(scenario,caplog):
    gate,clock=availability();actions=[];checks=[];injected=[False]
    at=datetime(2026,11,21,15,tzinfo=timezone(timedelta(hours=9)))
    def status(*a):
        checks.append(clock())
        return MAINT if scenario=='startup' and clock()<60 else OPEN
    bot=SimpleNamespace(logger=logging.getLogger('maintenance-test'),symbols=['ltc_jpy'],
        positions={'ltc_jpy':'long'},entry_prices={'ltc_jpy':100.},entry_times={'ltc_jpy':at},
        entry_sizes={'ltc_jpy':1.},short_profit_lock_armed={},reentry_block_until={},
        initial_capital=1000.,total_profit=0.,test_mode=True,is_backtest=False,
        exchange_settings_gmo={'live_trade':True,'api_key':'dummy','api_secret':'dummy'},
        gmo_api=SimpleNamespace(availability=gate,_public_request=status),
        send_notification=lambda *a:None,save_positions=lambda:None,
        update_sr_zones_for_symbol=lambda *a,**kw:None,_generate_final_report=lambda *a:None,
        save_trade_logs_to_excel=lambda *a:None,_display_current_positions=lambda:None,
        _refresh_live_stats=lambda *a:None,_queue_close_alerts=lambda:actions.append('alert_check'))
    bot._exchange_status_ready=MethodType(LiveExecutionMixin._exchange_status_ready,bot)
    def account():gate.require_available();return {'actualProfitLoss':1000.}
    bot._live_margin_available=account
    verify,_=method(BOT,'_verify_api_connection');bot._verify_api_connection=MethodType(verify,bot)
    def positions():
        if scenario=='during_cycle' and actions.count('positions_reconciled')==1 and not injected[0]:
            injected[0]=True;gate.observe(MAINT)
        gate.require_available();actions.append('positions_reconciled')
    bot.verify_positions=positions
    def balance():gate.require_available();return 1000.
    bot.get_total_balance=balance
    def inputs(*a):actions.append('analysis');raise KeyboardInterrupt()
    bot._prepare_live_inputs=inputs
    run,ns=method(BOT,'run_live')
    ns['record_bot_event']=lambda *a:None;ns['close_recorder']=lambda *a:None
    ns['now_jst']=lambda:at+timedelta(seconds=clock())
    ns['dt']=SimpleNamespace(datetime=SimpleNamespace(now=lambda tz:ns['now_jst']()))
    ns['time']=SimpleNamespace(time=clock,sleep=clock.advance)
    run(bot)
    assert checks==[0,60]
    assert actions[-1]=='analysis' and actions[-2]=='positions_reconciled'
    assert bot.positions=={'ltc_jpy':'long'} and bot.entry_sizes=={'ltc_jpy':1.}
    assert not gate.blocked and not any(r.levelno>=logging.ERROR for r in caplog.records)


@pytest.mark.parametrize('response',[
    {'status':False,'data':{'status':'OPEN'}}, {'status':'0','data':{'status':'OPEN'}},
    {'status':0,'data':{'status':'UNKNOWN'}}, {'status':0,'data':None},
])
def test_malformed_status_response_blocks_entries_without_claiming_maintenance(response):
    gate,_=availability()
    assert not gate.poll(lambda:response) and not gate.blocked and not gate.entries_allowed
