"""Ticker freshness at acquisition/use/cache, with no external requests or orders."""
import ast
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import logging
from types import MethodType,SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import requests

from market_quotes import MarketQuote,QuoteUnavailable,checked_quote,parse_ticker,price_value
from test_bot_execution_integration import BOT,TREE,method
from test_execution import service
from test_execution_recovery import bot_for
from test_exit_signals import position_bot


AT=datetime(2026,11,22,12,tzinfo=timezone.utc)
BAD_PRICES=[0,-1,None,True,False,np.bool_(True),'','bad','NaN','Infinity',float('nan'),float('inf'),float('-inf'),10**400,'1e400','1e-4000']


def ticker(price='100', *, at=AT, symbol='LTC_JPY'):
    return {'status':0,'data':[{'symbol':symbol,'last':price,'bid':'99','ask':'101',
                               'timestamp':at.isoformat() if isinstance(at,datetime) else at}]}


@pytest.mark.parametrize('value',BAD_PRICES)
def test_nonpositive_nonfinite_boolean_and_overflow_prices_are_invalid(value):
    assert price_value(value) is None
    with pytest.raises(QuoteUnavailable,match='invalid_price'):parse_ticker(ticker(value),'LTC_JPY',AT)


@pytest.mark.parametrize('seconds,reason',[(0,None),(90,None),(-.001,'future_timestamp'),(90.000001,'stale_timestamp')])
def test_exchange_age_boundaries_are_explicit(seconds,reason):
    payload=ticker(at=AT-timedelta(seconds=seconds))
    if reason:
        with pytest.raises(QuoteUnavailable,match=reason):parse_ticker(payload,'LTC_JPY',AT)
    else:assert parse_ticker(payload,'LTC_JPY',AT).price==100


@pytest.mark.parametrize('at',[None,True,'not-a-time','2026-11-22T12:00:00',{},1234,
                             '0001-01-01T00:00:00+14:00','9999-12-31T23:59:59-14:00'])
def test_missing_naive_and_invalid_exchange_times_are_rejected(at):
    with pytest.raises(QuoteUnavailable,match='invalid_timestamp'):parse_ticker(ticker(at=at),'LTC_JPY',AT)


@pytest.mark.parametrize('payload',[None,[],{'status':False,'data':[]},{'status':1,'data':[]},
    {'status':0,'data':None},{'status':0,'data':[]},{'status':0,'data':[None]},
    ticker(symbol='XRP_JPY'),{'status':0,'data':ticker()['data']*2}])
def test_wrong_status_shape_or_ambiguous_symbol_never_becomes_a_quote(payload):
    with pytest.raises(QuoteUnavailable):parse_ticker(payload,'LTC_JPY',AT)


def test_single_object_and_timezone_offsets_keep_the_original_exchange_instant():
    payload=ticker(at=AT.astimezone(timezone(timedelta(hours=9))))
    payload['data']=payload['data'][0]
    assert parse_ticker(payload,'LTC_JPY',AT)==MarketQuote(100.,AT)


def price_reader(responses):
    fn,ns=method(BOT,'get_current_price');items=iter(responses);calls=[];closed=[];events=[]
    ns['now_jst']=lambda:AT
    ns['record_bot_event']=lambda *a:events.append(a)
    ns['time']=SimpleNamespace(time=lambda:0,sleep=lambda _:None)
    def get(*a,**kw):
        calls.append((a,kw));payload=next(items)
        if isinstance(payload,Exception):raise payload
        return SimpleNamespace(json=lambda:payload,raise_for_status=lambda:None,close=lambda:closed.append(1))
    ns['HTTP_SESSION']=SimpleNamespace(get=get)
    bot=SimpleNamespace(symbol_mapping={'ltc_jpy':'LTC_JPY'},last_api_call=0,rate_limit_delay=0,logger=logging.getLogger('quote-test'))
    return bot,lambda **kw:fn(bot,'ltc_jpy',**kw),calls,closed,events


@pytest.mark.parametrize('first',[ticker(at=AT-timedelta(seconds=91)),ticker(price='NaN'),requests.Timeout('credential-must-not-leak')])
def test_actual_getter_retries_reads_then_returns_only_validated_price(first):
    _,read,calls,closed,events=price_reader([first,ticker('101')])
    assert read()==101 and len(calls)==2
    assert len(closed)==(1 if isinstance(first,Exception) else 2)
    assert any(event[1]=='input_error' for event in events)
    assert 'credential-must-not-leak' not in str(events)


def test_actual_getter_exhaustion_raises_instead_of_returning_zero():
    _,read,calls,closed,_=price_reader([ticker(at=AT-timedelta(seconds=91))]*3)
    with pytest.raises(QuoteUnavailable,match='stale_timestamp'):read()
    assert len(calls)==len(closed)==3


def test_actual_getter_can_preserve_timestamp_for_consumers_that_recheck_age():
    _,read,_,_,_=price_reader([ticker(at=AT-timedelta(seconds=90))])
    quote=read(with_timestamp=True)
    assert quote==MarketQuote(100.,AT-timedelta(seconds=90))
    with pytest.raises(QuoteUnavailable,match='stale_timestamp'):checked_quote(quote,AT+timedelta(microseconds=1))


def test_actual_getter_retries_malformed_json_without_exposing_exception_text():
    fn,ns=method(BOT,'get_current_price');events=[];closed=[];calls=[]
    def get(*a,**kw):
        calls.append(1)
        def invalid():raise ValueError('credential-must-not-leak')
        return SimpleNamespace(json=invalid,close=lambda:closed.append(1))
    ns['HTTP_SESSION']=SimpleNamespace(get=get);ns['now_jst']=lambda:AT
    ns['record_bot_event']=lambda *a:events.append(a)
    bot=SimpleNamespace(symbol_mapping={'ltc_jpy':'LTC_JPY'},last_api_call=0,rate_limit_delay=0,logger=logging.getLogger('quote-test'))
    with pytest.raises(QuoteUnavailable,match='invalid_json'):fn(bot,'ltc_jpy')
    assert len(calls)==len(closed)==3 and 'credential-must-not-leak' not in str(events)


def exit_bot(side='long',hours=1):
    bot,_=position_bot();bot.exchange_settings_gmo={'live_trade':True}
    bot.positions={'ltc_jpy':side};bot.entry_prices={'ltc_jpy':100.};bot.entry_sizes={'ltc_jpy':5.}
    bot.entry_times={'ltc_jpy':AT-timedelta(hours=hours)};bot.short_profit_lock_armed={'ltc_jpy':False}
    bot.notification_settings={'enabled':True};bot.notices=[];bot.closes=[]
    bot.send_notification=lambda *a,**kw:bot.notices.append((a,kw)) or True
    bot._close_live_position=lambda *a:bot.closes.append(a)
    levels={'take_profit_price':105. if side=='long' else 95.,'stop_loss_price':90. if side=='long' else 110.}
    bot._last_valid_exit_levels={('ltc_jpy',side,100.):deepcopy(levels)}
    bot.calculate_dynamic_exit_levels=lambda *a:deepcopy(levels)
    for name in ('_check_exit_conditions','_evaluate_exit_event','_apply_short_profit_lock'):
        fn,ns=method(BOT,name);ns['now_jst']=lambda:AT
        setattr(bot,name,MethodType(fn,bot))
    return bot


@pytest.mark.parametrize('side',['long','short'])
@pytest.mark.parametrize('invalid',BAD_PRICES)
def test_invalid_current_price_cannot_use_bar_extremes_or_change_protection(side,invalid):
    bot=exit_bot(side);bot.get_current_price=lambda _:invalid
    before=deepcopy(bot._last_valid_exit_levels)
    # Both exits are crossed by historical extremes; an invalid quote blocks them.
    data=pd.DataFrame({'high':[120.],'low':[80.]})
    bot._check_exit_conditions('ltc_jpy',None,None,data)
    assert not bot.closes and bot.positions['ltc_jpy']==side
    assert bot._last_valid_exit_levels==before and not bot.short_profit_lock_armed['ltc_jpy']
    assert len(bot.notices)==1 and bot.notices[0][0][2]=='error'


def test_unavailable_alert_is_deduplicated_and_valid_quote_reports_recovery(caplog):
    caplog.set_level(logging.INFO)
    bot=exit_bot()
    def unavailable(_):raise QuoteUnavailable('stale_timestamp')
    bot.get_current_price=unavailable
    for _ in range(3):bot._check_exit_conditions('ltc_jpy',None,None,pd.DataFrame())
    assert len(bot.notices)==1 and not bot.closes
    assert sum('exit quote unavailable' in r.message for r in caplog.records)==1
    bot.get_current_price=lambda _:106.
    bot._check_exit_conditions('ltc_jpy',None,None,pd.DataFrame())
    assert len(bot.closes)==1 and bot.closes[0][1]=='利益確定'
    assert [notice[0][2] for notice in bot.notices]==['error','info']
    assert not bot._exit_quote_failures


def test_failed_alert_enqueue_is_retried_with_same_outage_identity():
    bot=exit_bot();bot.get_current_price=lambda _:0;calls=[]
    def send(*a,**kw):calls.append(kw['notification_key']);return len(calls)>1
    bot.send_notification=send
    for _ in range(3):bot._check_exit_conditions('ltc_jpy',None,None,pd.DataFrame())
    assert len(calls)==2 and calls[0]==calls[1] and not bot.closes


def test_backtest_cannot_clear_live_outage_state_or_send_its_alerts():
    from backtest_data import backtest_state
    bot=exit_bot();bot.total_profit=0.;bot.get_current_price=lambda _:0
    bot._check_exit_conditions('ltc_jpy',None,None,pd.DataFrame())
    original=bot._exit_quote_failures;before=deepcopy(original)
    assert len(bot.notices)==1
    with backtest_state(bot):
        bot.entry_sizes['ltc_jpy']=5.
        assert not bot._exit_quote_failures
        bot._check_exit_conditions('ltc_jpy',None,None,pd.DataFrame())
        bot._reset_exit_adjustments('ltc_jpy')
        assert len(bot.notices)==1
    assert bot._exit_quote_failures is original and original==before


def test_near_threshold_retry_does_not_reuse_previous_price_after_failure():
    bot=exit_bot();quotes=iter([104.99,QuoteUnavailable('stale_timestamp'),106.]);reads=[];evaluated=[]
    def read(_):
        value=next(quotes);reads.append(value)
        if isinstance(value,Exception):raise value
        return value
    bot.get_current_price=read;original=bot._evaluate_exit_event
    def evaluate(*a,**kw):evaluated.append(a[-1]);return original(*a,**kw)
    bot._evaluate_exit_event=evaluate
    bot._check_exit_conditions('ltc_jpy',None,None,pd.DataFrame())
    assert len(reads)==2 and evaluated==[104.99] and not bot.closes


def test_near_threshold_valid_observation_still_closes_normally():
    bot=exit_bot();quotes=iter([104.99,105.01]);bot.get_current_price=lambda _:next(quotes)
    bot._check_exit_conditions('ltc_jpy',None,None,pd.DataFrame())
    assert len(bot.closes)==1 and bot.closes[0][1]=='利益確定'


@pytest.mark.parametrize('hours,expected',[(71.99,False),(72,True)])
def test_holding_time_exit_is_independent_of_price_availability(hours,expected):
    bot=exit_bot(hours=hours);bot.get_current_price=lambda _:None
    bot._check_exit_conditions('ltc_jpy',None,None,pd.DataFrame({'high':[120.],'low':[80.]}))
    assert bool(bot.closes) is expected
    if expected:assert bot.closes[0][1]=='長時間保有による強制決済'


def test_use_time_recheck_rejects_quote_that_aged_after_acquisition():
    bot=exit_bot();bot.get_current_price=lambda _:MarketQuote(106.,AT-timedelta(seconds=90,microseconds=1))
    bot._check_exit_conditions('ltc_jpy',None,None,pd.DataFrame())
    assert not bot.closes and bot._exit_quote_failures['ltc_jpy']['reason']=='stale_timestamp'


@pytest.mark.parametrize('quote,expected',[(MarketQuote(101.,AT-timedelta(seconds=10)),True),
    (MarketQuote(float('inf'),AT),False),(MarketQuote(101.,AT-timedelta(seconds=91)),False),
    (MarketQuote(101.,AT+timedelta(seconds=1)),False),(101.,False)])
def test_cache_requires_verified_metadata_and_keeps_exchange_timestamp(quote,expected):
    owner=next(n for n in TREE.body if isinstance(n,ast.ClassDef) and n.name=='PriceCacheRefresher')
    fn,ns=method(owner,'run_forever');saved=[]
    class Clock:
        @staticmethod
        def now(tz):return AT
    ns['dt']=SimpleNamespace(datetime=Clock);ns['timezone']=timezone
    ns['upsert_price_cache']=lambda *a:saved.append(a)
    refresher=SimpleNamespace(symbols=['ltc_jpy'],get_price=lambda _:quote,interval=30,logger=logging.getLogger('quote-test'))
    with patch('time.sleep',side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):fn(refresher)
    assert bool(saved) is expected
    if expected:assert saved==[('ltc_jpy',101.,quote.quoted_at)]


@pytest.mark.parametrize('side',['BUY','SELL'])
def test_real_close_path_still_finishes_time_exit_once_when_all_quotes_are_stale(store,side):
    svc,api=service(store);svc.submit('LTC_JPY',side,5)
    bot=bot_for(svc);bot._sync_live_positions();bot.exchange_settings_gmo={'live_trade':True}
    bot.entry_times['ltc_jpy']=AT-timedelta(hours=73)
    bot.calculate_dynamic_exit_levels=lambda *a:{'take_profit_price':105. if side=='BUY' else 95.,'stop_loss_price':90. if side=='BUY' else 110.}
    closed=[];gets=[]
    for name in ('get_current_price','_get_exit_quote','_check_exit_conditions','_evaluate_exit_event','_apply_short_profit_lock'):
        fn,ns=method(BOT,name);ns['now_jst']=lambda:AT
        if name=='get_current_price':
            def get(*a,**kw):
                gets.append(1)
                return SimpleNamespace(json=lambda:ticker(at=AT-timedelta(seconds=91)),raise_for_status=lambda:None,close=lambda:closed.append(1))
            ns['HTTP_SESSION']=SimpleNamespace(get=get)
        setattr(bot,name,MethodType(fn,bot))
    bot.last_api_call=0;bot.rate_limit_delay=0
    bot._check_exit_conditions('ltc_jpy',None,None,pd.DataFrame())
    assert len(gets)==len(closed)==3 and len(api.posts)==2 and not api.position_rows
    assert not svc.store.pending('test') and not svc.store.close_batches('test')
    assert not bot._exit_quote_failures
    bot._check_exit_conditions('ltc_jpy',None,None,pd.DataFrame())
    assert len(api.posts)==2
