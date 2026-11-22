"""Exercise display failures through the actual live loop and fake order ledger."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import logging
import sys
from types import MethodType,SimpleNamespace

import pandas as pd
import pytest
import requests

from market_quotes import QuoteUnavailable
from test_bot_execution_integration import BOT,method
from test_execution import records,service
from test_execution_recovery import bot_for
from test_exit_signals import position_bot
from trading_execution import ExchangeMaintenance


AT=datetime(2026,11,22,12,13,tzinfo=timezone(timedelta(hours=9)))
SYMBOLS=['ltc_jpy','xrp_jpy']


def display_bot():
    bot,_=position_bot()
    bot.symbols=SYMBOLS
    bot.positions={symbol:'long' for symbol in SYMBOLS}
    bot.entry_prices={symbol:100. for symbol in SYMBOLS}
    bot.entry_times={symbol:AT-timedelta(hours=1) for symbol in SYMBOLS}
    fn,ns=method(BOT,'_display_current_positions')
    ns['dt']=SimpleNamespace(datetime=SimpleNamespace(now=lambda tz:AT))
    bot._display_current_positions=MethodType(fn,bot)
    return bot


@pytest.mark.parametrize('error',[QuoteUnavailable('stale_timestamp'),requests.Timeout('secret-must-not-leak'),RuntimeError('secret-must-not-leak')])
def test_one_display_quote_failure_does_not_skip_next_symbol_or_change_state(error,caplog):
    caplog.set_level(logging.INFO)
    bot=display_bot();visited=[]
    before=deepcopy((bot.positions,bot.entry_prices,bot.entry_times))
    def price(symbol):
        visited.append(symbol)
        if symbol==SYMBOLS[0]:raise error
        return 101.
    bot.get_current_price=price
    bot._display_current_positions()
    assert visited==SYMBOLS and (bot.positions,bot.entry_prices,bot.entry_times)==before
    assert any('xrp_jpy: long (' in item.message for item in caplog.records)
    assert 'secret-must-not-leak' not in caplog.text
    assert not any(item.levelno>=logging.ERROR for item in caplog.records)


@pytest.mark.parametrize('price',[None,0,float('nan'),float('inf'),True])
def test_invalid_display_price_is_not_shown_as_profit_or_zero_valuation(price,caplog):
    caplog.set_level(logging.INFO);bot=display_bot();visited=[]
    def read(symbol):visited.append(symbol);return price if symbol==SYMBOLS[0] else 101.
    bot.get_current_price=read;bot._display_current_positions()
    assert visited==SYMBOLS
    assert any('ltc_jpy' in item.message and '未表示' in item.message for item in caplog.records)
    assert not any(item.message.startswith('ltc_jpy:') and '%' in item.message for item in caplog.records)


def test_bad_display_metadata_does_not_hide_other_positions(caplog):
    bot=display_bot();bot.entry_times['ltc_jpy']='invalid-time';visited=[]
    def price(symbol):visited.append(symbol);return 101.
    bot.get_current_price=price;bot._display_current_positions()
    assert visited==SYMBOLS and 'position display unavailable (TypeError)' in caplog.text
    assert bot.entry_times['ltc_jpy']=='invalid-time'


def test_confirmed_maintenance_still_propagates_to_shared_wait_path():
    bot=display_bot();visited=[]
    def unavailable(symbol):visited.append(symbol);raise ExchangeMaintenance()
    bot.get_current_price=unavailable
    with pytest.raises(ExchangeMaintenance):bot._display_current_positions()
    assert visited==['ltc_jpy']


def wire_live_loop(bot, *, cycles=2):
    clock={'seconds':0,'cycles':0};actions=[];waits=[]
    def now():return AT+timedelta(seconds=clock['seconds'])
    def sleep(seconds):
        waits.append(seconds);clock['seconds']+=seconds;clock['cycles']+=1
        if clock['cycles']>=cycles:raise KeyboardInterrupt()
    bot.initial_capital=1000.;bot.total_profit=0.;bot.test_mode=True;bot.is_backtest=False
    bot.exchange_settings_gmo={'live_trade':True}
    bot._verify_api_connection=lambda:True;bot._exchange_status_ready=lambda:True
    bot.get_total_balance=lambda:1000.
    bot.update_sr_zones_for_symbol=lambda *a,**kw:None
    bot._generate_final_report=lambda *a:None;bot.save_trade_logs_to_excel=lambda *a:None
    bot.check_backup_needed=lambda:None;bot.check_monthly_increase=lambda:None
    bot._store_recent_candles=lambda *a:None;bot._entry_direction=lambda *a:None
    data=pd.DataFrame({'timestamp':[AT-timedelta(minutes=45),AT-timedelta(minutes=30),AT-timedelta(minutes=15)],
                       'high':[101.]*3,'low':[99.]*3,'close':[100.]*3})
    def inputs(symbol,*a):actions.append(('monitor',symbol,clock['cycles']));return data.copy(),pd.DataFrame()
    bot._prepare_live_inputs=inputs
    bot.calculate_dynamic_exit_levels=lambda symbol,df,side,entry:{
        'take_profit_price':105. if side=='long' else 95.,
        'stop_loss_price':90. if side=='long' else 110.}
    for name in ('_display_current_positions','_check_exit_conditions','_evaluate_exit_event','_apply_short_profit_lock'):
        fn,ns=method(BOT,name);ns['now_jst']=now
        ns['dt']=SimpleNamespace(datetime=SimpleNamespace(now=lambda tz:now()))
        ns['record_bot_event']=lambda *a:None
        setattr(bot,name,MethodType(fn,bot))
    run,ns=method(BOT,'run_live');ns['now_jst']=now
    ns['dt']=SimpleNamespace(datetime=SimpleNamespace(now=lambda tz:now()))
    ns['time']=SimpleNamespace(time=lambda:clock['seconds'],sleep=sleep)
    def no_record(*a):bot._record_inputs=False  # No files, HTTP, SMTP or real bot startup.
    ns['record_bot_event']=no_record;ns['close_recorder']=lambda *a:None
    return lambda:run(bot),actions,waits


@pytest.mark.parametrize('failed_summary',['positions','balance','both'])
def test_actual_loop_isolates_whole_summary_failures_and_completes_next_cycle(failed_summary,caplog):
    caplog.set_level(logging.INFO)
    bot=display_bot();bot.entry_sizes={symbol:5. for symbol in SYMBOLS}
    bot.short_profit_lock_armed={};bot.reentry_block_until={};bot.send_notification=lambda *a,**kw:None
    bot.verify_positions=lambda:None;bot._refresh_live_stats=lambda *a:None
    run,actions,waits=wire_live_loop(bot)
    bot.get_current_price=lambda _:100.
    def bad_positions():actions.append(('display',));raise RuntimeError('secret-must-not-leak')
    original=bot._display_current_positions
    if failed_summary in ('positions','both'):bot._display_current_positions=bad_positions
    else:
        def report():actions.append(('display',));return original()
        bot._display_current_positions=report
    balance_calls=[]
    def balance():
        balance_calls.append(1)
        if len(balance_calls)>1 and failed_summary in ('balance','both'):raise RuntimeError('secret-must-not-leak')
        return 1000.
    bot.get_total_balance=balance
    run()
    assert [item[:2] for item in actions if item[0]=='monitor']==[('monitor',symbol) for _ in range(2) for symbol in SYMBOLS]
    assert actions[:2]==[('monitor','ltc_jpy',0),('monitor','xrp_jpy',0)]
    assert len(waits)==2 and all(0<seconds<=28 for seconds in waits)
    assert sum('次のサイクルまで' in item.message for item in caplog.records)==2
    assert not any(item.levelno>=logging.ERROR for item in caplog.records)
    assert 'secret-must-not-leak' not in caplog.text


def test_maintenance_from_summary_uses_shared_wait_instead_of_normal_error_path(caplog):
    bot=display_bot();bot.entry_sizes={symbol:5. for symbol in SYMBOLS}
    bot.short_profit_lock_armed={};bot.reentry_block_until={};bot.send_notification=lambda *a,**kw:None
    bot.verify_positions=lambda:None;bot._refresh_live_stats=lambda *a:None
    run,actions,waits=wire_live_loop(bot)
    bot.get_current_price=lambda _:100.;checks=[]
    gate=SimpleNamespace(blocked=False,seconds_until_probe=lambda:60)
    bot.gmo_api.availability=gate
    bot._queue_close_alerts=lambda:checks.append('maintenance_alert_check')
    def summary():gate.blocked=True;raise ExchangeMaintenance()
    bot._display_current_positions=summary
    run()
    assert [item[:2] for item in actions]==[('monitor',symbol) for symbol in SYMBOLS]
    assert waits==[1,30] and len(checks)==2
    assert not any(item.levelno>=logging.ERROR for item in caplog.records)


@pytest.mark.parametrize('side',['BUY','SELL'])
@pytest.mark.parametrize('expired_bad_quote',[False,True])
def test_actual_loop_protects_other_symbol_and_closes_expired_position_without_price(store,monkeypatch,side,expired_bad_quote,caplog):
    monkeypatch.setitem(sys.modules,'db',SimpleNamespace(engine=store.engine))
    svc,api=service(store);original_public=api._public_request
    def public(path,params=None):
        response=original_public(path,params)
        if path=='/v1/symbols':response['data'].append({**response['data'][0],'symbol':'XRP_JPY'})
        return response
    api._public_request=public
    api.now=(AT-timedelta(hours=73 if expired_bad_quote else 1)).isoformat()
    svc.submit('LTC_JPY',side,5,context={'bot_symbol':'ltc_jpy'})
    api.now=(AT-timedelta(hours=1)).isoformat()
    svc.submit('XRP_JPY',side,5,context={'bot_symbol':'xrp_jpy'})
    api.now=AT.isoformat()
    bot=bot_for(svc);bot.symbols=SYMBOLS
    bot.symbol_mapping={'ltc_jpy':'LTC_JPY','xrp_jpy':'XRP_JPY'}
    bot._sync_live_positions()
    direction='long' if side=='BUY' else 'short'
    bot.opposite_narrow_state={'ltc_jpy':{direction:True}}
    levels={'take_profit_price':105. if side=='BUY' else 95.,'stop_loss_price':90. if side=='BUY' else 110.}
    bot._last_valid_exit_levels={('ltc_jpy',direction,100.):deepcopy(levels)}
    if side=='SELL':bot.short_profit_lock_armed['ltc_jpy']=True
    run,actions,waits=wire_live_loop(bot)
    reads=[]
    def price(symbol):
        reads.append(symbol)
        if symbol=='ltc_jpy':raise QuoteUnavailable('stale_timestamp')
        return 85. if side=='BUY' else 115.
    bot.get_current_price=price
    def verify():bot._recover_live_orders();bot._sync_live_positions()
    bot.verify_positions=verify
    run()
    assert len(api.posts)==(4 if expired_bad_quote else 3)
    assert not store.pending('test') and not store.close_batches('test')
    trades=records(store,'trades')
    assert len(trades)==(2 if expired_bad_quote else 1)
    assert all(float(row['exit_price'])==100. for row in trades)  # Actual mock fills, not displayed quotes.
    xrp=next(row for row in trades if row['symbol']=='xrp_jpy')
    assert xrp['raw']['exit_reason']=='損切り'
    if expired_bad_quote:
        ltc=next(row for row in trades if row['symbol']=='ltc_jpy')
        assert ltc['raw']['exit_reason']=='長時間保有による強制決済' and not api.position_rows
    else:
        assert bot.positions['ltc_jpy']==direction and bot.entry_sizes['ltc_jpy']==5
        assert bot._last_valid_exit_levels[('ltc_jpy',direction,100.)]==levels
        assert bot.opposite_narrow_state['ltc_jpy'][direction]
        assert any(item[:2]==('monitor','ltc_jpy') and item[2]==1 for item in actions)
    assert len(waits)==2 and all(0<seconds<=28 for seconds in waits)
    assert not any(item.levelno>=logging.ERROR for item in caplog.records)
