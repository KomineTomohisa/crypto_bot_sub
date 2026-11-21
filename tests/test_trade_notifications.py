"""Receipts must explain actual fills, without extra exchange I/O or guessed P/L."""
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
from types import MethodType

import pandas as pd
import pytest

from trade_notifications import compose_execution_notice, entry_snapshot
from test_bot_execution_integration import BOT, method
from test_execution import service, records
from test_execution_recovery import bot_for, two_positions
from trading_execution import ExecutionError
from notification_outbox import NotificationOutboxWorker


AT = datetime(2026, 11, 21, 3, tzinfo=timezone.utc)


def receipt(settle='CLOSE', side='SELL'):
    context = {'exit_reason': '損切り', 'decision_bar': '2026-11-21T11:30:00+09:00',
               'decision_price': '100', 'scores': {'buy_score': .9, 'sell_score': .7,
                   'buy_score_scaled': .8, 'sell_score_scaled': .6, 'macd_score_long': .85, 'macd_score_short': .4},
               'notification': {'indicators': {'RSI': 60, 'ADX': 30, 'ATR': 1},
                   'signals': {'sell_signal': True, 'buy_signal': True},
                   'previous_bar': '2026-11-21T11:15:00+09:00'}}
    return {'finalized': True, 'success': True, 'status': 'EXECUTED',
            'client_order_id': 'fake-client-order', 'order_id': '10001',
            'executed_size': 2., 'execution_price': 110., 'executed_at': AT, 'execution_fee': Decimal('.2'),
            'intent': {'symbol': 'LTC_JPY', 'settle_type': settle, 'side': side,
                       'status': 'EXECUTED', 'size': Decimal(2), 'context': context},
            'trades': [{'entry_price': Decimal(100), 'size': Decimal(2), 'pnl': Decimal(20),
                        'holding_hours': Decimal(2), 'closed_at': AT,
                        'entry_fee': Decimal('.1'), 'exit_fee': Decimal('.2'),
                        'net_pnl_before_carry': Decimal('19.7')}] if settle=='CLOSE' else []}


@pytest.mark.parametrize('settle,side,direction,action', [
    ('OPEN', 'BUY', 'ロング', 'エントリー'), ('OPEN', 'SELL', 'ショート', 'エントリー'),
    ('CLOSE', 'SELL', 'ロング', '決済'), ('CLOSE', 'BUY', 'ショート', '決済')])
def test_receipts_describe_position_direction_and_actual_fills(settle, side, direction, action):
    result=receipt(settle, side); before=deepcopy(result)
    subject,body,kind=compose_execution_notice(result)
    assert direction in subject and action in subject and kind==('entry' if settle=='OPEN' else 'exit')
    assert '平均約定価格: 110 円' in body and '約定数量: 2 LTC' in body
    assert '2026-11-21 12:00:00 JST' in body and result==before


def test_entry_uses_sell_score_for_long_and_buy_score_for_short():
    for side,expected in [('BUY','0.6'),('SELL','0.8')]:
        _,body,_=compose_execution_notice(receipt('OPEN',side))
        assert f'採用方向の総合スコア: {expected}' in body
        assert '確定足2本で同方向のシグナルが連続' in body
        assert 'RSI: 60' in body and '新規注文の取引手数料: 0.2 円' in body


def test_close_pnl_is_ledger_based_weighted_and_costs_are_separate():
    result=receipt()
    other=deepcopy(result['trades'][0]);other.update(entry_price=Decimal(200),size=Decimal(1),pnl=Decimal(-5),
        entry_fee=Decimal('.3'),exit_fee=Decimal('-.1'),net_pnl_before_carry=Decimal('-5.2'),holding_hours=Decimal(3))
    result['trades'].append(other)
    subject,body,_=compose_execution_notice(result, remaining=Decimal(4))
    assert '売買損益 +15円' in subject and '売買損益（手数料控除前）: +15 円' in body
    assert '決済分の平均建値: 133.33333333 円' in body and '売買損益率: +3.75 %' in body
    assert '配賦済み新規手数料: 0.4 円' in body and '決済手数料: 0.1 円' in body
    assert '手数料控除後損益（日次費用を除く）: +14.5 円' in body
    assert '保有時間: 2〜3 時間' in body and '元の決済対象の残数量: 4 LTC' in body


@pytest.mark.parametrize('field', ['entry_fee','exit_fee','net_pnl_before_carry'])
def test_unknown_cost_is_not_presented_as_zero_or_final_net(field):
    result=receipt();result['trades'][0][field]=None
    _,body,_=compose_execution_notice(result)
    label={'entry_fee':'配賦済み新規手数料','exit_fee':'決済手数料','net_pnl_before_carry':'手数料控除後損益（日次費用を除く）'}[field]
    assert f'{label}: 未確定' in body and '日次費用は取引別に配賦していません' in body


def test_partial_canceled_order_is_not_announced_as_fully_executed():
    result=receipt();result['executed_size']=1.;result['status']=result['intent']['status']='CANCELED'
    subject,body,_=compose_execution_notice(result)
    assert '一部約定' in subject and '注文数量: 2 LTC' in body and '約定数量: 1 LTC' in body
    assert '残りの注文数量は約定していません' in body


@pytest.mark.parametrize('settle,side',[('OPEN','BUY'),('CLOSE','BUY')])
def test_zero_fill_is_not_reported_as_a_successful_trade(settle,side):
    result=receipt(settle,side);result.update(executed_size=0.,success=False,execution_price=None,trades=[])
    subject,body,_=compose_execution_notice(result,recovered=True)
    assert '約定なし' in subject and '約定なしで終了' in body
    assert '平均約定価格' not in body and '売買損益（手数料控除前）' not in body


def test_old_context_and_missing_trade_fields_remain_readable_without_invented_reasons():
    result=receipt('OPEN','BUY');result['intent']['context']={}
    _,body,_=compose_execution_notice(result,recovered=True)
    assert '応答待ち後の注文照合' in body and '旧形式の注文' in body
    assert '確定足2本' not in body and '採用方向の総合スコア: 未取得' in body
    result=receipt();result['trades']=[{'pnl':None,'size':None}]
    _,body,_=compose_execution_notice(result)
    assert '売買損益（手数料控除前）: 未取得 円' in body


def test_snapshot_and_output_whitelist_data_and_keep_small_prices():
    snapshot=pd.Series({'sell_signal':True,'buy_signal':False,'password':'never-copy'})
    saved=entry_snapshot(snapshot,{'rsi':60,'atr':float('nan'),'password':'never-copy'})
    assert 'password' not in str(saved) and saved['indicators']['ATR'] is None
    result=receipt('OPEN','BUY');result['execution_price']=.00001234
    result['intent']['context']['notification']['indicators']['password']='never-copy'
    _,body,_=compose_execution_notice(result)
    assert '平均約定価格: 0.00001234 円' in body and 'never-copy' not in body


def test_unfinalized_order_cannot_be_rendered_as_confirmed():
    result=receipt();result['finalized']=False
    with pytest.raises(ValueError):compose_execution_notice(result)


@pytest.mark.parametrize('size',[None,float('nan'),-1,3])
def test_invalid_execution_quantity_cannot_be_presented_as_zero_or_completed(size):
    result=receipt();result['executed_size']=size
    with pytest.raises(ValueError):compose_execution_notice(result)


def notified_bot(svc):
    bot=bot_for(svc);bot.notification_settings={'enabled':True,'send_on_entry':True,'send_on_exit':True,'send_on_error':False}
    bot.mail=[]
    bot.send_notification=lambda *a:bot.mail.append(a) or True
    bot._configured_trade_channels=lambda:['email']
    bot._prepare_notification_route=lambda channel:['fixture@example.invalid']
    bot._deliver_outbox_notification=lambda channel,subject,body,kind,*rest:bot.send_notification(subject,body,kind)
    # These existing tests isolate receipt content/routing. Dedicated outbox
    # tests use the real background dispatcher, blocked transports and restarts.
    def drain():
        if not bot.notification_settings.get('enabled',False):return
        worker=getattr(bot,'_trade_notification_worker',None)
        if worker is None:
            worker=NotificationOutboxWorker(svc.store,svc.account,bot._outbox_enabled,
                bot._prepare_notification_route,bot._deliver_outbox_notification,bot.logger,poll_seconds=3600)
            bot._trade_notification_worker=worker
        for _ in range(8):
            if not worker.process_one():break
    bot._queue_trade_notifications=drain
    return bot


def open_order(bot, **kw):
    return bot._execute_live_entry('ltc_jpy','buy',10,
        signal_raw={'decision_bar':'2026-11-21T11:30:00+09:00','decision_price':'88'},rsi=60,atr=1,adx=30,**kw)


def test_immediate_entry_and_repeated_decision_send_one_receipt(store):
    svc,api=service(store);bot=notified_bot(svc)
    bot.entry_scores['ltc_jpy']={'sell_score_scaled':.7,'buy_score_scaled':.9}
    bot._entry_decision_snapshot=pd.Series({'sell_signal':True})
    result=open_order(bot)
    assert result['success'] and len(bot.mail)==1 and len(api.posts)==1
    assert '平均約定価格: 100 円' in bot.mail[0][1] and '採用方向の総合スコア: 0.7' in bot.mail[0][1]
    assert result['execution_fee']==Decimal(1)
    assert records(store,'bot_order_intents')[0]['context']['notification']['indicators']['RSI']==60
    assert not bot._notify_live_execution(result) and len(bot.mail)==1
    assert not open_order(bot)['success'] and len(bot.mail)==1 and len(api.posts)==1


def test_lost_response_entry_uses_saved_snapshot_in_recovery_receipt(store):
    svc,api=service(store);bot=notified_bot(svc)
    bot._entry_decision_snapshot=pd.Series({'sell_signal':True})
    original=api._request
    api.lose_response=True
    def blocked(method,path,params=None,data=None):
        if path=='/v1/orders':return {'status':-1}
        return original(method,path,params,data)
    api._request=blocked
    with pytest.raises(ExecutionError):open_order(bot)
    assert not bot.mail and len(api.posts)==1
    api._request=original
    restarted=notified_bot(svc)
    restarted._recover_live_orders()
    assert len(restarted.mail)==1 and '応答待ち後' in restarted.mail[0][1]
    assert 'RSI: 60' in restarted.mail[0][1] and '平均約定価格: 100 円' in restarted.mail[0][1]
    restarted._recover_live_orders()
    assert len(restarted.mail)==1 and len(api.posts)==1


@pytest.mark.parametrize('position_side,opposite',[('BUY','SELL'),('SELL','BUY')])
def test_split_close_receipts_use_recorded_trades_after_positions_are_cleared(store,position_side,opposite):
    svc,api=service(store);two_positions(api)
    for row in api.position_rows:row['side']=position_side
    bot=notified_bot(svc);bot._sync_live_positions()
    api.fill_price=Decimal(110)
    bot._close_live_position('ltc_jpy','利益確定',None,[])
    assert len(api.posts)==2 and all(p['side']==opposite for p in api.posts)
    assert bot.positions['ltc_jpy'] is None and len(bot.mail)==2
    assert all('決済理由: 利益確定' in mail[1] and '決済分の平均建値: 90 円' in mail[1] for mail in bot.mail)
    assert all('配賦済み新規手数料: 未確定' in mail[1] for mail in bot.mail)
    assert all('通知範囲: この決済注文' in mail[1] for mail in bot.mail)
    assert all(mail[2]=='exit' for mail in bot.mail)
    bot._recover_live_orders()
    assert len(bot.mail)==2


@pytest.mark.parametrize('flag',['enabled','send_on_entry'])
def test_entry_settings_are_honored(store,flag):
    svc,api=service(store);bot=notified_bot(svc);bot.notification_settings[flag]=False
    assert open_order(bot)['success'] and not bot.mail


def test_delivery_failure_cannot_change_recorded_fills_or_cause_a_second_post(store,caplog):
    svc,api=service(store);bot=notified_bot(svc)
    def fail(*a):raise RuntimeError('must-not-leak-secret')
    bot.send_notification=fail
    result=open_order(bot)
    assert result['success'] and len(api.posts)==1 and len(records(store,'bot_execution_fills'))==1
    assert 'must-not-leak-secret' not in caplog.text


def test_entry_direction_confirmation_is_saved_only_for_matching_bar(store):
    svc,api=service(store);bot=notified_bot(svc);bot.reentry_block_until={}
    fn,_=method(BOT,'_entry_direction');bot._entry_direction=MethodType(fn,bot)
    latest=pd.Series({'timestamp':pd.Timestamp('2026-11-21T11:30:00+09:00'),'sell_signal':True,'buy_signal':False})
    previous=latest.copy();previous['timestamp']-=pd.Timedelta(minutes=15)
    assert bot._entry_direction('ltc_jpy',latest,previous,pd.Timestamp('2026-11-21T11:45:00+09:00'))=='long'
    result=open_order(bot)
    assert 'previous_bar' in result['intent']['context']['notification']
    assert '確定足2本' in bot.mail[0][1]


@pytest.mark.parametrize('position_side',['BUY','SELL'])
def test_late_close_confirmation_keeps_reason_prices_and_partial_scope(store,position_side):
    svc,api=service(store);two_positions(api)
    for row in api.position_rows:row['side']=position_side
    bot=notified_bot(svc);bot._sync_live_positions()
    original=api._request;api.lose_response=True
    def blocked(method,path,params=None,data=None):
        if path=='/v1/orders':return {'status':-1}
        return original(method,path,params,data)
    api._request=blocked
    with pytest.raises(ExecutionError):bot._close_live_position('ltc_jpy','損切り',None,None)
    assert not bot.mail and len(api.posts)==1
    api._request=original;api.lose_response=False
    restarted=notified_bot(svc);restarted._recover_live_orders()
    assert len(restarted.mail)==2 and len(api.posts)==2
    assert '応答待ち後' in restarted.mail[0][1] and '決済理由: 損切り' in restarted.mail[0][1]
    assert '元の決済対象の残数量: 6 LTC' in restarted.mail[0][1]
    assert '元の決済対象の残数量: 0 LTC' in restarted.mail[1][1]
    assert not store.pending('test') and not store.close_batches('test')


def test_actual_entry_handler_uses_rich_receipt_once_without_legacy_template(store):
    svc,api=service(store);bot=notified_bot(svc)
    bot.exchange_settings_gmo={'live_trade':True};bot.initial_capital=200000;bot.total_profit=0
    bot.adjust_position_risk=lambda *a:(10,1000);bot.get_current_price=lambda _:999
    bot.check_sufficient_funds=lambda *a:True;bot._should_block_entry_by_sr=lambda *a:(False,'')
    bot._build_entry_explanation=lambda **kw:({},'',{})
    bot._current_strategy_version=lambda:'test';bot.log_entry=lambda *a:None
    bot.execute_order_with_confirmation=lambda symbol,side,size,**kw:bot._execute_live_entry(symbol,side,size,**kw)
    bot._get_margin_positions_safe=lambda _: {'status':0,'data':{'list':[]}}
    fn,ns=method(BOT,'_handle_entry')
    ns['compose_signal_message']=lambda *a:(_ for _ in ()).throw(AssertionError('legacy notification must not run'))
    signal=pd.Series({'RSI':60.,'CCI':100.,'ATR':1.,'ADX':30.,'plus_di14':40.,'minus_di14':10.,
        'EMA_short':100.,'EMA_long':99.,'buy_score':.9,'sell_score':.7,'sell_signal':True,
        'close':88.,'timestamp':pd.Timestamp('2026-11-21T11:30:00+09:00')})
    fn(bot,'ltc_jpy','long',signal,{},[])
    assert len(bot.mail)==1 and '平均約定価格: 100 円' in bot.mail[0][1]
    assert '判断足の終値: 88 円' in bot.mail[0][1] and '採用方向の総合スコア: 0.7' in bot.mail[0][1]
    assert len(api.posts)==1 and bot.entry_prices['ltc_jpy']==100


def test_formatting_failure_does_not_undo_entry_or_repeat_post(store,monkeypatch):
    svc,api=service(store);bot=notified_bot(svc)
    def fail(*a,**kw):raise ValueError('invalid display')
    monkeypatch.setattr('notification_outbox.compose_execution_notice',fail)
    assert open_order(bot)['success']
    assert len(api.posts)==len(records(store,'bot_execution_fills'))==1
    assert not bot.mail


@pytest.mark.parametrize('change',[{'finalized':False},{'already_submitted':True}])
def test_nonfinal_and_already_processed_results_do_not_send(store,change):
    svc,_=service(store);bot=notified_bot(svc);result=receipt();result.update(change)
    assert bot._notify_live_execution(result) is False and not bot.mail


def test_old_or_mismatched_confirmation_is_not_presented_as_two_candles(store):
    svc,api=service(store);bot=notified_bot(svc)
    bot._entry_notification_context={'ltc_jpy':{'direction':'long','decision_bar':'2026-11-20T11:30:00+09:00','previous_bar':'2026-11-20T11:15:00+09:00'}}
    result=open_order(bot)
    assert 'previous_bar' not in result['intent']['context']['notification']
    assert '確定足2本' not in bot.mail[0][1]
