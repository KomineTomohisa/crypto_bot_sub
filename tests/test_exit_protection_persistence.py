"""Restart the actual exit/sync methods against files and a fake exchange."""
from decimal import Decimal
import json
import logging
from types import MethodType

import pandas as pd
import pytest

from backtest_data import backtest_state
from test_bot_execution_integration import BOT, method
from test_execution import FakeBot, service
from test_exit_signals import position_bot, position, frame
from strategy_decisions import next_entry_time


SYMBOL='ltc_jpy'


def file_bot(tmp_path, svc=None):
    if svc is None:
        original, _ = position_bot()
        svc=original._live_execution_service
    bot=FakeBot(svc)
    bot.base_dir=str(tmp_path)
    bot.exchange_settings_gmo={'live_trade':True}
    bot.source='real'
    bot.total_profit=0.
    bot._ceil_to_next_15min=next_entry_time
    for name in ('calculate_dynamic_exit_levels','_get_dynamic_profit_loss_settings','_apply_short_profit_lock'):
        fn,_=method(BOT,name)
        setattr(bot,name,MethodType(fn,bot))
    bot._apply_sr_exit_levels=lambda **kw: {'take_profit_price':kw['base_take_profit'],'stop_loss_price':kw['base_stop_loss']}
    return bot,svc.gateway.api


def calculate(bot, *, opposite=True):
    data=frame()
    side=bot.positions[SYMBOL]
    if opposite:data['buy_signal' if side=='long' else 'sell_signal']=[True,False,True,True,False]
    return bot.calculate_dynamic_exit_levels(SYMBOL,data,side,bot.entry_prices[SYMBOL])


def restart(bot,tmp_path):
    other,_=file_bot(tmp_path,bot._live_execution_service)
    other._sync_live_positions()
    return other


@pytest.mark.parametrize('side',['BUY','SELL'])
def test_calculation_is_saved_immediately_and_missing_atr_uses_restored_levels(tmp_path,side):
    bot,api=file_bot(tmp_path)
    api.position_rows=[position(api,'1',side=side)]
    bot._sync_live_positions()
    expected=calculate(bot)
    if side=='SELL':
        bot._apply_short_profit_lock(SYMBOL,'short',100.,99.,expected['stop_loss_price'])
    recovered=restart(bot,tmp_path)
    direction='long' if side=='BUY' else 'short'
    assert recovered.opposite_narrow_state[SYMBOL][direction] is True
    assert recovered.short_profit_lock_armed[SYMBOL] is (side=='SELL')
    assert recovered.calculate_dynamic_exit_levels(SYMBOL,pd.DataFrame(),direction,100.)==expected
    assert not api.posts


def test_restored_narrowing_can_unlock_and_new_state_survives_second_restart(tmp_path):
    bot,api=file_bot(tmp_path);api.position_rows=[position(api,'1')]
    bot._sync_live_positions();calculate(bot)
    recovered=restart(bot,tmp_path)
    same=frame(buy=[False,False,True,True,False])
    unlocked=recovered.calculate_dynamic_exit_levels(SYMBOL,same,'short',100.)
    assert not recovered.opposite_narrow_state[SYMBOL]['short']
    again=restart(recovered,tmp_path)
    assert not again.opposite_narrow_state[SYMBOL]['short']
    assert again._last_valid_exit_levels[(SYMBOL,'short',100.)]==unlocked


@pytest.mark.parametrize('partial_after_restart',[False,True])
def test_partial_close_keeps_state_when_representative_id_disappears(tmp_path,partial_after_restart):
    bot,api=file_bot(tmp_path);api.position_rows=[position(api,'1'),position(api,'2')]
    bot._sync_live_positions();expected=calculate(bot)
    bot._apply_short_profit_lock(SYMBOL,'short',100.,99.,expected['stop_loss_price'])
    if partial_after_restart:bot=restart(bot,tmp_path)
    api.position_rows=[position(api,'2',size='3')]
    recovered=restart(bot,tmp_path)
    assert recovered.position_ids[SYMBOL]=='2' and recovered.entry_sizes[SYMBOL]==3
    assert recovered.opposite_narrow_state[SYMBOL]['short'] and recovered.short_profit_lock_armed[SYMBOL]
    assert recovered._last_valid_exit_levels[(SYMBOL,'short',100.)]==expected


def test_partial_close_with_different_average_keeps_flags_without_relabeling_old_prices(tmp_path):
    bot,api=file_bot(tmp_path)
    api.position_rows=[position(api,'1'),{**position(api,'2'),'price':'120'}]
    bot._sync_live_positions();calculate(bot)
    api.position_rows=[api.position_rows[1]]
    recovered=restart(bot,tmp_path)
    assert recovered.entry_prices[SYMBOL]==120
    assert recovered.opposite_narrow_state[SYMBOL]['short'] is True
    assert not getattr(recovered,'_last_valid_exit_levels',{})


@pytest.mark.parametrize('change',['replacement','addition','side','price','opened_at','quantity_increase'])
@pytest.mark.parametrize('during_restart',[False,True])
def test_changed_position_identity_cannot_inherit_old_state(tmp_path,change,during_restart):
    bot,api=file_bot(tmp_path);api.position_rows=[position(api,'1')]
    bot._sync_live_positions();calculate(bot)
    bot.short_profit_lock_armed[SYMBOL]=True;bot._persist_exit_protection(SYMBOL)
    row=api.position_rows[0]
    if change=='replacement':row['positionId']='9'
    elif change=='addition':api.position_rows.append(position(api,'2'))
    elif change=='side':row['side']='BUY'
    elif change=='price':row['price']='101'
    elif change=='opened_at':row['timestamp']='2026-01-01T00:00:00Z'
    else:row['size']='6'
    if during_restart:bot=restart(bot,tmp_path)
    else:bot._sync_live_positions()
    assert not any(getattr(bot,'opposite_narrow_state',{}).get(SYMBOL,{}).values())
    assert not bot.short_profit_lock_armed[SYMBOL]
    assert not getattr(bot,'_last_valid_exit_levels',{})


def test_verified_flat_removes_only_its_snapshot_and_reopening_starts_fresh(tmp_path):
    bot,api=file_bot(tmp_path);api.position_rows=[position(api,'1')]
    bot._sync_live_positions();calculate(bot)
    state=bot._exit_protection_store
    state.snapshots['other-symbol']={'untouched':True};state.save()
    api.position_rows=[];bot._sync_live_positions()
    assert SYMBOL not in state.snapshots and state.snapshots['other-symbol']=={'untouched':True}
    api.position_rows=[position(api,'9')]
    recovered=restart(bot,tmp_path)
    assert not any(getattr(recovered,'opposite_narrow_state',{}).get(SYMBOL,{}).values())
    assert not getattr(recovered,'_last_valid_exit_levels',{})


def test_failed_startup_snapshot_keeps_saved_file_until_exchange_recovers(tmp_path):
    bot,api=file_bot(tmp_path);api.position_rows=[position(api,'1')]
    bot._sync_live_positions();expected=calculate(bot)
    path=bot._exit_protection_store.path;before=path.read_bytes()
    other,_=file_bot(tmp_path,bot._live_execution_service)
    api.fail_read=True;other._sync_live_positions()
    assert path.read_bytes()==before and SYMBOL in other._position_sync_failed
    api.fail_read=False;other._sync_live_positions()
    assert other._last_valid_exit_levels[(SYMBOL,'short',100.)]==expected
    assert not other._position_sync_failed


def test_unrecorded_disappearance_does_not_erase_persisted_protection(tmp_path,store):
    svc,api=service(store);svc.submit('LTC_JPY','SELL',5)
    bot,_=file_bot(tmp_path,svc);bot._sync_live_positions();calculate(bot)
    before=bot._exit_protection_store.path.read_bytes()
    api.position_rows=[];bot._sync_live_positions()
    assert SYMBOL in bot._position_sync_failed
    assert bot._exit_protection_store.path.read_bytes()==before


def test_account_scope_and_paper_backtest_never_restore_or_overwrite_live_file(tmp_path):
    bot,api=file_bot(tmp_path);api.position_rows=[position(api,'1')]
    bot._sync_live_positions();calculate(bot)
    path=bot._exit_protection_store.path;before=path.read_bytes()
    other,_=file_bot(tmp_path,bot._live_execution_service)
    other.exchange_settings_gmo={'live_trade':False};other._sync_live_positions()
    assert not getattr(other,'_last_valid_exit_levels',{}) and path.read_bytes()==before
    with backtest_state(bot):
        bot.entry_sizes={SYMBOL:5};calculate(bot)
        assert not bot._persist_exit_protection(SYMBOL)
    assert path.read_bytes()==before
    other._live_execution_service.account='different-account'
    other.exchange_settings_gmo={'live_trade':True};other._sync_live_positions()
    assert not getattr(other,'_last_valid_exit_levels',{})
    assert other._exit_protection_store.path!=path and path.read_bytes()==before


@pytest.mark.parametrize('corruption',['truncated','version','account'])
def test_unreadable_file_is_visible_and_not_overwritten_until_read_recovers(tmp_path,corruption,caplog):
    bot,api=file_bot(tmp_path);api.position_rows=[position(api,'1')]
    bot._sync_live_positions();expected=calculate(bot)
    path=bot._exit_protection_store.path;good=path.read_bytes();data=json.loads(good)
    if corruption=='truncated':bad=b'{broken'
    else:
        data['version' if corruption=='version' else 'account']=9 if corruption=='version' else 'other'
        bad=json.dumps(data).encode()
    path.write_bytes(bad);other=restart(bot,tmp_path)
    assert path.read_bytes()==bad
    assert any('persistence unavailable' in record.message for record in caplog.records)
    path.write_bytes(good);other._sync_live_positions()
    assert other._last_valid_exit_levels[(SYMBOL,'short',100.)]==expected


@pytest.mark.parametrize('field,value',[('opposite_narrow','yes'),('entry_price',float('nan')),('levels',{'take_profit_price':110,'stop_loss_price':90}),('levels',{'take_profit_price':True,'stop_loss_price':101})])
def test_invalid_symbol_state_is_discarded_instead_of_applied(tmp_path,field,value,caplog):
    bot,api=file_bot(tmp_path);api.position_rows=[position(api,'1')]
    bot._sync_live_positions();calculate(bot)
    path=bot._exit_protection_store.path;data=json.loads(path.read_bytes())
    data['symbols'][SYMBOL][field]=value;path.write_text(json.dumps(data),encoding='utf-8')
    recovered=restart(bot,tmp_path)
    assert not any(getattr(recovered,'opposite_narrow_state',{}).get(SYMBOL,{}).values())
    assert not getattr(recovered,'_last_valid_exit_levels',{})
    assert any(record.levelno>=logging.WARNING for record in caplog.records)


def test_atomic_write_failure_preserves_previous_file_and_retries_dirty_state(tmp_path,monkeypatch):
    import exit_protection
    bot,api=file_bot(tmp_path);api.position_rows=[position(api,'1')]
    bot._sync_live_positions();path=bot._exit_protection_store.path;before=path.read_bytes()
    original=exit_protection.os.replace
    def failure(*args):raise OSError('controlled disk failure')
    monkeypatch.setattr(exit_protection.os,'replace',failure)
    expected=calculate(bot)
    assert path.read_bytes()==before and not list(tmp_path.glob('.exit-protection-*.tmp'))
    assert bot._last_valid_exit_levels[(SYMBOL,'short',100.)]==expected
    monkeypatch.setattr(exit_protection.os,'replace',original)
    assert bot._persist_exit_protection(SYMBOL)
    assert restart(bot,tmp_path)._last_valid_exit_levels[(SYMBOL,'short',100.)]==expected


def test_unchanged_state_does_not_rewrite_file_or_restore_old_flags_each_cycle(tmp_path,monkeypatch):
    import exit_protection
    bot,api=file_bot(tmp_path);api.position_rows=[position(api,'1')]
    bot._sync_live_positions();calculate(bot)
    recovered=restart(bot,tmp_path);calls=[];original=exit_protection.os.replace
    def record(*args):calls.append(args);return original(*args)
    monkeypatch.setattr(exit_protection.os,'replace',record)
    recovered._sync_live_positions();calculate(recovered)
    assert not calls
    recovered.opposite_narrow_state[SYMBOL]['short']=False
    recovered._sync_live_positions()
    assert not recovered.opposite_narrow_state[SYMBOL]['short'] and len(calls)==1


def test_short_profit_lock_arm_is_persisted_even_without_valid_candles(tmp_path):
    bot,api=file_bot(tmp_path);api.position_rows=[position(api,'1')]
    bot._sync_live_positions()
    assert bot._apply_short_profit_lock(SYMBOL,'short',100.,99.,101.)==pytest.approx(99.75)
    recovered=restart(bot,tmp_path)
    assert recovered.short_profit_lock_armed[SYMBOL] is True
    assert not getattr(recovered,'_last_valid_exit_levels',{})


@pytest.mark.parametrize('legacy_value',[False,True,'false'])
def test_repeated_verification_cannot_reload_stale_legacy_profit_lock(tmp_path,legacy_value):
    bot,api=file_bot(tmp_path);api.position_rows=[position(api,'1')]
    bot._sync_live_positions();bot._apply_short_profit_lock(SYMBOL,'short',100.,99.,101.)
    (tmp_path/'positions.json').write_text(json.dumps({SYMBOL:{'position_id':'1','short_profit_lock_armed':legacy_value}}),encoding='utf-8')
    other,_=file_bot(tmp_path,bot._live_execution_service)
    for name in ('verify_positions','_restore_short_profit_lock_state'):
        fn,_=method(BOT,name);setattr(other,name,MethodType(fn,other))
    other._recover_live_orders=lambda:None
    other.verify_positions()
    assert other.short_profit_lock_armed[SYMBOL]
    other.verify_positions()
    assert other.short_profit_lock_armed[SYMBOL]


@pytest.mark.parametrize('legacy_value,expected',[(True,True),(False,False),('true',False),('false',False)])
def test_newer_verified_legacy_arm_is_not_cleared_by_older_unarmed_snapshot(tmp_path,legacy_value,expected):
    bot,api=file_bot(tmp_path);api.position_rows=[position(api,'1')]
    bot._sync_live_positions()  # Last dedicated save was unarmed.
    (tmp_path/'positions.json').write_text(json.dumps({SYMBOL:{'position_id':'1','short_profit_lock_armed':legacy_value}}),encoding='utf-8')
    other,_=file_bot(tmp_path,bot._live_execution_service)
    for name in ('verify_positions','_restore_short_profit_lock_state'):
        fn,_=method(BOT,name);setattr(other,name,MethodType(fn,other))
    other._recover_live_orders=lambda:None
    other.verify_positions()
    assert other.short_profit_lock_armed[SYMBOL] is expected
    assert other._exit_protection_store.snapshots[SYMBOL]['short_profit_lock'] is expected


@pytest.mark.parametrize('side',['BUY','SELL'])
def test_restored_stop_closes_once_with_missing_candles_and_clears_durable_state(tmp_path,store,side):
    svc,api=service(store);svc.submit('LTC_JPY',side,5)
    bot,_=file_bot(tmp_path,svc);bot._sync_live_positions();levels=calculate(bot)
    recovered=restart(bot,tmp_path)
    price=levels['stop_loss_price']+(-1 if side=='BUY' else 1)
    recovered.get_current_price=lambda _:price
    api.fill_price=Decimal(str(price))
    for name in ('_check_exit_conditions','_evaluate_exit_event'):
        fn,_=method(BOT,name);setattr(recovered,name,MethodType(fn,recovered))
    recovered._check_exit_conditions(SYMBOL,None,None,pd.DataFrame())
    assert len(api.posts)==2 and not api.position_rows
    assert not svc.store.pending('test') and not svc.store.close_batches('test')
    assert SYMBOL not in recovered._exit_protection_store.snapshots
    recovered._check_exit_conditions(SYMBOL,None,None,pd.DataFrame())
    assert len(api.posts)==2
