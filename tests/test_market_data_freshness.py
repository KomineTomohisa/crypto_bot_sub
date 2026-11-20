"""UTC API partitions, mutable cache refresh, and position protection on feed failure."""
from types import SimpleNamespace, MethodType
import json
import logging
import os

import numpy as np
import pandas as pd
import pytest

from strategy_inputs import candle_api_dates, candle_cache_valid, candle_partition_complete, entry_input_quality
from test_bot_execution_integration import BOT, method


@pytest.mark.parametrize('at,first,last', [
    ('2026-11-20T00:00:00+09:00','20261117','20261119'),
    ('2026-11-20T08:59:00+09:00','20261117','20261119'),
    ('2026-11-20T09:00:00+09:00','20261117','20261120'),
    ('2026-12-01T00:00:00+09:00','20261128','20261130'),
])
def test_utc_partitions_cover_jst_history_without_future_day(at, first, last):
    dates = candle_api_dates(at)
    assert dates[0] == first and dates[-1] == last
    assert dates == sorted(set(dates))


@pytest.mark.parametrize('interval', ['5min','15min','1hour'])
def test_mutable_day_refreshes_after_30_seconds(interval):
    at = pd.Timestamp('2026-11-20T08:00:29+09:00')
    assert candle_cache_valid(interval, '20261119', at, at.timestamp()-29)
    assert not candle_cache_valid(interval, '20261119', at, at.timestamp()-30)
    assert not candle_cache_valid(interval, '20261120', at, at.timestamp())
    assert not candle_cache_valid(interval, '20261119', at, at.timestamp()+1)


@pytest.mark.parametrize('interval,before,after',[
    ('1hour','2026-11-20T09:59:55+09:00','2026-11-20T10:00:02+09:00'),
    ('15min','2026-11-20T10:14:55+09:00','2026-11-20T10:15:02+09:00'),
    ('1hour','2026-11-20T08:59:55+09:00','2026-11-20T09:00:02+09:00'),
])
def test_just_closed_bar_is_refetched_even_if_cache_is_under_30_seconds(interval,before,after):
    assert not candle_cache_valid(interval,'20261119' if before[11:13]=='08' else '20261120',pd.Timestamp(after),pd.Timestamp(before).timestamp())


def test_incomplete_previous_day_is_not_sealed_across_utc_midnight():
    at = pd.Timestamp('2026-11-20T09:10:00+09:00')
    old = pd.Timestamp('2026-11-20T08:59:59+09:00').timestamp()
    sealed = pd.Timestamp('2026-11-20T09:01:00+09:00').timestamp()
    assert not candle_cache_valid('1hour', '20261119', at, old)
    assert candle_cache_valid('1hour', '20261119', at, sealed)


def raw(at, minutes):
    times = pd.date_range(at.normalize()-pd.Timedelta(days=2), at.floor(f'{minutes}min'), freq=f'{minutes}min')
    return pd.DataFrame({'timestamp':times,'open':100.,'high':101.,'low':99.,'close':100.,'volume':1.})


@pytest.mark.parametrize('at', ['2026-11-20T00:00:00+09:00','2026-11-20T08:59:00+09:00','2026-11-20T09:00:00+09:00'])
def test_complete_current_history_is_ready(at):
    at = pd.Timestamp(at)
    assert entry_input_quality(raw(at,15),raw(at,60),at)['ready']


@pytest.mark.parametrize('failure', ['gap15','gap60','stale_hour','no_forming_bar','nan','negative','conflict','invalid_high','unavailable'])
def test_bad_or_stale_input_never_allows_entry(failure):
    at=pd.Timestamp('2026-11-20T12:15:00+09:00');a=raw(at,15);b=raw(at,60)
    if failure=='gap15':a=a.drop(index=12)
    if failure=='gap60':b=b.drop(index=12)
    if failure=='stale_hour':b=b.iloc[:-3]
    if failure=='no_forming_bar':a=a.iloc[:-1]
    if failure=='nan':a.loc[10,'close']=np.nan
    if failure=='negative':a.loc[10,'volume']=-1
    if failure=='conflict':a=pd.concat([a,a.iloc[[10]].assign(close=100.5)])
    if failure=='invalid_high':a.loc[10,'high']=98
    if failure=='unavailable':a=None
    assert not entry_input_quality(a,b,at)['ready']


@pytest.mark.parametrize('failure',['unavailable','stale','feature_exception','fetch_exception'])
@pytest.mark.parametrize('held',[False,True])
def test_actual_live_input_gate_keeps_held_position_monitoring(failure,held):
    fn,ns=method(BOT,'_prepare_live_inputs');at=pd.Timestamp('2026-11-20T12:15:00+09:00')
    ns['now_jst']=lambda:at
    ns['record_bot_event']=lambda *a:None
    a,b=raw(at,15),raw(at,60)
    if failure=='stale':b=b.iloc[:-3]
    if failure=='unavailable':a=None
    def fetch(symbol):
        if failure=='fetch_exception':raise TimeoutError()
        return a,b
    def prepare(*a):raise ValueError('bad feature')
    ns['prepare_strategy_inputs']=prepare
    calls=[]
    bot=SimpleNamespace(_get_market_data=fetch,positions={'x': 'long' if held else None},
        logger=logging.getLogger('test'),_check_exit_conditions=lambda *args:calls.append(args))
    assert fn(bot,'x',{},[]) == (None,None)
    assert len(calls) == int(held)
    if held:assert calls[0][-1].empty  # no stale bar extremes used for protection


def test_actual_live_gate_passes_good_inputs_to_shared_pipeline():
    fn,ns=method(BOT,'_prepare_live_inputs');at=pd.Timestamp('2026-11-20T12:15:00+09:00')
    ns['now_jst']=lambda:at
    ns['prepare_strategy_inputs']=lambda bot,s,a,b,t:(a,b)
    a,b=raw(at,15),raw(at,60)
    bot=SimpleNamespace(_get_market_data=lambda s:(a,b),positions={'x':None},_market_quality_state={'x':'unavailable'})
    result=fn(bot,'x',{},[])
    assert result[0] is a and result[1] is b
    assert not bot._market_quality_state


def test_actual_fetch_uses_only_required_utc_days_and_no_fallback():
    fn,ns=method(BOT,'_get_market_data');at=pd.Timestamp('2026-11-20T00:01:00+09:00');ns['now_jst']=lambda:at
    calls=[]
    def cached(symbol,interval,date,**kwargs):
        calls.append((interval,date,kwargs))
        return raw(at,15 if interval=='15min' else 60)
    bot=SimpleNamespace(logger=logging.getLogger('test'),get_cached_data=cached)
    a,b=fn(bot,'x')
    assert not a.empty and not b.empty
    assert len(calls)==6
    assert {date for _,date,_ in calls}=={'20261117','20261118','20261119'}
    assert all(kwargs=={'fallback_days':0} for _,_,kwargs in calls)


def payload(close=101):
    return {'success':1,'data':{'candlestick':[{'type':'1hour','ohlcv':[['100','102','99',str(close),'1',1795089600000]]}]}}


@pytest.mark.parametrize('interval',['15min','1hour'])
def test_actual_cache_refresh_and_utc_default(tmp_path,interval):
    fn,ns=method(BOT,'get_cached_data');at=pd.Timestamp('2026-11-20T08:00:29+09:00');ns['now_jst']=lambda:at
    path=tmp_path/f'x_{interval}_20261119.json';path.write_text(json.dumps(payload(100)))
    os.utime(path,(at.timestamp()-31,at.timestamp()-31))
    calls=[]
    def get(url,**kwargs):
        calls.append(url)
        return SimpleNamespace(status_code=200,json=lambda:payload(101))
    ns['requests']=SimpleNamespace(get=get,exceptions=__import__('requests').exceptions)
    bot=SimpleNamespace(logger=logging.getLogger('test'),cache_dir=str(tmp_path),rate_limit_delay=0)
    result=fn(bot,'x',interval,fallback_days=0)
    assert calls==[f'https://public.bitbank.cc/x/candlestick/{interval}/20261119']
    assert result.close.iloc[-1]==101
    # A recent cache must not trigger another request.
    os.utime(path,(at.timestamp()-10,at.timestamp()-10))
    fn(bot,'x',interval,fallback_days=0)
    assert len(calls)==1
    # Future partition is refused before HTTP or previous-day fallback.
    assert fn(bot,'x',interval,'20261120').empty
    assert len(calls)==1


def test_actual_cache_does_not_backfill_invalid_raw_prices(tmp_path):
    fn,ns=method(BOT,'get_cached_data');at=pd.Timestamp('2026-11-20T08:00:00+09:00');ns['now_jst']=lambda:at
    bad=payload();bad['data']['candlestick'][0]['ohlcv'][0][3]='NaN'
    ns['requests']=SimpleNamespace(get=lambda *a,**kw:SimpleNamespace(status_code=200,json=lambda:bad),exceptions=__import__('requests').exceptions)
    bot=SimpleNamespace(logger=logging.getLogger('test'),cache_dir=str(tmp_path),rate_limit_delay=0)
    assert fn(bot,'x','1hour','20261119',fallback_days=0).empty
    assert not list(tmp_path.glob('*.json'))


def test_incomplete_old_cache_is_refetched_even_with_recent_mtime(tmp_path):
    fn,ns=method(BOT,'get_cached_data');at=pd.Timestamp('2026-11-20T12:00:00+09:00');ns['now_jst']=lambda:at
    path=tmp_path/'x_1hour_20261119.json';path.write_text(json.dumps(payload(100)))
    os.utime(path,(at.timestamp()-10,at.timestamp()-10))
    calls=[]
    ns['requests']=SimpleNamespace(get=lambda *a,**kw:(calls.append(a) or SimpleNamespace(status_code=200,json=lambda:payload(101))),exceptions=__import__('requests').exceptions)
    bot=SimpleNamespace(logger=logging.getLogger('test'),cache_dir=str(tmp_path),rate_limit_delay=0)
    assert fn(bot,'x','1hour','20261119',fallback_days=0).close.iloc[-1]==101
    assert len(calls)==1


@pytest.mark.parametrize('interval,minutes',[('15min',15),('1hour',60)])
def test_past_partition_requires_all_expected_utc_timestamps(interval,minutes):
    times=pd.date_range('2026-11-19',periods=1440//minutes,freq=f'{minutes}min',tz='UTC')
    data=payload();data['data']['candlestick'][0]['ohlcv']=[['100','102','99','101','1',t.value//1_000_000] for t in times]
    assert candle_partition_complete(data,interval,'20261119')
    data['data']['candlestick'][0]['ohlcv'].pop()
    assert not candle_partition_complete(data,interval,'20261119')


def test_actual_position_protection_closes_on_quote_when_candles_are_missing():
    at=pd.Timestamp('2026-11-20T12:15:00+09:00')
    closes=[]
    bot=SimpleNamespace(logger=logging.getLogger('test'),positions={'x':'long'},
        entry_prices={'x':100.},entry_sizes={'x':1},entry_times={'x':at-pd.Timedelta(hours=1)},
        short_profit_lock_armed={},opposite_narrow_state={},
        _last_valid_exit_levels={('x','long',100.):{'take_profit_price':105.,'stop_loss_price':95.}},
        _get_market_data=lambda s:(None,None),get_current_price=lambda s:90.,
        _get_dynamic_profit_loss_settings=lambda *a:{'long_stop_loss':1.05,'long_profit_take':0.95},
        exchange_settings_gmo={'live_trade':True},_close_live_position=lambda *args:closes.append(args))
    for name in ['_prepare_live_inputs','_check_exit_conditions','_evaluate_exit_event','_apply_short_profit_lock','calculate_dynamic_exit_levels']:
        fn,ns=method(BOT,name);ns['now_jst']=lambda:at
        setattr(bot,name,MethodType(fn,bot))
    assert bot._prepare_live_inputs('x',{},[]) == (None,None)
    assert len(closes)==1 and closes[0][1]=='損切り'
