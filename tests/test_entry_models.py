from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research.entry_models import (LEGACY, TREND, closed_pair, entry_rows,
                                   model_spec, prepare_model, trend_flags)
from research.replay_engine import compare_core, simulate
from research.search_backtests import DailyDecisionCache, make_bot, offline_io_guard
from strategy_decisions import entry_side
from strategy_inputs import historical_observation, prepare_strategy_inputs
from test_research_replay import START, END, synthetic


AT = pd.Timestamp('2026-11-23T09:00:00+09:00')
COSTS = dict(name='entry-model-test',fee_bps={'ltc_jpy':3},spread_bps=2,
             slippage_bps=1,carry_bps_per_day=4)


def row(side='long', minutes=15):
    values = dict(close=103.,EMA_short=102.,EMA_long=100.,plus_di14=35.,minus_di14=15.,
                  MACD_histogram=.2,ADX=25.,buy_signal=False,sell_signal=False)
    if side == 'short':
        values.update(close=97.,EMA_short=98.,plus_di14=15.,minus_di14=35.,MACD_histogram=-.2)
    return pd.Series(dict(values,timestamp=AT-pd.Timedelta(minutes=minutes)))


@pytest.mark.parametrize('side',['long','short'])
def test_direction_is_symmetric_and_uses_shared_mapping(side):
    rows = entry_rows(row(side),row(side,30),AT,TREND)
    assert entry_side(*rows,AT) == side
    assert not any(row(side)[c] for c in ('buy_signal','sell_signal'))


@pytest.mark.parametrize('adx,allowed',[(24.999,False),(25.,True),(100.,True),(100.001,False),(-1.,False)])
def test_fixed_adx_bounds(adx,allowed):
    latest = row();latest['ADX']=adx
    assert trend_flags(latest) == (allowed,False)


@pytest.mark.parametrize('column',['close','EMA_short','EMA_long','ADX','plus_di14','minus_di14','MACD_histogram'])
@pytest.mark.parametrize('bad',[None,np.nan,np.inf,-np.inf,True,'25'])
def test_bad_features_block_entry_on_either_confirmation_bar(column,bad):
    for minutes in (15,30):
        latest,previous=row(),row(minutes=30)
        target=latest if minutes==15 else previous
        target[column]=bad
        assert entry_side(*entry_rows(latest,previous,AT,TREND),AT) is None


@pytest.mark.parametrize('column',['close','EMA_short','EMA_long'])
@pytest.mark.parametrize('bad',[0.,-1.])
def test_nonpositive_prices_are_rejected(column,bad):
    latest=row();latest[column]=bad
    assert trend_flags(latest)==(False,False)


@pytest.mark.parametrize('column,value',[('EMA_short',100.),('close',102.),('plus_di14',15.),
                                       ('MACD_histogram',0.),('minus_di14',101.)])
def test_equal_or_conflicting_direction_does_not_get_a_signal(column,value):
    latest=row();latest[column]=value
    assert trend_flags(latest)==(False,False)


def test_opposite_directions_cannot_confirm_each_other():
    assert entry_side(*entry_rows(row(),row('short',30),AT,TREND),AT) is None


def test_candidate_still_obeys_post_exit_cooldown():
    rows=entry_rows(row(),row(minutes=30),AT,TREND)
    assert entry_side(*rows,AT,AT+pd.Timedelta(minutes=15)) is None
    assert entry_side(*rows,AT,AT)=='long'


@pytest.mark.parametrize('column',['close','EMA_short','EMA_long','ADX','plus_di14','minus_di14','MACD_histogram'])
def test_missing_required_feature_does_not_fall_back_to_legacy(column):
    latest=row().drop(column);previous=row(minutes=30)
    latest['buy_signal']=previous['buy_signal']=True
    assert entry_side(*entry_rows(latest,previous,AT,TREND),AT) is None


@pytest.mark.parametrize('minutes',[0,5,20,30,45,-15])
def test_missing_stale_or_forming_latest_bar_blocks_entry(minutes):
    assert entry_side(*entry_rows(row(minutes=minutes),row(minutes=30),AT,TREND),AT) is None


def test_out_of_order_missing_and_timezone_equivalent_timestamps():
    assert not closed_pair(row(minutes=30),row(),AT)
    assert not closed_pair(row().drop('timestamp'),row(minutes=30),AT)
    assert closed_pair(row(),row(minutes=30),AT.tz_convert('UTC'))
    assert not closed_pair(row(),row(minutes=30),pd.NaT)


@pytest.mark.parametrize('model',[None,'trend','',{},True])
def test_invalid_model_is_rejected_before_cache_is_read(model):
    with pytest.raises(ValueError):prepare_model(SimpleNamespace(),model)
    with pytest.raises(ValueError):entry_rows(row(),row(minutes=30),AT,model)


def test_model_spec_is_fixed_and_returns_independent_values():
    spec=model_spec(TREND);spec['conditions']['adx_min']=0
    assert model_spec(TREND)['conditions']['adx_min']==25
    assert model_spec()['name']==LEGACY


def test_legacy_does_not_reinterpret_or_mutate_signals():
    latest,previous=row(),row(minutes=30)
    latest['buy_signal']=previous['buy_signal']=True
    old=(latest.copy(),previous.copy())
    rows=entry_rows(latest,previous,AT)
    assert entry_side(*rows,AT)=='short'
    rows[0]['buy_signal']=False
    pd.testing.assert_series_equal(latest,old[0]);pd.testing.assert_series_equal(previous,old[1])


class StaticCache:
    def __init__(self,frame):self.frame=frame
    def prepare(self,*args):return self.frame,pd.DataFrame({'sentinel':[42]})


def test_opening_adapter_copies_only_closed_pair_and_blanks_forming_flags():
    frame=pd.DataFrame([row(minutes=45),row(minutes=30),row(),row(minutes=0)])
    frame.loc[frame.index[-1],['buy_signal','sell_signal']]=True
    frame.loc[frame.index[-1],['ADX','MACD_histogram']]=np.inf
    original=frame.copy(deep=True);cache=StaticCache(frame)
    result,hours=prepare_model(cache,TREND)(None,'ltc_jpy',frame,pd.DataFrame(),AT)
    assert entry_side(result.iloc[-2],result.iloc[-3],AT)=='long'
    assert not result.iloc[-1][['buy_signal','sell_signal']].any()
    pd.testing.assert_series_equal(result.iloc[0],original.iloc[0])
    pd.testing.assert_frame_equal(frame,original)
    assert hours.sentinel.iloc[0]==42


def test_existing_position_observation_retains_exact_original_signals_and_object():
    frame=pd.DataFrame([row(minutes=30),row(),row(minutes=0)])
    cache=StaticCache(frame)
    at=AT+pd.Timedelta(minutes=15)-pd.Timedelta(nanoseconds=1)
    result,_=prepare_model(cache,TREND)(None,'ltc_jpy',frame,pd.DataFrame(),at)
    assert result is frame


def test_too_little_history_cannot_open():
    frame=pd.DataFrame([row(),row(minutes=0)])
    frame[['buy_signal','sell_signal']]=True
    result,_=prepare_model(StaticCache(frame),TREND)(None,'ltc_jpy',frame,pd.DataFrame(),AT)
    assert not result[['buy_signal','sell_signal']].any().any()
    assert frame[['buy_signal','sell_signal']].all().all()


def test_legacy_prepare_is_direct_original_method():
    cache=StaticCache(pd.DataFrame())
    assert prepare_model(cache)==cache.prepare
    assert prepare_model(cache,LEGACY)==cache.prepare


@pytest.mark.parametrize('model',[LEGACY,TREND])
def test_real_source_account_and_exit_observations_are_unchanged_by_adapter(tmp_path,model):
    frames=synthetic();cache=DailyDecisionCache(frames,cache_dir=tmp_path)
    adapted=prepare_model(cache,model)
    exits=[]
    def observed(bot,symbol,intraday,hourly,at):
        left,hours=adapted(bot,symbol,intraday,hourly,at)
        raw,_=cache.prepare(bot,symbol,intraday,hourly,at)
        if pd.Timestamp(at)!=intraday.timestamp.iloc[-1]:
            pd.testing.assert_frame_equal(left,raw,check_exact=True)
            exits.append(at)
        return left,hours
    bot=make_bot(frames,START,END,symbols=['ltc_jpy'],prepare=observed)
    with offline_io_guard():bot.backtest(1,costs=COSTS)
    assert bot.captured['trades'], 'Must test real fills, not an empty replay'
    assert exits
    if model==LEGACY:
        replay=simulate(frames,START,END,COSTS,cache,symbols=['ltc_jpy'])
        compare_core(bot.captured,replay)
    else:
        assert {t['type'] for t in bot.captured['trades']}=={'long','short'}
        for trade in bot.captured['trades']:
            at=pd.Timestamp(trade['entry_time']);full=cache.day(trade['symbol'],at)
            index=int(full.timestamp.searchsorted(at))
            rows=entry_rows(full.iloc[index-1],full.iloc[index-2],at,TREND)
            assert entry_side(*rows,at)==trade['type']


def test_new_entry_source_cached_and_uncached_feature_paths_have_identical_fills(tmp_path):
    frames=synthetic();cached=DailyDecisionCache(frames,cache_dir=tmp_path)
    reference=SimpleNamespace(prepare=prepare_strategy_inputs)
    results=[]
    for cache in (cached,reference):
        bot=make_bot(frames,START,END,symbols=['ltc_jpy'],prepare=prepare_model(cache,TREND))
        with offline_io_guard():bot.backtest(1,costs=COSTS)
        results.append(bot.captured)
    assert results[0]['trades'] and results[1]['trades']
    # Compare all captured fields, including exit diagnostics and valuation.
    from research.search_backtests import compare_trade_paths
    compare_trade_paths(*results)


def test_future_ohlc_changes_do_not_affect_closed_entry_rows(tmp_path):
    frames=synthetic();at=START+pd.Timedelta(hours=8)
    changed=deepcopy(frames)
    for (symbol,interval),frame in changed.items():
        frame.loc[frame.timestamp.ge(at),['open','high','low','close']]*=10
    results=[]
    for index,values in enumerate((frames,changed)):
        cache=DailyDecisionCache(values,cache_dir=tmp_path/str(index))
        raw=historical_observation(values['ltc_jpy','15min'],at,at_open=True)
        visible=values['ltc_jpy','1hour']
        visible=visible.loc[visible.timestamp+pd.Timedelta(hours=1)<=at]
        with offline_io_guard():
            prepared,_=prepare_model(cache,TREND)(None,'ltc_jpy',raw,visible,at)
        results.append(prepared.iloc[-3:-1].reset_index(drop=True))
    pd.testing.assert_frame_equal(*results,check_exact=True)
