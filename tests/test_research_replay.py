"""Certify research acceleration before comparing strategy candidates."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from research.search_backtests import DailyDecisionCache, make_bot, compare_trade_paths
from research.replay_engine import simulate, compare_core, case_prepare, install_case
from strategy_inputs import historical_observation, prepare_strategy_inputs


START=pd.Timestamp('2026-10-20',tz='Asia/Tokyo')
END=START+pd.Timedelta(days=1)


def synthetic():
    times=pd.date_range(START-pd.Timedelta(days=3),END,freq='15min',inclusive='left')
    i=np.arange(len(times))
    price=100+np.sin(i*.16)*4+i*.003
    frame=pd.DataFrame(dict(timestamp=times,open=price,close=price+.05,high=price+.3,
                            low=price-.3,volume=100+(i%17)*8))
    hours=frame.set_index('timestamp').resample('1h').agg(dict(open='first',close='last',high='max',low='min',volume='sum')).reset_index()
    return {('ltc_jpy','15min'):frame,('ltc_jpy','1hour'):hours}


@pytest.mark.parametrize('at_open',[False,True])
@pytest.mark.parametrize('hour',[0,8,16])
def test_cached_consumed_rows_match_real_prefix_computation(tmp_path,at_open,hour):
    frames=synthetic()
    cache=DailyDecisionCache(frames);cache.cache_dir=tmp_path
    bot=make_bot(frames,START,END,symbols=['ltc_jpy'])
    timestamp=START+pd.Timedelta(hours=hour)
    at=timestamp if at_open else timestamp+pd.Timedelta(minutes=15)-pd.Timedelta(nanoseconds=1)
    raw=historical_observation(frames['ltc_jpy','15min'],timestamp,at_open=at_open)
    hours=frames['ltc_jpy','1hour']
    hours=hours.loc[hours.timestamp+pd.Timedelta(hours=1)<=at]
    actual,_=prepare_strategy_inputs(bot,'ltc_jpy',raw,hours,at)
    accelerated,_=cache.prepare(bot,'ltc_jpy',raw,hours,at)
    if at_open:
        actual=actual.iloc[:-1].tail(5)
        accelerated=accelerated.iloc[:-1]
        forming=cache.prepare(bot,'ltc_jpy',raw,hours,at)[0].iloc[-1]
        assert pd.isna(forming['ADX']) and not forming['buy_signal'] and forming['volume']==0
    else:
        actual=actual.tail(6)
    pd.testing.assert_frame_equal(actual.reset_index(drop=True),
        accelerated.reindex(columns=actual.columns).reset_index(drop=True),check_dtype=False,check_exact=True)


@pytest.mark.parametrize('profile',['baseline','no_short_lock','short_lock_1pct','tp_120pct'])
@pytest.mark.parametrize('direction',['both','short_only','long_only'])
def test_research_account_and_orders_match_source_execution(tmp_path,profile,direction):
    frames=synthetic()
    case=dict(direction=direction,adx_min=25,exit_profile=profile)
    cache=DailyDecisionCache(frames);cache.cache_dir=tmp_path
    costs=dict(name='test',fee_bps={'ltc_jpy':3},spread_bps=2,slippage_bps=1,carry_bps_per_day=4)
    source=make_bot(frames,START,END,symbols=['ltc_jpy'],prepare=case_prepare(cache,case))
    install_case(source,case)
    source.backtest(1,costs=costs)
    replay=simulate(frames,START,END,costs,cache,case=case,symbols=['ltc_jpy'])
    compare_core(source.captured,replay)


def test_future_price_mutation_cannot_change_earlier_consumed_rows(tmp_path):
    frames=synthetic()
    at=START+pd.Timedelta(hours=8)
    original=DailyDecisionCache(frames);original.cache_dir=tmp_path/'a';original.cache_dir.mkdir()
    changed={key:value.copy() for key,value in frames.items()}
    mask=changed['ltc_jpy','15min'].timestamp.gt(at)
    changed['ltc_jpy','15min'].loc[mask,['open','high','low','close']]*=10
    modified=DailyDecisionCache(changed);modified.cache_dir=tmp_path/'b';modified.cache_dir.mkdir()
    left=original.day('ltc_jpy',at);right=modified.day('ltc_jpy',at)
    left=left.loc[left.timestamp.le(at)].reset_index(drop=True)
    right=right.loc[right.timestamp.le(at)].reset_index(drop=True)
    pd.testing.assert_frame_equal(left,right,check_exact=True)
