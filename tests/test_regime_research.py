from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research.regime_filters import allowed, gate_pair
from research.replay_engine import filtered_rows, install_case, case_prepare, simulate, compare_core
from research.search_backtests import DailyDecisionCache, make_bot
from strategy_decisions import entry_side
from test_research_replay import synthetic, START, END


AT = pd.Timestamp('2026-11-22T09:00:00+09:00')


def pair():
    common=dict(EMA_short=102.,EMA_long=100.,plus_di14=30.,minus_di14=20.,ATR=.4,close=100.,ADX=35.,buy_signal=True,sell_signal=True)
    return [pd.Series(dict(common,timestamp=AT-pd.Timedelta(minutes=m))) for m in (15,30)]


@pytest.mark.parametrize('context,side,expected', [
    ('ema_follow','long',True),('ema_follow','short',False),('ema_fade','long',False),('ema_fade','short',True),
    ('di_follow','long',True),('di_follow','short',False),('di_fade','long',False),('di_fade','short',True)])
def test_direction_filters_use_closed_direction(context, side, expected):
    assert allowed(*pair(),AT,side,dict(entry_context=context)) is expected


@pytest.mark.parametrize('index',[0,1])
@pytest.mark.parametrize('value',[None,np.nan,np.inf,True])
def test_each_closed_row_must_have_valid_filtered_features(index,value):
    rows=pair();rows[index]['EMA_short']=value
    assert not allowed(*rows,AT,'long',dict(entry_context='ema_follow'))


def test_direction_filter_runs_before_short_priority_so_long_can_remain():
    rows=pair();case=dict(direction='both',adx_min=30,exit_profile='no_short_lock',entry_context='ema_follow')
    filtered=filtered_rows(*rows,case,AT)
    assert entry_side(*filtered,AT)=='long'
    assert all(row.buy_signal for row in rows)


@pytest.mark.parametrize('hour,session,expected',[(0,'jst_17_01',True),(1,'jst_17_01',False),(1,'jst_01_09',True),
    (8,'jst_01_09',True),(9,'jst_01_09',False),(9,'jst_09_17',True),(17,'jst_09_17',False),(17,'jst_17_01',True)])
def test_sessions_use_entry_observation_jst_not_signal_bar(hour,session,expected):
    at=AT.normalize()+pd.Timedelta(hours=hour)
    assert allowed(*pair(),at.tz_convert('UTC'),'short',dict(session=session)) is expected


def test_volatility_bounds_apply_to_both_closed_rows():
    rows=pair()
    assert allowed(*rows,AT,'short',dict(atr_band='quiet'))
    rows[1]['ATR']=.5
    assert not allowed(*rows,AT,'short',dict(atr_band='quiet'))
    assert allowed(*rows,AT,'short',dict(atr_band='normal'))
    rows[0]['close']=0
    assert not allowed(*rows,AT,'short',dict(atr_band='normal'))


def test_adx_rising_cannot_use_a_forming_bar():
    rows=pair();rows[1]['ADX']=40
    assert not allowed(*rows,AT,'short',dict(entry_context='adx_rising'))
    rows[1]['ADX']=30
    assert allowed(*rows,AT,'short',dict(entry_context='adx_rising'))


SCENARIOS=[dict(entry_context=c) for c in ('none','ema_follow','ema_fade','di_follow','di_fade','adx_rising')]
SCENARIOS += [dict(atr_band=b) for b in ('quiet','normal')]
SCENARIOS += [dict(session=s) for s in ('jst_01_09','jst_09_17','jst_17_01')]
SCENARIOS += [dict(entry_context='ema_fade',atr_band='normal',session='jst_17_01')]


@pytest.mark.parametrize('gates',SCENARIOS)
def test_context_volatility_and_time_gates_match_actual_source_replay(tmp_path,gates):
    frames=synthetic();cache=DailyDecisionCache(frames,cache_dir=tmp_path)
    case=dict(direction='both',adx_min=25,exit_profile='no_short_lock',**gates)
    costs=dict(name='test',fee_bps={'ltc_jpy':3},spread_bps=2,slippage_bps=1,carry_bps_per_day=4)
    source=make_bot(frames,START,END,symbols=['ltc_jpy'],prepare=case_prepare(cache,case));install_case(source,case)
    source.backtest(1,costs=costs)
    result=simulate(frames,START,END,costs,cache,case,symbols=['ltc_jpy'])
    compare_core(source.captured,result)


def test_cache_eviction_preserves_exact_feature_rows(tmp_path):
    from research.regime_search import BoundedFeatureCache
    frames=synthetic()
    bounded=BoundedFeatureCache(frames,cache_dir=tmp_path/'bounded');bounded.max_days=1
    normal=DailyDecisionCache(frames,cache_dir=tmp_path/'normal')
    for day in (START-pd.Timedelta(days=1),START,START-pd.Timedelta(days=1)):
        pd.testing.assert_frame_equal(bounded.day('ltc_jpy',day),normal.day('ltc_jpy',day),check_exact=True)
        assert len(bounded.days)<=1
