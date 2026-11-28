from types import SimpleNamespace

import pytest

from research.replay_engine import install_case, case_prepare, simulate, compare_core
from research.search_backtests import DailyDecisionCache, make_bot
from test_research_replay import synthetic, START, END


@pytest.mark.parametrize('direction', ['both', 'short_only'])
@pytest.mark.parametrize('tp,sl', [(.8,1.),(1.2,1.),(1.,.8),(1.,1.2)])
def test_expanded_exit_scenarios_match_actual_backtest(tmp_path, direction, tp, sl):
    frames = synthetic()
    cache = DailyDecisionCache(frames, cache_dir=tmp_path / 'features')
    case = dict(direction=direction, adx_min=30, exit_profile='no_short_lock', tp_multiplier=tp, sl_multiplier=sl)
    costs = dict(name='test', fee_bps={'ltc_jpy':3}, spread_bps=2, slippage_bps=1, carry_bps_per_day=4)
    source = make_bot(frames, START, END, symbols=['ltc_jpy'], prepare=case_prepare(cache, case))
    install_case(source, case)
    source.backtest(1, costs=costs)
    result = simulate(frames, START, END, costs, cache, case=case, symbols=['ltc_jpy'])
    compare_core(source.captured, result)
    assert result['trades'], 'Comparison must contain completed trades'


@pytest.mark.parametrize('bad', [True, None, '1', float('nan'), float('inf'), .49, 1.51])
def test_invalid_sweep_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        install_case(SimpleNamespace(), dict(exit_profile='baseline', tp_multiplier=bad))


def test_explicit_cache_directory_preserves_previous_study(tmp_path):
    cache = DailyDecisionCache(synthetic(), cache_dir=tmp_path / 'new-study')
    cache.day('ltc_jpy', START)
    assert cache.cache_dir == tmp_path / 'new-study'
    assert len(list(cache.cache_dir.glob('*.pkl.gz'))) == 1
