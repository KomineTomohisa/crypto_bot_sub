"""Keep research scenarios meaningful after entry policies are pinned."""
from types import SimpleNamespace

from research.replay_engine import install_case, simulate, compare_core
from research.search_backtests import DailyDecisionCache, make_bot
from strategy_policy import CANDIDATE, position_policy
from test_research_replay import synthetic, START, END


def test_one_percent_research_case_does_not_silently_use_legacy_trigger():
    bot = SimpleNamespace(short_profit_lock_armed={})
    install_case(bot, dict(exit_profile='short_lock_1pct'))
    # The research variant must become an explicit entry policy.
    from strategy_policy import active_policy
    bot.position_policies = {'ltc_jpy': active_policy(bot).snapshot()}
    assert position_policy(bot, 'ltc_jpy').lock_trigger == 1.0


def test_implemented_candidate_matches_searched_candidate_on_same_prices(tmp_path):
    frames = synthetic()
    cache = DailyDecisionCache(frames)
    cache.cache_dir = tmp_path
    costs = dict(name='test', fee_bps={'ltc_jpy': 3}, spread_bps=2, slippage_bps=1, carry_bps_per_day=4)
    source = make_bot(frames, START, END, symbols=['ltc_jpy'], prepare=cache.prepare)
    source.backtest(1, policy=CANDIDATE, costs=costs)
    expected = simulate(frames, START, END, costs, cache,
                        case=dict(direction='both', adx_min=30, exit_profile='no_short_lock'),
                        symbols=['ltc_jpy'])
    compare_core(source.captured, expected)
    assert source.captured['trades'], 'Parity must include actual simulated trades'
