import logging
from types import SimpleNamespace
import pytest
from position_monitor import wait_with_position_monitoring
from trading_execution import ExchangeMaintenance
from test_exit_quote_validation import price_reader,ticker
from market_quotes import QuoteUnavailable


def setup(check=None):
    timer=[0.];calls=[]
    def sleep(n):timer[0]+=n
    def callback(s,*args,**kw):
        calls.append((timer[0],s,kw));assert args[-1].empty
        if check:check(s,timer)
    bot=SimpleNamespace(symbols=['a','b'],positions={'a':'short','b':'long'},
        exchange_settings_gmo={'live_trade':True},is_backtest=False,logger=logging.getLogger('monitor'),_check_exit_conditions=callback)
    def run(seconds):wait_with_position_monitoring(bot,seconds,{},[],clock=lambda:timer[0],sleep=sleep)
    return bot,timer,calls,run


def test_wait_checks_both_positions_every_two_seconds_without_new_thread():
    bot,timer,calls,run=setup();run(28)
    assert len(calls)==28 and timer[0]==28
    assert [t for t,s,k in calls if s=='a']==list(range(2,29,2))
    assert all(k=={'monitor_only':True} for t,s,k in calls)


def test_a_failed_symbol_does_not_hide_other_positions():
    def fail(s,t):
        if s=='a':raise RuntimeError('secret must not be logged')
    bot,timer,calls,run=setup(fail);run(4)
    assert [s for t,s,k in calls]==['a','b','a','b']


def test_slow_close_does_not_start_concurrent_duplicate_checks(caplog):
    def slow(s,t):
        if s=='a':t[0]+=6
    bot,timer,calls,run=setup(slow);run(4)
    assert [s for t,s,k in calls]==['a','b']
    assert 'exceeded' in caplog.text


def test_maintenance_returns_to_main_availability_gate():
    def maintenance(s,t):raise ExchangeMaintenance()
    bot,timer,calls,run=setup(maintenance);run(30)
    assert len(calls)==1 and timer[0]==2


@pytest.mark.parametrize('live,backtest',[(False,False),(True,True)])
def test_paper_and_backtest_waits_remain_unchanged(live,backtest):
    bot,timer,calls,run=setup();bot.exchange_settings_gmo['live_trade']=live;bot.is_backtest=backtest;run(28)
    assert not calls and timer[0]==28


def test_no_positions_avoids_price_requests():
    bot,timer,calls,run=setup();bot.positions={};run(28)
    assert not calls and timer[0]==28


def test_closed_positions_are_not_polled_again():
    bot,timer,calls,run=setup()
    def close(*a,**kw):bot.positions.clear()
    bot._check_exit_conditions=close;run(28)
    assert timer[0]==28


@pytest.mark.parametrize('value',[-1,True,float('nan'),float('inf'),'2'])
def test_invalid_wait_rejected(value):
    *_,run=setup()
    with pytest.raises(ValueError):run(value)


def test_monitor_quote_attempt_does_not_retry_bad_price():
    bot,read,calls,closed,events=price_reader([ticker('NaN'),ticker('101')])
    with pytest.raises(QuoteUnavailable):read(max_attempts=1,request_timeout=2)
    assert len(calls)==1 and calls[0][1]['timeout']==2


def test_default_quote_behavior_still_retries():
    bot,read,calls,closed,events=price_reader([ticker('NaN'),ticker('101')])
    assert read()==101 and len(calls)==2 and calls[0][1]['timeout']==10
