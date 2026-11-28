from types import MethodType
import position_monitor
from test_bot_execution_integration import BOT,method
from test_display_monitoring_isolation import display_bot,wire_live_loop,SYMBOLS


def test_actual_live_loop_closes_during_wait_without_another_analysis(monkeypatch):
    bot=display_bot();bot.entry_sizes={s:5. for s in SYMBOLS}
    bot.short_profit_lock_armed={};bot.reentry_block_until={}
    bot.send_notification=lambda *a,**k:None
    bot.verify_positions=lambda:None;bot._refresh_live_stats=lambda *a:None
    run,actions,_=wire_live_loop(bot)
    now=[0.];closed=[];reads=[]
    def price(symbol):
        reads.append((now[0],symbol))
        return 85. if symbol==SYMBOLS[0] and now[0]>=2 else 100.
    bot.get_current_price=price
    def close(symbol,*args):
        closed.append((now[0],symbol));bot.positions[symbol]=None
    bot._close_live_position=close
    original=position_monitor.wait_with_position_monitoring
    def sleep(seconds):
        now[0]+=seconds
        if now[0]>=8:raise KeyboardInterrupt()
    def scheduler(*args):return original(*args,clock=lambda:now[0],sleep=sleep)
    monkeypatch.setattr(position_monitor,'wait_with_position_monitoring',scheduler)
    fn,_=method(BOT,'_wait_with_position_monitoring');bot._wait_with_position_monitoring=MethodType(fn,bot)
    run()
    assert closed==[(2.,SYMBOLS[0])]
    assert [a[:2] for a in actions if a[0]=='monitor']==[('monitor',s) for s in SYMBOLS]
    assert (4.,SYMBOLS[1]) in reads and (6.,SYMBOLS[1]) in reads


def test_fast_quote_method_passes_only_the_short_budget():
    fn,_=method(BOT,'_get_monitor_exit_quote')
    calls=[]
    from types import SimpleNamespace
    bot=SimpleNamespace(get_current_price=lambda *a,**k:calls.append((a,k)) or 1.)
    assert fn(bot,'eth_jpy')==1.
    assert calls==[(('eth_jpy',),dict(with_timestamp=True,max_attempts=1,request_timeout=2))]
