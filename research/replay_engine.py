"""Research event loop: shared strategy decisions/costs, certified against source."""
import heapq
from copy import deepcopy

import numpy as np
import pandas as pd

from backtest_costs import BacktestAccount
from strategy_decisions import entry_side, next_entry_time
from strategy_policy import StrategyPolicy, active_policy
from research.search_backtests import make_bot, offline_io_guard


BASELINE = dict(direction='both', adx_min=0, exit_profile='baseline')


def install_case(bot, case):
    if case['exit_profile']=='no_short_lock':
        bot._apply_short_profit_lock = lambda symbol,position,entry_price,current_price,stop_loss_price: float(stop_loss_price)
    elif case['exit_profile']=='short_lock_1pct':
        bot.SHORT_PROFIT_LOCK_TRIGGER_PCT = 1.
        bot.strategy_policy = StrategyPolicy(lock_trigger=1.)
    elif case['exit_profile']=='tp_120pct':
        original = bot.calculate_dynamic_exit_levels
        def levels(symbol, history, side, entry):
            result = dict(original(symbol,history,side,entry))
            result['take_profit_price'] = entry + (result['take_profit_price']-entry)*1.2
            return result
        bot.calculate_dynamic_exit_levels = levels
    elif case['exit_profile']!='baseline':
        raise ValueError('Unknown exit profile')


def filtered_rows(latest, previous, case):
    rows = [latest.copy(), previous.copy()]
    for row in rows:
        if case['direction']=='short_only':row['sell_signal']=False
        if case['direction']=='long_only':row['buy_signal']=False
        if case['adx_min'] and not float(row.get('ADX',np.nan)) >= case['adx_min']:
            row['buy_signal']=row['sell_signal']=False
    return rows


def case_prepare(cache, case):
    def prepare(bot,symbol,intraday,hourly,observed_at):
        result,hours = cache.prepare(bot,symbol,intraday,hourly,observed_at)
        if pd.Timestamp(observed_at)==intraday.timestamp.iloc[-1]:
            if case['direction']=='short_only':result['sell_signal']=False
            if case['direction']=='long_only':result['buy_signal']=False
            if case['adx_min']:
                invalid=~result.ADX.ge(case['adx_min'])
                result.loc[invalid,['buy_signal','sell_signal']]=False
        return result,hours
    return prepare


class ObservedAccount(BacktestAccount):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.peak=self.initial_cash
        self.max_drawdown=0.;self.max_drawdown_pct=0.;self.daily={}

    def observe(self):
        equity = self.cash + sum(p['reserved_capital']+(1 if p['side']=='long' else -1)*p['size']*(p['mark_price']-p['entry_price']) for p in self.positions.values())
        self.peak=max(self.peak,equity)
        self.max_drawdown=max(self.max_drawdown,self.peak-equity)
        self.max_drawdown_pct=max(self.max_drawdown_pct,(self.peak-equity)/self.peak*100)
        if self.now is not None:self.daily[str(self.now.date())]=equity

    def advance(self,*a,**kw):
        result=super().advance(*a,**kw);self.observe();return result

    def mark(self,*a,**kw):
        result=super().mark(*a,**kw);self.observe();return result

    def open(self,*a,**kw):
        result=super().open(*a,**kw);self.observe();return result

    def close(self,*a,**kw):
        result=super().close(*a,**kw);self.observe();return result


def simulate(frames,start,end,costs,cache,case=None,symbols=None):
    case=case or BASELINE
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    bot=make_bot(frames,start,end,symbols=symbols)
    bot.position_policies = {}
    install_case(bot,case)
    account=ObservedAccount(bot.initial_capital,bot.symbols,costs)
    trades=[];positions={};last_rows={}

    def history(symbol,timestamp,at):
        full=cache.day(symbol,at)
        idx=int(full.timestamp.searchsorted(timestamp,side='right'))
        return full.iloc[max(0,idx-6):idx]

    def stream(symbol):
        raw=frames[symbol,'15min']
        raw=raw.loc[raw.timestamp.ge(start)&raw.timestamp.lt(end)]
        position=None;cooldown=None
        for row in raw.itertuples(index=False):
            at=row.timestamp+pd.Timedelta(minutes=15) if position else row.timestamp
            yield at, 0 if position else 1
            account.advance(at)
            account.mark(symbol,row.close if position else row.open,at)
            last_rows[symbol]=row
            if position is None:
                if cooldown is not None and at<cooldown:continue
                full=cache.day(symbol,at)
                idx=int(full.timestamp.searchsorted(row.timestamp,side='left'))
                if idx<2:continue
                buy=full.buy_signal.to_numpy();sell=full.sell_signal.to_numpy()
                if not ((buy[idx-1] and buy[idx-2]) or (sell[idx-1] and sell[idx-2])):
                    continue
                latest,previous=filtered_rows(full.iloc[idx-1],full.iloc[idx-2],case)
                direction=entry_side(latest,previous,at,cooldown)
                if direction is None:continue
                fill=account.fill_price(row.open,direction)
                size=bot.adjust_order_size(symbol,bot.TRADE_SIZE/fill)
                if not np.isfinite(size) or size<=0:continue
                opened=account.open(symbol,direction,row.open,size,at)
                if opened is None:continue
                bot._reset_exit_adjustments(symbol)
                bot.short_profit_lock_armed[symbol]=False
                bot.position_policies[symbol]=active_policy(bot).snapshot()
                bot.entry_sizes[symbol]=size
                position=dict(side=direction,entry_price=fill,entry_time=at,size=size,
                              balance_after_entry=account.cash,entry_row=latest)
            else:
                event_history=history(symbol,row.timestamp,at-pd.Timedelta(nanoseconds=1))
                decision=bot._evaluate_exit_event(symbol,event_history,position['side'],position['entry_price'],position['entry_time'],at,row.close)
                if decision.reason is None:continue
                execution=account.close(symbol,row.close,at)
                trades.append(dict(**execution,symbol=symbol,type=position['side'],
                    entry_price=position['entry_price'],entry_time=position['entry_time'],
                    exit_time=at,size=position['size'],exit_reason=decision.reason,
                    holding_hours=(at-position['entry_time']).total_seconds()/3600,
                    balance_after_entry=position['balance_after_entry'],balance_after_exit=account.cash))
                position=None;cooldown=next_entry_time(at)
        positions[symbol]=position

    streams={s:stream(s) for s in sorted(bot.symbols)}
    heap=[]
    with offline_io_guard():
        for symbol,iterator in streams.items():
            at,priority=next(iterator);heapq.heappush(heap,(at,priority,symbol))
        while heap:
            _,_,symbol=heapq.heappop(heap)
            try:at,priority=next(streams[symbol])
            except StopIteration:pass
            else:heapq.heappush(heap,(at,priority,symbol))
        account.advance(end)
        for symbol in account.positions:
            account.mark(symbol,last_rows[symbol].close,end)
    trades.sort(key=lambda t:(t['exit_time'],t['symbol']))
    values=[t['profit'] for t in trades]
    gains=sum(v for v in values if v>0);losses=-sum(v for v in values if v<0)
    equity=account.summary()
    monthly={};prior=account.initial_cash
    for day,value in sorted(account.daily.items()):
        month=day[:7];monthly[month]=monthly.get(month,0)+(value-prior);prior=value
    return dict(trades=trades,valuation=equity,
        open_positions={s:{k:v for k,v in pos.items() if k not in ('next_carry',)} for s,pos in account.positions.items()},
        metrics=dict(trades=len(trades),profit_factor=gains/losses if losses else None,
            win_rate_pct=sum(v>0 for v in values)/len(values)*100 if values else 0,
            net_equity_change=equity['net_equity_change'],realized_profit=equity['realized_profit'],
            max_observation_drawdown=account.max_drawdown,max_observation_drawdown_pct=account.max_drawdown_pct,
            negative_cash_observed=equity['negative_cash_observed'],monthly_equity_changes=monthly))


def compare_core(source,experiment,tolerance=1e-7):
    left,right=source['trades'],experiment['trades']
    if len(left)!=len(right):raise AssertionError(f'Trade count {len(left)} != {len(right)}')
    numeric=['entry_price','exit_price','size','profit','profit_pct','gross_profit','entry_fee',
             'exit_fee','carry_fee','holding_hours','balance_after_entry','balance_after_exit']
    for a,b in zip(left,right):
        for key in numeric:
            if not np.isclose(a[key],b[key],rtol=0,atol=tolerance):raise AssertionError((key,a[key],b[key]))
        for key in ('entry_time','exit_time'):
            if pd.Timestamp(a[key])!=pd.Timestamp(b[key]):raise AssertionError((key,a[key],b[key]))
        for key in ('symbol','type','exit_reason'):
            if str(a[key])!=str(b[key]):raise AssertionError((key,a[key],b[key]))
    for key in ('ending_available_cash','reserved_capital','unrealized_profit','realized_profit','open_costs_paid'):
        if not np.isclose(source['valuation'][key],experiment['valuation'][key],rtol=0,atol=tolerance):
            raise AssertionError((key,source['valuation'][key],experiment['valuation'][key]))
