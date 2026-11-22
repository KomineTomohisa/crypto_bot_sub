"""Scenario costs and chronological cash accounting. No live I/O or fee lookup."""
from copy import deepcopy
import math

import pandas as pd


def positive(value, *, zero=False):
    if isinstance(value, bool):
        raise ValueError('Boolean is not a cost or price')
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError('Expected a finite numeric value') from None
    if not math.isfinite(result) or (result < 0 if zero else result <= 0):
        raise ValueError('Expected a finite positive value')
    return result


def stamp(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError('Cost accounting requires timezone-aware timestamps')
    return result.tz_convert('Asia/Tokyo')


class BacktestAccount:
    def __init__(self, capital, symbols, config=None):
        self.cash = positive(capital)
        self.initial_cash = self.cash
        self.positions = {}
        self.closed = []
        self.now = None
        self.enabled = config is not None
        self.config = deepcopy(config) if self.enabled else {
            'name': 'legacy_zero_cost', 'fee_bps': {s: 0 for s in symbols},
            'spread_bps': 0, 'slippage_bps': 0, 'carry_bps_per_day': 0,
        }
        required = {'name', 'fee_bps', 'spread_bps', 'slippage_bps', 'carry_bps_per_day'}
        if (not isinstance(self.config, dict) or not required.issubset(self.config)
                or set(self.config) - required - {'source', 'notes'}):
            raise ValueError('Incomplete or unknown backtest cost configuration')
        if not isinstance(self.config['name'], str) or not self.config['name'].strip():
            raise ValueError('Cost scenario needs a name')
        fees = self.config['fee_bps']
        if not isinstance(fees, dict) or not set(symbols).issubset(fees):
            raise ValueError('Explicit fee_bps required for every symbol')
        for value in list(fees.values()) + [self.config[k] for k in ('spread_bps', 'slippage_bps', 'carry_bps_per_day')]:
            if positive(value, zero=True) >= 10000:
                raise ValueError('Cost basis points must be below 10000')
        self.friction = float(self.config['spread_bps']) / 20000 + float(self.config['slippage_bps']) / 10000
        if self.friction >= 1:
            raise ValueError('Execution friction must be below 100%')
        self.fees_paid = 0.
        self.carry_paid = 0.
        self.negative_cash_observed = False

    def fill_price(self, reference, side, *, closing=False):
        if side not in ('long', 'short'):
            raise ValueError('Unknown position side')
        buy = (side == 'long') != closing
        return positive(positive(reference) * (1 + self.friction if buy else 1 - self.friction))

    def advance(self, at):
        at = stamp(at)
        if self.now is not None and at < self.now:
            raise ValueError('Backtest account events must be chronological')
        for pos in self.positions.values():
            while pos['next_carry'] <= at:
                amount = pos['size'] * pos['mark_price'] * float(self.config['carry_bps_per_day']) / 10000
                pos['carry_fee'] += amount
                self.carry_paid += amount
                self.cash -= amount
                self.negative_cash_observed |= self.cash < 0
                pos['carry_events'].append({
                    'at': pos['next_carry'].isoformat(), 'amount': amount,
                    'reference_price': pos['mark_price'], 'reference_at': pos['mark_at'].isoformat(),
                })
                pos['next_carry'] += pd.Timedelta(days=1)
        self.now = at

    def open(self, symbol, side, reference, size, at):
        self.advance(at)
        if symbol in self.positions:
            raise ValueError('Position already open')
        fill = self.fill_price(reference, side)
        size = positive(size)
        reserve = positive(size * fill)
        fee = reserve * float(self.config['fee_bps'][symbol]) / 10000
        if self.cash < reserve + fee:
            return None
        cutoff = stamp(at).normalize() + pd.Timedelta(hours=6)
        if cutoff <= stamp(at):
            cutoff += pd.Timedelta(days=1)
        pos = dict(side=side, size=size, entry_price=fill, entry_reference=positive(reference),
                   entry_time=stamp(at), reserved_capital=reserve, entry_fee=fee,
                   carry_fee=0., carry_events=[], next_carry=cutoff,
                   mark_price=positive(reference), mark_at=stamp(at))
        self.cash -= reserve + fee
        self.fees_paid += fee
        self.positions[symbol] = pos
        return pos

    def mark(self, symbol, reference, at):
        self.advance(at)
        if symbol in self.positions:
            self.positions[symbol].update(mark_price=positive(reference), mark_at=stamp(at))

    def close(self, symbol, reference, at):
        self.advance(at)
        pos = self.positions[symbol]
        fill = self.fill_price(reference, pos['side'], closing=True)
        direction = 1 if pos['side'] == 'long' else -1
        gross = direction * pos['size'] * (fill - pos['entry_price'])
        reference_gross = direction * pos['size'] * (positive(reference) - pos['entry_reference'])
        fee = pos['size'] * fill * float(self.config['fee_bps'][symbol]) / 10000
        net = gross - pos['entry_fee'] - fee - pos['carry_fee']
        self.cash += pos['reserved_capital'] + gross - fee
        self.negative_cash_observed |= self.cash < 0
        self.fees_paid += fee
        result = dict(exit_price=fill, gross_profit=gross, reference_gross_profit=reference_gross,
                      execution_cost=reference_gross - gross, entry_fee=pos['entry_fee'],
                      exit_fee=fee, carry_fee=pos['carry_fee'], carry_events=deepcopy(pos['carry_events']),
                      profit=net, profit_pct=net / pos['reserved_capital'] * 100,
                      cost_status='scenario_estimate' if self.enabled else 'costs_excluded')
        self.closed.append(result)
        del self.positions[symbol]
        return result

    def summary(self):
        reserve = sum(p['reserved_capital'] for p in self.positions.values())
        open_costs = sum(p['entry_fee'] + p['carry_fee'] for p in self.positions.values())
        realized = sum(p['profit'] for p in self.closed)
        unrealized = sum((1 if p['side'] == 'long' else -1) * p['size'] * (p['mark_price'] - p['entry_price'])
                         for p in self.positions.values())
        if not math.isclose(self.cash + reserve, self.initial_cash + realized - open_costs, abs_tol=1e-6):
            raise ValueError('Backtest cost cash accounting does not reconcile')
        return dict(costs_included=self.enabled, cost_status='scenario_estimate' if self.enabled else 'costs_excluded',
                    closed_gross_profit=sum(p['gross_profit'] for p in self.closed),
                    closed_reference_gross_profit=sum(p['reference_gross_profit'] for p in self.closed),
                    closed_execution_cost=sum(p['execution_cost'] for p in self.closed),
                    cost_assumptions=deepcopy(self.config), execution_fees_paid=self.fees_paid,
                    carry_fees_paid=self.carry_paid, open_costs_paid=open_costs,
                    realized_profit=realized, ending_available_cash=self.cash, reserved_capital=reserve,
                    unrealized_profit=unrealized, ending_equity=self.cash + reserve + unrealized,
                    net_equity_change=realized + unrealized - open_costs,
                    negative_cash_observed=self.negative_cash_observed,
                    valuation_excludes_future_exit_costs=True,
                    carry_valuation='last_observed_reference_before_06JST_cutoff')
