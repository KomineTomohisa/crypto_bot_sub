"""Read-only forward comparison. Virtual accounts have no execution gateway.

Quotes and SR are sampled once per observation and shared by both books. Fills,
spread, carry valuation and the conservative cash reserve are estimates, not
exchange margin/fill emulation. A failed observer never controls live trading.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import queue
import threading
from types import FunctionType, MethodType, SimpleNamespace
import uuid

import numpy as np
import pandas as pd

from backtest_costs import BacktestAccount, positive, stamp
from market_quotes import checked_quote, parse_ticker
from strategy_decisions import entry_side, next_entry_time
from strategy_policy import BASELINE, CANDIDATE, policy
from strategy_recording import ObservationJournal, canonical, code_fingerprint, digest, json_value
from trading_execution import ExecutionError, fallback_exit, normalize_size


METHODS = ('_get_dynamic_profit_loss_settings', 'calculate_dynamic_exit_levels',
           '_sr_exit_params', '_apply_sr_exit_levels', '_sr_entry_guard_params',
           '_should_block_entry_by_sr', '_evaluate_exit_event', '_apply_short_profit_lock')


def isolated_strategy(methods, selected):
    """Bind only decision methods to fresh state, without the live bot or API."""
    from strategy_decisions import exit_decision, short_profit_stop, jst_timestamp
    state = SimpleNamespace(is_backtest=True, _record_inputs=False,
                            strategy_policy=selected, position_policies={},
                            short_profit_lock_armed={}, opposite_narrow_state={},
                            _last_valid_exit_levels={}, _exit_level_fallback_state={},
                            logger=logging.getLogger('strategy.shadow'), zones={}, at=None)

    def zones(symbol, **kwargs):
        key = digest({'symbol': symbol, **kwargs})
        return deepcopy(state.zones[key])

    namespace = dict(pd=pd, np=np, math=__import__('math'), fallback_exit=fallback_exit,
                     exit_decision=exit_decision, short_profit_stop=short_profit_stop,
                     jst_timestamp=jst_timestamp, record_bot_event=lambda *a, **k: None,
                     fetch_sr_zones=zones, default_source_from_env=lambda _: 'real',
                     now_jst=lambda: state.at)
    for name in METHODS:
        fn = methods[name]
        clone = FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)
        clone.__kwdefaults__ = fn.__kwdefaults__
        setattr(state, name, MethodType(clone, state))
    return state


class ShadowComparison:
    """Synchronous deterministic core; the worker owns all calls and persistence."""
    def __init__(self, account, symbols, capital, trade_size, costs, methods, *, fingerprint=None):
        self.symbols = sorted(symbols)
        self.trade_size = positive(trade_size)
        self.methods = methods
        self.identity = dict(schema='strategy_shadow_v1', account=str(account),
                             symbols=self.symbols, capital=positive(capital),
                             trade_size=self.trade_size, costs=deepcopy(costs),
                             code=fingerprint or code_fingerprint(),
                             policies=[BASELINE.snapshot(), CANDIDATE.snapshot()],
                             execution_model='observed_last_plus_scenario_costs_full_notional_reserve')
        self.books = {}
        for selected in (BASELINE, CANDIDATE):
            self.books[selected.version] = dict(
                account=BacktestAccount(capital, self.symbols, costs),
                strategy=isolated_strategy(methods, selected), cooldown={}, entry_bar={}, trades=[], entries={})
        self.last = {}
        self.observations = 0
        self.gaps = []
        self.actual = {}
        self.last_input = None

    def sr_queries(self, symbol):
        state = self.books[BASELINE.version]['strategy']
        entry = state._sr_entry_guard_params()
        exit_ = state._sr_exit_params()
        return [dict(symbol=symbol, timeframe=str(cfg['timeframe']), lookback_days=int(cfg['lookback_days']),
                     user_id=None, source='real', min_strength=float(cfg.get('min_strength', cfg.get('min_strength_fallback', 8))),
                     limit=int(cfg['limit'])) for cfg in (entry, exit_)]

    def observe(self, symbol, frame, quote, at, zones, rule):
        at = stamp(at)
        quote = checked_quote(quote, at)
        if symbol not in self.symbols:
            raise ValueError('Unknown shadow symbol')
        if symbol in self.last and at <= stamp(self.last[symbol]):
            return False
        if any(b['account'].now is not None and at < b['account'].now for b in self.books.values()):
            return False
        frame = frame.copy(deep=True)
        fresh = bool(len(frame) >= 3 and 'timestamp' in frame
                     and stamp(frame.iloc[-1]['timestamp']).floor('15min') == at.floor('15min'))
        if not fresh:
            frame = pd.DataFrame()
            self.gap('stale_or_missing_features', symbol, at)
        # Completed high/low from before an entry cannot cause a virtual close.
        if len(frame):
            for col in ('close', 'high', 'low'):
                frame.loc[frame.index[-1], col] = quote.price
        for name, book in self.books.items():
            account, state = book['account'], book['strategy']
            state.at, state.zones = at, zones
            account.mark(symbol, quote.price, at)
            pos = account.positions.get(symbol)
            if pos is not None:
                result = state._evaluate_exit_event(symbol, frame, pos['side'], pos['entry_price'],
                                                    pos['entry_time'], at, quote.price)
                if result.reason:
                    closed = account.close(symbol, quote.price, at)
                    book['trades'].append(dict(symbol=symbol, side=pos['side'], entry_time=pos['entry_time'],
                                               exit_time=at, entry_price=pos['entry_price'], size=pos['size'],
                                               reason=result.reason, strategy_policy=state.position_policies[symbol], **closed))
                    book['cooldown'][symbol] = next_entry_time(at)
            elif fresh:
                latest, previous = frame.iloc[-2], frame.iloc[-3]
                direction = entry_side(latest, previous, at, book['cooldown'].get(symbol), policy=state.strategy_policy)
                bar = str(latest['timestamp'])
                if direction and book['entry_bar'].get(symbol) != bar:
                    # No empty SR substitution on a failed read: both books skip entry.
                    if any(digest(q) not in zones for q in self.sr_queries(symbol)):
                        continue
                    blocked, _ = state._should_block_entry_by_sr(symbol, direction, quote.price)
                    if blocked:
                        continue
                    fill = account.fill_price(quote.price, direction)
                    try:
                        size = normalize_size(self.trade_size / fill, rule['sizeStep'], rule['minOrderSize'])
                    except ExecutionError:
                        self.gap('below_minimum_order_size', symbol, at)
                        continue
                    if size > positive(rule['maxOrderSize']):
                        continue
                    opened = account.open(symbol, direction, quote.price, float(size), at)
                    if opened:
                        book['entry_bar'][symbol] = bar
                        book['entries'][symbol + '|' + bar + '|' + direction] = dict(price=fill, at=at, size=float(size))
                        state.position_policies[symbol] = state.strategy_policy.snapshot()
                        state.short_profit_lock_armed[symbol] = False
                        state.opposite_narrow_state[symbol] = {'long': False, 'short': False}
                        state._last_valid_exit_levels = {k: v for k, v in state._last_valid_exit_levels.items() if k[0] != symbol}
        self.last[symbol] = at.isoformat()
        self.observations += 1
        self.last_input = dict(symbol=symbol, observed_at=at.isoformat(), quoted_at=str(quote.quoted_at),
                               price=quote.price, features=digest(frame), sr=digest(zones), rule=deepcopy(rule))
        return True

    def gap(self, reason, symbol, at):
        self.gaps.append(dict(reason=reason, symbol=symbol, at=str(at)))
        self.gaps = self.gaps[-1000:]

    def actual_result(self, payload):
        # Exchange results are evidence only; never a candidate fill source.
        self.actual[payload['event_id']] = json_value(payload)

    def report(self):
        books = {name: dict(**book['account'].summary(), trades=len(book['trades']),
                            open_positions=len(book['account'].positions)) for name, book in self.books.items()}
        baseline, candidate = books[BASELINE.version], books[CANDIDATE.version]
        trades = {t['trade_id']: t for p in self.actual.values() for t in p.get('trades', []) if t.get('trade_id')}
        actual_profit = sum(float(t['pnl']) for t in trades.values() if t.get('pnl') is not None)
        differences = []
        for receipt in self.actual.values():
            if receipt.get('purpose') != 'entry':
                continue
            key = receipt['symbol'] + '|' + str(receipt.get('decision_bar')) + '|' + ('long' if receipt['side'] == 'BUY' else 'short')
            real_price = receipt['result'].get('execution_price')
            for name, book in self.books.items():
                simulated = book['entries'].get(key)
                differences.append(dict(event_id=receipt['event_id'], policy=name,
                                        same_signal_entry=simulated is not None,
                                        actual_minus_estimated_price=float(real_price) - simulated['price'] if real_price is not None and simulated else None))
        return dict(observations=self.observations, books=books,
                    candidate_minus_baseline_equity=candidate['ending_equity'] - baseline['ending_equity'],
                    actual_order_results=len(self.actual), actual_recorded_pnl=actual_profit,
                    actual_comparison_status='different_holdings_and_fills_not_a_controlled_profit_comparison',
                    baseline_shadow_realized_minus_actual_recorded_pnl=baseline['realized_profit'] - actual_profit,
                    entry_fill_differences=differences[-100:],
                    gaps=len(self.gaps), gaps_are_unobserved_prices=True,
                    fills_are_estimates=True, last_input=self.last_input)

    def snapshot(self):
        books = {}
        for name, book in self.books.items():
            account, state = book['account'], book['strategy']
            books[name] = dict(account={key: deepcopy(getattr(account, key)) for key in (
                'cash', 'initial_cash', 'positions', 'closed', 'now', 'fees_paid', 'carry_paid', 'negative_cash_observed')},
                cooldown=book['cooldown'], entry_bar=book['entry_bar'], trades=book['trades'], entries=book['entries'],
                position_policies=state.position_policies, armed=state.short_profit_lock_armed,
                narrow=state.opposite_narrow_state,
                levels=[[list(k), v] for k, v in state._last_valid_exit_levels.items()])
        return json_value(dict(identity=self.identity, books=books, last=self.last, observations=self.observations,
                               gaps=self.gaps, actual=self.actual, last_input=self.last_input, report=self.report()))

    def restore(self, saved):
        if saved['identity'] != json_value(self.identity) or set(saved['books']) != set(self.books):
            raise ValueError('Shadow configuration/source changed; use a new comparison directory')
        for name, stored in saved['books'].items():
            book, state = self.books[name], self.books[name]['strategy']
            data = deepcopy(stored['account'])
            data['now'] = stamp(data['now']) if data['now'] else None
            for pos in data['positions'].values():
                for key in ('entry_time', 'next_carry', 'mark_at'):
                    pos[key] = stamp(pos[key])
                if pos['side'] not in ('long', 'short'):
                    raise ValueError('Invalid persisted virtual position')
                for key in ('entry_price', 'entry_reference', 'size', 'reserved_capital', 'mark_price'):
                    positive(pos[key])
            for key, value in data.items():
                if key not in ('cash', 'initial_cash', 'positions', 'closed', 'now', 'fees_paid', 'carry_paid', 'negative_cash_observed'):
                    raise ValueError('Unknown virtual account field')
                setattr(book['account'], key, value)
            book['account'].summary()  # Cash identity must still reconcile.
            book['cooldown'] = {s: stamp(t) for s, t in stored['cooldown'].items()}
            book['entry_bar'], book['trades'] = stored['entry_bar'], stored['trades']
            book['entries'] = stored['entries']
            state.position_policies = {s: policy(p).snapshot() for s, p in stored['position_policies'].items()}
            for s in data['positions']:
                if state.position_policies.get(s) != policy(name).snapshot():
                    raise ValueError('Virtual position lost its entry policy')
            if any(type(v) is not bool for v in stored['armed'].values()):
                raise ValueError('Invalid saved profit lock')
            state.short_profit_lock_armed, state.opposite_narrow_state = stored['armed'], stored['narrow']
            state._last_valid_exit_levels = {tuple(k): v for k, v in stored['levels']}
        self.last, self.observations, self.gaps, self.actual, self.last_input = (saved[k] for k in ('last', 'observations', 'gaps', 'actual', 'last_input'))


class ShadowFile:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'state.json'
        # OS lock releases on crash; a second bot cannot overwrite this study.
        self.lock = (self.directory / 'writer.lock').open('a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                self.lock.seek(0)
                if not self.lock.read(1):
                    self.lock.write(b'0'); self.lock.flush()
                self.lock.seek(0)
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise

    def load(self):
        if not self.path.exists():
            return None
        content = json.loads(self.path.read_text(encoding='utf-8'))
        if digest(content['state']) != content['sha256']:
            raise ValueError('Corrupt shadow state; comparison must not silently reset')
        return content['state']

    def save(self, core):
        state = core.snapshot()
        tmp = self.directory / ('.state.' + uuid.uuid4().hex + '.tmp')
        try:
            with tmp.open('xb') as stream:
                os.chmod(tmp, 0o600)
                stream.write(canonical(dict(state=state, sha256=digest(state))))
                stream.flush(); os.fsync(stream.fileno())
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)

    def close(self):
        self.lock.close()


class PublicShadowReader:
    """Only public GET endpoints. No key, private API or trading service."""
    def __init__(self, mapping):
        import requests
        self.mapping = dict(mapping)
        self.session = requests.Session()
        self.rules = None

    def get(self, endpoint, params=None):
        with self.session.get('https://api.coin.z.com/public/v1/' + endpoint, params=params, timeout=(3, 5)) as response:
            response.raise_for_status()
            data = response.json()
        if data.get('status') != 0:
            raise ValueError('Shadow public API unavailable')
        return data

    def read(self, symbol):
        status = self.get('status')['data']
        if status.get('status') != 'OPEN':
            raise ValueError('Exchange is not open for shadow observations')
        if self.rules is None:
            self.rules = {r['symbol']: r for r in self.get('symbols')['data']}
        result = self.get('ticker', {'symbol': self.mapping[symbol]})
        at = datetime.now(timezone.utc)
        rule = self.rules[self.mapping[symbol]]
        return parse_ticker(result, self.mapping[symbol], at), at, {k: rule[k] for k in ('sizeStep', 'minOrderSize', 'maxOrderSize')}

    def close(self):
        self.session.close()


class ShadowWorker:
    def __init__(self, build, directory, reader, sr_reader, logger, *, capacity=64):
        self.queue = queue.Queue(maxsize=capacity)
        self.build, self.directory, self.reader, self.sr_reader, self.logger = build, directory, reader, sr_reader, logger
        self.stopping = threading.Event()
        self.failed = False
        self.dropped = 0
        self.thread = threading.Thread(target=self.run, name='strategy-shadow', daemon=True)
        self.thread.start()

    def offer(self, kind, payload):
        if self.failed or self.stopping.is_set():
            return False
        try:
            self.queue.put_nowait((kind, payload))
            return True
        except queue.Full:
            self.dropped += 1
            self.logger.warning('Shadow queue full; observation dropped (count=%s)', self.dropped)
            return False

    def run(self):
        file = None
        journal = None
        try:
            core = self.build()
            file = ShadowFile(self.directory)
            saved = file.load()
            if saved is not None:
                core.restore(saved)
                core.gap('restart_unobserved_interval', '*', datetime.now(timezone.utc))
            journal = ObservationJournal(Path(self.directory) / 'observations',
                                         {'identity': core.identity, 'initial_state': core.snapshot()})
            def gap(reason, symbol, at):
                journal.append('shadow_gap', dict(reason=reason, symbol=symbol, at=str(at)))
                core.gap(reason, symbol, at)
            while not self.stopping.is_set() or not self.queue.empty():
                try:
                    kind, payload = self.queue.get(timeout=.1)
                except queue.Empty:
                    continue
                try:
                    if kind == 'actual':
                        journal.append('shadow_actual_result', payload)
                        core.actual_result(payload)
                    else:
                        symbol, frame = payload
                        try:
                            quote, at, rule = self.reader.read(symbol)
                        except Exception as exc:
                            gap('quote_unavailable_' + type(exc).__name__, symbol, datetime.now(timezone.utc))
                            file.save(core)
                            continue
                        zones = {}
                        for query in core.sr_queries(symbol):
                            try:
                                rows = self.sr_reader(**query) or []
                                # Match the live SR observation contract. Database
                                # IDs (UUID) and unrelated columns are not inputs
                                # to either decision and must not enter the journal.
                                fields = ('zone_type', 'price_low', 'price_high', 'strength', 'touches', 'last_touched_at')
                                zones[digest(query)] = [{key: row.get(key) for key in fields} for row in rows]
                            except Exception as exc:
                                gap('sr_unavailable_' + type(exc).__name__, symbol, at)
                        # SR reads may take time; validate quote freshness again.
                        at = datetime.now(timezone.utc)
                        try:
                            checked_quote(quote, at)
                        except Exception as exc:
                            gap('expired_during_sr_' + type(exc).__name__, symbol, at)
                            file.save(core)
                            continue
                        journal.append('shadow_input', dict(symbol=symbol, frame=frame, quote=quote,
                                                            zones=zones, rule=rule), observed_at=at)
                        core.observe(symbol, frame, quote, at, zones, rule)
                    if self.dropped:
                        gap('queue_dropped_' + str(self.dropped), '*', datetime.now(timezone.utc))
                        self.dropped = 0
                    file.save(core)
                    journal.append('shadow_checkpoint', dict(state_sha256=digest(core.snapshot()), report=core.report()))
                    self.logger.info('Shadow comparison: %s', json.dumps(json_value(core.report()), ensure_ascii=False))
                finally:
                    self.queue.task_done()
        except Exception as exc:
            self.failed = True
            self.logger.error('Shadow comparison disabled (%s); live trading unaffected', type(exc).__name__)
        finally:
            if journal is not None:
                try:
                    journal.close({'observer_failed': self.failed})
                except Exception:
                    self.logger.warning('Shadow journal close failed; live trading unaffected')
            if file:
                file.close()
            self.reader.close()

    def close(self, timeout=1):
        self.stopping.set()
        self.thread.join(timeout)


def offer_observation(bot, symbol, frame):
    """Bounded work in the live loop: copy six rows and enqueue, never I/O."""
    if not getattr(bot, 'shadow_costs', None) or not bot.exchange_settings_gmo.get('live_trade'):
        return
    try:
        worker = getattr(bot, '_shadow_worker', None)
        if worker is None:
            account = bot._live_execution_service.account
            methods = {name: getattr(type(bot), name) for name in METHODS}
            config = dict(account=account, symbols=list(bot.symbols), capital=bot.initial_capital,
                          trade_size=bot.TRADE_SIZE, costs=deepcopy(bot.shadow_costs), methods=methods)
            directory = Path(bot.base_dir) / 'shadow' / hashlib.sha256(str(account).encode()).hexdigest()[:20]
            from db import fetch_sr_zones
            worker = ShadowWorker(lambda: ShadowComparison(**config), directory,
                                  PublicShadowReader(bot.symbol_mapping), fetch_sr_zones, bot.logger)
            bot._shadow_worker = worker
        copied = frame.tail(6).copy(deep=True) if frame is not None else pd.DataFrame()
        worker.offer('observation', (symbol, copied))
    except Exception as exc:
        bot.logger.warning('Shadow observation unavailable (%s); live trading unaffected', type(exc).__name__)


def verify_shadow_journal(path):
    """Replay captured virtual observations offline, with no bot startup or I/O."""
    import ast
    from market_quotes import MarketQuote
    from strategy_recording import read_journal
    events = read_journal(path)
    metadata = events[0]['payload']
    identity = metadata['identity']
    if identity['code'] != code_fingerprint():
        raise ValueError('Shadow replay requires the matching source version')
    tree = ast.parse(Path(__file__).with_name('crypto_bot_sub.py').read_text(encoding='utf-8-sig'))
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CryptoTradingBot')
    functions = [n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    namespace = {'pd': pd, 'np': np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), 'shadow_decision_methods', 'exec'), namespace)
    core = ShadowComparison(identity['account'], identity['symbols'], identity['capital'],
                            identity['trade_size'], identity['costs'], {n: namespace[n] for n in METHODS})
    core.restore(metadata['initial_state'])
    checkpoints = 0
    for event in events[1:]:
        kind, payload = event['kind'], event['payload']
        if kind == 'shadow_actual_result':
            core.actual_result(payload)
        elif kind == 'shadow_gap':
            core.gap(**payload)
        elif kind == 'shadow_input':
            frame = pd.DataFrame(payload['frame']['rows'], columns=payload['frame']['columns'])
            if len(frame):
                frame['timestamp'] = pd.to_datetime(frame['timestamp'])
            quote = MarketQuote(**payload['quote'])
            core.observe(payload['symbol'], frame, quote, event['observed_at'], payload['zones'], payload['rule'])
        elif kind == 'shadow_checkpoint':
            if digest(core.snapshot()) != payload['state_sha256']:
                raise ValueError('Shadow replay differs at checkpoint ' + str(event['sequence']))
            checkpoints += 1
    return dict(checkpoints=checkpoints, report=core.report())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Replay a shadow observation journal without orders or network')
    parser.add_argument('journal')
    arguments = parser.parse_args()
    print(json.dumps(json_value(verify_shadow_journal(arguments.journal)), ensure_ascii=False, indent=2))
