"""Offline experiments using the actual historical execution methods.

No bot startup, DB, dotenv, orders or notifications are imported. Current-bar
entry indicators are deliberately absent: entry consumes two CLOSED rows only.
The daily feature cache is certified against the uncached path before searches.
"""
import ast
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import hashlib
import logging
import math
from pathlib import Path
import time
from types import MethodType

import numpy as np
import pandas as pd

from backtest_data import backtest_state, load_backtest_candles, excel_backtest_inputs
from live_execution import LiveExecutionMixin
from replay_observations import offline_signal_bot
from signal_scoring import MACD_SCORING_VERSION, macd_scoring_spec
from strategy_decisions import (DECISION_VERSION, entry_side, exit_decision,
    short_profit_stop, jst_timestamp, next_entry_time, symbol_priority)
from strategy_inputs import (FEATURE_HISTORY_VERSION, feature_history_spec,
    prepare_strategy_inputs, historical_observation, RAW_COLUMNS)
from strategy_recording import code_fingerprint, json_value, record_bot_event
from trading_execution import fallback_exit

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ['ltc_jpy','xrp_jpy','sol_jpy','doge_jpy','eth_jpy','bcc_jpy','ada_jpy']


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def read_frames(directory):
    frames = {}
    for symbol in SYMBOLS:
        for interval in ('15min', '1hour'):
            chunks = []
            for path in sorted(Path(directory).glob(f'{symbol}_{interval}_*.json')):
                payload = json.loads(path.read_text())
                rows = next(c['ohlcv'] for c in payload['data']['candlestick'] if c['type'] == interval)
                df = pd.DataFrame(rows, columns=['open','high','low','close','volume','timestamp'])
                df['timestamp'] = pd.to_datetime(df.timestamp, unit='ms', utc=True).dt.tz_convert('Asia/Tokyo')
                for col in RAW_COLUMNS[1:]:
                    df[col] = pd.to_numeric(df[col], errors='raise')
                chunks.append(df)
            frame = pd.concat(chunks).drop_duplicates().sort_values('timestamp').reset_index(drop=True)
            if frame.timestamp.duplicated().any():
                raise ValueError('Conflicting raw candle timestamps')
            values = frame[['open','high','low','close','volume']]
            if (not np.isfinite(values.to_numpy()).all() or not values[['open','high','low','close']].gt(0).all().all()
                    or not frame.volume.ge(0).all() or not frame.high.ge(values[['open','close','low']].max(axis=1)).all()
                    or not frame.low.le(values[['open','close','high']].min(axis=1)).all()):
                raise ValueError('Invalid raw candle values')
            frames[symbol, interval] = frame
    return frames


def make_bot(frames, start, end, *, symbols=None, prepare=None):
    bot = offline_signal_bot({'live_trade':False,'is_backtest':True,'entry_thresholds':{}})
    bot.logger = logging.getLogger('offline-search')
    bot.logger.addHandler(logging.NullHandler()); bot.logger.propagate = False
    bot.symbols = symbols or SYMBOLS
    bot.initial_capital = 200000.; bot.TRADE_SIZE = 40000.; bot.total_profit = 0.
    bot.entry_sizes = {}; bot.entry_prices = {}; bot.entry_times = {}; bot.positions = {}
    bot.opposite_narrow_state = {}; bot._last_valid_exit_levels = {}; bot.short_profit_lock_armed = {}
    bot.SHORT_PROFIT_LOCK_TRIGGER_PCT = .75; bot.SHORT_PROFIT_LOCK_LEVEL_PCT = .25
    bot._record_inputs = False
    tree = ast.parse((ROOT/'crypto_bot_sub.py').read_text(encoding='utf-8-sig'))
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='CryptoTradingBot')
    initializer = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name=='__init__')
    for node in ast.walk(initializer):
        if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Attribute):
            if node.targets[0].attr in ('min_order_sizes','entry_thresholds','sentiment'):
                setattr(bot, node.targets[0].attr, ast.literal_eval(node.value))
    required = {'backtest','_run_backtest','adjust_order_size','calculate_dynamic_exit_levels',
                '_evaluate_exit_event','_apply_short_profit_lock','_last5_flags_str','_get_dynamic_profit_loss_settings'}
    nodes = [n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name in required]
    namespace = dict(globals(), now_jst=lambda: pd.Timestamp(end).to_pydatetime())
    namespace['prepare_strategy_inputs'] = prepare or prepare_strategy_inputs
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])), 'actual_search_methods', 'exec'), namespace)
    for name in required:
        setattr(bot, name, MethodType(namespace[name], bot))
    def cached(symbol, interval, date, **kwargs):
        frame = frames[symbol, interval]
        return frame.loc[frame.timestamp.dt.strftime('%Y%m%d') == date].copy()
    bot.get_cached_data = cached
    bot._reset_exit_adjustments = MethodType(LiveExecutionMixin._reset_exit_adjustments, bot)
    bot._apply_sr_exit_levels = lambda **kw: dict(take_profit_price=kw['base_take_profit'], stop_loss_price=kw['base_stop_loss'], applied=False)
    bot.log_entry = bot.log_exit = bot.save_positions = lambda *a, **kw: None
    bot.save_trade_logs_to_excel = bot._generate_excel_report_from_trade_logs = lambda *a, **kw: None
    bot.captured = {}
    bot.output_trade_summary = lambda logs: bot.captured.update(trades=deepcopy(logs))
    def save(results, days, start_profit):
        bot.captured.update(results=deepcopy(results), valuation=deepcopy(bot._backtest_summary))
    bot.save_backtest_result = save
    return bot


class DailyDecisionCache:
    """Causal closed-row cache; no forming indicators are supplied to entries."""
    def __init__(self, frames):
        self.frames = frames
        self.days = {}
        self.bot = offline_signal_bot({'live_trade':False,'is_backtest':True,'entry_thresholds':{}})
        self.bot.logger = logging.getLogger('cached-features')
        self.bot.logger.addHandler(logging.NullHandler()); self.bot.logger.propagate = False
        self.bot._record_inputs = False
        self.cache_dir = ROOT/'outputs/backtests/search_20261122/feature_cache'
        self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.fingerprint = str(code_fingerprint(ROOT))
        self.adapter_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def day(self, symbol, observed_at):
        day = pd.Timestamp(observed_at).normalize()
        key = symbol, day
        if key not in self.days:
            raw = self.frames[symbol, '15min']
            raw = raw.loc[raw.timestamp.ge(day-pd.Timedelta(days=2)) & raw.timestamp.lt(day+pd.Timedelta(days=1))].copy()
            identity = hashlib.sha256(raw.timestamp.array.asi8.tobytes()+raw[['open','high','low','close','volume']].to_numpy(dtype='float64').tobytes()).hexdigest()
            expected = dict(input_hash=identity,source=self.fingerprint,adapter=self.adapter_hash)
            path=self.cache_dir/f'{symbol}_{day:%Y%m%d}.pkl.gz'
            metadata=path.with_suffix('.meta.json')
            if path.exists() and metadata.exists():
                saved=json.loads(metadata.read_text())
                if (all(saved.get(k)==v for k,v in expected.items())
                        and saved.get('sha256')==hashlib.sha256(path.read_bytes()).hexdigest()):
                    # Only self-generated, checksum-verified local caches are read.
                    self.days[key]=pd.read_pickle(path,compression='gzip')
                    return self.days[key]
            # Signals do not read hourly columns; check this source assumption.
            signals = self.bot.generate_signals_with_sentiment(symbol, self.bot.build_features(raw), pd.DataFrame())
            self.days[key] = signals.reset_index(drop=True)
            self.days[key].to_pickle(path,compression='gzip')
            write_json(metadata,{**expected,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        return self.days[key]

    def prepare(self, bot, symbol, intraday, hourly, observed_at):
        at = pd.Timestamp(observed_at)
        timestamp = intraday.timestamp.iloc[-1]
        at_open = at == timestamp
        full = self.day(symbol, at)
        if at_open:
            # Only two preceding closed rows are used for entry. Blank every
            # non-raw forming value so the cache cannot expose future OHLCV.
            past = full.loc[full.timestamp.lt(timestamp)].tail(5)
            forming = intraday[RAW_COLUMNS].tail(1).copy()
            forming['buy_signal'] = False
            forming['sell_signal'] = False
            result = pd.concat([past, forming], ignore_index=True)
        else:
            result = full.loc[full.timestamp.le(timestamp)].tail(6).copy().reset_index(drop=True)
        return result, pd.DataFrame()


def compare_trade_paths(left, right):
    # Forming signal-history plots are intentionally not part of the cache's
    # certification. Orders, state, accounting and closed trade evidence are.
    for key in ('trades','results','valuation'):
        if json_value(left[key]) != json_value(right[key]):
            raise AssertionError('Cached/uncached mismatch: '+key)


@contextmanager
def offline_io_guard():
    import socket
    import smtplib
    import requests
    import sqlalchemy
    import dotenv
    def forbidden(*a, **kw):
        raise RuntimeError('Network, DB, dotenv and SMTP forbidden during research computation')
    targets = [(socket.socket,'connect'),(socket,'create_connection'),(smtplib,'SMTP'),
               (requests.sessions.Session,'request'),(sqlalchemy,'create_engine'),(dotenv,'load_dotenv')]
    original = [(owner,name,getattr(owner,name)) for owner,name in targets]
    try:
        for owner,name in targets:
            setattr(owner,name,forbidden)
        yield
    finally:
        for owner,name,value in original:
            setattr(owner,name,value)


def execute(frames, start, end, costs, cache=None, symbols=None):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    bot = make_bot(frames,start,end,symbols=symbols,prepare=cache.prepare if cache else None)
    begun = time.monotonic()
    with offline_io_guard():
        bot.backtest((end-start).days, costs=costs)
    return bot.captured, time.monotonic()-begun
