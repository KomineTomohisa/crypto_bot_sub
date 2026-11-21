"""Exercise the actual bot methods without importing its startup/notification modules."""

import ast
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import datetime as dt
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import time
import math
import sys
import numpy as np
import pandas as pd
import pytest
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from trading_execution import ExchangeGateway, ExecutionError, ExchangeMaintenance, fallback_exit
from exchange_availability import ExchangeAvailability
from signal_scoring import MACD_SCORING_VERSION, causal_macd_scores, macd_scoring_spec
from backtest_data import backtest_state, load_backtest_candles, excel_backtest_inputs
from strategy_decisions import DECISION_VERSION, entry_side, exit_decision, short_profit_stop, jst_timestamp, next_entry_time, symbol_priority
from strategy_inputs import FEATURE_HISTORY_VERSION, feature_history_spec, prepare_strategy_inputs, historical_observation, candle_api_dates, candle_cache_valid, candle_partition_complete, entry_input_quality
from strategy_recording import record_bot_event, close_recorder
from test_execution import service, FakeBot, records

SOURCE = Path("crypto_bot_sub.py").read_text(encoding="utf-8-sig")
TREE = ast.parse(SOURCE)
BOT = next(
    n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "CryptoTradingBot"
)
API = next(
    n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "GMOCoinAPI"
)


def method(owner, name):
    node = deepcopy(
        next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)
    )
    ns = {
        "pd": pd,
        "np": np,
        "ExchangeGateway": ExchangeGateway,
        "ExecutionError": ExecutionError,
        "ExchangeMaintenance": ExchangeMaintenance,
        "ExchangeAvailability": ExchangeAvailability,
        "logging": logging,
        "fallback_exit": fallback_exit,
        "MACD_SCORING_VERSION": MACD_SCORING_VERSION,
        "causal_macd_scores": causal_macd_scores,
        "macd_scoring_spec": macd_scoring_spec,
        "backtest_state": backtest_state,
        "load_backtest_candles": load_backtest_candles,
        "excel_backtest_inputs": excel_backtest_inputs,
        "DECISION_VERSION": DECISION_VERSION,
        "entry_side": entry_side,
        "exit_decision": exit_decision,
        "short_profit_stop": short_profit_stop,
        "jst_timestamp": jst_timestamp,
        "next_entry_time": next_entry_time,
        "symbol_priority": symbol_priority,
        "FEATURE_HISTORY_VERSION": FEATURE_HISTORY_VERSION,
        "feature_history_spec": feature_history_spec,
        "prepare_strategy_inputs": prepare_strategy_inputs,
        "historical_observation": historical_observation,
        "candle_api_dates": candle_api_dates,
        "candle_cache_valid": candle_cache_valid,
        "candle_partition_complete": candle_partition_complete,
        "entry_input_quality": entry_input_quality,
        "os": __import__('os'),
        "random": __import__('random'),
        "datetime": datetime,
        "record_bot_event": record_bot_event,
        "close_recorder": close_recorder,
        "time": SimpleNamespace(sleep=lambda _: None, time=time.time),
        "json": json,
        "requests": requests,
        "_is_dns_temp_failure": lambda e: False,
        "math": math,
        "dt": dt,
        "JST": timezone(timedelta(hours=9)),
        "now_jst": lambda: datetime.now(timezone(timedelta(hours=9))),
        "HTTPAdapter": HTTPAdapter,
        "Retry": Retry,
        "hashlib": hashlib,
        "uuid": __import__('uuid'),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            "actual_bot_method",
            "exec",
        ),
        ns,
    )
    return ns[name], ns


@pytest.mark.parametrize("failure", ["timeout", "http500", "invalid_json"])
def test_actual_request_post_is_never_retried(failure):
    fn, ns = method(API, "_request")
    calls = []

    def post(*a, **kw):
        calls.append(kw)
        if failure == "timeout":
            raise requests.exceptions.ReadTimeout("test")
        if failure == "http500":
            raise requests.exceptions.HTTPError("500")
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: (_ for _ in ()).throw(ValueError()),
            close=lambda: None,
        )

    ns["HTTP_SESSION"] = SimpleNamespace(post=post)
    api = SimpleNamespace(
        base_url="https://offline.invalid",
        api_key="dummy",
        _sign=lambda *a: "signature",
    )
    assert fn(api, "POST", "/v1/order", data={"size": "1"})["status"] == -1
    assert len(calls) == 1


def test_http_adapter_has_no_post_or_connect_retry():
    node = deepcopy(
        next(
            n
            for n in TREE.body
            if isinstance(n, ast.FunctionDef) and n.name == "_make_session"
        )
    )
    ns = {"requests": requests, "HTTPAdapter": HTTPAdapter, "Retry": Retry}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            "session",
            "exec",
        ),
        ns,
    )
    session = ns["_make_session"]()
    policy = session.get_adapter("https://").max_retries
    assert "POST" not in policy.allowed_methods and policy.connect == 0
    session.close()


@pytest.mark.parametrize(
    "index",
    [pd.RangeIndex(6), pd.RangeIndex(1, 7), pd.date_range("2026-01-01", periods=6)],
)
def test_actual_ema_filter_is_index_invariant(index):
    fn, _ = method(BOT, "generate_signals_with_sentiment")
    bot = SimpleNamespace(
        logger=logging.getLogger("test"), generate_signals=lambda symbol, df, h: df
    )
    frame = pd.DataFrame(
        {
            "open": 102.0,
            "high": 103.0,
            "low": 101.0,
            "close": 102.0,
            "volume": 1.0,
            "EMA_long": 100.0,
            "buy_signal": True,
            "sell_signal": True,
        },
        index=index,
    )
    result = fn(bot, "ltc_jpy", frame, pd.DataFrame())
    assert not result.buy_signal.any() and not result.sell_signal.any()


def test_actual_ema_invalid_values_are_blocked():
    fn, _ = method(BOT, "generate_signals_with_sentiment")
    bot = SimpleNamespace(
        logger=logging.getLogger("test"), generate_signals=lambda symbol, df, h: df
    )
    frame = pd.DataFrame(
        {
            "open": [100.0] * 4,
            "high": [100.0] * 4,
            "low": [100.0] * 4,
            "close": [100.0] * 4,
            "volume": [1.0] * 4,
            "EMA_long": [0.0, np.nan, np.inf, 100.0],
            "buy_signal": [True] * 4,
            "sell_signal": [True] * 4,
        }
    )
    result = fn(bot, "ltc_jpy", frame, pd.DataFrame())
    assert result.buy_signal.tolist() == [False, False, False, True]


def test_actual_equity_uses_equity_field(store):
    svc, api = service(store)
    bot = FakeBot(svc)
    bot.exchange_settings_gmo = {"live_trade": True}
    fn, _ = method(BOT, "get_total_balance")
    assert fn(bot) == 100000


def test_actual_funds_check_blocks_small_yen_balance_both_sides(store, monkeypatch):
    svc, api = service(store)
    bot = FakeBot(svc)
    bot.exchange_settings_gmo = {"live_trade": True}
    original = api._request

    def request(m, p, params=None, data=None):
        if p == "/v1/account/margin":
            return {
                "status": 0,
                "data": {
                    "actualProfitLoss": "1000",
                    "availableAmount": "1000",
                    "margin": "0",
                    "profitLoss": "0",
                },
            }
        return original(m, p, params, data)

    api._request = request
    monkeypatch.setenv("GMO_MARGIN_LEVERAGE", "2")
    fn, _ = method(BOT, "check_sufficient_funds")
    assert not fn(bot, "ltc_jpy", "buy", 100)
    assert not fn(bot, "ltc_jpy", "sell", 100)


def test_live_risk_sizing_does_not_round_up_to_exchange_minimum(store):
    svc, api = service(store)
    bot = FakeBot(svc)
    bot.exchange_settings_gmo = {"live_trade": True}
    bot.TRADE_SIZE = 40
    fn, _ = method(BOT, "adjust_position_risk")
    with pytest.raises(ExecutionError):
        fn(bot, "ltc_jpy", "long", 100, 200000)
    assert not api.posts


def test_live_risk_sizing_reports_actual_rounded_notional(store):
    svc, api = service(store)
    bot = FakeBot(svc)
    bot.exchange_settings_gmo = {"live_trade": True}
    bot.TRADE_SIZE = 1050
    fn, _ = method(BOT, "adjust_position_risk")
    assert fn(bot, "ltc_jpy", "short", 100, 200000) == (10, 1000)


def test_actual_handle_entry_preserves_execution_price_time_and_snapshot(store):
    svc, api = service(store)
    bot = FakeBot(svc)
    bot.exchange_settings_gmo = {"live_trade": True}
    bot.initial_capital = 200000
    bot.total_profit = 0
    bot.notification_settings = {"send_on_entry": False}
    bot.adjust_position_risk = lambda *a: (10, 1000)
    bot.get_current_price = lambda symbol: 999
    bot.check_sufficient_funds = lambda *a: True
    bot._should_block_entry_by_sr = lambda *a: (False, "")
    bot._build_entry_explanation = lambda **kw: ({}, "", {})
    bot._current_strategy_version = lambda: "test"
    bot.execute_order_with_confirmation = lambda symbol, order_type, size, **kw: (
        bot._execute_live_entry(symbol, order_type, size, **kw)
    )
    bot._get_margin_positions_safe = lambda symbol: {"status": 0, "data": {"list": []}}
    bot.log_entry = lambda *a: None
    bot.sentiment = {}
    fn, _ = method(BOT, "_handle_entry")
    signal = pd.Series(
        {
            "RSI": 60.0,
            "CCI": 100.0,
            "ATR": 1.0,
            "ADX": 30.0,
            "plus_di14": 40.0,
            "minus_di14": 10.0,
            "EMA_short": 100.0,
            "EMA_long": 99.0,
            "buy_score": 0.5,
            "sell_score": 0.7,
            "close": 88.0,
            "timestamp": pd.Timestamp("2026-11-16 01:00:00+09:00"),
            "adx_score_long": 0.33,
        }
    )
    logs = []
    fn(bot, "ltc_jpy", "long", signal, {}, logs)
    assert bot.entry_prices["ltc_jpy"] == 100
    assert bot.entry_times["ltc_jpy"] == datetime.fromisoformat(api.now)
    assert records(store, "positions")[0]["avg_entry_price"] == 100
    assert (
        records(store, "bot_order_intents")[0]["context"]["scores"]["adx_score_long"]
        == 0.33
    )
    assert logs[0]["price"] == 100


def test_actual_dynamic_exit_fallback_retains_last_valid_level():
    fn, _ = method(BOT, "calculate_dynamic_exit_levels")
    base, _ = method(BOT, "_get_dynamic_profit_loss_settings")
    bot = SimpleNamespace(logger=logging.getLogger("test"))
    bot._get_dynamic_profit_loss_settings = lambda *args: base(bot, *args)
    bot._apply_sr_exit_levels = lambda **kw: {
        "take_profit_price": kw["base_take_profit"],
        "stop_loss_price": kw["base_stop_loss"],
    }
    frame = pd.DataFrame(
        {
            "ATR": [50.0] * 6,
            "ADX": [30.0] * 6,
            "buy_signal": [False] * 6,
            "sell_signal": [False] * 6,
        }
    )
    valid = fn(bot, "ltc_jpy", frame, "long", 1000)
    assert valid["stop_loss_price"] < 1000 < valid["take_profit_price"]
    assert fn(bot, "ltc_jpy", frame.iloc[:0], "long", 1000) == valid


def test_pending_close_recovery_keeps_residual_position(store):
    svc, api = service(store)
    bot = FakeBot(svc)
    svc.submit(
        "LTC_JPY", "BUY", 10, decision_key="open", context={"bot_symbol": "ltc_jpy"}
    )
    bot._sync_live_positions()
    pos = deepcopy(api.position_rows[0])
    api.fill_ratio = Decimal(".6")
    api.fill_price = Decimal(110)
    api.hide_fills = True
    result = svc.submit(
        "LTC_JPY",
        "SELL",
        10,
        position_id=pos["positionId"],
        context={"bot_symbol": "ltc_jpy", "baseline": pos},
    )
    assert not result["finalized"]
    api.hide_fills = False
    svc.recover()
    bot._sync_live_positions()
    assert bot.positions["ltc_jpy"] == "long" and bot.entry_sizes["ltc_jpy"] == 4


def test_main_dispatches_live_exit_to_execution_service():
    fn, _ = method(BOT, "_check_exit_conditions")
    called = []
    bot = SimpleNamespace(
        positions={"ltc_jpy": "long"},
        entry_prices={"ltc_jpy": 100},
        entry_times={"ltc_jpy": datetime.now(timezone.utc)},
        entry_sizes={"ltc_jpy": 10},
        exchange_settings_gmo={"live_trade": True},
        logger=logging.getLogger("test"),
        calculate_dynamic_exit_levels=lambda *a: {
            "take_profit_price": 110,
            "stop_loss_price": 90,
        },
        get_current_price=lambda *a: 111,
        _apply_short_profit_lock=lambda **kw: kw["stop_loss_price"],
        _close_live_position=lambda *a: called.append(a),
        short_profit_lock_armed={},
    )
    evaluate, _ = method(BOT, "_evaluate_exit_event")
    bot._evaluate_exit_event = lambda *a, **kw: evaluate(bot, *a, **kw)
    fn(bot, "ltc_jpy", {}, [], pd.DataFrame({"high": [112], "low": [100]}))
    assert len(called) == 1


def test_live_close_partial_then_full_updates_memory_and_stats(store, monkeypatch):
    svc, api = service(store)
    bot = FakeBot(svc)
    bot.notification_settings = {"send_on_exit": False}
    bot._ceil_to_next_15min = lambda now: now + timedelta(minutes=15)
    monkeypatch.setitem(sys.modules, "db", SimpleNamespace(engine=store.engine))
    svc.submit(
        "LTC_JPY", "BUY", 10, decision_key="open", context={"bot_symbol": "ltc_jpy"}
    )
    bot._sync_live_positions()
    api.fill_ratio = Decimal(".6")
    api.fill_price = Decimal(110)
    stats = {}
    logs = []
    bot._close_live_position("ltc_jpy", "test", stats, logs)
    assert bot.entry_sizes["ltc_jpy"] == 4 and bot.positions["ltc_jpy"] == "long"
    assert bot.total_profit == 60 and stats["total_trades"] == 1 and len(logs) == 1
    api.fill_ratio = Decimal(1)
    bot._close_live_position("ltc_jpy", "test", stats, logs)
    assert bot.positions["ltc_jpy"] is None and bot.entry_sizes["ltc_jpy"] == 0
    assert bot.total_profit == 100 and stats["total_trades"] == 2 and len(logs) == 2


def test_short_profit_lock_restore_keeps_matching_position_only(store):
    svc, api = service(store)
    bot = FakeBot(svc)
    svc.submit(
        "LTC_JPY", "SELL", 10, decision_key="open", context={"bot_symbol": "ltc_jpy"}
    )
    bot.short_profit_lock_armed["ltc_jpy"] = True
    bot._saved_short_lock_position_ids = {"ltc_jpy": api.position_rows[0]["positionId"]}
    bot._sync_live_positions()
    assert bot.short_profit_lock_armed["ltc_jpy"] is True
    bot._position_sync_failed = {"ltc_jpy"}
    pos = deepcopy(api.position_rows[0])
    svc.submit(
        "LTC_JPY",
        "BUY",
        10,
        position_id=pos["positionId"],
        context={"bot_symbol": "ltc_jpy", "baseline": pos},
    )
    bot._sync_live_positions()
    assert not bot.short_profit_lock_armed["ltc_jpy"]
    assert not bot._position_sync_failed


def test_unrecorded_position_disappearance_blocks_new_entry(store):
    svc, api = service(store)
    bot = FakeBot(svc)
    svc.submit(
        "LTC_JPY", "BUY", 10, decision_key="open", context={"bot_symbol": "ltc_jpy"}
    )
    bot._sync_live_positions()
    api.position_rows = []
    bot._sync_live_positions()
    assert "ltc_jpy" in bot._position_sync_failed
    assert not bot._execute_live_entry("ltc_jpy", "buy", 10, signal_raw={})["success"]


def test_same_decision_after_close_does_not_create_phantom_position(store):
    svc, api = service(store)
    bot = FakeBot(svc)
    raw = {"decision_bar": "2026-11-16T01:00:00+09:00", "decision_price": "88"}
    first = bot._execute_live_entry("ltc_jpy", "buy", 10, signal_raw=raw)
    pos = deepcopy(api.position_rows[0])
    svc.submit(
        "LTC_JPY",
        "SELL",
        10,
        position_id=pos["positionId"],
        context={"bot_symbol": "ltc_jpy", "baseline": pos},
    )
    bot._sync_live_positions()
    result = bot._execute_live_entry("ltc_jpy", "buy", 10, signal_raw=raw)
    assert not result["success"] and bot.positions["ltc_jpy"] is None
    assert len(api.posts) == 2


def test_whole_bot_import_and_live_recovery_use_local_db_only(
    store, monkeypatch, tmp_path
):
    import importlib.util
    import dotenv
    from conftest import TEST_URL

    monkeypatch.setenv("DATABASE_URL", TEST_URL)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr(sys, "argv", ["crypto_bot_sub.py", "backtest"])
    db_spec = importlib.util.spec_from_file_location("db", "db.py")
    db_module = importlib.util.module_from_spec(db_spec)
    monkeypatch.setitem(sys.modules, "db", db_module)
    db_spec.loader.exec_module(db_module)
    db_module.engine.dispose()
    db_module.engine = store.engine
    spec = importlib.util.spec_from_file_location(
        "execution_test_whole_bot", "crypto_bot_sub.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    svc, api = service(store)
    bot = object.__new__(module.CryptoTradingBot)
    bot.__dict__.update(FakeBot(svc).__dict__)
    bot.exchange_settings_gmo = {"live_trade": True}
    bot.base_dir = str(tmp_path)
    bot.notification_settings = {"enabled": False}
    bot.save_positions = lambda: None
    bot.verify_positions()
    assert bot.get_total_balance() == 100000
    assert not api.posts
