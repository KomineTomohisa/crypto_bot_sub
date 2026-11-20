"""Exit decisions use actual signal directions and the same closed candle's DI pair."""

from types import SimpleNamespace
import logging
import numpy as np
import pandas as pd
import pytest

from test_bot_execution_integration import BOT, method
from test_execution import FakeBot, FakeExchange
from trading_execution import ExchangeGateway


def bot_and_exit():
    calculate, _ = method(BOT, "calculate_dynamic_exit_levels")
    settings, _ = method(BOT, "_get_dynamic_profit_loss_settings")
    bot = SimpleNamespace(logger=logging.getLogger("exit-test"))
    bot._get_dynamic_profit_loss_settings = lambda *args: settings(bot, *args)
    bot._apply_sr_exit_levels = lambda **kw: {
        "take_profit_price": kw["base_take_profit"],
        "stop_loss_price": kw["base_stop_loss"],
    }
    return bot, lambda df, side: calculate(bot, "ltc_jpy", df, side, 1000)


def frame(buy=None, sell=None, plus=25.0, minus=25.0):
    return pd.DataFrame(
        {
            "ATR": [50.0] * 5,
            "ADX": [30.0] * 5,
            "buy_signal": buy if buy is not None else [False] * 5,
            "sell_signal": sell if sell is not None else [False] * 5,
            "plus_di14": [float(plus)] * 5,
            "minus_di14": [float(minus)] * 5,
        }
    )


@pytest.mark.parametrize(
    "side,opposite", [("long", "buy_signal"), ("short", "sell_signal")]
)
def test_only_opposite_direction_three_of_four_tightens_stop(side, opposite):
    bot, calculate = bot_and_exit()
    neutral = calculate(frame(), side)
    data = frame()
    data[opposite] = [True, False, True, True, False]
    narrowed = calculate(data, side)
    assert bot.opposite_narrow_state["ltc_jpy"][side] is True
    if side == "long":
        assert neutral["stop_loss_price"] < narrowed["stop_loss_price"] < 1000
    else:
        assert 1000 < narrowed["stop_loss_price"] < neutral["stop_loss_price"]


@pytest.mark.parametrize(
    "side,same", [("long", "sell_signal"), ("short", "buy_signal")]
)
def test_same_direction_does_not_arm_narrowing(side, same):
    bot, calculate = bot_and_exit()
    neutral = calculate(frame(), side)
    data = frame()
    data[same] = [True] * 5
    result = calculate(data, side)
    assert not bot.opposite_narrow_state["ltc_jpy"][side]
    assert result["stop_loss_price"] == neutral["stop_loss_price"]


@pytest.mark.parametrize(
    "side,opposite,same",
    [("long", "buy_signal", "sell_signal"), ("short", "sell_signal", "buy_signal")],
)
def test_two_same_direction_closed_candles_unlock_previous_narrowing(
    side, opposite, same
):
    bot, calculate = bot_and_exit()
    data = frame()
    data[opposite] = [True] * 5
    calculate(data, side)
    assert bot.opposite_narrow_state["ltc_jpy"][side]
    unlocked = frame()
    unlocked[same] = [False, False, True, True, False]
    result = calculate(unlocked, side)
    assert not bot.opposite_narrow_state["ltc_jpy"][side]
    _, baseline = bot_and_exit()
    assert result["stop_loss_price"] == baseline(frame(), side)["stop_loss_price"]


@pytest.mark.parametrize(
    "side,opposite", [("long", "buy_signal"), ("short", "sell_signal")]
)
def test_forming_candle_does_not_supply_third_opposite_signal(side, opposite):
    bot, calculate = bot_and_exit()
    data = frame()
    data[opposite] = [True, True, False, False, True]
    calculate(data, side)
    assert not bot.opposite_narrow_state["ltc_jpy"][side]


@pytest.mark.parametrize(
    "side,opposite", [("long", "buy_signal"), ("short", "sell_signal")]
)
def test_narrowing_persists_without_unlock_signal(side, opposite):
    bot, calculate = bot_and_exit()
    data = frame()
    data[opposite] = [True] * 5
    initial = calculate(data, side)
    result = calculate(frame(), side)
    assert bot.opposite_narrow_state["ltc_jpy"][side]
    assert result["stop_loss_price"] == initial["stop_loss_price"]


@pytest.mark.parametrize(
    "side,favorable,adverse",
    [("long", (40.0, 10.0), (10.0, 40.0)), ("short", (10.0, 40.0), (40.0, 10.0))],
)
def test_canonical_di_columns_activate_adverse_trend_adjustment(
    side, favorable, adverse
):
    _, calculate = bot_and_exit()
    aligned = calculate(frame(plus=favorable[0], minus=favorable[1]), side)
    opposed = calculate(frame(plus=adverse[0], minus=adverse[1]), side)
    if side == "long":
        assert aligned["stop_loss_price"] < opposed["stop_loss_price"] < 1000
        assert 1000 < opposed["take_profit_price"] < aligned["take_profit_price"]
    else:
        assert 1000 < opposed["stop_loss_price"] < aligned["stop_loss_price"]
        assert aligned["take_profit_price"] < opposed["take_profit_price"] < 1000


def test_di_uses_closed_candle_even_when_forming_candle_is_opposite():
    _, calculate = bot_and_exit()
    data = frame(plus=40, minus=10)
    baseline = calculate(data, "long")
    data.loc[4, ["plus_di14", "minus_di14"]] = [1, 99]
    assert calculate(data, "long") == baseline


def test_only_forming_candle_cannot_activate_di_adjustment():
    _, calculate = bot_and_exit()
    data = frame(plus=10, minus=40).iloc[:1]
    baseline = calculate(data.drop(columns=["plus_di14", "minus_di14"]), "long")
    assert calculate(data, "long") == baseline


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0])
def test_invalid_closed_di_does_not_pair_values_from_different_candles(bad):
    _, calculate = bot_and_exit()
    baseline = calculate(frame().drop(columns=["plus_di14", "minus_di14"]), "long")
    data = frame(plus=10, minus=40)
    data.loc[3, "plus_di14"] = bad
    result = calculate(data, "long")
    assert result["stop_loss_price"] == baseline["stop_loss_price"]
    assert result["take_profit_price"] == baseline["take_profit_price"]


def test_canonical_di_wins_over_legacy_alias_and_legacy_only_remains_supported():
    _, calculate = bot_and_exit()
    canonical = frame(plus=40, minus=10)
    baseline = calculate(canonical, "long")
    canonical["DI+"] = 10.0
    canonical["DI-"] = 40.0
    assert calculate(canonical, "long") == baseline
    legacy = frame(plus=40, minus=10).rename(
        columns={"plus_di14": "DI+", "minus_di14": "DI-"}
    )
    assert calculate(legacy, "long") == baseline


def position_bot():
    api = FakeExchange()
    store = SimpleNamespace(
        position_context=lambda *args: {}, open_position_ids=lambda *args: []
    )
    service = SimpleNamespace(gateway=ExchangeGateway(api), store=store, account="test")
    return FakeBot(service), api


def position(api, pid, side="SELL", size="5"):
    return {
        "positionId": pid,
        "symbol": "LTC_JPY",
        "side": side,
        "size": size,
        "price": "100",
        "timestamp": api.now,
    }


def arm(bot):
    bot.opposite_narrow_state = {
        "ltc_jpy": {"long": True, "short": True},
        "eth_jpy": {"long": True, "short": False},
    }
    bot._last_valid_exit_levels = {
        ("ltc_jpy", "short", 100): {"stop_loss_price": 101},
        ("eth_jpy", "long", 100): {"stop_loss_price": 99},
    }
    bot.short_profit_lock_armed["ltc_jpy"] = True


def test_full_close_clears_only_that_symbols_exit_adjustments():
    bot, api = position_bot()
    api.position_rows = [position(api, "1")]
    bot._sync_live_positions()
    arm(bot)
    api.position_rows = []
    bot._sync_live_positions()
    assert bot.opposite_narrow_state["ltc_jpy"] == {"long": False, "short": False}
    assert bot.opposite_narrow_state["eth_jpy"]["long"] is True
    assert list(bot._last_valid_exit_levels) == [("eth_jpy", "long", 100)]


def test_partial_close_preserves_adjustments_when_representative_position_changes():
    bot, api = position_bot()
    api.position_rows = [position(api, "1"), position(api, "2")]
    bot._sync_live_positions()
    arm(bot)
    api.position_rows = [position(api, "2", size="3")]
    bot._sync_live_positions()
    assert bot.entry_sizes["ltc_jpy"] == 3
    assert bot.position_ids["ltc_jpy"] == "2"
    assert bot.opposite_narrow_state["ltc_jpy"]["short"] is True
    assert bot.short_profit_lock_armed["ltc_jpy"] is True
    assert ("ltc_jpy", "short", 100) in bot._last_valid_exit_levels


@pytest.mark.parametrize("new_side", ["BUY", "SELL"])
def test_replacement_position_cannot_inherit_previous_adjustments(new_side):
    bot, api = position_bot()
    api.position_rows = [position(api, "1")]
    bot._sync_live_positions()
    arm(bot)
    api.position_rows = [position(api, "9", side=new_side)]
    bot._sync_live_positions()
    assert bot.opposite_narrow_state["ltc_jpy"] == {"long": False, "short": False}
    assert not bot.short_profit_lock_armed["ltc_jpy"]
    assert ("ltc_jpy", "short", 100) not in bot._last_valid_exit_levels


def test_temporary_position_read_failure_does_not_clear_adjustments():
    bot, api = position_bot()
    api.position_rows = [position(api, "1")]
    bot._sync_live_positions()
    arm(bot)
    api.fail_read = True
    bot._sync_live_positions()
    assert bot.opposite_narrow_state["ltc_jpy"]["short"]
    assert bot.short_profit_lock_armed["ltc_jpy"]
    assert ("ltc_jpy", "short", 100) in bot._last_valid_exit_levels


def test_paper_entry_starts_with_fresh_exit_adjustments():
    bot, api = position_bot()
    arm(bot)
    bot.exchange_settings_gmo = {"live_trade": False}
    bot.initial_capital = 200000
    bot.total_profit = 0
    bot.notification_settings = {"send_on_entry": False}
    bot.get_current_price = lambda symbol: 100
    bot.adjust_position_risk = lambda *args: (10, 1000)
    bot.check_sufficient_funds = lambda *args: True
    bot._should_block_entry_by_sr = lambda *args: (False, "")
    bot._build_entry_explanation = lambda **kw: ({}, "", {})
    bot._current_strategy_version = lambda: "test"
    bot.execute_order_with_confirmation = lambda *args, **kw: {
        "success": True,
        "executed_size": 10,
    }
    bot._get_margin_positions_safe = lambda *args: {"status": 0, "data": {"list": []}}
    bot.log_entry = lambda *args: None
    bot.sentiment = {}
    signal = pd.Series(
        {
            "RSI": 60.0,
            "CCI": 100.0,
            "ATR": 50.0,
            "ADX": 30.0,
            "plus_di14": 40.0,
            "minus_di14": 10.0,
            "EMA_short": 100.0,
            "EMA_long": 99.0,
            "buy_score": 0.3,
            "sell_score": 0.7,
            "close": 100.0,
            "timestamp": pd.Timestamp("2026-11-16T00:00:00+09:00"),
        }
    )
    handle, _ = method(BOT, "_handle_entry")
    handle(bot, "ltc_jpy", "long", signal, {}, [])
    assert bot.opposite_narrow_state["ltc_jpy"] == {"long": False, "short": False}
    assert ("ltc_jpy", "short", 100) not in bot._last_valid_exit_levels
    assert not api.posts
