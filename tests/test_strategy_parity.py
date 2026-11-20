"""The real live exit wrapper and replay must share decisions and lock state."""

from types import SimpleNamespace
import logging

import pandas as pd
import pytest

from strategy_decisions import (
    entry_side,
    exit_decision,
    short_profit_stop,
    next_entry_time,
)
from test_backtest_continuity import START, END, candles, replay
from test_bot_execution_integration import BOT, method


def wire_decisions(bot):
    for name in ("_evaluate_exit_event", "_apply_short_profit_lock"):
        function, _ = method(BOT, name)
        setattr(
            bot, name, lambda *a, _function=function, **kw: _function(bot, *a, **kw)
        )
    bot.SHORT_PROFIT_LOCK_TRIGGER_PCT = 0.75
    bot.SHORT_PROFIT_LOCK_LEVEL_PCT = 0.25
    bot.save_positions = lambda: None


@pytest.mark.parametrize("side", ["long", "short"])
def test_market_exit_uses_observed_price_instead_of_the_threshold(side):
    data = candles(side)
    data.loc[97:, "close"] = 111 if side == "long" else 89
    bot, run = replay(data)
    run()
    trade = bot.trades[0]
    assert trade["exit_price"] == (111 if side == "long" else 89)
    assert trade["profit"] == 110
    assert trade["exit_reason"] == "利益確定"


def test_rounded_quantity_determines_realized_profit():
    bot, run = replay(candles())
    bot.adjust_order_size = lambda symbol, size: 7
    run()
    trade = bot.trades[0]
    assert trade["size"] == 7
    assert trade["profit"] == 35
    assert trade["balance_after_exit"] == 10035


@pytest.mark.parametrize("side", ["long", "short"])
def test_live_and_replay_issue_the_same_exit_reason_for_the_same_snapshot(side):
    data = candles(side)
    data.loc[97:, "close"] = 106 if side == "long" else 94
    replay_bot, run = replay(data)
    run()
    expected = replay_bot.trades[0]
    check, namespace = method(BOT, "_check_exit_conditions")
    at = expected["exit_time"]
    namespace["now_jst"] = lambda: at.to_pydatetime()
    row_index = 97
    bot = SimpleNamespace(
        logger=logging.getLogger("live-parity"),
        positions={"ltc_jpy": side},
        entry_prices={"ltc_jpy": 100},
        entry_sizes={"ltc_jpy": 10},
        entry_times={"ltc_jpy": expected["entry_time"]},
        short_profit_lock_armed={"ltc_jpy": False},
        exchange_settings_gmo={"live_trade": True},
        get_current_price=lambda _: float(data.iloc[row_index]["close"]),
    )
    wire_decisions(bot)
    bot.calculate_dynamic_exit_levels = lambda *a: {
        "take_profit_price": 105 if side == "long" else 95,
        "stop_loss_price": 90 if side == "long" else 110,
    }
    decisions = []
    bot._close_live_position = lambda symbol, reason, stats, logs: decisions.append(
        reason
    )
    check(bot, "ltc_jpy", {}, [], data.iloc[: row_index + 1])
    assert decisions == [expected["exit_reason"]]


def test_short_profit_protection_arms_and_exits_identically_in_replay():
    data = candles("short")
    data.loc[96:, "close"] = 99.2
    data.loc[98:, "close"] = 99.8
    bot, run = replay(data)
    run()
    assert len(bot.trades) == 1
    trade = bot.trades[0]
    assert trade["exit_reason"] == "利益保護"
    assert trade["exit_price"] == 99.8
    assert trade["profit"] == pytest.approx(2)


@pytest.mark.parametrize("side", ["long", "short"])
def test_both_threshold_touches_choose_stop_reason_when_order_is_unknown(side):
    result = exit_decision(
        side=side,
        take_profit=105 if side == "long" else 95,
        base_stop=90 if side == "long" else 110,
        stop_loss=90 if side == "long" else 110,
        current_price=100,
        high=111,
        low=89,
        holding_hours=1,
    )
    assert result.reason == "損切り"
    assert result.ambiguous_touches is True


def test_current_quote_can_trigger_an_exit_even_when_signal_feed_did_not_touch():
    result = exit_decision(
        side="long",
        take_profit=105,
        base_stop=90,
        stop_loss=90,
        current_price=89,
        high=101,
        low=99,
        holding_hours=1,
    )
    assert result.reason == "損切り"


@pytest.mark.parametrize(
    "hours,expected", [(71.99, None), (72, "長時間保有による強制決済")]
)
def test_time_exit_uses_the_injected_observation_clock(hours, expected):
    result = exit_decision(
        side="long",
        take_profit=105,
        base_stop=90,
        stop_loss=90,
        current_price=100,
        high=101,
        low=99,
        holding_hours=hours,
    )
    assert result.reason == expected


def test_profit_lock_never_uses_the_high_from_before_arming():
    stop, armed = short_profit_stop("short", 100, 99.2, 110, False)
    assert armed is True
    assert stop == 99.75
    result = exit_decision(
        side="short",
        take_profit=95,
        base_stop=110,
        stop_loss=stop,
        current_price=99.2,
        high=101,
        low=99,
        holding_hours=1,
        profit_lock_armed=armed,
    )
    assert result.reason is None


@pytest.mark.parametrize("mode", ["live", "paper", "backtest"])
def test_entry_priority_and_closed_bar_checks_are_execution_independent(mode):
    # The kernel deliberately has no mode argument.
    latest = {
        "timestamp": START + pd.Timedelta(minutes=15),
        "buy_signal": True,
        "sell_signal": True,
    }
    previous = {"timestamp": START, "buy_signal": True, "sell_signal": True}
    at = START + pd.Timedelta(minutes=30)
    assert entry_side(latest, previous, at) == "short"
    assert entry_side(latest, previous, at, at + pd.Timedelta(minutes=1)) is None
    assert entry_side(latest, previous, at - pd.Timedelta(minutes=1)) is None
    assert entry_side(latest, previous, at + pd.Timedelta(minutes=15)) is None


def test_nan_signal_is_not_a_positive_entry_signal():
    latest = {"timestamp": START + pd.Timedelta(minutes=15), "buy_signal": float("nan")}
    previous = {"timestamp": START, "buy_signal": True}
    assert entry_side(latest, previous, START + pd.Timedelta(minutes=30)) is None


def test_cooldown_is_the_same_next_boundary_for_live_and_replay():
    assert next_entry_time(START) == START + pd.Timedelta(minutes=15)
    assert next_entry_time(
        START + pd.Timedelta(minutes=14, seconds=59)
    ) == START + pd.Timedelta(minutes=15)


def test_the_real_replay_applies_the_live_72_hour_exit():
    data = pd.concat([candles(), candles()], ignore_index=True)
    data["timestamp"] = pd.date_range(
        END - pd.Timedelta(days=4), periods=384, freq="15min"
    )
    data["open"] = data["close"] = 100.0
    data["buy_signal"] = data["sell_signal"] = False
    data.loc[:1, "sell_signal"] = True
    bot, run = replay(data, days=4)
    run()
    assert len(bot.trades) == 1
    assert bot.trades[0]["holding_hours"] == 72
    assert bot.trades[0]["exit_reason"] == "長時間保有による強制決済"


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("price", [89.0, 99.2, 99.8, 106.0])
def test_actual_decision_method_has_identical_results_and_state_in_all_modes(
    side, price
):
    frame = candles(side).iloc[:4].copy()
    outcomes = []
    for mode in ("live", "paper", "backtest"):
        bot = SimpleNamespace(
            logger=logging.getLogger("mode-parity"),
            is_backtest=mode == "backtest",
            short_profit_lock_armed={"ltc_jpy": False},
            exchange_settings_gmo={"live_trade": mode == "live"},
        )
        wire_decisions(bot)
        bot.calculate_dynamic_exit_levels = lambda *a: {
            "take_profit_price": 105 if side == "long" else 95,
            "stop_loss_price": 90 if side == "long" else 110,
        }
        decision = bot._evaluate_exit_event(
            "ltc_jpy",
            frame,
            side,
            100,
            START,
            START + pd.Timedelta(hours=1),
            price,
        )
        outcomes.append((decision, dict(bot.short_profit_lock_armed)))
    assert outcomes[0] == outcomes[1] == outcomes[2]


def test_high_and_low_before_entry_cannot_trigger_the_new_positions_exit():
    frame = candles().iloc[:1].copy()
    frame["high"] = 120
    frame["low"] = 80
    bot = SimpleNamespace(
        logger=logging.getLogger("pre-entry"), short_profit_lock_armed={}
    )
    wire_decisions(bot)
    bot.calculate_dynamic_exit_levels = lambda *a: {
        "take_profit_price": 105,
        "stop_loss_price": 90,
    }
    decision = bot._evaluate_exit_event(
        "ltc_jpy",
        frame,
        "long",
        100,
        START + pd.Timedelta(minutes=5),
        START + pd.Timedelta(minutes=10),
        100,
    )
    assert decision.reason is None


def test_insufficient_cash_cannot_create_an_unfunded_position():
    bot, run = replay(candles())
    bot.initial_capital = 500
    result = run()["ltc_jpy"]
    assert result["trades"] == 0
    assert result["open_position"] is None
    assert bot.valuation["ending_available_cash"] == 500


def test_all_symbols_share_one_chronological_account_regardless_of_input_order():
    outputs = []
    for symbols in (["eth_jpy", "ltc_jpy"], ["ltc_jpy", "eth_jpy"]):
        bot, run = replay(candles())
        bot.symbols = symbols
        bot.initial_capital = 1500
        result = run()
        # Both try to open 1000 JPY at the same instant. Deterministic tie priority
        # lets one enter; the other cannot borrow tomorrow's closing proceeds.
        assert result["eth_jpy"]["trades"] == 1
        assert result["ltc_jpy"]["trades"] == 0
        assert bot.valuation["ending_available_cash"] == 1550
        outputs.append((result, bot.trades, bot.valuation))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("side", ["long", "short"])
def test_cash_and_reserved_notional_reconcile_to_realized_profit(side):
    bot, run = replay(candles(side))
    bot.adjust_order_size = lambda symbol, size: 7
    result = run()["ltc_jpy"]
    assert bot.valuation["ending_available_cash"] + result["reserved_capital"] == (
        bot.initial_capital + result["symbol_profit"]
    )
