"""Replay the actual backtest without network, startup code, or real orders."""

from copy import deepcopy
import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from live_execution import LiveExecutionMixin
from test_bot_execution_integration import BOT, method


START = pd.Timestamp("2026-11-14T00:00:00+09:00")
END = START + pd.Timedelta(days=2)


def candles(side="long"):
    timestamps = pd.date_range(START, END, freq="15min", inclusive="left")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100.0,
            "close": 100.0,
            "high": 101.0,
            "low": 99.0,
            "volume": 10.0,
            "buy_signal": False,
            "sell_signal": False,
            "RSI": 55.0,
            "CCI": 25.0,
            "ATR": 1.0,
            "ADX": 30.0,
        }
    )
    # Enter at 23:45 on day 1; the 00:15 candle is observed at 00:30 on day 2.
    flag = "sell_signal" if side == "long" else "buy_signal"
    frame.loc[93:94, flag] = True
    frame.loc[97:, "close"] = 105.0 if side == "long" else 95.0
    return frame


def replay(
    frame, *, hourly=True, fail_exit=False, starting_backtest=False, bulk=False, days=2,
    costs=None, capital=10000.0, symbols=None
):
    bot = SimpleNamespace(
        symbols=symbols or ["ltc_jpy"],
        initial_capital=capital,
        total_profit=0.0,
        TRADE_SIZE=1000.0,
        entry_sizes={"ltc_jpy": 123.0},
        opposite_narrow_state={"ltc_jpy": {"long": True, "short": True}},
        _last_valid_exit_levels={("ltc_jpy", "long", 300): {"original": True}},
        _exit_level_fallback_state={("ltc_jpy", "long", 300): ("missing_atr", "previous_levels")},
        _entry_notification_context={"ltc_jpy": {"direction": "long", "decision_bar": "original"}},
        short_profit_lock_armed={"ltc_jpy": True},
        is_backtest=starting_backtest,
        sentiment={"bullish": 12.0},
        logger=logging.getLogger("backtest-continuity"),
    )
    bot.before = deepcopy(
        {
            key: getattr(bot, key)
            for key in (
                "total_profit",
                "entry_sizes",
                "opposite_narrow_state",
                "_last_valid_exit_levels",
                "_exit_level_fallback_state",
                "_entry_notification_context",
                "short_profit_lock_armed",
                "is_backtest",
            )
        }
    )
    bot.requests = []
    bot.exit_observations = []

    def cached(symbol, timeframe, date_str, fallback_days=3):
        bot.requests.append((timeframe, date_str, fallback_days))
        if timeframe == "1hour" and not hourly:
            return pd.DataFrame()
        if bulk:
            return frame.iloc[::-1].copy()
        day = pd.Timestamp(date_str).date()
        return frame.loc[frame.timestamp.dt.date == day].copy()

    def levels(symbol, history, side, entry_price):
        if fail_exit:
            raise RuntimeError("intentional replay failure")
        state = bot.opposite_narrow_state[symbol]
        bot.exit_observations.append((history.timestamp.iloc[-1], deepcopy(state)))
        state[side] = True
        return {
            "take_profit_price": 105.0 if side == "long" else 95.0,
            "stop_loss_price": 90.0 if side == "long" else 110.0,
        }

    bot.get_cached_data = cached

    def build(data):
        # Controlled indicators/signals for execution tests. Production receives
        # raw OHLCV only and computes them through the actual feature methods.
        if data.empty:
            return data.copy()
        extras = frame.drop(columns=["open", "high", "low", "close", "volume"])
        return data.merge(extras, on="timestamp", how="left")

    bot.build_features = build
    bot.generate_signals_with_sentiment = lambda symbol, data, hours: data.copy()
    bot.adjust_order_size = lambda symbol, size: size
    bot.calculate_dynamic_exit_levels = levels
    for name in ("_evaluate_exit_event", "_apply_short_profit_lock"):
        function, _ = method(BOT, name)
        setattr(
            bot, name, lambda *a, _function=function, **kw: _function(bot, *a, **kw)
        )
    bot._reset_exit_adjustments = lambda symbol: (
        LiveExecutionMixin._reset_exit_adjustments(bot, symbol)
    )
    bot.log_entry = lambda *a: None
    bot.log_exit = lambda *a: None
    flags, _ = method(BOT, "_last5_flags_str")
    bot._last5_flags_str = lambda *a: flags(bot, *a)
    bot.output_trade_summary = lambda logs: setattr(bot, "trades", deepcopy(logs))

    def save_excel(logs, history):
        bot.excel_trades = deepcopy(logs)
        bot.history = deepcopy(history)

    bot.save_trade_logs_to_excel = save_excel

    def report(logs, days):
        bot.report_trades = deepcopy(logs)

    bot._generate_excel_report_from_trade_logs = report

    def save_result(results, days, start):
        bot.saved = deepcopy(results)
        bot.valuation = deepcopy(bot._backtest_summary)

    bot.save_backtest_result = save_result

    backtest, namespace = method(BOT, "backtest")
    namespace["now_jst"] = lambda: END.to_pydatetime()
    if any(getattr(node, "name", None) == "_run_backtest" for node in BOT.body):
        run, run_namespace = method(BOT, "_run_backtest")
        run_namespace["now_jst"] = namespace["now_jst"]
        bot.run_namespace = run_namespace
        bot._run_backtest = lambda *a, **kw: run(bot, *a, **kw)
    return bot, lambda: backtest(bot, days, costs=costs)


@pytest.mark.parametrize("side", ["long", "short"])
def test_position_and_entry_metadata_survive_midnight(side):
    bot, run = replay(candles(side))
    result = run()["ltc_jpy"]
    assert result["trades"] == 1
    assert len(bot.trades) == 1
    trade = bot.trades[0]
    assert trade["type"] == side
    assert trade["entry_time"] == START + pd.Timedelta(hours=23, minutes=45)
    assert trade["exit_time"] == START + pd.Timedelta(days=1, minutes=30)
    assert trade["holding_hours"] == 0.75
    assert trade["size"] == 10
    assert trade["entry_rsi"] == 55
    assert trade["balance_after_entry"] == 9000
    assert trade["balance_after_exit"] == 10050
    assert result["symbol_profit"] == 50
    assert result["open_position"] is None


@pytest.mark.parametrize("side", ["long", "short"])
def test_period_end_retains_open_position_and_marks_unrealized_separately(side):
    data = candles(side)
    data.loc[97:, "close"] = 103 if side == "long" else 97
    bot, run = replay(data)
    result = run()["ltc_jpy"]
    assert result["trades"] == 0
    assert bot.trades == []
    assert result["symbol_profit"] == 0
    assert result["unrealized_profit"] == 30
    assert result["reserved_capital"] == 1000
    position = result["open_position"]
    assert position["side"] == side
    assert position["size"] == 10
    assert position["entry_price"] == 100
    assert position["mark_price"] == (103 if side == "long" else 97)
    assert pd.Timestamp(position["mark_time"]) == END
    assert position["unrealized_profit"] == 30
    assert bot.saved["ltc_jpy"]["open_position"] == position


def test_entry_can_use_closed_signals_on_opposite_sides_of_midnight():
    data = candles()
    data["sell_signal"] = False
    data.loc[95:96, "sell_signal"] = True
    bot, run = replay(data)
    assert run()["ltc_jpy"]["trades"] == 1
    assert bot.trades[0]["entry_time"] == START + pd.Timedelta(days=1, minutes=15)


def test_protection_state_is_carried_to_the_next_day():
    data = candles()
    # Enter at 23:30; first exit evaluation arms protection at 23:45.
    data["sell_signal"] = False
    data.loc[92:93, "sell_signal"] = True
    bot, run = replay(data)
    run()
    assert bot.exit_observations[0][1]["long"] is False
    assert bot.exit_observations[1][0].date() == (START + pd.Timedelta(days=1)).date()
    assert bot.exit_observations[1][1]["long"] is True


def test_missing_day_does_not_drop_an_open_position_or_replay_previous_data():
    data = candles()
    data.loc[data.timestamp >= START + pd.Timedelta(days=1), "timestamp"] += (
        pd.Timedelta(days=1)
    )
    bot, run = replay(data)
    result = run()["ltc_jpy"]
    assert result["trades"] == 0
    assert result["open_position"]["side"] == "long"
    assert result["processed_bars"] == 96
    assert all(fallback == 0 for _, _, fallback in bot.requests)


@pytest.mark.parametrize("starting_backtest", [False, True])
def test_backtest_restores_runtime_state_and_can_be_repeated(starting_backtest):
    bot, run = replay(candles(), starting_backtest=starting_backtest)
    result = run()
    for key, value in bot.before.items():
        assert getattr(bot, key) == value, key
    assert run() == result


def test_replay_errors_are_reported_and_runtime_state_is_restored():
    bot, run = replay(candles(), fail_exit=True)
    with pytest.raises(RuntimeError, match="intentional replay failure"):
        run()
    assert not hasattr(bot, "saved")
    for key, value in bot.before.items():
        assert getattr(bot, key) == value, key


@pytest.mark.parametrize("side", ["long", "short"])
def test_daily_and_bulk_data_produce_identical_trades_signals_and_valuation(side):
    daily, daily_run = replay(candles(side))
    bulk, bulk_run = replay(candles(side), bulk=True)
    assert daily_run() == bulk_run()
    assert daily.trades == bulk.trades
    pd.testing.assert_frame_equal(daily.history["ltc_jpy"], bulk.history["ltc_jpy"])


def test_partial_day_does_not_reprocess_the_previous_days_last_96_bars():
    bot, run = replay(candles().iloc[:106])
    result = run()["ltc_jpy"]
    assert result["processed_bars"] == 106
    assert result["trades"] == 1
    assert len(bot.trades) == 1


def test_warmup_is_used_for_first_entry_but_never_traded_or_exported():
    data = candles()
    warmup = data.iloc[:96].copy()
    warmup["timestamp"] -= pd.Timedelta(days=1)
    warmup["sell_signal"] = True
    warmup.loc[95, "sell_signal"] = False
    bot, run = replay(pd.concat([warmup, data], ignore_index=True))
    result = run()["ltc_jpy"]
    assert result["processed_bars"] == 192
    assert all(t["entry_time"] >= START for t in bot.trades)
    assert bot.history["ltc_jpy"].index.min() == START.tz_localize(None)
    # The warmup's last signal is false: no entry is fabricated at the start.
    assert bot.trades[0]["entry_time"] == START + pd.Timedelta(hours=23, minutes=45)


def test_first_bar_can_enter_using_the_final_two_warmup_signals():
    data = candles()
    data["sell_signal"] = False
    warmup = data.iloc[:2].copy()
    warmup["timestamp"] = pd.date_range(
        START - pd.Timedelta(minutes=30), periods=2, freq="15min"
    )
    warmup["sell_signal"] = True
    bot, run = replay(pd.concat([warmup, data], ignore_index=True))
    assert run()["ltc_jpy"]["trades"] == 1
    assert bot.trades[0]["entry_time"] == START


def test_data_outside_the_requested_period_cannot_change_the_result():
    data = candles()
    future = data.copy()
    future["timestamp"] += pd.Timedelta(days=2)
    future["open"] = 500
    future["close"] = 500
    bot, run = replay(pd.concat([data, future], ignore_index=True), bulk=True)
    clean, clean_run = replay(data)
    assert run() == clean_run()
    assert bot.trades == clean.trades


def test_missing_bar_cannot_be_treated_as_two_consecutive_entry_signals():
    bot, run = replay(candles().drop(index=95))
    result = run()["ltc_jpy"]
    assert result["trades"] == 0
    assert result["open_position"] is None


def test_reentry_wait_survives_midnight_and_allows_only_the_next_bar():
    data = candles()
    data["sell_signal"] = False
    data.loc[[90, 91, 94, 95, 96], "sell_signal"] = True
    data.loc[95:, "close"] = 105
    bot, run = replay(data)
    assert run()["ltc_jpy"]["trades"] == 2
    assert bot.trades[0]["exit_time"] == START + pd.Timedelta(days=1)
    assert bot.trades[1]["entry_time"] == START + pd.Timedelta(days=1, minutes=15)


def test_missing_hourly_data_fails_explicitly_instead_of_saving_a_partial_result():
    bot, run = replay(candles(), hourly=False)
    with pytest.raises(ValueError, match="ltc_jpy"):
        run()
    assert not hasattr(bot, "saved")


def test_run_period_is_fixed_once_even_if_the_clock_crosses_midnight():
    bot, run = replay(candles())
    bot.symbols = ["ltc_jpy", "eth_jpy"]
    reads = []

    def clock():
        reads.append(True)
        return (END + pd.Timedelta(days=len(reads) - 1)).to_pydatetime()

    bot.run_namespace["now_jst"] = clock
    results = run()
    assert len(reads) == 1
    assert all(result["processed_bars"] == 192 for result in results.values())


def test_saved_json_contains_the_open_position_and_gross_valuation(tmp_path):
    import json
    import os
    from datetime import timedelta

    data = candles()
    data.loc[97:, "close"] = 103
    bot, run = replay(data)
    save, namespace = method(BOT, "save_backtest_result")
    namespace.update(os=os, timedelta=timedelta, now_jst=lambda: END.to_pydatetime())
    bot.log_dir = str(tmp_path)
    bot.save_backtest_result = lambda *args: save(bot, *args)
    result = run()
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["total_profit"] == 0
    assert saved["valuation"]["unrealized_profit"] == 30
    assert saved["valuation"]["reserved_capital"] == 1000
    assert saved["valuation"]["costs_included"] is False
    assert saved["valuation"]["live_equivalent"] is False
    assert saved["valuation"]["execution_model"] == "15min_ohlc_market_close_proxy"
    assert saved["period"] == {
        "start": START.isoformat(),
        "end_exclusive": END.isoformat(),
    }
    assert saved["results"] == result
    assert saved["start_date"] == "2026-11-14"
    assert saved["end_date"] == "2026-11-15"


def test_both_excel_exports_receive_naive_jst_and_replay_keeps_aware_times():
    bot, run = replay(candles())
    run()
    assert bot.trades[0]["entry_time"].tzinfo is not None
    assert bot.excel_trades[0]["entry_time"].tzinfo is None
    assert bot.excel_trades[0]["entry_time"].hour == 23
    assert bot.report_trades == bot.excel_trades
    assert bot.history["ltc_jpy"].index.tz is None


@pytest.mark.parametrize("days", [0, -1, True, 1.5])
def test_invalid_test_period_is_rejected_before_data_access(days):
    fn, _ = method(BOT, "backtest")
    with pytest.raises(ValueError):
        fn(SimpleNamespace(), days)
