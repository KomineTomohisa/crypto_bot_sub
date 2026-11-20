"""Candle chunking must not alter the continuous timeline or runtime state."""

import pandas as pd
import pytest

from backtest_data import backtest_state, merge_candle_chunks, excel_backtest_inputs
from test_backtest_continuity import START, END, candles


def test_daily_bulk_shuffled_and_overlapping_chunks_have_identical_timeline():
    data = candles()
    expected = merge_candle_chunks([data], START, END)
    for chunks in (
        [data.iloc[:96], data.iloc[96:]],
        [data.iloc[80:], data.iloc[:100]],
        [data.iloc[::-1]],
        [data.iloc[96:], data.iloc[:96], data.iloc[20:140]],
    ):
        pd.testing.assert_frame_equal(merge_candle_chunks(chunks, START, END), expected)


def test_naive_jst_utc_and_jst_timestamps_represent_the_same_events():
    data = candles()
    expected = merge_candle_chunks([data], START, END)
    for timestamps in (
        data.timestamp.dt.tz_localize(None),
        data.timestamp.dt.tz_convert("UTC"),
    ):
        changed = data.copy()
        changed["timestamp"] = timestamps
        pd.testing.assert_frame_equal(
            merge_candle_chunks([changed], START, END), expected
        )


def test_conflicting_duplicate_candles_are_rejected():
    data = candles()
    conflict = data.iloc[[0]].copy()
    conflict["close"] = 102
    with pytest.raises(ValueError, match="Conflicting"):
        merge_candle_chunks([data, conflict], START, END)


@pytest.mark.parametrize(
    "column,value",
    [
        ("open", 0),
        ("close", float("nan")),
        ("close", float("inf")),
        ("timestamp", pd.NaT),
    ],
)
def test_unusable_timestamps_and_valuation_prices_fail_explicitly(column, value):
    data = candles()
    data.loc[0, column] = value
    with pytest.raises(ValueError):
        merge_candle_chunks([data], START, END)


def test_state_context_restores_original_objects_even_on_failure():
    from types import SimpleNamespace

    sizes = {"ltc_jpy": 50}
    bot = SimpleNamespace(total_profit=123.0, entry_sizes=sizes)
    original = vars(bot).copy()
    with pytest.raises(RuntimeError):
        with backtest_state(bot):
            bot.entry_sizes["ltc_jpy"] = 10
            bot.opposite_narrow_state["ltc_jpy"] = {"long": True}
            bot.total_profit += 100
            raise RuntimeError("test")
    assert vars(bot) == original
    assert bot.entry_sizes is sizes


def test_excel_timestamps_are_jst_values_without_modifying_original_records():
    from pandas.io.formats.excel import ExcelFormatter

    trades = [{"entry_time": START.tz_convert("UTC"), "exit_time": END}]
    history = {"ltc_jpy": pd.DataFrame({"buy_signal": [False]}, index=[START])}
    exported, exported_history = excel_backtest_inputs(trades, history)
    assert exported[0]["entry_time"] == START.tz_localize(None)
    assert exported[0]["exit_time"] == END.tz_localize(None)
    assert trades[0]["entry_time"].tzinfo is not None
    assert history["ltc_jpy"].index.tz is not None
    # Exercise pandas' real Excel cell conversion, which rejects aware timestamps.
    list(ExcelFormatter(pd.DataFrame(exported)).get_formatted_cells())
    list(ExcelFormatter(exported_history["ltc_jpy"]).get_formatted_cells())
