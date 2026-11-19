"""Compare actual feature methods for identical timestamped live/replay inputs."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from replay_observations import offline_signal_bot
from strategy_inputs import (
    feature_input,
    historical_observation,
    prepare_strategy_inputs,
    history_quality,
)
from test_macd_causality import raw_candles


def hourly_candles(raw):
    return (
        raw.set_index("timestamp")
        .resample("1h")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )


def signal_config(mode="live"):
    return {
        "live_trade": mode == "live",
        "is_backtest": mode == "backtest",
        "entry_thresholds": {},
    }


@pytest.mark.parametrize(
    "symbol",
    ["ltc_jpy", "xrp_jpy", "sol_jpy", "doge_jpy", "eth_jpy", "bcc_jpy", "ada_jpy"],
)
@pytest.mark.parametrize("offset", [288, 323])
def test_actual_live_and_backtest_features_scores_and_flags_match(symbol, offset):
    raw = raw_candles(500)
    at = raw.timestamp.iloc[offset]
    observed = historical_observation(raw, at, at_open=True)
    hours = hourly_candles(raw)
    # The live adapter can have a much larger response/cache than the replay.
    live_raw = pd.concat([observed, raw.loc[raw.timestamp > at]], ignore_index=True)
    live = prepare_strategy_inputs(
        offline_signal_bot(signal_config()), symbol, live_raw, hours, at
    )
    replay = prepare_strategy_inputs(
        offline_signal_bot(signal_config("backtest")),
        symbol,
        observed,
        hours.loc[hours.timestamp < at],
        at,
    )
    assert_frame_equal(live[0], replay[0])
    assert_frame_equal(live[1], replay[1])
    assert live[0].timestamp.min() == at.normalize() - pd.Timedelta(days=2)


def test_prices_before_window_and_after_observation_cannot_change_decisions():
    raw = raw_candles(600)
    at = raw.timestamp.iloc[440]
    changed = raw.copy()
    outside = (raw.timestamp < at.normalize() - pd.Timedelta(days=2)) | (
        raw.timestamp > at
    )
    changed.loc[outside, ["open", "high", "low", "close", "volume"]] *= 100
    original = prepare_strategy_inputs(
        offline_signal_bot(signal_config()), "doge_jpy", raw, hourly_candles(raw), at
    )
    modified = prepare_strategy_inputs(
        offline_signal_bot(signal_config()),
        "doge_jpy",
        changed,
        hourly_candles(changed),
        at,
    )
    assert_frame_equal(original[0], modified[0])
    assert_frame_equal(original[1], modified[1])


def test_opening_observation_has_no_future_high_low_close_or_volume():
    raw = raw_candles()
    at = raw.timestamp.iloc[96]
    observed = historical_observation(raw, at, at_open=True)
    assert observed.timestamp.max() == at
    assert (
        observed.iloc[-1]["open"]
        == observed.iloc[-1]["high"]
        == observed.iloc[-1]["low"]
        == observed.iloc[-1]["close"]
    )
    assert observed.iloc[-1]["volume"] == 0
    assert_frame_equal(observed.iloc[:-1], raw.iloc[:96])


def test_unclosed_hourly_candle_cannot_change_the_hourly_features():
    raw = raw_candles(500)
    hours = hourly_candles(raw)
    at = raw.timestamp.iloc[323] + pd.Timedelta(minutes=7)
    changed = hours.copy()
    changed.loc[hours.timestamp + pd.Timedelta(hours=1) > at, "close"] *= 100
    first = prepare_strategy_inputs(
        offline_signal_bot(signal_config()), "eth_jpy", raw, hours, at
    )
    second = prepare_strategy_inputs(
        offline_signal_bot(signal_config()), "eth_jpy", raw, changed, at
    )
    assert_frame_equal(first[0], second[0])
    assert_frame_equal(first[1], second[1])


def test_naive_jst_and_utc_inputs_have_identical_features():
    raw = raw_candles(400)
    at = raw.timestamp.iloc[323]
    baseline = prepare_strategy_inputs(
        offline_signal_bot(signal_config()), "eth_jpy", raw, hourly_candles(raw), at
    )
    for kind in ("naive", "utc"):
        converted = raw.copy()
        converted["timestamp"] = (
            raw.timestamp.dt.tz_localize(None)
            if kind == "naive"
            else raw.timestamp.dt.tz_convert("UTC")
        )
        result = prepare_strategy_inputs(
            offline_signal_bot(signal_config()),
            "eth_jpy",
            converted,
            hourly_candles(converted),
            at,
        )
        assert_frame_equal(baseline[0], result[0])
        assert_frame_equal(baseline[1], result[1])


@pytest.mark.parametrize("failure", ["missing_column", "missing_time", "conflict"])
def test_invalid_history_fails_explicitly(failure):
    raw = raw_candles()
    if failure == "missing_column":
        raw = raw.drop(columns="open")
    elif failure == "missing_time":
        raw.loc[0, "timestamp"] = pd.NaT
    else:
        duplicate = raw.iloc[[0]].copy()
        duplicate["close"] += 1
        raw = pd.concat([raw, duplicate], ignore_index=True)
    with pytest.raises(ValueError):
        feature_input(raw, pd.Timestamp("2026-10-02T12:00:00+09:00"), 15)


def test_backtest_really_uses_the_bounded_pipeline_at_each_observation():
    from test_backtest_continuity import candles, replay

    bot, run = replay(candles().iloc[:5])
    original = bot.run_namespace["prepare_strategy_inputs"]
    calls = []

    def observed(bot, symbol, intraday, hourly, at):
        calls.append((intraday.copy(), at))
        return original(bot, symbol, intraday, hourly, at)

    bot.run_namespace["prepare_strategy_inputs"] = observed
    run()
    assert len(calls) == 5
    for frame, at in calls:
        assert frame.timestamp.max() == at
        assert frame.iloc[-1]["close"] == frame.iloc[-1]["open"]
        assert frame.iloc[-1]["volume"] == 0


def test_missing_market_intervals_are_counted_separately_from_journal_integrity():
    raw = raw_candles(400)
    at = raw.timestamp.iloc[323]
    selected = feature_input(raw, at, 15)
    assert history_quality(selected, at, 15)["missing_closed_intervals"] == 0
    missing = selected.drop(index=range(10, 20))
    assert history_quality(missing, at, 15)["missing_closed_intervals"] == 10
