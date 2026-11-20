"""Future bars must not rewrite MACD scores or earlier signal decisions."""

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from test_bot_execution_integration import BOT, method
from signal_scoring import MACD_SCORING_VERSION, causal_macd_scores, macd_scoring_spec


def raw_candles(n=240):
    t = np.arange(n, dtype=float)
    close = 100 + 0.025 * t + 3 * np.sin(t / 7) + np.cos(t / 3)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-10-01", periods=n, freq="15min", tz="Asia/Tokyo"
            ),
            "open": close - 0.1,
            "high": close + 0.3 + 0.1 * np.cos(t / 5),
            "low": close - 0.3 - 0.1 * np.sin(t / 5),
            "close": close,
            "volume": 100 + 10 * np.sin(t / 4) + t,
        }
    )


def signal_bot(mode="live", symbol="doge_jpy"):
    bot = SimpleNamespace(
        logger=logging.getLogger("macd-test"),
        exchange_settings_gmo={"live_trade": mode == "live"},
        is_backtest=mode == "backtest",
    )
    bot._apply_rule_thresholds = lambda *a, **kw: None
    build, _ = method(BOT, "build_features")
    generate, _ = method(BOT, "generate_signals")
    return lambda raw: build(bot, raw.copy()), lambda features: generate(
        bot, symbol, features.copy(), pd.DataFrame()
    )


@pytest.mark.parametrize("mode", ["live", "paper", "backtest"])
def test_appending_future_bars_does_not_rewrite_macd_or_signals(mode):
    build, generate = signal_bot(mode)
    features = build(raw_candles())
    full = generate(features)
    for n in [96, 144, 192]:
        prefix = generate(features.iloc[:n])
        cols = ["macd_score_long", "macd_score_short", "buy_signal", "sell_signal"]
        assert_frame_equal(prefix[cols], full.iloc[:n][cols])


def test_changing_future_prices_does_not_change_past_features_or_decisions():
    build, generate = signal_bot()
    raw = raw_candles()
    changed = raw.copy()
    changed.loc[144:, ["open", "high", "low", "close", "volume"]] *= 100
    original = build(raw)
    altered = build(changed)
    assert_frame_equal(original.iloc[:144], altered.iloc[:144])
    a, b = generate(original), generate(altered)
    assert_frame_equal(a.iloc[:144], b.iloc[:144])


def test_forming_bar_cannot_change_previous_closed_macd_score():
    build, generate = signal_bot()
    raw = raw_candles(144)
    changed = raw.copy()
    changed.loc[143, ["open", "high", "low", "close"]] *= 10
    old = generate(build(raw))
    new = generate(build(changed))
    cols = ["macd_score_long", "macd_score_short", "buy_signal", "sell_signal"]
    assert_frame_equal(old.iloc[:-1][cols], new.iloc[:-1][cols])


@pytest.mark.parametrize("index_kind", ["range", "offset", "datetime"])
def test_history_is_exactly_prior_fifty_bars_and_preserves_index(index_kind):
    build, generate = signal_bot()
    features = build(raw_candles(64))
    values = np.arange(64, dtype=float)
    values[50] = 30.0
    features["MACD_histogram"] = values
    features["MACD"] = values
    features["MACD_signal"] = 0.0
    if index_kind == "offset":
        features.index += 100
    if index_kind == "datetime":
        features.index = features.timestamp
    result = generate(features)
    # At bar 50, the reference population is 0..49; the current value 30 is excluded.
    low, high = np.quantile(np.arange(50, dtype=float), [0.2, 0.8])
    expected = (30 - low) / (high - low)
    assert result.macd_score_long.iloc[50] == pytest.approx(expected)
    assert result.macd_score_short.iloc[50] == pytest.approx(1 - expected)
    assert result.index.equals(features.index)


def test_short_history_and_flat_market_cannot_create_entry_signals():
    build, generate = signal_bot()
    features = build(raw_candles(64))
    result = generate(features.iloc[:50])
    assert not result[["buy_signal", "sell_signal"]].any().any()
    assert result[["macd_score_long", "macd_score_short"]].isna().all().all()
    constant = features.copy()
    constant[["MACD", "MACD_signal", "MACD_histogram"]] = 0.0
    result = generate(constant)
    assert not result[["buy_signal", "sell_signal"]].any().any()
    assert result[["macd_score_long", "macd_score_short"]].isna().all().all()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_missing_or_nonfinite_macd_blocks_entry_until_valid_history_recovers(bad):
    build, generate = signal_bot()
    features = build(raw_candles(144))
    features.loc[60, "MACD_histogram"] = bad
    result = generate(features)
    assert not result.loc[60:110, ["buy_signal", "sell_signal"]].any().any()
    assert result.loc[60:110, "macd_score_long"].isna().all()
    assert pd.notna(result.loc[111, "macd_score_long"])


@pytest.mark.parametrize("column", ["MACD", "MACD_signal", "MACD_histogram"])
def test_missing_macd_column_is_not_treated_as_an_observed_zero(column):
    build, generate = signal_bot()
    result = generate(build(raw_candles(144)).drop(columns=column))
    assert not result[["buy_signal", "sell_signal"]].any().any()
    assert not result.macd_score_ready.any()
    assert result[["macd_score_long", "macd_score_short"]].isna().all().all()


def test_score_is_bounded_and_long_short_are_complements():
    raw = raw_candles(144)
    build, _ = signal_bot()
    result = causal_macd_scores(build(raw))
    valid = result.loc[result.macd_score_ready, ["macd_score_long", "macd_score_short"]]
    assert len(valid) > 0
    assert ((valid >= 0) & (valid <= 1)).all().all()
    np.testing.assert_allclose(valid.macd_score_long + valid.macd_score_short, 1)


def test_old_observations_outside_reference_window_do_not_change_current_score():
    build, generate = signal_bot()
    features = build(raw_candles(144))
    expected = generate(features).iloc[-1]
    features.loc[:92, ["MACD", "MACD_signal", "MACD_histogram"]] *= 10000
    actual = generate(features).iloc[-1]
    assert actual.macd_score_long == expected.macd_score_long
    assert actual.macd_score_short == expected.macd_score_short


def test_macd_definition_is_the_same_across_execution_modes():
    scores = []
    for mode in ["live", "paper", "backtest"]:
        build, generate = signal_bot(mode)
        scores.append(
            generate(build(raw_candles(144)))[
                ["macd_score_long", "macd_score_short", "macd_score_ready"]
            ]
        )
    assert_frame_equal(scores[0], scores[1])
    assert_frame_equal(scores[0], scores[2])


def test_scoring_version_changes_strategy_hash_and_is_saved_with_signal():
    import hashlib, json

    version, ns = method(BOT, "_current_strategy_version")
    fixed = pd.Timestamp("2026-11-16T00:00:00+09:00").to_pydatetime()
    ns["now_jst"] = lambda: fixed
    bot = SimpleNamespace(entry_thresholds={"score_long_min": 0.55})
    old_base = {"date": "2026-11-16", "thresholds": bot.entry_thresholds}
    old_hash = hashlib.sha1(json.dumps(old_base, sort_keys=True).encode()).hexdigest()[
        :7
    ]
    assert version(bot) != "2026-11-16+" + old_hash
    explain, _ = method(BOT, "_build_entry_explanation")
    _, _, raw = explain(
        SimpleNamespace(),
        "doge_jpy",
        "long",
        pd.Series({"RSI": 65.0, "ADX": 30.0}),
        100,
    )
    assert raw["scoring"]["macd"] == macd_scoring_spec()
    assert raw["scoring"]["macd"]["version"] == MACD_SCORING_VERSION


def test_backtest_result_records_scoring_definition(tmp_path):
    import json, os
    from datetime import timedelta

    save, ns = method(BOT, "save_backtest_result")
    ns.update(os=os, timedelta=timedelta)
    bot = SimpleNamespace(
        log_dir=str(tmp_path), total_profit=0, logger=logging.getLogger("test")
    )
    save(bot, {}, 1, 0)
    result = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert result["scoring"]["macd"] == macd_scoring_spec()
