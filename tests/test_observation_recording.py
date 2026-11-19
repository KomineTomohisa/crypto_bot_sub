"""Journal integrity, real feature replay, failures, and secret-free inputs."""

from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from replay_observations import offline_signal_bot, verify_session
from strategy_inputs import prepare_strategy_inputs
from strategy_recording import (
    ObservationJournal,
    BackgroundJournal,
    RecordingError,
    close_recorder,
    code_fingerprint,
    digest,
    read_blob,
    read_journal,
    record_bot_event,
    runtime_versions,
)
from test_bot_execution_integration import BOT, method
from test_macd_causality import raw_candles
from test_strategy_inputs import hourly_candles, signal_config
from test_strategy_parity import wire_decisions


def recorded_bot(tmp_path):
    bot = offline_signal_bot(signal_config())
    bot._record_inputs = True
    bot.log_dir = str(tmp_path)
    return bot


def capture_session(tmp_path):
    bot = recorded_bot(tmp_path)
    raw = raw_candles(400)
    at = raw.timestamp.iloc[323]
    signals, _ = prepare_strategy_inputs(bot, "doge_jpy", raw, hourly_candles(raw), at)
    bot.reentry_block_until = {"doge_jpy": None}
    direction, _ = method(BOT, "_entry_direction")
    direction(bot, "doge_jpy", signals.iloc[-2], signals.iloc[-3], at)
    wire_decisions(bot)
    bot.short_profit_lock_armed = {"doge_jpy": False}
    bot.calculate_dynamic_exit_levels = lambda *a: {
        "take_profit_price": 95,
        "stop_loss_price": 110,
    }
    bot._evaluate_exit_event(
        "doge_jpy", raw.iloc[:324], "short", 100, at - pd.Timedelta(hours=1), at, 99.2
    )
    bot._evaluate_exit_event(
        "doge_jpy",
        raw.iloc[:324],
        "short",
        100,
        at - pd.Timedelta(hours=1),
        at + pd.Timedelta(minutes=1),
        99.8,
    )
    record_bot_event(
        bot, "input_error", {"source": "gmo_ticker", "error_type": "Timeout"}, at
    )
    close_recorder(bot)
    return bot._input_recorder.path


def test_actual_recorded_features_and_exit_state_replay_without_network_or_startup(
    tmp_path, monkeypatch
):
    path = capture_session(tmp_path)
    import dotenv
    import sqlalchemy

    monkeypatch.setattr(
        dotenv,
        "load_dotenv",
        lambda *a, **kw: pytest.fail("dotenv must not load in replay"),
    )
    monkeypatch.setattr(
        sqlalchemy,
        "create_engine",
        lambda *a, **kw: pytest.fail("DB must not connect in replay"),
    )
    before = path.read_bytes()
    result = verify_session(path)
    assert result["checked"]["feature_snapshot"] == 1
    assert result["checked"]["exit_decision"] == 2
    assert result["checked"]["entry_decision"] == 1
    assert result["linked_entry_decisions"] == 1
    assert result["checked"]["input_error"] == 1
    assert result["live_equivalent"] is False
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "damage", ["remove", "reorder", "duplicate", "edit", "partial_line", "missing_end"]
)
def test_missing_reordered_or_damaged_observations_are_not_certified(tmp_path, damage):
    path = capture_session(tmp_path)
    lines = path.read_bytes().splitlines(keepends=True)
    if damage == "remove":
        lines.pop(2)
    elif damage == "reorder":
        lines[1], lines[2] = lines[2], lines[1]
    elif damage == "duplicate":
        lines.insert(2, lines[1])
    elif damage == "edit":
        event = json.loads(lines[1])
        event["payload"]["signals_hash"] = "0" * 64
        lines[1] = json.dumps(event).encode() + b"\n"
    elif damage == "partial_line":
        lines[-1] = lines[-1][:20]
    else:
        lines.pop()
    path.write_bytes(b"".join(lines))
    with pytest.raises(RecordingError):
        verify_session(path)


def test_missing_blob_is_not_silently_skipped(tmp_path):
    path = capture_session(tmp_path)
    event = read_journal(path)[1]
    blob = path.parent / "blobs" / f"{event['payload']['inputs']}.json.gz"
    blob.write_bytes(b"corrupt")
    with pytest.raises(RecordingError, match="blob"):
        verify_session(path)


def test_replay_detects_a_validly_written_but_wrong_expected_decision(tmp_path):
    journal = ObservationJournal(
        tmp_path,
        {
            "code_fingerprint": code_fingerprint(),
            "runtime_versions": runtime_versions(),
        },
    )
    at = pd.Timestamp("2026-11-17T00:30:00+09:00")
    journal.append(
        "entry_decision",
        {
            "latest": {"timestamp": at - pd.Timedelta(minutes=15), "buy_signal": True},
            "previous": {
                "timestamp": at - pd.Timedelta(minutes=30),
                "buy_signal": True,
            },
            "blocked_until": None,
            "direction": "long",
        },
        at,
    )
    journal.close()
    with pytest.raises(RecordingError, match="Entry replay mismatch"):
        verify_session(journal.path)


def test_code_mismatch_is_rejected_before_replay(tmp_path):
    journal = ObservationJournal(tmp_path, {"code_fingerprint": "wrong"})
    journal.close()
    with pytest.raises(RecordingError, match="Code fingerprint"):
        verify_session(journal.path)


def test_concurrent_records_remain_ordered_and_complete(tmp_path):
    journal = ObservationJournal(tmp_path, {"test": True})
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: journal.append("input_error", {"index": i}), range(40)))
    journal.close()
    events = read_journal(journal.path)
    assert len(events) == 42
    assert {event["payload"]["index"] for event in events[1:-1]} == set(range(40))


def test_sensitive_fields_never_enter_a_journal(tmp_path):
    bot = recorded_bot(tmp_path)
    record_bot_event(bot, "runtime_start", {"positions": {}})
    record_bot_event(bot, "runtime_start", {"api_secret": "secret-must-not-be-written"})
    close_recorder(bot)
    assert bot._input_recording_failed is True
    assert b"secret-must-not-be-written" not in bot._input_recorder.path.read_bytes()
    with pytest.raises(RecordingError):
        read_journal(bot._input_recorder.path)


def test_recording_failure_does_not_interrupt_exit_protection(tmp_path):
    bot = recorded_bot(tmp_path)
    bot._input_recorder = SimpleNamespace(
        failed=False,
        append=lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")),
    )
    wire_decisions(bot)
    bot.short_profit_lock_armed = {"ltc_jpy": False}
    bot.calculate_dynamic_exit_levels = lambda *a: {
        "take_profit_price": 105,
        "stop_loss_price": 90,
    }
    raw = raw_candles().iloc[:3]
    decision = bot._evaluate_exit_event(
        "ltc_jpy",
        raw,
        "long",
        100,
        raw.timestamp.iloc[0],
        raw.timestamp.iloc[-1],
        89,
    )
    assert decision.reason == "損切り"
    assert bot._input_recording_failed is True
    assert bot._input_recorder.failed is True


def test_failed_append_cannot_later_appear_as_a_complete_session(tmp_path, monkeypatch):
    journal = ObservationJournal(tmp_path, {"test": True})
    original = Path.open

    def fail(path, *args, **kwargs):
        if path == journal.path and args and args[0] == "ab":
            raise OSError("disk full")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", fail)
        with pytest.raises(OSError):
            journal.append("input_error", {"reason": "timeout"})
    journal.close()
    with pytest.raises(RecordingError):
        read_journal(journal.path)


def test_actual_price_getter_records_only_public_quote_fields(tmp_path):
    bot = recorded_bot(tmp_path)
    bot.symbol_mapping = {"ltc_jpy": "LTC_JPY"}
    bot.last_api_call = 0
    bot.rate_limit_delay = 0
    function, namespace = method(BOT, "get_current_price")
    namespace["HTTP_SESSION"] = SimpleNamespace(
        get=lambda *a, **kw: SimpleNamespace(
            raise_for_status=lambda: None,
            close=lambda: None,
            json=lambda: {
                "status": 0,
                "api_secret": "never-copy-whole-responses",
                "data": [
                    {
                        "symbol": "LTC_JPY",
                        "last": "100",
                        "bid": "99",
                        "ask": "101",
                        "timestamp": "2026-11-17T00:00:00Z",
                    }
                ],
            },
        )
    )
    assert function(bot, "ltc_jpy") == 100
    close_recorder(bot)
    events = read_journal(bot._input_recorder.path)
    assert events[1]["kind"] == "ticker_response"
    assert events[1]["payload"]["tickers"][0]["bid"] == "99"
    assert "never-copy-whole-responses" not in bot._input_recorder.path.read_text(
        encoding="utf-8"
    )


def test_blob_ids_cannot_escape_the_recording_directory(tmp_path):
    with pytest.raises(RecordingError):
        read_blob(tmp_path / "session.jsonl", "../anything")


def test_request_failures_are_recorded_in_the_order_the_bot_observed_them(tmp_path):
    import requests

    bot = recorded_bot(tmp_path)
    bot.symbol_mapping = {"ltc_jpy": "LTC_JPY"}
    bot.last_api_call = bot.rate_limit_delay = 0
    getter, namespace = method(BOT, "get_current_price")
    namespace["HTTP_SESSION"] = SimpleNamespace(
        get=lambda *a, **kw: (_ for _ in ()).throw(requests.exceptions.Timeout())
    )
    assert getter(bot, "ltc_jpy") == 0
    close_recorder(bot)
    errors = [
        event
        for event in read_journal(bot._input_recorder.path)
        if event["kind"] == "input_error"
    ]
    assert [event["payload"]["attempt"] for event in errors] == [1, 2, 3]
    assert all(event["payload"]["error_type"] == "Timeout" for event in errors)


def test_backtest_does_not_modify_live_recording_or_its_feature_reference(tmp_path):
    from backtest_data import backtest_state

    bot = recorded_bot(tmp_path)
    bot.total_profit = 0
    raw = raw_candles(400)
    at = raw.timestamp.iloc[323]
    prepare_strategy_inputs(bot, "ltc_jpy", raw, hourly_candles(raw), at)
    journal = bot._input_recorder
    journal.flush()
    before = journal.path.read_bytes()
    refs = dict(bot._feature_event_refs)
    with backtest_state(bot):
        prepare_strategy_inputs(bot, "ltc_jpy", raw, hourly_candles(raw), at)
    assert journal.path.read_bytes() == before
    assert bot._feature_event_refs == refs
    close_recorder(bot)


def test_daily_rotation_closes_old_session_and_links_new_session(tmp_path):
    from datetime import timedelta

    bot = recorded_bot(tmp_path)
    record_bot_event(bot, "runtime_start", {"positions": {}})
    old = bot._input_recorder
    old.day -= timedelta(days=1)
    record_bot_event(bot, "input_error", {"reason": "test"})
    current = bot._input_recorder
    close_recorder(bot)
    assert current is not old
    old.close()
    assert read_journal(old.path)[-1]["kind"] == "session_end"
    assert read_journal(current.path)[0]["payload"]["previous_session"] == old.session


def test_runtime_version_mismatch_is_rejected(tmp_path):
    versions = {**runtime_versions(), "pandas": "different"}
    journal = ObservationJournal(
        tmp_path, {"code_fingerprint": code_fingerprint(), "runtime_versions": versions}
    )
    journal.close()
    with pytest.raises(RecordingError, match="Runtime versions"):
        verify_session(journal.path)


def test_atomic_blob_write_can_repair_a_corrupt_copy_without_changing_identity(
    tmp_path,
):
    journal = ObservationJournal(tmp_path, {"test": True})
    identity = journal.blob({"value": 123})
    (journal.blobs / f"{identity}.json.gz").write_bytes(b"broken")
    assert journal.blob({"value": 123}) == identity
    assert read_blob(journal.path, identity) == {"value": 123}
    assert list(journal.blobs.glob("*.tmp")) == []
    journal.close()


def test_unknown_event_kind_is_not_treated_as_verified(tmp_path):
    journal = ObservationJournal(
        tmp_path,
        {
            "code_fingerprint": code_fingerprint(),
            "runtime_versions": runtime_versions(),
        },
    )
    journal.append("unknown_future_event", {})
    journal.close()
    with pytest.raises(RecordingError, match="Unsupported observation"):
        verify_session(journal.path)


def test_cli_replays_in_a_fresh_process_with_network_db_and_dotenv_forbidden(tmp_path):
    import os
    import subprocess
    import sys

    path = capture_session(tmp_path)
    guard = """
import runpy, socket, sys
import dotenv, sqlalchemy
def forbidden(*args, **kwargs):
    raise AssertionError('Network, DB and dotenv are forbidden during replay')
socket.socket.connect = forbidden
socket.create_connection = forbidden
dotenv.load_dotenv = forbidden
sqlalchemy.create_engine = forbidden
sys.argv = ['replay_observations.py', sys.argv[1]]
runpy.run_path('replay_observations.py', run_name='__main__')
"""
    environment = {**os.environ, "DATABASE_URL": "must-not-be-read"}
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", guard, str(path)],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["checked"]["feature_snapshot"] == 1
    assert output["checked"]["entry_decision"] == 1
    assert output["checked"]["exit_decision"] == 2


def test_slow_disk_does_not_block_the_real_exit_decision(tmp_path, monkeypatch):
    import threading
    import time

    entered, release = threading.Event(), threading.Event()
    original = ObservationJournal.append

    def slow(self, kind, payload, *a, **kw):
        if kind == "input_error" and payload.get("reason") == "hold":
            entered.set()
            assert release.wait(5)
        return original(self, kind, payload, *a, **kw)

    monkeypatch.setattr(ObservationJournal, "append", slow)
    bot = recorded_bot(tmp_path)
    bot._input_recorder = BackgroundJournal(
        tmp_path,
        {
            "code_fingerprint": code_fingerprint(),
            "runtime_versions": runtime_versions(),
        },
    )
    bot._input_recorder.append("input_error", {"reason": "hold"})
    try:
        assert entered.wait(2)
        wire_decisions(bot)
        bot.short_profit_lock_armed = {"ltc_jpy": False}
        bot.calculate_dynamic_exit_levels = lambda *a: {
            "take_profit_price": 105,
            "stop_loss_price": 90,
        }
        raw = raw_candles().iloc[:3]
        started = time.monotonic()
        decision = bot._evaluate_exit_event(
            "ltc_jpy",
            raw,
            "long",
            100,
            raw.timestamp.iloc[0],
            raw.timestamp.iloc[-1],
            89,
        )
        assert decision.reason == "損切り"
        assert time.monotonic() - started < 0.5
        assert not release.is_set()
    finally:
        release.set()
        close_recorder(bot)
    assert verify_session(bot._input_recorder.path)["checked"]["exit_decision"] == 1


def test_queue_overflow_is_bounded_and_cannot_be_certified_complete(
    tmp_path, monkeypatch
):
    import threading

    entered, release = threading.Event(), threading.Event()
    original = ObservationJournal.append

    def slow(self, kind, payload, *a, **kw):
        if kind == "input_error" and payload.get("reason") == "hold":
            entered.set()
            assert release.wait(5)
        return original(self, kind, payload, *a, **kw)

    monkeypatch.setattr(ObservationJournal, "append", slow)
    journal = BackgroundJournal(tmp_path, {"test": True}, capacity=1)
    journal.append("input_error", {"reason": "hold"})
    try:
        assert entered.wait(2)
        journal.append("input_error", {"reason": "queued"})
        with pytest.raises(RecordingError, match="queue full"):
            journal.append("input_error", {"reason": "overflow"})
    finally:
        release.set()
        journal.close()
    assert journal.dropped == 1
    with pytest.raises(RecordingError, match="incomplete"):
        read_journal(journal.path)


def test_entry_must_reference_the_same_actual_feature_values(tmp_path):
    path = capture_session(tmp_path)
    events = read_journal(path)
    altered = ObservationJournal(path.parent, events[0]["payload"])
    for event in events[1:-1]:
        payload = event["payload"]
        if event["kind"] == "entry_decision":
            # Direction alone would still match: RSI is not read by entry_side.
            payload["latest"]["RSI"] += 1
        altered.append(event["kind"], payload, event["observed_at"])
    altered.close()
    with pytest.raises(RecordingError, match="Entry/feature snapshot mismatch"):
        verify_session(altered.path)
