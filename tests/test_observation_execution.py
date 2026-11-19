"""Local PostgreSQL + fake exchange: recording must not alter actual-fill handling."""

from decimal import Decimal
import sys
from types import SimpleNamespace

from strategy_decisions import next_entry_time
from strategy_recording import close_recorder, read_journal
from test_execution import FakeBot, service


def test_funds_and_real_fill_outcomes_are_recorded_without_credentials(
    store, tmp_path, monkeypatch
):
    svc, api = service(store)
    bot = FakeBot(svc)
    bot._record_inputs = True
    bot.log_dir = str(tmp_path)
    bot.exchange_settings_gmo = {
        "live_trade": True,
        "api_secret": "do-not-record-settings",
    }
    bot.notification_settings = {"send_on_exit": False}
    bot._ceil_to_next_15min = next_entry_time
    bot.entry_scores = {"ltc_jpy": {}}
    result = bot._execute_live_entry(
        "ltc_jpy",
        "buy",
        10,
        signal_raw={
            "decision_bar": "2026-11-16T01:00:00+09:00",
            "decision_price": "100",
        },
    )
    assert result["success"]
    monkeypatch.setitem(sys.modules, "db", SimpleNamespace(engine=store.engine))
    api.fill_price = Decimal(110)
    bot._close_live_position("ltc_jpy", "利益確定", {}, [])
    close_recorder(bot)
    events = read_journal(bot._input_recorder.path)
    funds = next(
        event["payload"] for event in events if event["kind"] == "funds_snapshot"
    )
    assert "bid" in funds["ticker"] and "ask" in funds["ticker"]
    assert "sizeStep" in funds["rule"] and "takerFee" in funds["rule"]
    assert "availableAmount" in funds["margin"]
    outcomes = [event["payload"] for event in events if event["kind"] == "order_result"]
    assert [outcome["purpose"] for outcome in outcomes] == ["entry", "close"]
    assert Decimal(outcomes[-1]["trades"][0]["pnl"]) == 100
    assert "do-not-record-settings" not in bot._input_recorder.path.read_text(
        encoding="utf-8"
    )


def test_audit_disk_failure_does_not_duplicate_or_prevent_a_confirmed_order(store):
    svc, api = service(store)
    bot = FakeBot(svc)
    bot._record_inputs = True
    bot._input_recorder = SimpleNamespace(
        failed=False, append=lambda *a, **kw: (_ for _ in ()).throw(OSError("full"))
    )
    bot.entry_scores = {"ltc_jpy": {}}
    raw = {"decision_bar": "2026-11-16T01:00:00+09:00", "decision_price": "100"}
    first = bot._execute_live_entry("ltc_jpy", "buy", 10, signal_raw=raw)
    second = bot._execute_live_entry("ltc_jpy", "buy", 10, signal_raw=raw)
    assert first["success"] is True
    assert second["success"] is False
    assert len(api.posts) == 1
    assert bot.positions["ltc_jpy"] == "long"
    assert bot._input_recording_failed is True
