from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import logging
import pytest
from sqlalchemy import text

from trading_execution import (
    ExecutionError,
    ExchangeGateway,
    ExecutionService,
    decimal,
    normalize_size,
    required_margin,
    fallback_exit,
)
from live_execution import LiveExecutionMixin


class FakeExchange:
    api_key = "test-account"

    def __init__(self):
        self.posts = []
        self.orders = {}
        self.fills = {}
        self.position_rows = []
        self.lose_response = False
        self.fill_ratio = Decimal(1)
        self.hide_fills = False
        self.fill_price = Decimal("100")
        self.fee = Decimal("1")
        self.status = "EXECUTED"
        self.now = datetime.now(timezone.utc).isoformat()
        self.fail_read = False
        self.reject = False
        self.reject_code = "ERR-201"

    def _public_request(self, path, params=None):
        if path == "/v1/symbols":
            return {
                "status": 0,
                "data": [
                    {
                        "symbol": "LTC_JPY",
                        "sizeStep": "1",
                        "minOrderSize": "1",
                        "maxOrderSize": "1000",
                        "takerFee": "0.0003",
                    }
                ],
            }
        return {
            "status": 0,
            "data": [
                {"symbol": "LTC_JPY", "ask": "101", "bid": "99", "timestamp": self.now}
            ],
        }

    def _request(self, method, path, params=None, data=None):
        if method == "POST":
            self.posts.append(deepcopy(data))
            if self.reject:
                return {"status": 1, "messages": [{"message_code": self.reject_code}]}
            oid = str(100 + len(self.posts))
            cid = data["clientOrderId"]
            close = "settlePosition" in data
            requested = decimal(
                data["settlePosition"][0]["size"] if close else data["size"]
            )
            qty = requested * self.fill_ratio
            pid = (
                str(data["settlePosition"][0]["positionId"])
                if close
                else str(200 + len(self.posts))
            )
            status = self.status if self.fill_ratio == 1 else "CANCELED"
            row = {
                "orderId": oid,
                "clientOrderId": cid,
                "symbol": data["symbol"],
                "side": data["side"],
                "settleType": "CLOSE" if close else "OPEN",
                "size": str(requested),
                "executedSize": str(qty),
                "status": status,
                "timestamp": self.now,
            }
            self.orders[cid] = row
            pnl = Decimal(0)
            if close:
                pos = next(p for p in self.position_rows if str(p["positionId"]) == pid)
                pnl = (
                    (self.fill_price - decimal(pos["price"]))
                    * qty
                    * (1 if pos["side"] == "BUY" else -1)
                )
                pos["size"] = str(decimal(pos["size"]) - qty)
                self.position_rows = [
                    p for p in self.position_rows if decimal(p["size"]) > 0
                ]
            elif qty:
                self.position_rows.append(
                    {
                        "positionId": pid,
                        "symbol": data["symbol"],
                        "side": data["side"],
                        "size": str(qty),
                        "price": str(self.fill_price),
                        "timestamp": self.now,
                    }
                )
            self.fills[oid] = (
                [
                    {
                        "executionId": str(300 + len(self.posts)),
                        "orderId": oid,
                        "clientOrderId": cid,
                        "positionId": pid,
                        "symbol": data["symbol"],
                        "side": data["side"],
                        "settleType": row["settleType"],
                        "size": str(qty),
                        "price": str(self.fill_price),
                        "fee": str(self.fee),
                        "lossGain": str(pnl),
                        "timestamp": self.now,
                    }
                ]
                if qty
                else []
            )
            if self.lose_response:
                self.lose_response = False
                raise TimeoutError("Response lost AFTER exchange acceptance")
            return {"status": 0, "data": oid}
        if self.fail_read:
            return {"status": -1}
        if path == "/v1/orders":
            rows = [
                o
                for o in self.orders.values()
                if (
                    str(o["orderId"]) == params.get("orderId")
                    or o["clientOrderId"] == params.get("clientOrderId")
                )
            ]
            return {"status": 0, "data": {"list": rows}}
        if path == "/v1/executions":
            return {
                "status": 0,
                "data": {
                    "list": []
                    if self.hide_fills
                    else deepcopy(self.fills.get(params["orderId"], []))
                },
            }
        if path == "/v1/openPositions":
            return {
                "status": 0,
                "data": {
                    "list": deepcopy(
                        [
                            p
                            for p in self.position_rows
                            if p["symbol"] == params["symbol"]
                        ]
                    )
                },
            }
        if path == "/v1/account/margin":
            return {
                "status": 0,
                "data": {
                    "actualProfitLoss": "100000",
                    "availableAmount": "50000",
                    "margin": "40000",
                    "profitLoss": "10000",
                },
            }
        if path == "/v1/account/exchangeFee/history":
            return {
                "status": 0,
                "data": {"list": [], "lastFinalizedDate": params["toDate"]},
            }
        raise AssertionError((method, path, params))


def service(store, api=None):
    api = api or FakeExchange()
    return ExecutionService(ExchangeGateway(api), store, "test"), api


def records(store, table):
    with store.engine.connect() as conn:
        return [
            dict(r) for r in conn.execute(text("SELECT * FROM " + table)).mappings()
        ]


def test_migration_is_repeatable_and_preserves_old_rows(store):
    from pathlib import Path

    with store.engine.begin() as conn:
        conn.execute(
            text("INSERT INTO trades(trade_id,pnl,raw) VALUES('legacy',12,'{}')")
        )
    with store.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.exec_driver_sql(
            Path("migrations/001_execution_accounting.sql").read_text()
        )
    row = records(store, "trades")[0]
    assert row["pnl"] == 12 and row["accounting_version"] == "legacy_estimate"
    assert row["net_pnl_before_carry"] is None


def test_lost_response_then_reconcile_no_second_post(store):
    svc, api = service(store)
    api.lose_response = True
    result = svc.submit(
        "LTC_JPY", "BUY", "10", decision_key="bar1", context={"bot_symbol": "ltc_jpy"}
    )
    assert result["success"] and result["execution_price"] == 100
    again = svc.submit("LTC_JPY", "BUY", "10", decision_key="bar1")
    assert len(api.posts) == 1 and again["executed_size"] == 10
    assert len(records(store, "fills")) == 1


def test_restart_after_post_and_before_accept_commit(store, monkeypatch):
    svc, api = service(store)
    original = store.accept
    monkeypatch.setattr(
        store,
        "accept",
        lambda *a: (_ for _ in ()).throw(RuntimeError("DB down after POST")),
    )
    with pytest.raises(RuntimeError):
        svc.submit(
            "LTC_JPY",
            "BUY",
            "10",
            decision_key="bar1",
            context={"bot_symbol": "ltc_jpy"},
        )
    assert len(api.posts) == 1
    monkeypatch.setattr(store, "accept", original)
    recovered = ExecutionService(ExchangeGateway(api), store, "test").recover()
    assert recovered[0]["executed_size"] == 10 and len(api.posts) == 1


def test_unknown_order_blocks_new_exposure(store):
    svc, api = service(store)
    api.reject = True
    with pytest.raises(ExecutionError):
        svc.submit("LTC_JPY", "BUY", "10", decision_key="bar1")
    with pytest.raises(ExecutionError):
        svc.submit("LTC_JPY", "BUY", "10", decision_key="bar2")
    assert len(api.posts) == 1


def test_documented_rejection_releases_pending_but_does_not_retry_same_decision(store):
    svc, api = service(store)
    api.reject = True
    api.reject_code = "ERR-5106"
    result = svc.submit("LTC_JPY", "BUY", 10, decision_key="rejected")
    assert result["finalized"] and result["status"] == "REJECTED"
    assert not store.pending("test")
    svc.submit("LTC_JPY", "BUY", 10, decision_key="rejected")
    assert len(api.posts) == 1


def test_pending_account_order_cannot_be_reused_for_different_symbol(store):
    svc, api = service(store)
    api.reject = True
    with pytest.raises(ExecutionError):
        svc.submit("LTC_JPY", "BUY", 10, decision_key="pending")
    with pytest.raises(ExecutionError):
        svc.submit("ETH_JPY", "BUY", 1, decision_key="other-symbol")
    assert len(api.posts) == 1


def test_database_failure_before_send_means_no_post(store, monkeypatch):
    svc, api = service(store)
    monkeypatch.setattr(
        store,
        "prepare",
        lambda *a: (_ for _ in ()).throw(RuntimeError("DB unavailable")),
    )
    with pytest.raises(RuntimeError):
        svc.submit("LTC_JPY", "BUY", 10)
    assert not api.posts


def test_concurrent_claim_only_sends_once(store):
    svc, api = service(store)

    def submit():
        try:
            return svc.submit(
                "LTC_JPY",
                "BUY",
                10,
                decision_key="same",
                context={"bot_symbol": "ltc_jpy"},
            )
        except ExecutionError:
            # Another sender may be between claim and exchange acceptance. Reads can retry.
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [f.result() for f in [pool.submit(submit), pool.submit(submit)]]
    svc.recover()
    assert len(api.posts) == 1 and len(records(store, "fills")) == 1


def test_partial_close_remains_open_and_fees_are_allocated(store):
    svc, api = service(store)
    opening = svc.submit(
        "LTC_JPY", "BUY", 10, decision_key="open", context={"bot_symbol": "ltc_jpy"}
    )
    pos = deepcopy(api.position_rows[0])
    api.fill_ratio = Decimal(".6")
    api.fill_price = Decimal("110")
    closing = svc.submit(
        "LTC_JPY",
        "SELL",
        10,
        position_id=pos["positionId"],
        context={"bot_symbol": "ltc_jpy", "baseline": pos, "exit_reason": "test"},
    )
    assert closing["finalized"] and closing["executed_size"] == 6
    assert records(store, "positions")[0]["size"] == 4
    trade = records(store, "trades")[0]
    assert trade["gross_pnl"] == 60 and trade["entry_fee"] == Decimal(".6")
    assert trade["exit_fee"] == 1 and trade["net_pnl_before_carry"] == Decimal("58.4")
    svc.reconcile(store.get(closing["client_order_id"]))
    assert len(records(store, "trades")) == 1 and len(records(store, "fills")) == 2
    api.fill_ratio = Decimal(1)
    last = deepcopy(api.position_rows[0])
    svc.submit(
        "LTC_JPY",
        "SELL",
        4,
        position_id=last["positionId"],
        context={"bot_symbol": "ltc_jpy", "baseline": last},
    )
    assert records(store, "positions")[0]["size"] == 0
    trades = records(store, "trades")
    assert sum(t["gross_pnl"] for t in trades) == 100
    assert sum(t["entry_fee"] for t in trades) == 1


def test_canceled_zero_does_not_use_unrelated_position(store):
    svc, api = service(store)
    api.fill_ratio = Decimal(0)
    api.position_rows = [
        {
            "positionId": "99",
            "symbol": "LTC_JPY",
            "side": "SELL",
            "size": "5",
            "price": "100",
            "timestamp": api.now,
        }
    ]
    result = svc.submit("LTC_JPY", "BUY", 10, decision_key="zero")
    assert not result["success"] and result["finalized"]
    assert not records(store, "fills") and not records(store, "trades")


def test_missing_executions_not_finalized_or_resubmitted(store):
    svc, api = service(store)
    api.hide_fills = True
    first = svc.submit(
        "LTC_JPY", "BUY", 10, decision_key="missing", context={"bot_symbol": "ltc_jpy"}
    )
    assert not first["finalized"] and not first["success"]
    api.hide_fills = False
    svc.recover()
    assert not store.pending("test") and len(api.posts) == 1


def test_transaction_rollback_after_execution_read_is_recoverable(store, monkeypatch):
    svc, api = service(store)
    original = store._project_position
    monkeypatch.setattr(
        store,
        "_project_position",
        lambda *a: (_ for _ in ()).throw(RuntimeError("Projection failed")),
    )
    with pytest.raises(RuntimeError):
        svc.submit(
            "LTC_JPY",
            "BUY",
            10,
            decision_key="rollback",
            context={"bot_symbol": "ltc_jpy"},
        )
    assert not records(store, "fills") and not records(store, "bot_execution_fills")
    monkeypatch.setattr(store, "_project_position", original)
    svc.recover()
    assert len(api.posts) == 1 and len(records(store, "fills")) == 1


def test_legacy_position_entry_fee_remains_unknown(store):
    svc, api = service(store)
    pos = {
        "positionId": "900",
        "symbol": "LTC_JPY",
        "side": "BUY",
        "size": "10",
        "price": "90",
        "timestamp": api.now,
    }
    api.position_rows = [deepcopy(pos)]
    svc.submit(
        "LTC_JPY",
        "SELL",
        10,
        position_id="900",
        context={"bot_symbol": "ltc_jpy", "baseline": pos},
    )
    trade = records(store, "trades")[0]
    assert trade["pnl"] == 100 and trade["entry_fee"] is None
    assert trade["net_pnl_before_carry"] is None


def test_trade_short_pnl_and_maker_rebate(store):
    svc, api = service(store)
    api.fee = Decimal("-1")
    result = svc.submit(
        "LTC_JPY", "SELL", 10, decision_key="short", context={"bot_symbol": "ltc_jpy"}
    )
    pos = deepcopy(api.position_rows[0])
    api.fill_price = Decimal("90")
    svc.submit(
        "LTC_JPY",
        "BUY",
        10,
        position_id=pos["positionId"],
        context={"bot_symbol": "ltc_jpy", "baseline": pos},
    )
    trade = records(store, "trades")[0]
    assert trade["gross_pnl"] == 100 and trade["net_pnl_before_carry"] == 102


def test_fee_resync_upserts_and_tracks_finalized_date(store):
    svc, api = service(store)
    original = api._request
    row = {
        "symbol": "LTC_JPY",
        "feeType": "LEVERAGE_FEE",
        "deliveryDate": "20261115",
        "deliveryDatetime": "2026-11-15T21:00:00Z",
        "deliveryAmount": "-12",
    }
    api._request = lambda method, path, params=None, data=None: (
        {
            "status": 0,
            "data": {
                "list": [row]
                if params["fromDate"] <= "20261115" <= params["toDate"]
                else [],
                "lastFinalizedDate": "20261115",
            },
        }
        if "exchangeFee" in path
        else original(method, path, params, data)
    )
    from datetime import date

    svc.sync_fees(date(2026, 11, 16))
    svc.sync_fees(date(2026, 11, 16))
    assert len(records(store, "bot_account_fees")) == 1 and store.fee_through(
        "test"
    ) == date(2026, 11, 15)


@pytest.mark.parametrize("bad", [0, -1, "NaN", "Infinity", None])
def test_invalid_numbers_rejected(bad):
    with pytest.raises(ExecutionError):
        decimal(bad, positive=True)


def test_size_floor_and_margin_units():
    assert normalize_size("10.9", "1", "1") == 10
    with pytest.raises(ExecutionError):
        normalize_size(".9", "1", "1")
    assert required_margin(1, 300000, 2, 0) > 150000


@pytest.mark.parametrize("side", ["long", "short"])
def test_fallback_prices_correct_side(side):
    settings = {
        "long_stop_loss": ".972",
        "long_profit_take": "1.022",
        "short_stop_loss": "1.028",
        "short_profit_take": ".978",
    }
    result = fallback_exit(settings, side, 1000)
    if side == "long":
        assert result["stop_loss_price"] < 1000 < result["take_profit_price"]
    else:
        assert result["take_profit_price"] < 1000 < result["stop_loss_price"]


class FakeBot(LiveExecutionMixin):
    def __init__(self, svc):
        self._live_execution_service = svc
        self.gmo_api = svc.gateway.api
        self.symbol_mapping = {"ltc_jpy": "LTC_JPY"}
        self.symbols = ["ltc_jpy"]
        self.logger = logging.getLogger("test")
        self.positions = {"ltc_jpy": None}
        self.entry_sizes = {}
        self.entry_prices = {}
        self.entry_times = {}
        self.position_ids = {}
        self.entry_scores = {}
        self.short_profit_lock_armed = {}
        self.reentry_block_until = {}

    def save_positions(self):
        pass

    def _record_signal(self, **kw):
        return None


def test_margin_read_error_disallows_entry_but_keeps_position(store):
    svc, api = service(store)
    bot = FakeBot(svc)
    bot.positions["ltc_jpy"] = "long"
    bot.entry_sizes["ltc_jpy"] = 10
    api.fail_read = True
    bot._sync_live_positions()
    assert bot.positions["ltc_jpy"] == "long" and bot.entry_sizes["ltc_jpy"] == 10
    with pytest.raises(ExecutionError):
        bot._live_funds_check("ltc_jpy", "buy", 10)


def test_live_entry_uses_executed_price_not_decision_or_ticker(store):
    svc, api = service(store)
    bot = FakeBot(svc)
    bot._entry_decision_snapshot = {}
    bot.entry_scores = {"ltc_jpy": {"entry_adx": 30}}
    result = bot._execute_live_entry(
        "ltc_jpy",
        "buy",
        10,
        signal_raw={
            "decision_bar": "2026-11-16T01:00:00+09:00",
            "decision_price": "88",
        },
    )
    assert result["execution_price"] == 100 and bot.entry_prices["ltc_jpy"] == 100
    assert records(store, "positions")[0]["avg_entry_price"] == 100


def test_multiple_execution_prices_use_quantity_weighted_average(store):
    svc, api = service(store)
    original = api._request

    def request(method, path, params=None, data=None):
        response = original(method, path, params, data)
        if method == "POST":
            oid = response["data"]
            first = api.fills[oid][0]
            api.fills[oid] = [
                {**first, "size": "4", "price": "100", "fee": ".4"},
                {
                    **first,
                    "executionId": "second",
                    "size": "6",
                    "price": "110",
                    "fee": ".6",
                },
            ]
            api.position_rows[0]["price"] = "106"
        return response

    api._request = request
    result = svc.submit(
        "LTC_JPY", "BUY", 10, decision_key="weighted", context={"bot_symbol": "ltc_jpy"}
    )
    assert result["execution_price"] == 106
    assert records(store, "positions")[0]["avg_entry_price"] == 106
    assert len(records(store, "fills")) == 2


def test_mismatched_execution_is_not_recorded(store):
    svc, api = service(store)
    original = api._request

    def request(method, path, params=None, data=None):
        response = original(method, path, params, data)
        if path == "/v1/executions" and response["data"]["list"]:
            response["data"]["list"][0]["side"] = "SELL"
        return response

    api._request = request
    with pytest.raises(ExecutionError):
        svc.submit("LTC_JPY", "BUY", 10, decision_key="wrong-side")
    assert not records(store, "fills")
    assert store.pending("test")


def test_stale_reconciliation_cannot_reopen_finalized_order(store):
    svc, api = service(store)
    result = svc.submit(
        "LTC_JPY", "BUY", 10, decision_key="complete", context={"bot_symbol": "ltc_jpy"}
    )
    intent = store.get(result["client_order_id"])
    stale = {
        **api.orders[intent["client_order_id"]],
        "status": "ORDERED",
        "executedSize": "0",
    }
    store.record(intent, stale, [], False)
    assert store.get(intent["client_order_id"])["finalized"]
    assert records(store, "positions")[0]["size"] == 10


def test_regressing_exchange_quantity_does_not_erase_known_fills(store):
    svc, api = service(store)
    api.status = "ORDERED"
    result = svc.submit(
        "LTC_JPY",
        "BUY",
        10,
        decision_key="in-progress",
        context={"bot_symbol": "ltc_jpy"},
    )
    intent = store.get(result["client_order_id"])
    stale = {**api.orders[intent["client_order_id"]], "executedSize": "0"}
    with pytest.raises(ExecutionError):
        store.record(intent, stale, [], False)
    assert records(store, "positions")[0]["size"] == 10
    assert not store.get(intent["client_order_id"])["finalized"]
