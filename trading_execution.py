"""Exchange execution primitives. No bot, database, network or environment side effects."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation, ROUND_DOWN
import hashlib
import time

TERMINAL = frozenset({"EXECUTED", "CANCELED", "EXPIRED", "REJECTED"})
DECLINED_CODES = frozenset(
    {
        "ERR-5106",
        "ERR-5111",
        "ERR-5114",
        "ERR-5118",
        "ERR-5121",
        "ERR-5125",
        "ERR-5126",
        "ERR-5127",
        "ERR-5129",
    }
)


class ExecutionError(RuntimeError):
    pass


class ExecutionPending(ExecutionError):
    """An identified read snapshot can lag another exchange endpoint."""

    pass


class ExchangeMaintenance(ExecutionError):
    """Explicit exchange unavailability; only a local pre-send guard proves no POST."""
    def __init__(self, state='MAINTENANCE', *, not_sent=False):
        super().__init__('Exchange unavailable: ' + state)
        self.state, self.not_sent = state, not_sent


def decimal(value, *, positive=False):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ExecutionError("Invalid numeric exchange value") from None
    if not result.is_finite() or (positive and result <= 0):
        raise ExecutionError("Invalid numeric exchange value")
    return result


def timestamp(value):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        raise ExecutionError("Missing or invalid exchange timestamp") from None
    if result.tzinfo is None:
        raise ExecutionError("Exchange timestamp must include timezone")
    return result.astimezone(timezone.utc)


def account_key(api_key):
    if not api_key:
        raise ExecutionError("Missing exchange account identity")
    return hashlib.sha256(api_key.encode()).hexdigest()[:24]


def normalize_size(size, step, minimum):
    size, step, minimum = (
        decimal(size, positive=True),
        decimal(step, positive=True),
        decimal(minimum, positive=True),
    )
    quantity = (size / step).to_integral_value(rounding=ROUND_DOWN) * step
    if quantity < minimum:
        raise ExecutionError("Quantity below minimum; do not round exposure up")
    return quantity


def required_margin(size, price, leverage, taker_fee, buffer="0.005"):
    size, price = decimal(size, positive=True), decimal(price, positive=True)
    leverage = decimal(leverage, positive=True)
    fee, buffer = decimal(taker_fee), decimal(buffer)
    if not 1 <= leverage <= 2 or fee < 0 or buffer < 0:
        raise ExecutionError("Invalid margin configuration")
    notional = size * price
    return notional / leverage + notional * (fee * 2 + buffer)


def fallback_exit(settings, side, entry):
    """Preserve the normal strategy's distance mapping, including its inverted labels."""
    entry = decimal(entry, positive=True)
    if side not in ("long", "short"):
        raise ExecutionError("Invalid position side")
    tp = abs(decimal(settings[side + "_stop_loss"]) - 1)
    sl = abs(decimal(settings[side + "_profit_take"]) - 1)
    if not (0 < tp < 1 and 0 < sl < 1):
        raise ExecutionError("Invalid protective distances")
    return {
        "take_profit_price": float(entry * (1 + tp if side == "long" else 1 - tp)),
        "stop_loss_price": float(entry * (1 - sl if side == "long" else 1 + sl)),
        "take_profit_ratio": float(1 + tp if side == "long" else 1 - tp),
        "stop_loss_ratio": float(1 - sl if side == "long" else 1 + sl),
        "atr_value": 0,
        "risk_reward_ratio": float(tp / sl),
    }


class ExchangeGateway:
    def __init__(self, api):
        self.api = api

    def request(self, path, params=None):
        response = self.api._request("GET", path, params=params)
        from exchange_availability import maintenance_reason
        state = maintenance_reason(response)
        if state:
            availability = getattr(self.api, 'availability', None)
            if availability is not None:
                availability.observe(response)
            raise ExchangeMaintenance(state)
        if not isinstance(response, dict) or response.get("status") != 0:
            codes = []
            if isinstance(response, dict):
                for message in response.get('messages', []):
                    code = message.get('message_code') if isinstance(message, dict) else None
                    if isinstance(code, str) and code.startswith('ERR-') and code[4:].isdigit():
                        codes.append(code)
            raise ExecutionError("Exchange read not confirmed: " + path + " codes=" + ','.join(codes))
        return response.get("data")

    def pages(self, path, params):
        result = []
        seen_positions = set()
        for page in range(1, 1001):
            data = self.request(path, {**params, "page": page, "count": 100})
            if path == '/v1/openPositions' and data == {}:
                return result
            # GMO can omit list for an empty, successful paginated position response.
            # Require the exact requested pagination; malformed/null lists stay unconfirmed.
            if (path == '/v1/openPositions' and isinstance(data, dict)
                    and set(data) == {'pagination'}
                    and isinstance(data['pagination'], dict)
                    and type(data['pagination'].get('currentPage')) is int
                    and type(data['pagination'].get('count')) is int
                    and data['pagination'] == {'currentPage': page, 'count': 100}):
                return result
            if not isinstance(data, dict) or not isinstance(data.get("list"), list):
                raise ExecutionError("Invalid exchange list: data_type=" + type(data).__name__ +
                                     " list_type=" + type(data.get('list') if isinstance(data, dict) else None).__name__ +
                                     " list_present=" + str(isinstance(data, dict) and 'list' in data) +
                                     " pagination_type=" + type(data.get('pagination') if isinstance(data, dict) else None).__name__ +
                                     " pagination=" + str({key: value for key, value in (data.get('pagination', {}).items() if isinstance(data, dict) and isinstance(data.get('pagination'), dict) else []) if key in ('currentPage','count','totalPage') and type(value) is int}))
            rows = data["list"]
            if path == "/v1/openPositions":
                ids = [str(row.get("positionId")) for row in rows]
                if len(set(ids)) != len(ids) or seen_positions.intersection(ids):
                    raise ExecutionError(
                        "Position pagination changed or repeated; retry snapshot"
                    )
                seen_positions.update(ids)
            result.extend(rows)
            pagination = data.get("pagination", {})
            if not rows or len(rows) < 100:
                return result
            if pagination.get("totalPage") is not None and page >= int(
                pagination["totalPage"]
            ):
                return result
        raise ExecutionError("Exchange pagination limit exceeded")

    def positions(self, symbol):
        rows = self.pages("/v1/openPositions", {"symbol": symbol})
        for row in rows:
            if (
                row.get("symbol") != symbol
                or row.get("side") not in ("BUY", "SELL")
                or row.get("positionId") is None
            ):
                raise ExecutionError("Invalid position identity")
            decimal(row["size"], positive=True)
            decimal(row["price"], positive=True)
            timestamp(row["timestamp"])
        return rows

    def margin(self):
        data = self.request("/v1/account/margin")
        return {
            name: decimal(data[name])
            for name in ("actualProfitLoss", "availableAmount", "margin", "profitLoss")
        }

    def rule(self, symbol):
        response = self.api._public_request("/v1/symbols")
        if not isinstance(response, dict) or response.get("status") != 0:
            raise ExecutionError("Trading rules unavailable")
        data = response["data"]
        rows = data if isinstance(data, list) else data.get("list", [])
        row = next((r for r in rows if r.get("symbol") == symbol), None)
        if row is None:
            raise ExecutionError("Missing exchange trading rule")
        return row

    def order(self, intent):
        params = (
            {"orderId": intent["exchange_order_id"]}
            if intent.get("exchange_order_id")
            else {"clientOrderId": intent["client_order_id"]}
        )
        data = self.request("/v1/orders", params)
        rows = data.get("list", []) if isinstance(data, dict) else []
        rows = [
            r
            for r in rows
            if (
                str(r.get("clientOrderId")) == intent["client_order_id"]
                or (
                    intent.get("exchange_order_id")
                    and str(r.get("orderId")) == intent["exchange_order_id"]
                )
            )
        ]
        if not rows:
            raise ExecutionPending("Order not yet visible")
        if len(rows) != 1:
            raise ExecutionError("Order identity not confirmed")
        row = rows[0]
        if (
            row.get("clientOrderId") is not None
            and str(row["clientOrderId"]) != intent["client_order_id"]
        ):
            raise ExecutionError("Order belongs to another client intent")
        if (
            row.get("symbol") != intent["symbol"]
            or row.get("side") != intent["side"]
            or row.get("settleType") != intent["settle_type"]
        ):
            raise ExecutionError("Order attributes do not match intent")
        if decimal(row.get("size"), positive=True) != decimal(intent["size"]):
            raise ExecutionError("Order quantity does not match intent")
        return row

    def executions(self, intent, order):
        data = self.request("/v1/executions", {"orderId": str(order["orderId"])})
        if not isinstance(data, dict) or not isinstance(data.get("list"), list):
            raise ExecutionError("Invalid executions response")
        rows = data["list"]
        unique = {}
        for row in rows:
            if (
                str(row.get("orderId")) != str(order["orderId"])
                or row.get("symbol") != intent["symbol"]
                or row.get("side") != intent["side"]
                or row.get("settleType") != intent["settle_type"]
                or row.get("executionId") is None
                or row.get("positionId") is None
            ):
                raise ExecutionError("Execution identity mismatch")
            if (
                intent.get("position_id")
                and str(row["positionId"]) != intent["position_id"]
            ):
                raise ExecutionError("Execution belongs to another position")
            decimal(row["price"], positive=True)
            decimal(row["size"], positive=True)
            if row.get("fee") is not None:
                decimal(row["fee"])
            if intent["settle_type"] == "CLOSE":
                decimal(row["lossGain"])
            timestamp(row["timestamp"])
            key = str(row["executionId"])
            if key in unique and unique[key] != row:
                raise ExecutionError("Conflicting duplicate execution")
            unique[key] = row
        return list(unique.values())

    def daily_fees(self, start, end):
        result, cursor, seen = [], None, set()
        finalized = None
        for _ in range(1000):
            params = {
                "fromDate": start.strftime("%Y%m%d"),
                "toDate": end.strftime("%Y%m%d"),
                "count": 100,
            }
            if cursor:
                params["cursor"] = cursor
            data = self.request("/v1/account/exchangeFee/history", params)
            current = data.get("lastFinalizedDate")
            if finalized is not None and current != finalized:
                raise ExecutionError(
                    "Fee finalization advanced during pagination; retry from first page"
                )
            finalized = current
            for row in data.get("list", []):
                day = datetime.strptime(row["deliveryDate"], "%Y%m%d").date()
                if not start <= day <= end:
                    raise ExecutionError("Fee outside requested range")
                decimal(row["deliveryAmount"])
                timestamp(row["deliveryDatetime"])
                result.append(row)
            cursor = data.get("pagination", {}).get("cursor")
            if not cursor:
                return result, finalized
            if cursor in seen:
                raise ExecutionError("Repeated fee cursor")
            seen.add(cursor)
        raise ExecutionError("Fee pagination limit exceeded")


class ExecutionService:
    def __init__(self, gateway, store, account, *, retry_delays=(0.2, 0.5)):
        self.gateway, self.store, self.account = gateway, store, account
        self.retry_delays = retry_delays

    def submit(
        self, symbol, side, size, *, position_id=None, decision_key=None, context=None
    ):
        settle = "CLOSE" if position_id is not None else "OPEN"
        if side not in ("BUY", "SELL"):
            raise ExecutionError("Invalid order side")
        size = decimal(size, positive=True)
        availability = getattr(self.gateway.api, 'availability', None)
        if availability is not None:
            availability.require_available()
        body = {
            "symbol": symbol,
            "side": side,
            "executionType": "MARKET",
            "size": format(size, "f"),
        }
        if position_id is not None:
            body.pop("size")
            body["settlePosition"] = [
                {"positionId": int(position_id), "size": format(size, "f")}
            ]
        # One active intent per account/symbol, including unresolved close orders.
        intent = self.store.prepare(
            self.account,
            symbol,
            side,
            settle,
            size,
            position_id,
            decision_key,
            body,
            context or {},
        )
        if (
            intent["symbol"],
            intent["side"],
            intent["settle_type"],
            decimal(intent["size"]),
            intent["position_id"],
        ) != (
            symbol,
            side,
            settle,
            size,
            str(position_id) if position_id is not None else None,
        ):
            raise ExecutionError("Another unresolved order must be reconciled first")
        requested_batch = (context or {}).get("close_batch", {}).get("id")
        if requested_batch and intent["context"].get("close_batch", {}).get("id") != requested_batch:
            raise ExecutionError("Another close batch must finish first")
        if intent["finalized"]:
            return {
                **self.store.result(intent["client_order_id"]),
                "already_submitted": True,
            }
        if intent["status"] == "PREPARED" and self.store.claim(
            intent["client_order_id"]
        ):
            request = {**intent["request"], "clientOrderId": intent["client_order_id"]}
            # Persist SUBMITTING before I/O. Even a crash after this point must never blindly resend.
            try:
                response = self.gateway.api._request(
                    "POST",
                    "/v1/closeOrder" if settle == "CLOSE" else "/v1/order",
                    data=request,
                )
            except ExchangeMaintenance as exc:
                if exc.not_sent:
                    # This wrapper raised before entering HTTP, after our DB claim.
                    # Never apply this to a response or an exception after sending.
                    self.store.reject(intent['client_order_id'])
                    raise
                response = {"status": -1}
            except Exception:
                response = {"status": -1}
            order_id = None
            if isinstance(response, dict) and response.get("status") == 0:
                data = response.get("data")
                order_id = data.get("orderId") if isinstance(data, dict) else data
            elif isinstance(response, dict) and response.get("status") == 1:
                codes = {m.get("message_code") for m in response.get("messages", [])}
                if codes and codes.issubset(DECLINED_CODES):
                    self.store.reject(intent["client_order_id"])
                    return self.store.result(intent["client_order_id"])
            self.store.accept(
                intent["client_order_id"],
                str(order_id) if order_id is not None else None,
            )
        return self.reconcile(self.store.get(intent["client_order_id"]))

    def reconcile(self, intent):
        # Only GET snapshots are retried. POST and the persisted sending claim are
        # outside this loop, including when the original response was lost.
        started = time.monotonic()
        for attempt in range(len(self.retry_delays) + 1):
            try:
                result = self._reconcile_once(intent)
            except ExecutionPending:
                if attempt == len(self.retry_delays) or time.monotonic() - started >= 2:
                    raise
            else:
                if result["finalized"] or attempt == len(self.retry_delays) or time.monotonic() - started >= 2:
                    return result
            time.sleep(self.retry_delays[attempt])
            intent = self.store.get(intent["client_order_id"])

    def _reconcile_once(self, intent):
        if intent["finalized"]:
            return self.store.result(intent["client_order_id"])
        order = self.gateway.order(intent)
        rows = self.gateway.executions(intent, order)
        quantity = sum((decimal(r["size"]) for r in rows), Decimal(0))
        reported = decimal(order.get("executedSize", "0"))
        if not 0 <= quantity <= decimal(intent["size"]) or not 0 <= reported <= decimal(intent["size"]):
            raise ExecutionError("Inconsistent executed quantity")
        if quantity > reported:
            raise ExecutionPending("Execution list ahead of order snapshot")
        if order["status"] == "EXECUTED" and reported != decimal(intent["size"]):
            raise ExecutionPending("Executed order total not yet complete")
        # A delayed executions endpoint must not turn missing fills into a completed order.
        finalized = order["status"] in TERMINAL and quantity == reported
        self.store.record(intent, order, rows, finalized)
        return self.store.result(intent["client_order_id"])

    def recover(self):
        results = []
        for intent in self.store.pending(self.account):
            if intent["status"] == "PREPARED" and self.store.abandon_prepared(
                intent["client_order_id"]
            ):
                # Persisted before the sending claim: no request was made.
                continue
            results.append(self.reconcile(self.store.get(intent["client_order_id"])))
        return results

    def sync_fees(self, today):
        through = self.store.fee_through(self.account)
        start = through + timedelta(days=1) if through else today - timedelta(days=7)
        while start <= today:
            end = min(start + timedelta(days=7), today)
            rows, finalized = self.gateway.daily_fees(start, end)
            confirmed = (
                min(end, datetime.strptime(finalized, "%Y%m%d").date())
                if finalized
                else None
            )
            self.store.record_fees(
                self.account,
                rows,
                confirmed if confirmed and confirmed >= start else None,
            )
            if confirmed is None or confirmed < end:
                break
            start = end + timedelta(days=1)
