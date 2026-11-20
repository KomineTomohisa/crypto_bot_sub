"""PostgreSQL persistence for execution v2; no environment loading or DDL on import."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from decimal import Decimal
import json
import uuid
from sqlalchemy import text

from trading_execution import decimal, timestamp, ExecutionError


def json_value(value):
    # Serializing before I/O also rejects NaN and unpersistable intent contexts.
    return json.dumps(
        value,
        default=lambda v: str(v) if isinstance(v, Decimal) else v.isoformat(),
        allow_nan=False,
    )


class ExecutionStore:
    def __init__(self, engine):
        self.engine = engine

    def check_schema(self):
        with self.engine.connect() as conn:
            conn.execute(text("SELECT client_order_id FROM bot_order_intents LIMIT 0"))
            conn.execute(text("SELECT net_pnl_before_carry FROM trades LIMIT 0"))

    def get(self, client_id):
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    text("SELECT * FROM bot_order_intents WHERE client_order_id=:id"),
                    {"id": client_id},
                )
                .mappings()
                .one()
            )
            return dict(row)

    def pending(self, account):
        with self.engine.connect() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    text(
                        "SELECT * FROM bot_order_intents WHERE account_key=:a AND NOT finalized ORDER BY created_at"
                    ),
                    {"a": account},
                ).mappings()
            ]

    def finalized_ids(self, account):
        """Startup backfill also covers a crash after commit but before recording."""
        with self.engine.connect() as conn:
            return list(conn.execute(text("""SELECT client_order_id FROM bot_order_intents
                WHERE account_key=:a AND finalized ORDER BY created_at,client_order_id"""),
                {"a": account}).scalars())

    def close_batches(self, account):
        """Batch metadata is committed with the first intent, before any POST."""
        with self.engine.connect() as conn:
            rows = conn.execute(text("""SELECT DISTINCT ON (context #>> '{close_batch,id}')
                context->'close_batch' AS batch FROM bot_order_intents
                WHERE account_key=:a AND settle_type='CLOSE' AND context ? 'close_batch'
                  AND NOT (context @> '{"close_batch_done": true}'::jsonb)
                ORDER BY context #>> '{close_batch,id}',created_at"""), {"a": account})
            return [r.batch for r in rows]

    def close_batch_remaining(self, account, batch):
        with self.engine.connect() as conn:
            closed = dict(conn.execute(text("""SELECT f.position_id,sum(f.size)
                FROM bot_execution_fills f JOIN bot_order_intents i USING(client_order_id)
                WHERE f.account_key=:a AND i.account_key=:a AND f.settle_type='CLOSE'
                  AND i.context #>> '{close_batch,id}'=:id GROUP BY f.position_id"""),
                {"a": account, "id": batch["id"]}).all())
        remaining = {str(p["positionId"]): decimal(p["size"], positive=True) - closed.get(str(p["positionId"]), Decimal(0))
                     for p in batch["positions"]}
        if any(size < 0 for size in remaining.values()):
            raise ExecutionError("Close batch exceeded target quantity")
        return remaining

    def finish_close_batch(self, account, batch):
        if any(self.close_batch_remaining(account, batch).values()):
            raise ExecutionError("Close batch still has unclosed targets")
        with self.engine.begin() as conn:
            conn.execute(text("""UPDATE bot_order_intents
                SET context=context || '{"close_batch_done": true}'::jsonb
                WHERE account_key=:a AND context #>> '{close_batch,id}'=:id"""),
                {"a": account, "id": batch["id"]})

    def prepare(
        self, account, symbol, side, settle, size, position_id, key, body, context
    ):
        params = {
            "a": account,
            "symbol": symbol,
            "side": side,
            "settle": settle,
            "size": size,
            "pid": str(position_id) if position_id is not None else None,
            "key": key,
            "body": json_value(body),
            "context": json_value(context),
            "id": uuid.uuid4().hex,
        }
        with self.engine.begin() as conn:
            # Serialize decisions for a symbol across processes, not just Python threads.
            conn.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:lock,0))"),
                {"lock": account},
            )
            row = (
                conn.execute(
                    text(
                        "SELECT * FROM bot_order_intents WHERE account_key=:a AND ((NOT finalized) OR (decision_key=:key AND :key IS NOT NULL)) ORDER BY finalized LIMIT 1"
                    ),
                    params,
                )
                .mappings()
                .first()
            )
            if row:
                return dict(row)
            batch = context.get('close_batch') if settle == 'CLOSE' else None
            if batch:
                active = conn.execute(text("""SELECT context #>> '{close_batch,id}'
                    FROM bot_order_intents WHERE account_key=:a AND symbol=:symbol
                      AND context ? 'close_batch'
                      AND NOT (context @> '{"close_batch_done": true}'::jsonb) LIMIT 1"""), params).scalar()
                if active is not None and active != batch['id']:
                    raise ExecutionError("Another close batch must finish first")
                # Recheck under the account lock. A second worker may have
                # completed this target after the caller read its snapshot.
                target = next((p for p in batch['positions'] if str(p['positionId']) == params['pid']), None)
                if target is None:
                    raise ExecutionError("Position is not a close batch target")
                closed = conn.execute(text("""SELECT COALESCE(sum(f.size),0)
                    FROM bot_execution_fills f JOIN bot_order_intents i USING(client_order_id)
                    WHERE f.account_key=:a AND i.account_key=:a AND f.position_id=:pid
                      AND f.settle_type='CLOSE' AND i.context #>> '{close_batch,id}'=:batch"""),
                    {**params, 'batch': batch['id']}).scalar_one()
                if decimal(target['size'], positive=True) - closed != decimal(size):
                    raise ExecutionError("Close batch target changed; refresh snapshot")
                projected = conn.execute(text("""SELECT p.size FROM positions p
                    WHERE p.position_id=:pid AND EXISTS (SELECT 1 FROM bot_execution_fills f
                        WHERE f.position_id=p.position_id AND f.account_key=:a)"""), params).scalar()
                if projected is not None and projected != decimal(size):
                    raise ExecutionError("Close position quantity changed; refresh snapshot")
            if settle == 'OPEN' and conn.execute(text("""SELECT 1 FROM bot_order_intents
                WHERE account_key=:a AND context ? 'close_batch'
                  AND NOT (context @> '{"close_batch_done": true}'::jsonb) LIMIT 1"""), params).first():
                raise ExecutionError("Unfinished close batch blocks new exposure")
            conn.execute(
                text("""INSERT INTO bot_order_intents(client_order_id,account_key,symbol,side,settle_type,size,position_id,decision_key,request,context)
                VALUES(:id,:a,:symbol,:side,:settle,:size,:pid,:key,CAST(:body AS jsonb),CAST(:context AS jsonb))"""),
                params,
            )
        return self.get(params["id"])

    def claim(self, client_id):
        with self.engine.begin() as conn:
            return (
                conn.execute(
                    text(
                        "UPDATE bot_order_intents SET status='SUBMITTING',updated_at=now() WHERE client_order_id=:id AND status='PREPARED' RETURNING client_order_id"
                    ),
                    {"id": client_id},
                ).first()
                is not None
            )

    def abandon_prepared(self, client_id):
        with self.engine.begin() as conn:
            return (
                conn.execute(
                    text(
                        "UPDATE bot_order_intents SET status='REJECTED',finalized=true,updated_at=now() WHERE client_order_id=:id AND status='PREPARED' RETURNING client_order_id"
                    ),
                    {"id": client_id},
                ).first()
                is not None
            )

    def accept(self, client_id, order_id):
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE bot_order_intents SET status=:s,exchange_order_id=:oid,updated_at=now() WHERE client_order_id=:id AND NOT finalized"
                ),
                {
                    "s": "ACCEPTED" if order_id else "UNKNOWN",
                    "oid": order_id,
                    "id": client_id,
                },
            )

    def reject(self, client_id):
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE bot_order_intents SET status='REJECTED',finalized=true,updated_at=now() WHERE client_order_id=:id AND status='SUBMITTING'"
                ),
                {"id": client_id},
            )

    def record(self, intent, order, rows, finalized):
        cid, account = intent["client_order_id"], intent["account_key"]
        context = intent["context"]
        with self.engine.begin() as conn:
            locked = conn.execute(
                text(
                    "SELECT finalized FROM bot_order_intents WHERE client_order_id=:id FOR UPDATE"
                ),
                {"id": cid},
            ).one()
            if locked.finalized:
                return
            known = conn.execute(
                text(
                    "SELECT COALESCE(sum(size),0) FROM bot_execution_fills WHERE client_order_id=:id"
                ),
                {"id": cid},
            ).scalar_one()
            if known > decimal(order.get("executedSize", "0")):
                raise ExecutionError("Exchange executed quantity regressed")
            # Legacy positions are adopted from the exchange snapshot taken BEFORE closing.
            baseline = context.get("baseline")
            if baseline:
                conn.execute(
                    text("""INSERT INTO bot_position_baselines(account_key,position_id,symbol,side,size,price,opened_at)
                    VALUES(:a,:pid,:symbol,:side,:size,:price,:opened) ON CONFLICT DO NOTHING"""),
                    {
                        "a": account,
                        "pid": str(baseline["positionId"]),
                        "symbol": intent["symbol"],
                        "side": baseline["side"],
                        "size": decimal(baseline["size"], positive=True),
                        "price": decimal(baseline["price"], positive=True),
                        "opened": timestamp(baseline["timestamp"]),
                    },
                )
            conn.execute(
                text("""INSERT INTO orders(order_id,symbol,side,type,size,status,requested_at,placed_at,raw)
                VALUES(:oid,:symbol,:side,'MARKET',:size,:status,:created,:placed,CAST(:raw AS jsonb))
                ON CONFLICT(order_id) DO UPDATE SET status=EXCLUDED.status,raw=EXCLUDED.raw"""),
                {
                    "oid": str(order["orderId"]),
                    "symbol": context.get("bot_symbol", intent["symbol"]),
                    "side": intent["side"],
                    "size": intent["size"],
                    "status": order["status"],
                    "created": intent["created_at"],
                    "placed": timestamp(order["timestamp"]),
                    "raw": json_value(
                        {
                            "accounting_version": "execution_v2",
                            "clientOrderId": cid,
                            "exchange": order,
                        }
                    ),
                },
            )
            for row in rows:
                params = {
                    "a": account,
                    "eid": str(row["executionId"]),
                    "cid": cid,
                    "oid": str(row["orderId"]),
                    "pid": str(row["positionId"]),
                    "symbol": row["symbol"],
                    "side": row["side"],
                    "settle": row["settleType"],
                    "price": decimal(row["price"]),
                    "size": decimal(row["size"]),
                    "fee": decimal(row["fee"]) if row.get("fee") is not None else None,
                    "pnl": decimal(row["lossGain"])
                    if row.get("lossGain") is not None
                    else None,
                    "ts": timestamp(row["timestamp"]),
                    "raw": json_value(row),
                }
                prior = conn.execute(
                    text(
                        "SELECT raw FROM bot_execution_fills WHERE account_key=:a AND execution_id=:eid"
                    ),
                    params,
                ).scalar()
                if prior is not None and prior != row:
                    raise ExecutionError("Previously recorded execution changed")
                conn.execute(
                    text("""INSERT INTO bot_execution_fills(account_key,execution_id,client_order_id,order_id,position_id,symbol,side,settle_type,price,size,fee,loss_gain,executed_at,raw)
                    VALUES(:a,:eid,:cid,:oid,:pid,:symbol,:side,:settle,:price,:size,:fee,:pnl,:ts,CAST(:raw AS jsonb)) ON CONFLICT DO NOTHING"""),
                    params,
                )
                params["fid"] = "GMO" + account + params["eid"]
                conn.execute(
                    text("""INSERT INTO fills(fill_id,order_id,price,size,fee,executed_at,raw)
                    VALUES(:fid,:oid,:price,:size,:fee,:ts,CAST(:raw AS jsonb)) ON CONFLICT(fill_id) DO NOTHING"""),
                    params,
                )
            for pid in {str(row["positionId"]) for row in rows}:
                self._project_position(conn, intent, pid)
            conn.execute(
                text(
                    "UPDATE bot_order_intents SET status=:status,exchange_order_id=:oid,finalized=:done,updated_at=now() WHERE client_order_id=:id"
                ),
                {
                    "id": cid,
                    "status": order["status"],
                    "oid": str(order["orderId"]),
                    "done": finalized,
                },
            )

    def _project_position(self, conn, intent, pid):
        account = intent["account_key"]
        ctx = intent["context"]
        params = {"a": account, "pid": pid}
        fills = list(
            conn.execute(
                text(
                    "SELECT * FROM bot_execution_fills WHERE account_key=:a AND position_id=:pid ORDER BY executed_at,execution_id"
                ),
                params,
            ).mappings()
        )
        opened = [f for f in fills if f["settle_type"] == "OPEN"]
        closed = [f for f in fills if f["settle_type"] == "CLOSE"]
        if opened:
            total = sum(f["size"] for f in opened)
            price = sum(f["price"] * f["size"] for f in opened) / total
            opened_at = min(f["executed_at"] for f in opened)
            side = opened[0]["side"]
            entry_fee = (
                sum(f["fee"] for f in opened)
                if all(f["fee"] is not None for f in opened)
                else None
            )
        else:
            baseline = (
                conn.execute(
                    text(
                        "SELECT * FROM bot_position_baselines WHERE account_key=:a AND position_id=:pid"
                    ),
                    params,
                )
                .mappings()
                .one()
            )
            total, price, opened_at, side = (
                baseline["size"],
                baseline["price"],
                baseline["opened_at"],
                baseline["side"],
            )
            entry_fee = None
        remainder = total - sum(f["size"] for f in closed)
        if remainder < 0:
            raise ExecutionError("Closed quantity exceeds known position quantity")
        open_intent = (
            conn.execute(
                text("SELECT context FROM bot_order_intents WHERE client_order_id=:id"),
                {"id": opened[0]["client_order_id"]},
            ).scalar_one()
            if opened
            else {}
        )
        conn.execute(
            text("""INSERT INTO positions(position_id,symbol,side,size,avg_entry_price,opened_at,updated_at,raw,source,user_id,open_signal_id)
            VALUES(:pid,:symbol,:side,:size,:price,:opened,now(),CAST(:raw AS jsonb),'real',:user,CAST(:signal AS uuid))
            ON CONFLICT(position_id) DO UPDATE SET size=EXCLUDED.size,avg_entry_price=EXCLUDED.avg_entry_price,updated_at=now(),
            raw=EXCLUDED.raw,opened_at=EXCLUDED.opened_at,open_signal_id=COALESCE(positions.open_signal_id,EXCLUDED.open_signal_id)"""),
            {
                "pid": pid,
                "symbol": ctx.get("bot_symbol", intent["symbol"]),
                "side": "long" if side == "BUY" else "short",
                "size": remainder,
                "price": price,
                "opened": opened_at,
                "user": ctx.get("user_id"),
                "signal": open_intent.get("signal_id"),
                "raw": json_value(
                    {
                        "accounting_version": "execution_v2",
                        "scores": open_intent.get("scores", {}),
                        "entry_fee": entry_fee,
                    }
                ),
            },
        )
        groups = defaultdict(list)
        for fill in closed:
            groups[fill["client_order_id"]].append(fill)
        for cid, group in groups.items():
            close_ctx = conn.execute(
                text("SELECT context FROM bot_order_intents WHERE client_order_id=:id"),
                {"id": cid},
            ).scalar_one()
            size = sum(f["size"] for f in group)
            exit_price = sum(f["size"] * f["price"] for f in group) / size
            pnl = sum(f["loss_gain"] for f in group)
            allocated_entry = (
                entry_fee * size / total if entry_fee is not None else None
            )
            exit_fee = (
                sum(f["fee"] for f in group)
                if all(f["fee"] is not None for f in group)
                else None
            )
            net = (
                pnl - allocated_entry - exit_fee
                if allocated_entry is not None and exit_fee is not None
                else None
            )
            end = max(f["executed_at"] for f in group)
            raw = {
                "accounting_version": "execution_v2",
                "entry_time": opened_at,
                "exit_reason": close_ctx.get("exit_reason"),
                "scores": open_intent.get("scores", close_ctx.get("scores", {})),
                "execution_ids": [f["execution_id"] for f in group],
                "gross_pnl": pnl,
                "entry_fee": allocated_entry,
                "exit_fee": exit_fee,
                "net_pnl_before_carry": net,
                "carry_allocation": "not_allocated",
                "entry_basis": "exchange_open_executions"
                if opened
                else "exchange_position_baseline",
            }
            conn.execute(
                text("""INSERT INTO trades(trade_id,symbol,side,entry_position_id,exit_order_id,entry_price,exit_price,size,pnl,pnl_pct,holding_hours,closed_at,raw,source,user_id,
                gross_pnl,entry_fee,exit_fee,net_pnl_before_carry,accounting_version,cost_status)
                VALUES(:tid,:symbol,:side,:pid,:oid,:entry,:exit,:size,:pnl,:pct,:hours,:end,CAST(:raw AS jsonb),'real',:user,:pnl,:ef,:xf,:net,'execution_v2',:cost)
                ON CONFLICT(trade_id) DO UPDATE SET exit_price=EXCLUDED.exit_price,size=EXCLUDED.size,pnl=EXCLUDED.pnl,pnl_pct=EXCLUDED.pnl_pct,
                holding_hours=EXCLUDED.holding_hours,closed_at=EXCLUDED.closed_at,raw=EXCLUDED.raw,gross_pnl=EXCLUDED.gross_pnl,entry_fee=EXCLUDED.entry_fee,
                exit_fee=EXCLUDED.exit_fee,net_pnl_before_carry=EXCLUDED.net_pnl_before_carry,cost_status=EXCLUDED.cost_status"""),
                {
                    "tid": "GMO" + cid + pid,
                    "symbol": ctx.get("bot_symbol", intent["symbol"]),
                    "side": "long" if side == "BUY" else "short",
                    "pid": pid,
                    "oid": group[0]["order_id"],
                    "entry": price,
                    "exit": exit_price,
                    "size": size,
                    "pnl": pnl,
                    "pct": pnl / (price * size) * 100,
                    "hours": Decimal(str(max(0, (end - opened_at).total_seconds())))
                    / 3600,
                    "end": end,
                    "raw": json_value(raw),
                    "user": ctx.get("user_id"),
                    "ef": allocated_entry,
                    "xf": exit_fee,
                    "net": net,
                    "cost": "carry_pending"
                    if net is not None
                    else "entry_or_execution_fee_unknown",
                },
            )

    def result(self, client_id):
        intent = self.get(client_id)
        with self.engine.connect() as conn:
            rows = list(
                conn.execute(
                    text(
                        "SELECT * FROM bot_execution_fills WHERE client_order_id=:id ORDER BY executed_at"
                    ),
                    {"id": client_id},
                ).mappings()
            )
            trades = [
                dict(r)
                for r in conn.execute(
                    text("SELECT * FROM trades WHERE trade_id LIKE :prefix"),
                    {"prefix": "GMO" + client_id + "%"},
                ).mappings()
            ]
        size = sum((r["size"] for r in rows), Decimal(0))
        return {
            "success": size > 0,
            "order_id": intent["exchange_order_id"],
            "client_order_id": client_id,
            "executed_size": float(size),
            "execution_price": float(sum(r["size"] * r["price"] for r in rows) / size)
            if size
            else None,
            "executed_at": min((r["executed_at"] for r in rows), default=None),
            "position_ids": sorted({r["position_id"] for r in rows}),
            "finalized": intent["finalized"],
            "status": intent["status"],
            "intent": intent,
            "trades": trades,
        }

    def position_context(self, account, pid):
        with self.engine.connect() as conn:
            value = conn.execute(
                text(
                    "SELECT i.context FROM bot_execution_fills f JOIN bot_order_intents i USING(client_order_id) WHERE f.account_key=:a AND f.position_id=:pid AND f.settle_type='OPEN' ORDER BY f.executed_at LIMIT 1"
                ),
                {"a": account, "pid": str(pid)},
            ).scalar()
        return value or {}

    def open_position_ids(self, account, symbol):
        with self.engine.connect() as conn:
            return [
                r[0]
                for r in conn.execute(
                    text("""SELECT DISTINCT p.position_id FROM positions p
                JOIN bot_execution_fills f ON f.position_id=p.position_id
                WHERE f.account_key=:a AND f.symbol=:symbol AND p.size>0"""),
                    {"a": account, "symbol": symbol},
                )
            ]

    def fee_through(self, account):
        with self.engine.connect() as conn:
            return conn.execute(
                text("SELECT through_date FROM bot_fee_sync WHERE account_key=:a"),
                {"a": account},
            ).scalar()

    def record_fees(self, account, rows, confirmed):
        with self.engine.begin() as conn:
            for row in rows:
                conn.execute(
                    text("""INSERT INTO bot_account_fees(account_key,symbol,business_date,fee_type,delivery_amount,delivered_at,raw)
                    VALUES(:a,:symbol,:day,:type,:amount,:ts,CAST(:raw AS jsonb))
                    ON CONFLICT(account_key,symbol,business_date,fee_type) DO UPDATE SET delivery_amount=EXCLUDED.delivery_amount,delivered_at=EXCLUDED.delivered_at,raw=EXCLUDED.raw"""),
                    {
                        "a": account,
                        "symbol": row["symbol"],
                        "day": datetime.strptime(row["deliveryDate"], "%Y%m%d").date(),
                        "type": row["feeType"],
                        "amount": decimal(row["deliveryAmount"]),
                        "ts": timestamp(row["deliveryDatetime"]),
                        "raw": json_value(row),
                    },
                )
            if confirmed:
                conn.execute(
                    text(
                        "INSERT INTO bot_fee_sync(account_key,through_date) VALUES(:a,:day) ON CONFLICT(account_key) DO UPDATE SET through_date=GREATEST(bot_fee_sync.through_date,EXCLUDED.through_date)"
                    ),
                    {"a": account, "day": confirmed},
                )

    def pnl_summary(self, account):
        with self.engine.connect() as conn:
            execution = conn.execute(
                text(
                    "SELECT COALESCE(sum(loss_gain) FILTER(WHERE settle_type='CLOSE'),0),sum(fee),count(*) FILTER(WHERE fee IS NULL) FROM bot_execution_fills WHERE account_key=:a"
                ),
                {"a": account},
            ).one()
            fees = conn.execute(
                text(
                    "SELECT COALESCE(sum(delivery_amount),0) FROM bot_account_fees WHERE account_key=:a"
                ),
                {"a": account},
            ).scalar()
        return {
            "gross_pnl": execution[0],
            "execution_fees": execution[1],
            "daily_fee_cashflow": fees,
            "fee_unknown_count": execution[2],
            "fee_through": self.fee_through(account),
            "net_recorded_cashflow": execution[0] - execution[1] + fees
            if execution[1] is not None and not execution[2]
            else None,
        }
