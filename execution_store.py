"""PostgreSQL persistence for execution v2; no environment loading or DDL on import."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
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
            conn.execute(text("SELECT batch_id,payload,completed FROM bot_close_batches LIMIT 0"))
            conn.execute(text("""SELECT alert_last_attempt_at,alert_last_attempt_kind,
                alert_warning_sent_at,alert_recovery_sent_at,alert_claim_token,alert_claim_until
                FROM bot_close_batches LIMIT 0"""))
            conn.execute(text('SELECT job_id,claim_token,delivered_at FROM bot_notification_outbox LIMIT 0'))

    @staticmethod
    def _insert_notifications(conn, account, event_key, preferences, kind, *, client_id=None,
                              recovered=False, subject=None, body=None):
        if not isinstance(preferences,dict) or preferences.get('enabled') is not True or preferences.get(kind,kind=='info') is not True:
            return 0
        if not isinstance(preferences.get('channels'),list):return 0
        count = 0
        for channel in ('email', 'line'):
            if channel not in preferences.get('channels', []):
                continue
            job_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f'crypto-bot:{account}:{event_key}:{channel}'))
            result = conn.execute(text("""INSERT INTO bot_notification_outbox
                (job_id,account_key,event_key,client_order_id,channel,kind,recovered,subject,body)
                VALUES(:id,:a,:key,:cid,:channel,:kind,:recovered,:subject,:body)
                ON CONFLICT(account_key,event_key,channel) DO NOTHING"""),
                {'id':job_id,'a':account,'key':event_key,'cid':client_id,'channel':channel,
                 'kind':kind,'recovered':recovered,'subject':subject,'body':body})
            count += result.rowcount
        return count

    def set_notification_preferences(self, account, client_id, preferences):
        """Adopt only unfinished pre-upgrade intents, never replay historical mail."""
        with self.engine.begin() as conn:
            conn.execute(text("""UPDATE bot_order_intents
                SET context=context || jsonb_build_object('notification_delivery',CAST(:prefs AS jsonb))
                WHERE account_key=:a AND client_order_id=:id AND NOT finalized
                  AND NOT(context ? 'notification_delivery')"""),
                {'a':account,'id':client_id,'prefs':json_value(preferences)})

    def enqueue_message(self, account, event_key, preferences, subject, body, kind):
        with self.engine.begin() as conn:
            return self._insert_notifications(conn,account,event_key,preferences,kind,subject=subject,body=body)

    def claim_notification(self, account, *, now=None, lease_seconds=600):
        with self.engine.begin() as conn:
            at = timestamp(now) if now is not None else conn.execute(text('SELECT clock_timestamp()')).scalar_one()
            row = conn.execute(text("""SELECT job_id FROM bot_notification_outbox
                WHERE account_key=:a AND delivered_at IS NULL AND terminal_reason IS NULL
                  AND next_attempt_at<=:now AND (claim_until IS NULL OR claim_until<=:now)
                ORDER BY created_at,job_id LIMIT 1 FOR UPDATE SKIP LOCKED"""),{'a':account,'now':at}).first()
            if row is None:
                return None
            return dict(conn.execute(text("""UPDATE bot_notification_outbox
                SET claim_token=:token,claim_until=:until,attempts=attempts+1
                WHERE account_key=:a AND job_id=:id RETURNING *"""),
                {'a':account,'id':row.job_id,'token':uuid.uuid4().hex,
                 'until':at+timedelta(seconds=lease_seconds)}).mappings().one())

    def freeze_notification(self, account, job, subject, body, route_hash, *, now=None):
        with self.engine.begin() as conn:
            at = timestamp(now) if now is not None else conn.execute(text('SELECT clock_timestamp()')).scalar_one()
            row = conn.execute(text("""UPDATE bot_notification_outbox
                SET subject=COALESCE(subject,:subject),body=COALESCE(body,:body),
                    route_hash=COALESCE(route_hash,:route),first_attempt_at=COALESCE(first_attempt_at,:now)
                WHERE account_key=:a AND job_id=:id AND claim_token=:token AND claim_until>:now
                  AND delivered_at IS NULL AND terminal_reason IS NULL RETURNING *"""),
                {'a':account,'id':job['job_id'],'token':job['claim_token'],'now':at,
                 'subject':subject,'body':body,'route':route_hash}).mappings().first()
            return dict(row) if row is not None else None

    def finish_notification(self, account, job, delivered, *, error_type=None, terminal_reason=None, now=None):
        with self.engine.begin() as conn:
            at = timestamp(now) if now is not None else conn.execute(text('SELECT clock_timestamp()')).scalar_one()
            delay = min(1800, 30 * 2**min(max(job['attempts']-1,0),6))
            row = conn.execute(text("""UPDATE bot_notification_outbox
                SET delivered_at=CASE WHEN :sent THEN :now ELSE delivered_at END,
                    next_attempt_at=:next,last_error_type=:error,terminal_reason=:terminal,
                    claim_token=NULL,claim_until=NULL
                WHERE account_key=:a AND job_id=:id AND claim_token=:token
                  AND delivered_at IS NULL AND terminal_reason IS NULL RETURNING job_id"""),
                {'a':account,'id':job['job_id'],'token':job['claim_token'],'sent':bool(delivered),
                 'now':at,'next':at+timedelta(seconds=delay),'error':error_type,'terminal':terminal_reason}).first()
            return row is not None

    def _queue_execution_notifications(self, conn, intent, *, recovered=False):
        preferences = intent['context'].get('notification_delivery') or {}
        kind = 'entry' if intent['settle_type']=='OPEN' else 'exit'
        return self._insert_notifications(conn,intent['account_key'],'execution:'+intent['client_order_id'],
                                          preferences,kind,client_id=intent['client_order_id'],recovered=recovered)

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
        """Read queued decisions, including old-process writes during deployment."""
        with self.engine.connect() as conn:
            registered = list(conn.execute(text("""SELECT payload FROM bot_close_batches
                WHERE account_key=:a AND NOT completed ORDER BY created_at,batch_id"""),
                {"a": account}).scalars())
            return registered + self._legacy_close_batches(conn, account)

    def claim_close_alerts(self, account, policy, *, now=None, limit=8):
        """Lease due notices across processes without holding locks during I/O."""
        with self.engine.begin() as conn:
            at = timestamp(now) if now is not None else conn.execute(text('SELECT clock_timestamp()')).scalar_one()
            params = {'a': account, 'now': at, 'stale': at-timedelta(seconds=policy.stall_seconds),
                      'reminder': at-timedelta(seconds=policy.reminder_seconds),
                      'retry': at-timedelta(seconds=policy.retry_seconds), 'limit': limit}
            rows = list(conn.execute(text("""SELECT b.batch_id,b.symbol,b.created_at,b.completed,
                    b.payload->'positions' AS positions,
                    EXISTS(SELECT 1 FROM bot_order_intents i WHERE i.account_key=b.account_key AND NOT i.finalized) AS has_pending
                FROM bot_close_batches b WHERE b.account_key=:a
                  AND (b.alert_claim_until IS NULL OR b.alert_claim_until<=:now)
                  AND (
                    (NOT b.completed AND b.created_at<=:stale
                      AND (b.alert_warning_sent_at IS NULL OR b.alert_warning_sent_at<=:reminder)
                      AND (b.alert_last_attempt_at IS NULL OR b.alert_last_attempt_at<=:retry))
                    OR
                    (b.completed AND b.alert_warning_sent_at IS NOT NULL AND b.alert_recovery_sent_at IS NULL
                      AND (b.alert_last_attempt_kind IS DISTINCT FROM 'recovery'
                           OR b.alert_last_attempt_at IS NULL OR b.alert_last_attempt_at<=:retry))
                  ) ORDER BY b.created_at,b.batch_id LIMIT :limit FOR UPDATE OF b SKIP LOCKED"""), params).mappings())
            jobs = []
            for row in rows:
                kind = 'recovery' if row['completed'] else 'warning'
                token = uuid.uuid4().hex
                conn.execute(text("""UPDATE bot_close_batches SET alert_last_attempt_at=:now,
                    alert_last_attempt_kind=:kind,alert_claim_token=:token,alert_claim_until=:until
                    WHERE account_key=:a AND batch_id=:id"""),
                    {'a':account,'id':row['batch_id'],'now':at,'kind':kind,'token':token,
                     'until':at+timedelta(seconds=policy.lease_seconds)})
                phase = ('order_reconciliation' if row['has_pending'] else
                         'position_confirmation' if row['positions'] is None else 'close_pending')
                jobs.append({'batch_id':row['batch_id'],'symbol':row['symbol'],'created_at':row['created_at'],
                             'observed_at':at,'kind':kind,'phase':phase,'claim_token':token})
            return jobs

    def close_alert_is_current(self, account, job, *, now=None):
        with self.engine.connect() as conn:
            row = conn.execute(text("""SELECT completed,alert_warning_sent_at,alert_recovery_sent_at
                FROM bot_close_batches WHERE account_key=:a AND batch_id=:id
                  AND alert_claim_token=:token AND alert_last_attempt_kind=:kind
                  AND alert_claim_until>COALESCE(CAST(:now AS timestamptz),clock_timestamp())"""),
                {'a':account,'id':job['batch_id'],'token':job['claim_token'],'kind':job['kind'],
                 'now':timestamp(now) if now is not None else None}).mappings().first()
        if row is None:return False
        if job['kind']=='warning':return not row['completed']
        return row['completed'] and row['alert_warning_sent_at'] is not None and row['alert_recovery_sent_at'] is None

    def finish_close_alert(self, account, job, delivered, *, now=None):
        with self.engine.begin() as conn:
            at = timestamp(now) if now is not None else conn.execute(text('SELECT clock_timestamp()')).scalar_one()
            row = conn.execute(text("""UPDATE bot_close_batches
                SET alert_last_attempt_at=:now,
                    alert_warning_sent_at=CASE WHEN :delivered AND :kind='warning' THEN :now ELSE alert_warning_sent_at END,
                    alert_recovery_sent_at=CASE WHEN :delivered AND :kind='recovery' THEN :now ELSE alert_recovery_sent_at END,
                    alert_claim_token=NULL,alert_claim_until=NULL
                WHERE account_key=:a AND batch_id=:id AND alert_claim_token=:token
                  AND alert_last_attempt_kind=:kind RETURNING batch_id"""),
                {'a':account,'id':job['batch_id'],'token':job['claim_token'],'kind':job['kind'],
                 'now':at,'delivered':bool(delivered)}).first()
            return row is not None

    @staticmethod
    def _legacy_close_batches(conn, account):
        return list(conn.execute(text("""SELECT DISTINCT ON (i.context #>> '{close_batch,id}')
            i.context->'close_batch' FROM bot_order_intents i
            WHERE i.account_key=:a AND i.settle_type='CLOSE' AND i.context ? 'close_batch'
              AND NOT (i.context @> '{"close_batch_done": true}'::jsonb)
              AND NOT EXISTS (SELECT 1 FROM bot_close_batches b
                  WHERE b.batch_id=i.context #>> '{close_batch,id}' AND b.account_key=i.account_key)
            ORDER BY i.context #>> '{close_batch,id}',i.created_at"""),
            {"a": account}).scalars())

    @staticmethod
    def _validate_close_positions(symbol, batch, positions):
        expected_side = {"long": "BUY", "short": "SELL"}.get(batch.get("side"))
        if not expected_side or not isinstance(positions, list) or not positions:
            raise ExecutionError("Invalid close batch targets")
        ids = set()
        for pos in positions:
            pid = str(pos.get("positionId"))
            if pos.get("positionId") is None or pid in ids or pos.get("symbol") != symbol or pos.get("side") != expected_side:
                raise ExecutionError("Invalid close batch position identity")
            ids.add(pid)
            decimal(pos.get("size"), positive=True)
            decimal(pos.get("price"), positive=True)
            timestamp(pos.get("timestamp"))

    def save_close_batch(self, account, symbol, batch):
        """Persist the request even before a position lookup can succeed."""
        if batch.get('positions') is None:
            ids = batch.get('target_ids')
            if (batch.get('side') not in ('long', 'short') or not isinstance(ids, list)
                    or any(not isinstance(pid, str) or not pid for pid in ids)
                    or len(set(ids)) != len(ids)):
                raise ExecutionError("Invalid unresolved close request")
            sizes = batch.get('target_sizes')
            if sizes is not None:
                if not isinstance(sizes, dict) or set(sizes) != set(ids):
                    raise ExecutionError("Invalid close request quantities")
                for size in sizes.values():
                    decimal(size, positive=True)
            if batch.get('target_total') is not None:
                decimal(batch['target_total'], positive=True)
        else:
            self._validate_close_positions(symbol, batch, batch['positions'])
        params = {"a": account, "symbol": symbol, "id": batch["id"], "payload": json_value(batch)}
        with self.engine.begin() as conn:
            conn.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:a,0))"), params)
            known = conn.execute(text("""SELECT account_key,symbol,payload,completed
                FROM bot_close_batches WHERE batch_id=:id"""), params).mappings().first()
            if known:
                if known['account_key'] != account or known['symbol'] != symbol or known['completed']:
                    raise ExecutionError("Close batch changed or completed")
                return known['payload']
            active = conn.execute(text("""SELECT payload FROM bot_close_batches
                WHERE account_key=:a AND symbol=:symbol AND NOT completed"""), params).scalar()
            if active is not None:
                return active
            # The previous process can still write a legacy intent after migration.
            legacy = next((b for b in self._legacy_close_batches(conn, account)
                           if b['symbol'] == batch['symbol']), None)
            if legacy is not None:
                batch = legacy
                params.update(id=batch['id'], payload=json_value(batch))
            return conn.execute(text("""INSERT INTO bot_close_batches(batch_id,account_key,symbol,payload)
                VALUES(:id,:a,:symbol,CAST(:payload AS jsonb)) RETURNING payload"""), params).scalar_one()

    def resolve_close_batch(self, account, symbol, batch_id, positions):
        """Bind one confirmed snapshot to the identities frozen at decision time."""
        params = {'a': account, 'symbol': symbol, 'id': batch_id}
        with self.engine.begin() as conn:
            conn.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:a,0))"), params)
            saved = conn.execute(text("""SELECT payload FROM bot_close_batches
                WHERE account_key=:a AND symbol=:symbol AND batch_id=:id AND NOT completed"""), params).scalar()
            if saved is None:
                raise ExecutionError("Close batch changed or completed")
            if saved.get('positions') is not None:
                return saved
            self._validate_close_positions(symbol, saved, positions)
            ids = {str(pos['positionId']) for pos in positions}
            if not saved['target_ids'] or ids != set(saved['target_ids']):
                raise ExecutionError("Close request target identities not confirmed")
            sizes = saved.get('target_sizes')
            if sizes is not None:
                if any(decimal(pos['size']) != decimal(sizes[str(pos['positionId'])]) for pos in positions):
                    raise ExecutionError("Close request target quantities changed")
            elif saved.get('target_total') is None:
                raise ExecutionError("Close request has no verified quantity")
            if saved.get('target_total') is not None:
                total = sum((decimal(pos['size']) for pos in positions), Decimal(0))
                if total != decimal(saved['target_total']):
                    raise ExecutionError("Close request total quantity changed")
            resolved = {**saved, 'positions': positions}
            return conn.execute(text("""UPDATE bot_close_batches SET payload=CAST(:payload AS jsonb)
                WHERE account_key=:a AND batch_id=:id RETURNING payload"""),
                {**params, 'payload': json_value(resolved)}).scalar_one()

    def close_batch_remaining(self, account, batch):
        if batch.get('positions') is None:
            raise ExecutionError("Close request targets are not yet confirmed")
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
            conn.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:a,0))"), {"a": account})
            if conn.execute(text("""SELECT 1 FROM bot_order_intents
                WHERE account_key=:a AND NOT finalized AND context #>> '{close_batch,id}'=:id LIMIT 1"""),
                {"a": account, "id": batch["id"]}).first():
                raise ExecutionError("Close batch still has a pending order")
            updated = conn.execute(text("""UPDATE bot_close_batches
                SET completed=true,completed_at=COALESCE(completed_at,now())
                WHERE account_key=:a AND batch_id=:id RETURNING batch_id"""),
                {"a": account, "id": batch["id"]}).first()
            if updated is None:
                raise ExecutionError("Close batch registration missing")
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
                saved = conn.execute(text("""SELECT batch_id,payload FROM bot_close_batches
                    WHERE account_key=:a AND symbol=:symbol AND NOT completed"""), params).mappings().first()
                if saved is None:
                    raise ExecutionError("Close batch changed or completed")
                if saved['batch_id'] != batch['id']:
                    raise ExecutionError("Another close batch must finish first")
                if saved['payload'] != json.loads(params['context'])['close_batch']:
                    raise ExecutionError("Close batch targets changed after registration")
                if batch.get('positions') is None:
                    raise ExecutionError("Close request targets are not yet confirmed")
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
            if settle == 'OPEN':
                registered = conn.execute(text("""SELECT 1 FROM bot_close_batches
                    WHERE account_key=:a AND NOT completed LIMIT 1"""), params).first()
                if registered or self._legacy_close_batches(conn, account):
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
            row = conn.execute(
                text(
                    "UPDATE bot_order_intents SET status='REJECTED',finalized=true,updated_at=now() WHERE client_order_id=:id AND status='SUBMITTING' RETURNING *"
                ),
                {"id": client_id},
            ).mappings().first()
            if row is not None:
                self._queue_execution_notifications(conn,dict(row))

    def record(self, intent, order, rows, finalized, *, recovered=False):
        cid, account = intent["client_order_id"], intent["account_key"]
        with self.engine.begin() as conn:
            locked = conn.execute(
                text(
                    "SELECT * FROM bot_order_intents WHERE client_order_id=:id FOR UPDATE"
                ),
                {"id": cid},
            ).mappings().one()
            if locked['finalized']:
                return
            # Recovery may have adopted notification settings since the caller's
            # snapshot. Register from the locked, persisted intent, not that copy.
            intent = dict(locked)
            context = intent['context']
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
            if finalized:
                self._queue_execution_notifications(conn,intent,recovered=recovered)

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
            "execution_fee": sum((r['fee'] for r in rows), Decimal(0))
            if rows and all(r['fee'] is not None for r in rows) else None,
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
