"""Live-only integration. Paper/backtest keep their own adapters and never create GMO orders."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import os
import uuid
from copy import deepcopy
from strategy_recording import record_bot_event
from trade_notifications import entry_snapshot

from trading_execution import (
    ExchangeGateway,
    ExecutionError,
    ExecutionPending,
    ExchangeMaintenance,
    ExecutionService,
    account_key,
    decimal,
    normalize_size,
    required_margin,
    timestamp,
)


class LiveExecutionMixin:
    def _execution(self):
        if not hasattr(self, "_live_execution_service"):
            from db import engine
            from execution_store import ExecutionStore

            store = ExecutionStore(engine)
            store.check_schema()
            self._live_execution_service = ExecutionService(
                ExchangeGateway(self.gmo_api), store, account_key(self.gmo_api.api_key)
            )
        return self._live_execution_service

    def _live_margin_available(self):
        return self._execution().gateway.margin()

    def _exchange_status_ready(self):
        api = getattr(self, 'gmo_api', None)
        availability = getattr(api, 'availability', None)
        if availability is None:
            return True  # Paper adapters and older test doubles have no exchange status.
        return availability.poll(lambda: api._public_request('/v1/status'))

    def _live_order_size(self, symbol, size):
        rule = self._execution().gateway.rule(self.symbol_mapping[symbol])
        quantity = normalize_size(size, rule["sizeStep"], rule["minOrderSize"])
        if quantity > decimal(rule["maxOrderSize"], positive=True):
            raise ExecutionError("Order above exchange maximum quantity")
        return quantity, rule

    def _live_funds_check(self, symbol, side, size):
        if not self._exchange_status_ready():
            record_bot_event(self, 'funds_blocked', {'symbol': symbol, 'reason': 'exchange_status_unconfirmed'})
            return False
        service = self._execution()
        if service.store.pending(service.account) or service.store.close_batches(service.account):
            # An unconfirmed account order may already have consumed margin.
            record_bot_event(self, "funds_blocked", {"symbol": symbol, "reason": "pending_order"})
            return False
        quantity, rule = self._live_order_size(symbol, size)
        response = self.gmo_api._public_request(
            "/v1/ticker", {"symbol": self.symbol_mapping[symbol]}
        )
        if response.get("status") != 0:
            record_bot_event(self, "funds_blocked", {"symbol": symbol, "reason": "ticker_status", "status": response.get("status")})
            return False
        row = next(
            r for r in response["data"] if r["symbol"] == self.symbol_mapping[symbol]
        )
        age = (datetime.now(timezone.utc) - timestamp(row["timestamp"])).total_seconds()
        if not 0 <= age <= 90:
            record_bot_event(self, "funds_blocked", {"symbol": symbol, "reason": "stale_ticker", "timestamp": row.get("timestamp"), "age_seconds": age})
            return False
        price = decimal(row["ask"] if side == "buy" else row["bid"], positive=True)
        leverage = decimal(os.getenv("GMO_MARGIN_LEVERAGE", "1"), positive=True)
        required = required_margin(quantity, price, leverage, rule["takerFee"])
        margin = service.gateway.margin()
        record_bot_event(self, "funds_snapshot", {
            "symbol": symbol, "side": side, "quantity": quantity,
            "rule": {key: rule.get(key) for key in ("sizeStep", "minOrderSize", "maxOrderSize", "takerFee")},
            "ticker": {key: row.get(key) for key in ("symbol", "timestamp", "bid", "ask", "last")},
            "leverage": leverage, "required": required,
            "margin": {key: margin.get(key) for key in ("availableAmount", "actualProfitLoss", "margin")},
        })
        if margin["availableAmount"] < required:
            self.logger.warning(
                "%s: insufficient margin: required=%s available=%s",
                symbol,
                required,
                margin["availableAmount"],
            )
            return False
        return True

    def _recover_live_orders(self):
        service = self._execution()
        self._queue_trade_notifications()
        if getattr(self, '_record_inputs', False) and not getattr(self, '_execution_events_backfilled', False):
            try:
                # Export a bounded number per cycle. Stable event IDs permit
                # consumers to deduplicate delivery across process restarts.
                count = 0
                if not hasattr(self, '_backfilled_result_ids'):
                    self._backfilled_result_ids = set()
                for client_id in service.store.finalized_ids(service.account):
                    if client_id in self._backfilled_result_ids:
                        continue
                    result = service.store.result(client_id)
                    if self._record_live_order_result(result):
                        self._backfilled_result_ids.add(client_id)
                        count += 1
                    else:
                        break
                    if count == 16:
                        break
                else:
                    self._execution_events_backfilled = True
            except Exception as exc:
                self.logger.error("Execution event backfill not completed (%s)", type(exc).__name__)
        try:
            pending = service.store.pending(service.account)
        except Exception as exc:
            self.logger.error(
                "Order ledger unavailable; new orders blocked (%s)", type(exc).__name__
            )
            self._queue_close_alerts()
            return
        for intent in pending:
            try:
                if intent["status"] == "PREPARED" and service.store.abandon_prepared(
                    intent["client_order_id"]
                ):
                    continue
                service.store.set_notification_preferences(service.account,intent['client_order_id'],self._notification_preferences())
                result = service.reconcile(service.store.get(intent["client_order_id"]), recovered=True)
                self._record_live_order_result(result)
                self._notify_live_execution(result, recovered=True)
            except ExchangeMaintenance:
                self._queue_close_alerts()
                raise
            except Exception as exc:
                # Continue protecting other positions. Unknown orders still block new exposure.
                self.logger.error(
                    "Unresolved execution intent %s (%s)",
                    intent["client_order_id"],
                    type(exc).__name__,
                )
                if not hasattr(self, "_warned_execution_intents"):
                    self._warned_execution_intents = set()
                if intent[
                    "client_order_id"
                ] not in self._warned_execution_intents and callable(
                    getattr(self, "send_notification", None)
                ):
                    self.send_notification(
                        "注文結果未確認",
                        f"{intent['symbol']}: 注文結果を照合中です。重複発注を防ぐため新規注文を抑止しています。",
                        "error",
                    )
                    self._warned_execution_intents.add(intent["client_order_id"])
        # A triggered close survives both a changed price and a process restart.
        # Resume before considering any new entry or waiting for another signal.
        try:
            batches = service.store.close_batches(service.account)
        except Exception as exc:
            self.logger.error("Close batch ledger unavailable (%s)", type(exc).__name__)
            batches = []
        for batch in batches:
            try:
                self._drive_live_close_batch(batch)
            except ExchangeMaintenance:
                self._queue_close_alerts()
                raise
            except Exception as exc:
                self.logger.error(
                    "Close batch not completed (%s): %s", type(exc).__name__,
                    str(exc) if isinstance(exc, ExecutionError) else "details suppressed",
                )
        self._queue_close_alerts()
        self._queue_fee_sync()

    def _queue_fee_sync(self):
        if (getattr(self, 'is_backtest', False) or
                not (getattr(self, 'exchange_settings_gmo', {}) or {}).get('live_trade', False) or
                getattr(self, '_fee_sync_stopped', False)):
            return
        try:
            from fee_sync import FeeSyncWorker
            worker = getattr(self, '_fee_sync_worker', None)
            if worker is None or not worker.thread.is_alive():
                worker = FeeSyncWorker(self._execution(), self.logger)
                self._fee_sync_worker = worker
            worker.wake()  # No DB or exchange reads on the monitoring thread.
        except Exception as exc:
            self.logger.error('Fee sync worker unavailable (%s)', type(exc).__name__)

    def _stop_fee_sync(self):
        self._fee_sync_stopped = True
        worker = getattr(self, '_fee_sync_worker', None)
        if worker is not None:
            worker.close(1)

    def _queue_close_alerts(self):
        settings = getattr(self, 'notification_settings', {}) or {}
        if not settings.get('enabled', False) or not settings.get('send_on_error', False):
            return
        if not callable(getattr(self, 'send_notification', None)):
            return
        try:
            from close_alerts import CloseAlertPolicy, CloseAlertWorker
            service = self._execution()
            worker = getattr(self, '_close_alert_worker', None)
            if worker is None or not worker.thread.is_alive():
                policy = CloseAlertPolicy.from_environment(self.logger)
                sender = getattr(self, '_send_notification_direct', self.send_notification)
                worker = CloseAlertWorker(service.store, service.account, sender, self.logger, policy)
                self._close_alert_worker = worker
            for job in service.store.claim_close_alerts(service.account, worker.policy):
                if not worker.submit(job):
                    service.store.finish_close_alert(service.account, job, False)
        except Exception as exc:
            self.logger.error('Close stall monitoring unavailable (%s)', type(exc).__name__)

    def _reset_exit_adjustments(self, symbol):
        """Clear per-trade exit state without disturbing other symbols."""
        getattr(self, '_exit_quote_failures', {}).pop(symbol, None)
        if hasattr(self, "opposite_narrow_state"):
            self.opposite_narrow_state[symbol] = {"long": False, "short": False}
        cache = getattr(self, "_last_valid_exit_levels", {})
        for key in list(cache):
            if key[0] == symbol:
                cache.pop(key, None)
        fallback_state = getattr(self, '_exit_level_fallback_state', {})
        for key in list(fallback_state):
            if key[0] == symbol:
                fallback_state.pop(key, None)

    def _exit_protection_error(self, stage, exc):
        if not hasattr(self, '_exit_protection_errors'):
            self._exit_protection_errors = {}
        kind = type(exc).__name__
        if self._exit_protection_errors.get(stage) != kind:
            self.logger.error('Exit protection persistence unavailable (%s; %s)', stage, kind)
        self._exit_protection_errors[stage] = kind

    def _exit_protection_recovered(self, stage):
        if getattr(self, '_exit_protection_errors', {}).pop(stage, None):
            self.logger.info('Exit protection persistence recovered (%s)', stage)

    def _exit_protection_file(self):
        if (getattr(self, 'is_backtest', False) or
                not (getattr(self, 'exchange_settings_gmo', {}) or {}).get('live_trade', False) or
                not getattr(self, 'base_dir', None)):
            return None
        try:
            from exit_protection import ExitProtectionFile
            account = self._execution().account
            store = getattr(self, '_exit_protection_store', None)
            if store is None or store.account != account:
                store = ExitProtectionFile(self.base_dir, account)
                store.load()
                self._exit_protection_store = store
            self._exit_protection_recovered('load')
            return store
        except Exception as exc:
            self._exit_protection_error('load', exc)
            return None

    def _persist_exit_protection(self, symbol):
        # A failed/absent exchange snapshot must never replace the last saved
        # state with guessed positions or erase another symbol's protection.
        if symbol in getattr(self, '_position_sync_failed', set()):
            return False
        rows = getattr(self, '_verified_position_snapshots', {}).get(symbol)
        if not rows:
            return False
        store = self._exit_protection_file()
        if store is None:
            return False
        if (store.account, symbol) not in getattr(self, '_exit_protection_restored', set()):
            return False  # Do not overwrite a snapshot not yet checked after a read failure.
        try:
            from exit_protection import make_snapshot
            side, entry = self.positions[symbol], self.entry_prices[symbol]
            snapshot = make_snapshot(
                store.account, symbol, rows, side, entry,
                getattr(self, 'opposite_narrow_state', {}).get(symbol, {}).get(side, False),
                self.short_profit_lock_armed.get(symbol, False),
                getattr(self, '_last_valid_exit_levels', {}).get((symbol, side, entry)),
            )
            store.update(symbol, snapshot)
            store.save()
            self._exit_protection_recovered('write')
            return True
        except Exception as exc:
            self._exit_protection_error('write', exc)
            return False

    def _sync_exit_protection(self, symbol, rows):
        store = self._exit_protection_file()
        if store is None:
            return
        try:
            if not hasattr(self, '_exit_protection_restored'):
                self._exit_protection_restored = set()
            key = (store.account, symbol)
            if not rows:
                store.update(symbol, None)
                store.save()
                self._exit_protection_restored.add(key)
                self._exit_protection_recovered('write')
                return
            if key not in self._exit_protection_restored:
                saved = store.snapshots.get(symbol)
                if saved is not None:
                    from exit_protection import restore_snapshot
                    try:
                        state = restore_snapshot(saved, store.account, symbol, rows,
                                                 self.positions[symbol], self.entry_prices[symbol])
                    except (ValueError, TypeError, KeyError, OverflowError):
                        state = None
                        self.logger.warning('%s: invalid saved exit protection discarded', symbol)
                    if state is not None:
                        side, entry = self.positions[symbol], self.entry_prices[symbol]
                        if not hasattr(self, 'opposite_narrow_state'):
                            self.opposite_narrow_state = {}
                        self.opposite_narrow_state[symbol] = {'long': False, 'short': False}
                        self.opposite_narrow_state[symbol][side] = state['opposite_narrow']
                        # Arming is monotonic for the same verified position.
                        # A newer legacy save can survive a failed protection-file
                        # write; never disarm that already verified protection.
                        self.short_profit_lock_armed[symbol] = (
                            state['short_profit_lock'] or self.short_profit_lock_armed.get(symbol, False)
                        )
                        if state['levels'] is not None:
                            if not hasattr(self, '_last_valid_exit_levels'):
                                self._last_valid_exit_levels = {}
                            self._last_valid_exit_levels[(symbol, side, entry)] = state['levels']
                        self.logger.info('%s: verified exit protection restored', symbol)
                    else:
                        self.logger.info('%s: saved exit protection does not match current positions', symbol)
                self._exit_protection_restored.add(key)
            # A temporary sync failure from a previous cycle is now resolved.
            if hasattr(self, '_position_sync_failed'):
                self._position_sync_failed.discard(symbol)
            self._persist_exit_protection(symbol)
        except Exception as exc:
            self._exit_protection_error('write', exc)

    def _sync_live_positions(self, symbol=None):
        service = self._execution()
        for sym in [symbol] if symbol else self.symbols:
            try:
                rows = service.gateway.positions(self.symbol_mapping[sym])
                sides = {r["side"] for r in rows}
                if len(sides) > 1:
                    raise ExecutionError(
                        "Both position directions exist; netting would hide exposure"
                    )
                if not rows:
                    if service.store.open_position_ids(
                        service.account, self.symbol_mapping[sym]
                    ):
                        raise ExecutionError(
                            "Position disappeared without recorded closing executions"
                        )
                    self.positions[sym] = None
                    self.entry_sizes[sym] = 0.0
                    self.entry_prices[sym] = 0.0
                    self.entry_times[sym] = None
                    self.position_ids[sym] = None
                    self.entry_scores[sym] = {}
                    self.short_profit_lock_armed[sym] = False
                    self._reset_exit_adjustments(sym)
                    getattr(self, "_exit_position_ids", {}).pop(sym, None)
                    getattr(self, "_verified_position_snapshots", {}).pop(sym, None)
                    if hasattr(self, "_position_sync_failed"):
                        self._position_sync_failed.discard(sym)
                    self._sync_exit_protection(sym, [])
                    continue
                size = sum(decimal(r["size"]) for r in rows)
                price = (
                    sum(decimal(r["size"]) * decimal(r["price"]) for r in rows) / size
                )
                side = "long" if rows[0]["side"] == "BUY" else "short"
                previous_side = self.positions.get(sym)
                previous_pid = self.position_ids.get(sym)
                saved_pid = getattr(self, "_saved_short_lock_position_ids", {}).get(sym)
                current_ids = frozenset(str(row["positionId"]) for row in rows)
                previous_ids = getattr(self, "_exit_position_ids", {}).get(sym)
                if previous_ids is None:
                    known_pid = previous_pid if previous_pid is not None else saved_pid
                    previous_ids = (
                        frozenset([str(known_pid)])
                        if known_pid is not None
                        else frozenset()
                    )
                continuous = bool(previous_ids) and current_ids <= previous_ids and previous_side in (None, side)
                prior_rows = getattr(self, '_verified_position_snapshots', {}).get(sym)
                if continuous and prior_rows:
                    from exit_protection import position_basis
                    prior_basis, current_basis = position_basis(prior_rows), position_basis(rows)
                    continuous = all(
                        all(row[key] == prior_basis[pid][key] for key in ('symbol', 'side', 'price', 'opened_at'))
                        and decimal(row['size']) <= decimal(prior_basis[pid]['size'])
                        for pid, row in current_basis.items()
                    )
                if not continuous:
                    self._reset_exit_adjustments(sym)
                if side != "short" or not continuous:
                    self.short_profit_lock_armed[sym] = False
                if not hasattr(self, "_exit_position_ids"):
                    self._exit_position_ids = {}
                self._exit_position_ids[sym] = current_ids
                if not hasattr(self, '_verified_position_snapshots'):
                    self._verified_position_snapshots = {}
                self._verified_position_snapshots[sym] = deepcopy(rows)
                self.positions[sym] = side
                self.entry_sizes[sym] = float(size)
                self.entry_prices[sym] = float(price)
                self.entry_times[sym] = min(timestamp(r["timestamp"]) for r in rows)
                self.position_ids[sym] = str(rows[0]["positionId"])
                context = service.store.position_context(
                    service.account, rows[0]["positionId"]
                )
                self.entry_scores[sym] = context.get(
                    "scores", self.entry_scores.get(sym, {})
                )
                self._sync_exit_protection(sym, rows)
            except ExchangeMaintenance:
                if not hasattr(self, '_position_sync_failed'):
                    self._position_sync_failed = set()
                self._position_sync_failed.update(self.symbols)
                raise  # Preserve all prior snapshots, quantities and protection state.
            except Exception as exc:
                # Missing data is not proof that a position has disappeared.
                self.logger.error(
                    "%s: position reconciliation failed (%s): %s", sym, type(exc).__name__,
                    str(exc) if isinstance(exc, ExecutionError) else "details suppressed"
                )
                if not hasattr(self, "_position_sync_failed"):
                    self._position_sync_failed = set()
                self._position_sync_failed.add(sym)
                continue
            if hasattr(self, "_position_sync_failed"):
                self._position_sync_failed.discard(sym)
        self.save_positions()

    def _execute_live_entry(self, symbol, order_type, size, signal_raw=None, **kwargs):
        if self.positions.get(symbol) is not None or symbol in getattr(
            self, "_position_sync_failed", set()
        ):
            return {
                "success": False,
                "error": "Position state does not permit a new entry",
            }
        service = self._execution()
        if not self._live_funds_check(symbol, order_type, size):
            return {"success": False, "error": "Margin or pending order check failed"}
        quantity, _ = self._live_order_size(symbol, size)
        # Include the decision bar in the persisted key so a repeated cycle cannot reopen it.
        raw = signal_raw or {}
        bar = raw.get("decision_bar")
        if not bar:
            raise ExecutionError("Live entry requires a decision bar identity")
        decimal(raw.get("decision_price"), positive=True)
        side = "BUY" if order_type == "buy" else "SELL"
        key = "entry:" + hashlib.sha256(f"{symbol}|{side}|{bar}".encode()).hexdigest()
        scores = dict(self.entry_scores.get(symbol, {}))
        sid = self._record_signal(
            symbol=symbol,
            timeframe="15m",
            side="long" if side == "BUY" else "short",
            price=float(raw["decision_price"]),
            strength_score=kwargs.get("strength_score"),
            indicators={
                "RSI": kwargs.get("rsi"),
                "ADX": kwargs.get("adx"),
                "ATR": kwargs.get("atr"),
                "DI+": kwargs.get("di_plus"),
                "DI-": kwargs.get("di_minus"),
                "EMA_fast": kwargs.get("ema_fast"),
                "EMA_slow": kwargs.get("ema_slow"),
            },
            strategy_id=kwargs.get("strategy_id"),
            version=kwargs.get("version"),
            raw=raw,
            df_row=getattr(self, "_entry_decision_snapshot", None),
            scores=scores,
        )
        context = {
            "bot_symbol": symbol,
            "scores": scores,
            "signal_id": sid or None,
            "user_id": getattr(self, "user_id", None),
            "decision_bar": str(bar),
            "decision_price": str(raw["decision_price"]),
            "feature_snapshot": getattr(self, "_feature_event_refs", {}).get(symbol),
            "notification_delivery": self._notification_preferences(),
        }
        try:
            context['notification'] = entry_snapshot(getattr(self, '_entry_decision_snapshot', None), kwargs)
        except Exception as exc:
            self.logger.warning('Entry receipt snapshot unavailable (%s)', type(exc).__name__)
            context['notification'] = {}
        confirmation = getattr(self, '_entry_notification_context', {}).get(symbol, {})
        if confirmation.get('direction') == ('long' if side == 'BUY' else 'short'):
            try:
                if timestamp(confirmation.get('decision_bar')) == timestamp(bar):
                    context['notification']['previous_bar'] = confirmation['previous_bar']
                    context['notification'].setdefault('signals', {})['sell_signal' if side == 'BUY' else 'buy_signal'] = True
            except (ExecutionError, TypeError, ValueError, KeyError):
                pass  # Optional notification evidence cannot alter an order decision.
        result = service.submit(
            self.symbol_mapping[symbol],
            side,
            quantity,
            decision_key=key,
            context=context,
        )
        self._record_live_order_result(result)
        if result.get("already_submitted"):
            return {
                **result,
                "success": False,
                "error": "This entry decision was already processed",
            }
        self._sync_live_positions(symbol)
        self._notify_live_execution(result)
        return result

    def _configured_trade_channels(self):
        settings=getattr(self,'notification_settings',{}) or {}
        email=settings.get('email') or {}
        channels=[]
        if all(email.get(key) for key in ('smtp_server','smtp_port','sender','password','recipient')):
            channels.append('email')
        if os.getenv('LINE_CHANNEL_ACCESS_TOKEN') and (os.getenv('LINE_DEFAULT_USER_ID') or getattr(self,'user_id',None) is not None):
            channels.append('line')
        return channels

    def _notification_preferences(self):
        settings = getattr(self, 'notification_settings', {}) or {}
        return {'enabled':bool(settings.get('enabled',False)),
                'entry':bool(settings.get('send_on_entry',True)), 'exit':bool(settings.get('send_on_exit',True)),
                'error':bool(settings.get('send_on_error',True)), 'info':True,
                'daily_report':bool(settings.get('daily_report',False)),
                'channels':self._configured_trade_channels()}

    def _outbox_enabled(self, kind):
        prefs=self._notification_preferences()
        return prefs['enabled'] and prefs.get(kind,False)

    def _prepare_notification_route(self, channel):
        if channel not in self._configured_trade_channels():return None
        if channel=='email':return self._resolve_recipients_for_mode()
        recipient=None
        if getattr(self,'user_id',None) is not None:
            from db import get_line_user_id
            recipient=get_line_user_id(self.user_id)
        return recipient or os.getenv('LINE_DEFAULT_USER_ID')

    def _deliver_outbox_notification(self, channel, subject, body, kind, job_id, route):
        if channel=='email':
            return self._send_email(subject,body,message_id=f'<{job_id}@crypto-bot.invalid>',recipients=route)
        return self._send_line(subject,body,retry_key=job_id,recipient=route)

    def _queue_trade_notifications(self):
        if not (getattr(self,'notification_settings',{}) or {}).get('enabled',False):return
        try:
            from notification_outbox import NotificationOutboxWorker
            service=self._execution()
            worker=getattr(self,'_trade_notification_worker',None)
            if worker is None or not worker.thread.is_alive():
                worker=NotificationOutboxWorker(service.store,service.account,self._outbox_enabled,
                    self._prepare_notification_route,self._deliver_outbox_notification,self.logger)
                self._trade_notification_worker=worker
            worker.wake()
        except Exception as exc:
            self.logger.error('Notification dispatch unavailable (%s)',type(exc).__name__)

    def _notify_live_execution(self, result, *, recovered=False):
        # Finalized fills and their per-channel jobs already committed atomically.
        if result.get('finalized') and not result.get('already_submitted'):
            self._queue_trade_notifications()
        return False

    def _enqueue_live_message(self, subject, body, kind='info', *, notification_key=None):
        try:
            from notification_outbox import safe_message
            service=self._execution()
            count=service.store.enqueue_message(service.account,notification_key or 'message:'+uuid.uuid4().hex,
                self._notification_preferences(),safe_message(subject),safe_message(body),kind)
            self._queue_trade_notifications()
            return bool(count)
        except Exception as exc:
            self.logger.error('Notification enqueue failed (%s)',type(exc).__name__)
            return False

    def _stop_trade_notifications(self):
        worker=getattr(self,'_trade_notification_worker',None)
        if worker is not None:worker.close(1)

    def _exit_quote_status(self, symbol, reason=None):
        failures = getattr(self, '_exit_quote_failures', {})
        state = failures.get(symbol)
        if reason is None and state is None:
            return
        if not hasattr(self, '_exit_quote_failures'):
            self._exit_quote_failures = failures
        live = ((getattr(self, 'exchange_settings_gmo', {}) or {}).get('live_trade', False)
                and not getattr(self, 'is_backtest', False))
        send = getattr(self, 'send_notification', None)
        if reason is not None:
            if state is None:
                state = {'id': uuid.uuid4().hex, 'reason': None, 'notified': False}
                failures[symbol] = state
            if state['reason'] != reason:
                self.logger.warning('%s: exit quote unavailable (%s); retaining position and protection', symbol, reason)
                state['reason'] = reason
            if live and callable(send) and not state['notified']:
                try:
                    state['notified'] = send(
                        f'{symbol} 決済価格の確認不可',
                        '有効な現在価格を取得できません。価格による利確・損切り判定を保留し、再取得を続けています。'
                        '建玉と保護状態は保持しています。保有期限による決済と保存済み決済要求の照合は継続します。',
                        'error', notification_key=f"exit-quote:{symbol}:{state['id']}:unavailable",
                    ) is True
                except Exception as exc:
                    self.logger.error('Exit quote alert enqueue failed (%s)', type(exc).__name__)
            return
        failures.pop(symbol, None)
        self.logger.info('%s: exit quote recovered; resuming price-based protection', symbol)
        if live and callable(send) and state['notified']:
            try:
                send(f'{symbol} 決済価格の確認復旧', '有効な現在価格を取得できました。価格による利確・損切り判定を再開します。',
                     'info', notification_key=f"exit-quote:{symbol}:{state['id']}:recovered")
            except Exception as exc:
                self.logger.error('Exit quote recovery notice enqueue failed (%s)', type(exc).__name__)

    def _record_live_order_result(self, result):
        """Use the same payload for immediate and recovered exchange outcomes."""
        intent = result["intent"]
        context = intent["context"]
        event_id = "order:" + hashlib.sha256(
            f"{intent['client_order_id']}|{result['status']}|{result['finalized']}|{result['executed_size']}".encode()
        ).hexdigest()
        if not hasattr(self, '_recorded_order_results'):
            self._recorded_order_results = set()
        if event_id in self._recorded_order_results:
            return True
        payload = {
            "event_id": event_id,
            "symbol": context.get("bot_symbol", intent["symbol"]),
            "purpose": "close" if intent["settle_type"] == "CLOSE" else "entry",
            "side": intent["side"], "reason": context.get("exit_reason"),
            "requested_size": intent["size"], "decision_bar": context.get("decision_bar"),
            "feature_snapshot": context.get("feature_snapshot"),
            "result": {key: result.get(key) for key in ("success", "status", "finalized", "executed_size", "execution_price", "executed_at", "position_ids", "client_order_id")},
            "trades": [{key: trade.get(key) for key in ("entry_price", "exit_price", "size", "pnl", "entry_fee", "exit_fee", "net_pnl_before_carry", "cost_status")} for trade in result.get("trades", [])],
        }
        if record_bot_event(self, "order_result", payload):
            self._recorded_order_results.add(event_id)
            return True
        return False

    def _close_live_position(self, symbol, reason, stats, trade_logs):
        service = self._execution()
        batches = [b for b in service.store.close_batches(service.account) if b["symbol"] == symbol]
        if batches:
            batch = batches[0]
        else:
            side = self.positions[symbol]
            snapshot = getattr(self, '_verified_position_snapshots', {}).get(symbol)
            if snapshot:
                sizes = {str(p['positionId']): str(p['size']) for p in snapshot}
                ids = sorted(sizes)
                total = str(sum((decimal(size) for size in sizes.values()), decimal(0)))
            else:
                known = getattr(self, '_exit_position_ids', {}).get(symbol)
                pid = getattr(self, 'position_ids', {}).get(symbol)
                ids = sorted(str(value) for value in known) if known is not None else ([str(pid)] if pid is not None else [])
                sizes = None
                size = getattr(self, 'entry_sizes', {}).get(symbol)
                total = str(size) if size else None
            batch = {"id": uuid.uuid4().hex, "symbol": symbol, "side": side,
                     "reason": reason, "positions": None,
                     "target_ids": ids, "target_sizes": sizes, "target_total": total,
                     "scores": self.entry_scores.get(symbol, {}),
                     "user_id": getattr(self, "user_id", None),
                     "feature_snapshot": getattr(self, "_feature_event_refs", {}).get(symbol)}
        self._drive_live_close_batch(batch, stats, trade_logs)

    def _drive_live_close_batch(self, batch, stats=None, trade_logs=None):
        service = self._execution()
        # A busy order slot must not discard the close decision. Registration is
        # independent of bot_order_intents and commits before any waiting return.
        batch = service.store.save_close_batch(
            service.account, self.symbol_mapping[batch['symbol']], batch
        )
        symbol, side, reason = batch["symbol"], batch["side"], batch["reason"]
        target_ids = set(batch.get('target_ids') or [str(p['positionId']) for p in batch.get('positions') or []])
        def blocking_pending():
            return [intent for intent in service.store.pending(service.account)
                    if intent['symbol'] == self.symbol_mapping[symbol]
                    or str(intent.get('position_id')) in target_ids]
        if blocking_pending():
            self.logger.warning(
                "%s: waiting for existing order reconciliation before another close",
                symbol,
            )
            return
        positions = service.gateway.positions(self.symbol_mapping[symbol])
        if batch.get('positions') is None:
            targets = set(batch['target_ids'])
            selected = [p for p in positions if str(p['positionId']) in targets]
            batch = service.store.resolve_close_batch(
                service.account, self.symbol_mapping[symbol], batch['id'], selected
            )
        remaining = service.store.close_batch_remaining(service.account, batch)
        current = {str(p["positionId"]): p for p in positions}
        # A disappearing target without our recorded fills is an unresolved
        # exposure, never permission to close a different/new position.
        for pid, size in remaining.items():
            if size and (pid not in current or decimal(current[pid]["size"]) != size):
                raise ExecutionError("Close batch position snapshot not confirmed")
        results = []
        for target in batch["positions"]:
            pid = str(target["positionId"])
            if not remaining[pid]:
                continue
            pos = current[pid]
            if pos["side"] != ("BUY" if side == "long" else "SELL"):
                raise ExecutionError("Position direction changed during close")
            context = {
                "bot_symbol": symbol,
                "baseline": pos,
                "exit_reason": reason,
                "scores": batch["scores"], "user_id": batch["user_id"],
                "feature_snapshot": batch.get("feature_snapshot"), "close_batch": batch,
                "notification_delivery": self._notification_preferences(),
            }
            try:
                result = service.submit(
                    self.symbol_mapping[symbol],
                    "SELL" if side == "long" else "BUY",
                    pos["size"], position_id=pos["positionId"], context=context,
                )
            except ExecutionPending:
                self.logger.warning("%s: close awaiting consistent exchange snapshots", symbol)
                break
            results.append(result)
            self._record_live_order_result(result)
            if not result["finalized"]:
                break
        self._sync_live_positions(symbol)
        # Stats are refreshed from committed records, never incremented on a retry.
        if stats is not None:
            self._refresh_live_stats(stats)
        complete = not blocking_pending() and not any(service.store.close_batch_remaining(service.account, batch).values())
        if complete:
            service.store.finish_close_batch(service.account, batch)
        if results:
            self.logger.info(
                "%s: close execution progress (%s); remaining=%s",
                symbol,
                reason,
                self.entry_sizes[symbol],
            )
            for result in results:
                for trade in result.get("trades", []) if trade_logs is not None else []:
                    trade_logs.append(
                        {
                            "symbol": symbol,
                            "type": side,
                            "entry_price": float(trade["entry_price"]),
                            "exit_price": float(trade["exit_price"]),
                            "size": float(trade["size"]),
                            "profit": float(trade["pnl"]),
                            "profit_pct": float(trade["pnl_pct"]),
                            "exit_reason": reason,
                            "holding_hours": float(trade["holding_hours"]),
                        }
                    )
                if result['success']:
                    self._notify_live_execution(result)
        if complete and self.positions.get(symbol) is None:
            now = datetime.now(timezone.utc)
            self.reentry_block_until[symbol] = self._ceil_to_next_15min(now)

    def _refresh_live_stats(self, stats):
        from db import engine
        from sqlalchemy import text

        # Existing gross-P/L counters remain explicitly gross. Net amounts live in the v2 ledger.
        with engine.connect() as conn:
            rows = list(
                conn.execute(
                    text("SELECT pnl,closed_at FROM trades WHERE source='real'")
                ).mappings()
            )
        today = (datetime.now(timezone.utc) + timedelta(hours=9)).date()
        for prefix, selected in [
            ("total_", rows),
            (
                "daily_",
                [
                    r
                    for r in rows
                    if (r["closed_at"] + timedelta(hours=9)).date() == today
                ],
            ),
        ]:
            stats[prefix + "trades"] = len(selected)
            stats[prefix + "wins"] = sum(r["pnl"] > 0 for r in selected)
            stats[prefix + "losses"] = sum(r["pnl"] < 0 for r in selected)
            stats[prefix + "profit"] = float(
                sum(r["pnl"] for r in selected if r["pnl"] > 0)
            )
            stats[prefix + "loss"] = float(
                -sum(r["pnl"] for r in selected if r["pnl"] < 0)
            )
        self.total_profit = float(sum(r["pnl"] for r in rows))
