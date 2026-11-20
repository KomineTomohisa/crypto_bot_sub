"""Live-only integration. Paper/backtest keep their own adapters and never create GMO orders."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import os
import time
import uuid
from strategy_recording import record_bot_event

from trading_execution import (
    ExchangeGateway,
    ExecutionError,
    ExecutionPending,
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

    def _live_order_size(self, symbol, size):
        rule = self._execution().gateway.rule(self.symbol_mapping[symbol])
        quantity = normalize_size(size, rule["sizeStep"], rule["minOrderSize"])
        if quantity > decimal(rule["maxOrderSize"], positive=True):
            raise ExecutionError("Order above exchange maximum quantity")
        return quantity, rule

    def _live_funds_check(self, symbol, side, size):
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
            return
        for intent in pending:
            try:
                if intent["status"] == "PREPARED" and service.store.abandon_prepared(
                    intent["client_order_id"]
                ):
                    continue
                result = service.reconcile(service.store.get(intent["client_order_id"]))
                self._record_live_order_result(result)
                if result["finalized"] and callable(
                    getattr(self, "send_notification", None)
                ):
                    self.send_notification(
                        "注文照合完了",
                        f"{intent['symbol']}: 約定数量 {result['executed_size']}、状態 {result['status']}",
                    )
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
            except Exception as exc:
                self.logger.error(
                    "Close batch not completed (%s): %s", type(exc).__name__,
                    str(exc) if isinstance(exc, ExecutionError) else "details suppressed",
                )
        now = time.monotonic()
        if now >= getattr(self, "_next_fee_sync", 0):
            self._next_fee_sync = now + 3600
            try:
                today = (
                    datetime.now(timezone.utc) + timedelta(hours=9) - timedelta(hours=6)
                ).date()
                service.sync_fees(today)
            except Exception as exc:
                self.logger.error(
                    "Daily exchange fees not confirmed (%s): %s", type(exc).__name__,
                    str(exc) if isinstance(exc, ExecutionError) else "details suppressed"
                )

    def _reset_exit_adjustments(self, symbol):
        """Clear per-trade exit state without disturbing other symbols."""
        if hasattr(self, "opposite_narrow_state"):
            self.opposite_narrow_state[symbol] = {"long": False, "short": False}
        cache = getattr(self, "_last_valid_exit_levels", {})
        for key in list(cache):
            if key[0] == symbol:
                cache.pop(key, None)

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
                    if hasattr(self, "_position_sync_failed"):
                        self._position_sync_failed.discard(sym)
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
                continuous = bool(
                    previous_ids.intersection(current_ids)
                ) and previous_side in (None, side)
                if not continuous:
                    self._reset_exit_adjustments(sym)
                if side != "short" or not continuous:
                    self.short_profit_lock_armed[sym] = False
                if not hasattr(self, "_exit_position_ids"):
                    self._exit_position_ids = {}
                self._exit_position_ids[sym] = current_ids
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
        }
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
        return result

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
            positions = service.gateway.positions(self.symbol_mapping[symbol])
            if not positions:
                self._sync_live_positions(symbol)
                return
            side = self.positions[symbol]
            if any(p["side"] != ("BUY" if side == "long" else "SELL") for p in positions):
                raise ExecutionError("Position direction changed during close")
            batch = {"id": uuid.uuid4().hex, "symbol": symbol, "side": side,
                     "reason": reason, "positions": positions,
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
        pending = service.store.pending(service.account)
        if pending:
            self.logger.warning(
                "%s: waiting for existing order reconciliation before another close",
                symbol,
            )
            return
        positions = service.gateway.positions(self.symbol_mapping[symbol])
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
        complete = not service.store.pending(service.account) and not any(service.store.close_batch_remaining(service.account, batch).values())
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
                if result["success"] and result["finalized"] and self.notification_settings.get(
                    "send_on_exit", True
                ):
                    gross = sum(t["pnl"] for t in result.get("trades", []))
                    self.send_notification(
                        f"{symbol} 決済約定",
                        f"理由: {reason}\n実約定数量: {result['executed_size']}\n"
                        f"売買損益（費用別）: {gross:,.2f}円\n残数量: {self.entry_sizes[symbol]}\n"
                        "手数料・日次費用は実明細で別途集計します。",
                    )
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
