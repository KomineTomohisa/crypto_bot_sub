"""Bounded fee-history backfill outside the position-monitoring thread."""
from datetime import datetime, timedelta, timezone
import threading
import time

from trading_execution import ExchangeMaintenance


class FeeSyncInterrupted(RuntimeError):
    """Stop between reads; an incomplete page set must never be confirmed."""


class FeeSyncWorker:
    def __init__(self, service, logger, *, clock=time.monotonic,
                 interval_seconds=3600, retry_seconds=60,
                 continuation_seconds=30, budget_seconds=30):
        self.service, self.logger, self.clock = service, logger, clock
        self.interval_seconds, self.retry_seconds = interval_seconds, retry_seconds
        self.continuation_seconds, self.budget_seconds = continuation_seconds, budget_seconds
        self.stopping = threading.Event()
        self.wakeup = threading.Event()
        self._next_run = 0.
        self.thread = threading.Thread(target=self._run, name='exchange-fee-sync', daemon=True)
        self.thread.start()

    def wake(self):
        self.wakeup.set()

    def _run(self):
        while not self.stopping.is_set():
            self.wakeup.wait(30)
            self.wakeup.clear()
            if self.stopping.is_set():
                break
            try:
                if self.clock() < self._next_run:
                    continue
                gate = getattr(self.service.gateway.api, 'availability', None)
                if gate is not None and gate.blocked:
                    continue  # Only the shared status poll may confirm OPEN.
                deadline = self.clock() + self.budget_seconds

                def check_continue():
                    if self.stopping.is_set() or self.clock() >= deadline:
                        raise FeeSyncInterrupted()
                    if gate is not None:
                        gate.require_available()

                # Keep the existing exchange business-day cutoff (06:00 JST).
                today = (datetime.now(timezone.utc) + timedelta(hours=3)).date()
                more = self.service.sync_fees(today, check_continue=check_continue, max_windows=1)
                delay = self.continuation_seconds if more else self.interval_seconds
                self._next_run = self.clock() + delay
            except ExchangeMaintenance:
                self._next_run = 0.  # Resume promptly after the shared gate confirms OPEN.
            except FeeSyncInterrupted:
                self._next_run = self.clock() + self.retry_seconds
                if not self.stopping.is_set():
                    self.logger.warning('Daily exchange fee sync deferred (time budget exhausted)')
            except Exception as exc:
                self._next_run = self.clock() + self.retry_seconds
                self.logger.error('Daily exchange fees not confirmed (%s)', type(exc).__name__)

    def close(self, timeout=1):
        self.stopping.set()
        self.wakeup.set()
        self.thread.join(timeout)
