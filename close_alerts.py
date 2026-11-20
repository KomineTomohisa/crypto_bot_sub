"""Bounded asynchronous delivery for durable close-stall notices; no I/O on import."""
from dataclasses import dataclass
from datetime import timedelta, timezone
import os
import queue
import threading


@dataclass(frozen=True)
class CloseAlertPolicy:
    stall_seconds: int = 120
    reminder_seconds: int = 1800
    retry_seconds: int = 300
    lease_seconds: int = 600

    def __post_init__(self):
        for value in (self.stall_seconds, self.reminder_seconds, self.retry_seconds, self.lease_seconds):
            if type(value) is not int or not 0 < value <= 604800:
                raise ValueError('Close alert intervals must be positive integers')

    @classmethod
    def from_environment(cls, logger):
        values = {}
        for field, name, default in (
            ('stall_seconds', 'CLOSE_STALL_ALERT_SECONDS', 120),
            ('reminder_seconds', 'CLOSE_STALL_REMINDER_SECONDS', 1800),
            ('retry_seconds', 'CLOSE_STALL_RETRY_SECONDS', 300),
        ):
            try:
                value = int(os.getenv(name, str(default)))
                if not 0 < value <= 604800:raise ValueError()
            except ValueError:
                logger.warning('Invalid close alert setting %s; using default', name)
                value = default
            values[field] = value
        return cls(**values)

    def message(self, job):
        short_id = job['batch_id'][:8]
        created_jst = job['created_at'].astimezone(timezone(timedelta(hours=9)))
        if job['kind'] == 'recovery':
            return (f"{job['symbol']} 決済滞留の解消",
                    f"決済予定 {short_id} の全対象について、記録済み約定から決済完了を確認しました。\n"
                    f"要求日時: {created_jst:%Y-%m-%d %H:%M:%S} JST", 'info')
        phase = {
            'order_reconciliation': '口座の未確定注文の照合待ち',
            'position_confirmation': '建玉照会・決済対象の確認待ち',
            'close_pending': '決済処理・残量の確認待ち',
        }[job['phase']]
        minutes = max(0, int((job['observed_at'] - job['created_at']).total_seconds() // 60))
        return (f"{job['symbol']} 決済が未完了です",
                f"決済予定 {short_id} は要求から約{minutes}分経過しています。\n"
                f"要求日時: {created_jst:%Y-%m-%d %H:%M:%S} JST\n確認状況: {phase}\n"
                'この通知のために注文を再送・追加発注することはありません。建玉と注文状態を確認してください。', 'error')


class CloseAlertWorker:
    """Network delivery and acknowledgement never run in the trading thread."""
    def __init__(self, store, account, send, logger, policy, *, capacity=8):
        self.store, self.account, self.send = store, account, send
        self.logger, self.policy = logger, policy
        self.queue = queue.Queue(maxsize=capacity)
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, name='close-stall-notifier', daemon=True)
        self.thread.start()

    def submit(self, job):
        if self.stopping.is_set() or not self.thread.is_alive():return False
        try:self.queue.put_nowait(job)
        except queue.Full:return False
        return True

    def _run(self):
        while not self.stopping.is_set():
            try:job = self.queue.get(timeout=0.05)
            except queue.Empty:continue
            try:
                delivered = False
                if self.store.close_alert_is_current(self.account, job):
                    try:delivered = self.send(*self.policy.message(job)) is True
                    except Exception as exc:
                        self.logger.error('Close alert delivery failed (%s)', type(exc).__name__)
                self.store.finish_close_alert(self.account, job, delivered)
            except Exception as exc:
                # The persisted lease allows a later process/cycle to retry.
                self.logger.error('Close alert acknowledgement failed (%s)', type(exc).__name__)
            finally:self.queue.task_done()

    def close(self, timeout=1):
        self.stopping.set()
        self.thread.join(timeout)
