"""Bounded delivery from the DB outbox; network I/O never runs in the trading loop."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
import threading

from trade_notifications import compose_execution_notice


def route_fingerprint(channel, destination):
    value = sorted(destination) if channel=='email' else destination
    return hashlib.sha256(json.dumps([channel,value],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def safe_message(value):
    value = re.sub(r'(?i)postgres(?:ql)?(?:\+\w+)?://\S+', '[DB接続情報省略]', str(value))
    return re.sub(r'(?im)([\"\x27]?(?:api[_-]?key|api[_-]?secret|secret|password|passphrase|access[_-]?token|authorization|database_url)[\"\x27]?\s*[:=]).*', r'\1[非公開]', value)


class NotificationOutboxWorker:
    def __init__(self, store, account, enabled, prepare_route, deliver, logger, *, poll_seconds=5):
        self.store,self.account=store,account
        self.enabled,self.prepare_route,self.deliver=enabled,prepare_route,deliver
        self.logger,self.poll_seconds=logger,poll_seconds
        self.stopping=threading.Event();self.wakeup=threading.Event()
        self.thread=threading.Thread(target=self._run,name='trade-notification-outbox',daemon=True)
        self.thread.start()

    def wake(self):self.wakeup.set()

    def process_one(self):
        job=self.store.claim_notification(self.account)
        if job is None:return False
        delivered=False;terminal=None;error=None
        try:
            if not self.enabled(job['kind']):
                error='DeliveryDisabled'
            else:
                route=self.prepare_route(job['channel'])
                if not route:
                    error='DestinationUnavailable'
                else:
                    digest=route_fingerprint(job['channel'],route)
                    if job['route_hash'] is not None and job['route_hash']!=digest:
                        terminal='destination_changed'
                    elif (job['channel']=='line' and job['first_attempt_at'] is not None
                          and datetime.now(timezone.utc)-job['first_attempt_at']>=timedelta(hours=23,minutes=59)):
                        terminal='line_retry_window_expired'
                    else:
                        subject,body=job['subject'],job['body']
                        if subject is None:
                            result=self.store.result(job['client_order_id'])
                            batch=(result['intent'].get('context') or {}).get('close_batch')
                            remaining=None
                            if batch:
                                try:remaining=sum(self.store.close_batch_remaining(self.account,batch).values())
                                except Exception:pass
                            subject,body,_=compose_execution_notice(result,recovered=job['recovered'],remaining=remaining)
                        frozen=self.store.freeze_notification(self.account,job,subject,body,digest)
                        if frozen is not None and not self.stopping.is_set():
                            # Use the original bytes for every retry, even if fees/positions later change.
                            delivered=self.deliver(job['channel'],frozen['subject'],frozen['body'],
                                                   job['kind'],job['job_id'],route) is True
                            if not delivered:error='DeliveryNotConfirmed'
        except Exception as exc:
            error=type(exc).__name__
        try:
            self.store.finish_notification(self.account,job,delivered,error_type=error,terminal_reason=terminal)
        except Exception as exc:
            self.logger.error('Notification acknowledgement failed (%s); persisted lease retained',type(exc).__name__)
        if terminal:
            self.logger.error('Notification requires review (%s; %s)',job['channel'],terminal)
        elif error:
            self.logger.warning('Notification deferred (%s; %s)',job['channel'],error)
        return True

    def _run(self):
        while not self.stopping.is_set():
            self.wakeup.wait(self.poll_seconds);self.wakeup.clear()
            if self.stopping.is_set():break
            try:
                for _ in range(8):
                    if self.stopping.is_set() or not self.process_one():break
            except Exception as exc:
                self.logger.error('Notification queue unavailable (%s)',type(exc).__name__)

    def close(self,timeout=1):
        self.stopping.set();self.wakeup.set();self.thread.join(timeout)
