"""Shared, monotonic exchange-status polling; no network or I/O on import."""
import threading
import time

from trading_execution import ExchangeMaintenance


def maintenance_reason(response):
    if not isinstance(response, dict):
        return None
    if type(response.get('status')) is not int:
        return None
    if response.get('status') == 0:
        data = response.get('data')
        if isinstance(data, dict) and data.get('status') in ('MAINTENANCE', 'PREOPEN'):
            return data['status']
        return None
    messages = response.get('messages')
    if not isinstance(messages, list) or not messages:
        return None
    states = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        code = message.get('message_code')
        if code in ('ERR-5201', 'ERR-5202'):
            states.append('MAINTENANCE')
        elif code == 'ERR-5203':
            states.append('PREOPEN')
        elif code is None and message.get('message_string') == 'MAINTENANCE. Please wait for a while':
            states.append('MAINTENANCE')
        else:
            return None  # Do not disguise authentication or mixed errors as maintenance.
    return 'MAINTENANCE' if 'MAINTENANCE' in states else 'PREOPEN'


class ExchangeAvailability:
    def __init__(self, logger, *, clock=time.monotonic, normal_seconds=30, paused_seconds=60):
        if normal_seconds <= 0 or paused_seconds <= 0:
            raise ValueError('Status intervals must be positive')
        self.logger, self.clock = logger, clock
        self.normal_seconds, self.paused_seconds = normal_seconds, paused_seconds
        self._lock = threading.RLock()
        self._state = 'UNKNOWN'
        self._blocked = False
        self._next_probe = 0.
        self._probing = False
        self._generation = 0

    @property
    def blocked(self):
        with self._lock:
            return self._blocked

    @property
    def entries_allowed(self):
        with self._lock:
            return self._state == 'OPEN' and not self._blocked

    def seconds_until_probe(self):
        with self._lock:
            return max(0., self._next_probe - self.clock())

    def require_available(self):
        with self._lock:
            if self._blocked:
                raise ExchangeMaintenance(self._state, not_sent=True)

    def observe(self, response):
        state = maintenance_reason(response)
        if state is None:
            return
        with self._lock:
            if not self._blocked or self._state != state:
                self.logger.warning('Exchange %s confirmed; preserving positions and orders; status probe every %ss', state, self.paused_seconds)
            self._blocked, self._state = True, state
            self._generation += 1
            self._next_probe = self.clock() + self.paused_seconds
        raise ExchangeMaintenance(state)  # A received response never proves a POST was unsent.

    def poll(self, fetch_status):
        with self._lock:
            now = self.clock()
            if self._probing or now < self._next_probe:
                return self.entries_allowed
            self._probing = True
            generation = self._generation
            self._next_probe = now + (self.paused_seconds if self._blocked else self.normal_seconds)
        try:
            response = fetch_status()
            state = maintenance_reason(response)
            if state:
                self.observe(response)
            if not (isinstance(response, dict) and type(response.get('status')) is int and response.get('status') == 0
                    and isinstance(response.get('data'), dict) and response['data'].get('status') == 'OPEN'):
                raise ValueError('Invalid status response')
            with self._lock:
                if generation != self._generation:
                    return False  # A newer private maintenance observation wins over this OPEN.
                was_blocked = self._blocked
                self._blocked, self._state = False, 'OPEN'
                self._next_probe = self.clock() + self.normal_seconds
                if was_blocked:
                    self.logger.info('Exchange OPEN confirmed; reconcile orders and positions before new entries')
                return True
        except ExchangeMaintenance:
            return False
        except Exception as exc:
            with self._lock:
                if not self._blocked:
                    self._state = 'UNKNOWN'
                self._next_probe = self.clock() + (self.paused_seconds if self._blocked else self.normal_seconds)
            self.logger.error('Exchange status probe failed (%s); new entries withheld', type(exc).__name__)
            return False
        finally:
            with self._lock:
                self._probing = False
