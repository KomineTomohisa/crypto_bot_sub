"""Append-only observations. No credentials, API access, orders, or DB connections."""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import queue
import threading
import time
import uuid

import numpy as np
import pandas as pd


SCHEMA = "strategy_observations_v1"
SOURCE_FILES = (
    "crypto_bot_sub.py",
    "signal_scoring.py",
    "strategy_decisions.py",
    "strategy_inputs.py",
    "strategy_recording.py",
    "live_execution.py",
    "close_alerts.py",
    "trade_notifications.py",
    "exchange_availability.py",
    "notification_outbox.py",
    "fee_sync.py",
    "exit_protection.py",
    "market_quotes.py",
    "trading_execution.py",
    "execution_store.py",
    "backtest_data.py",
    "db.py",
)
FORBIDDEN_KEYS = {
    "api_key",
    "api_secret",
    "authorization",
    "password",
    "passphrase",
    "token",
    "headers",
}


class RecordingError(RuntimeError):
    pass


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, pd.DataFrame):
        return {
            "columns": list(value.columns),
            "rows": json_value(value.to_dict("records")),
        }
    if isinstance(value, dict):
        if any(str(key).lower() in FORBIDDEN_KEYS for key in value):
            raise RecordingError("Sensitive field refused by observation recorder")
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)):
        return value
    raise RecordingError(f"Unsupported observation value: {type(value).__name__}")


def canonical(value):
    return json.dumps(
        json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def code_fingerprint(directory=None):
    root = Path(directory) if directory else Path(__file__).resolve().parent
    return digest(
        {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES
        }
    )


def runtime_versions():
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def signal_config(bot):
    # Explicit fields only. Never serialize exchange_settings_gmo or __dict__.
    return json_value(
        {
            "live_trade": bool(
                getattr(bot, "exchange_settings_gmo", {}).get("live_trade", False)
            ),
            "is_backtest": bool(getattr(bot, "is_backtest", False)),
            "entry_thresholds": getattr(bot, "entry_thresholds", {}),
            "trade_size": getattr(bot, "TRADE_SIZE", None),
            "initial_capital": getattr(bot, "initial_capital", None),
            "short_lock_trigger": getattr(bot, "SHORT_PROFIT_LOCK_TRIGGER_PCT", 0.75),
            "short_lock_level": getattr(bot, "SHORT_PROFIT_LOCK_LEVEL_PCT", 0.25),
        }
    )


class ObservationJournal:
    def __init__(
        self, directory, metadata, *, session=None, path=None, received_start=None
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.blobs = self.directory / "blobs"
        self.blobs.mkdir(exist_ok=True)
        self.session = session or uuid.uuid4().hex
        self.day = datetime.now(timezone.utc).date()
        self.path = (
            Path(path)
            if path
            else (
                self.directory
                / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{self.session}.jsonl"
            )
        )
        self.sequence = 0
        self.previous = None
        self.failed = False
        self.closed = False
        self.lock = threading.RLock()
        self.append("session_start", metadata, received=received_start)

    def blob(self, value):
        payload = canonical(value)
        identity = hashlib.sha256(payload).hexdigest()
        path = self.blobs / f"{identity}.json.gz"
        temporary = self.blobs / f".{identity}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as stream:
                stream.write(gzip.compress(payload, mtime=0))
                stream.flush()
                os.fsync(stream.fileno())
            # A reader sees either the old complete blob or the new complete blob.
            # Replacing a hash-addressed blob with the same verified payload also
            # repairs a previous interrupted write without changing its identity.
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return identity

    def append(self, kind, payload, observed_at=None, *, received=None):
        with self.lock:
            if self.closed:
                raise RecordingError("Observation journal is closed")
            self.sequence += 1
            event = {
                "schema": SCHEMA,
                "session": self.session,
                "sequence": self.sequence,
                "previous": self.previous,
                "kind": kind,
                "received_at": received[0]
                if received
                else datetime.now(timezone.utc).isoformat(),
                "received_monotonic_ns": received[1]
                if received
                else time.monotonic_ns(),
                "observed_at": json_value(observed_at),
                "payload": json_value(payload),
            }
            event["hash"] = digest(event)
            try:
                with self.path.open("ab") as stream:
                    stream.write(canonical(event) + b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception:
                self.failed = True
                raise
            self.previous = event["hash"]
            return event["hash"]

    def close(self, summary=None):
        with self.lock:
            if not self.closed:
                self.append(
                    "session_end", {"complete": not self.failed, **(summary or {})}
                )
                self.closed = True


class BackgroundJournal:
    """Bounded memory queue; trading threads never wait for filesystem writes."""

    def __init__(self, directory, metadata, logger=None, *, capacity=128):
        self.directory = Path(directory)
        self.session = uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        self.day = now.date()
        self.path = self.directory / f"{now:%Y%m%dT%H%M%S}_{self.session}.jsonl"
        self.received_start = (now.isoformat(), time.monotonic_ns())
        self.metadata = json_value(metadata)
        self.logger = logger or logging.getLogger(__name__)
        self.failed = False
        self.closed = False
        self.dropped = 0
        self._journal = None
        self._lock = threading.RLock()
        self._queue = queue.Queue(maxsize=capacity)
        self._thread = threading.Thread(
            target=self._write_loop, name="strategy-input-recorder", daemon=True
        )
        self._thread.start()

    @property
    def previous(self):
        return self._journal.previous if self._journal else None

    def _enqueue(self, kind, payload, observed_at=None, blob=None):
        with self._lock:
            if self.closed or not self._thread.is_alive():
                raise RecordingError("Background observation writer is unavailable")
            record_id = f"{self.session}:{uuid.uuid4().hex}"
            value = json_value({**payload, "record_id": record_id})
            received = (datetime.now(timezone.utc).isoformat(), time.monotonic_ns())
            try:
                self._queue.put_nowait(
                    (kind, value, json_value(observed_at), received, blob)
                )
            except queue.Full:
                self.failed = True
                self.dropped += 1
                raise RecordingError(
                    "Observation queue full; session is incomplete"
                ) from None
            return record_id

    def append(self, kind, payload, observed_at=None):
        return self._enqueue(kind, payload, observed_at)

    def append_snapshot(self, payload, inputs, observed_at):
        frozen = json_value(inputs)
        return self._enqueue(
            "feature_snapshot",
            {**payload, "inputs": digest(frozen)},
            observed_at,
            frozen,
        )

    def _write_loop(self):
        try:
            self._journal = ObservationJournal(
                self.directory,
                self.metadata,
                session=self.session,
                path=self.path,
                received_start=self.received_start,
            )
            while True:
                try:
                    job = self._queue.get(timeout=0.02)
                except queue.Empty:
                    if self.closed:
                        break
                    continue
                try:
                    if job[0] == "__barrier__":
                        job[1].set()
                        continue
                    kind, payload, observed_at, received, blob = job
                    if blob is not None:
                        identity = self._journal.blob(blob)
                        if identity != payload["inputs"]:
                            raise RecordingError("Snapshot identity changed")
                    self._journal.append(kind, payload, observed_at, received=received)
                except Exception as exc:
                    self.failed = True
                    self.logger.error(
                        "Background input write failed (%s)", type(exc).__name__
                    )
                finally:
                    self._queue.task_done()
            self._journal.failed = self._journal.failed or self.failed
            self._journal.close({"dropped_events": self.dropped})
        except Exception as exc:
            self.failed = True
            self.logger.error(
                "Background input recorder stopped (%s)", type(exc).__name__
            )

    def flush(self, timeout=5):
        barrier = threading.Event()
        try:
            self._queue.put(("__barrier__", barrier), timeout=timeout)
        except queue.Full:
            raise RecordingError("Observation flush timed out") from None
        if not barrier.wait(timeout):
            raise RecordingError("Observation flush timed out")

    def request_close(self):
        with self._lock:
            self.closed = True

    def close(self, timeout=2):
        self.request_close()
        self._thread.join(timeout)
        if self._thread.is_alive():
            self.failed = True
            return False
        return not self.failed


def read_journal(path, *, require_complete=True):
    events = []
    previous = None
    session = None
    last_monotonic = -1
    ended = False
    try:
        with Path(path).open("rb") as stream:
            for sequence, line in enumerate(stream, 1):
                if not line.endswith(b"\n"):
                    raise RecordingError("Truncated observation record")
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise RecordingError("Observation record must be an object")
                actual_hash = event.pop("hash")
                if actual_hash != digest(event):
                    raise RecordingError("Observation hash mismatch")
                if (
                    event["schema"] != SCHEMA
                    or event["sequence"] != sequence
                    or event["previous"] != previous
                ):
                    raise RecordingError("Observation sequence/chain mismatch")
                if session is not None and session != event["session"]:
                    raise RecordingError("Mixed observation sessions")
                if event["received_monotonic_ns"] < last_monotonic or ended:
                    raise RecordingError("Observation ordering mismatch")
                if sequence == 1 and event["kind"] != "session_start":
                    raise RecordingError("Missing observation session header")
                session = event["session"]
                previous = actual_hash
                last_monotonic = event["received_monotonic_ns"]
                ended = event["kind"] == "session_end"
                event["hash"] = actual_hash
                events.append(event)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RecordingError(
            f"Invalid observation journal ({type(exc).__name__})"
        ) from None
    if not events or (
        require_complete and (not ended or not events[-1]["payload"].get("complete"))
    ):
        raise RecordingError("Observation session is incomplete")
    return events


def read_blob(journal_path, identity):
    if (
        not isinstance(identity, str)
        or len(identity) != 64
        or any(c not in "0123456789abcdef" for c in identity)
    ):
        raise RecordingError("Invalid observation blob identity")
    try:
        payload = gzip.decompress(
            (Path(journal_path).parent / "blobs" / f"{identity}.json.gz").read_bytes()
        )
        if hashlib.sha256(payload).hexdigest() != identity:
            raise RecordingError("Observation blob hash mismatch")
        return json.loads(payload)
    except (OSError, ValueError, EOFError):
        raise RecordingError("Missing/corrupt observation blob") from None


def recorder_for(bot):
    if not getattr(bot, "_record_inputs", False) or getattr(bot, "is_backtest", False):
        return None
    old = getattr(bot, "_input_recorder", None)
    rotate = (
        old is not None
        and getattr(old, "day", datetime.now(timezone.utc).date())
        != datetime.now(timezone.utc).date()
    )
    if old is None or getattr(old, "closed", False) or rotate:
        if rotate and not old.closed:
            if isinstance(old, BackgroundJournal):
                old.request_close()
            else:
                old.close()
        metadata = {
            "code_fingerprint": code_fingerprint(),
            "signal_config": signal_config(bot),
            "runtime_versions": runtime_versions(),
        }
        if old is not None and getattr(old, "session", None):
            metadata.update(
                previous_session=old.session, previous_final_hash=old.previous
            )
        bot._input_recorder = BackgroundJournal(
            Path(bot.log_dir) / "input_events", metadata, bot.logger
        )
        if old is None and getattr(bot, "_input_recording_failed", False):
            bot._input_recorder.failed = True
    return bot._input_recorder


def record_bot_event(bot, kind, payload, observed_at=None):
    try:
        recorder = recorder_for(bot)
        if recorder and isinstance(payload, dict) and "symbol" in payload:
            payload = {
                **payload,
                "feature_snapshot": payload.get("feature_snapshot", getattr(bot, "_feature_event_refs", {}).get(
                    payload["symbol"]
                )),
            }
        return recorder.append(kind, payload, observed_at) if recorder else None
    except Exception as exc:
        recorder = getattr(bot, "_input_recorder", None)
        if recorder:
            recorder.failed = True
        bot._input_recording_failed = True
        bot.logger.error(
            "Input recording failed (%s); session cannot be certified complete",
            type(exc).__name__,
        )
        return None


def record_feature_snapshot(
    bot, symbol, observed_at, raw15, raw60, signals, hourly_features
):
    try:
        recorder = recorder_for(bot)
        if recorder is None:
            return None
        if hasattr(bot, "_feature_event_refs"):
            bot._feature_event_refs.pop(symbol, None)
        from strategy_inputs import feature_history_spec, history_quality

        inputs = {"intraday": json_value(raw15), "hourly": json_value(raw60)}
        payload = {
            "symbol": symbol,
            "config": signal_config(bot),
            "history_spec": feature_history_spec(),
            "quality": {
                "intraday": history_quality(raw15, observed_at, 15),
                "hourly": history_quality(raw60, observed_at, 60),
            },
            "signals_hash": digest(signals),
            "hourly_features_hash": digest(hourly_features),
        }
        if isinstance(recorder, BackgroundJournal):
            event_id = recorder.append_snapshot(payload, inputs, observed_at)
        else:
            event_id = recorder.append(
                "feature_snapshot",
                {**payload, "inputs": recorder.blob(inputs)},
                observed_at,
            )
        if not hasattr(bot, "_feature_event_refs"):
            bot._feature_event_refs = {}
        bot._feature_event_refs[symbol] = event_id
        return event_id
    except Exception as exc:
        recorder = getattr(bot, "_input_recorder", None)
        if recorder:
            recorder.failed = True
        bot._input_recording_failed = True
        bot.logger.error("Feature snapshot recording failed (%s)", type(exc).__name__)
        return None


def close_recorder(bot):
    recorder = getattr(bot, "_input_recorder", None)
    if recorder:
        try:
            finished = recorder.close()
            if finished is False or recorder.failed:
                bot._input_recording_failed = True
        except Exception as exc:
            bot._input_recording_failed = True
            bot.logger.error("Input recording close failed (%s)", type(exc).__name__)
        finally:
            bot._record_inputs = False
