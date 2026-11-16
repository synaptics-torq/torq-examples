from contextlib import contextmanager
import os
import time
import threading


_SUPPRESS_START_MARKER = 'TORQ_HEARTBEAT_SUPPRESS_START'
_SUPPRESS_END_MARKER = 'TORQ_HEARTBEAT_SUPPRESS_END'
_CONTROL_ENV_VAR = 'TORQ_HEARTBEAT_CONTROL'


class Heartbeat:
    """Emit heartbeat messages to prevent timeout detection.
    By default, emits a heartbeat every `interval` seconds.
    If a condition function is provided, only emits when the condition returns True.
    Usage (simple periodic):
        heartbeat = PeriodicHeartbeat(interval=60, on_heartbeat=callback)
        heartbeat.start()
        # ... long-running operation ...
        heartbeat.stop()
    Usage (activity-aware):
        def should_emit():
            # Check if activity has stopped
            return (now - last_output >= interval)
        heartbeat = PeriodicHeartbeat(interval=60, on_heartbeat=callback, should_emit=should_emit)
        heartbeat.start()
    """
    def __init__(self, interval=60, on_heartbeat=None, should_emit=None):
        """Initialize heartbeat monitor.
        Args:
            interval: Seconds between heartbeats (default: 60)
            on_heartbeat: Optional callback(elapsed_seconds) called on each heartbeat
            should_emit:  Optional condition function() that returns bool.
                          If provided, heartbeat is only emitted when it returns True.
                          If None, heartbeat is emitted every interval seconds.
        """
        self._interval = interval
        self._on_heartbeat = on_heartbeat
        self._should_emit = should_emit
        self._start_time = None
        self._stop_event = None
        self._thread = None

    def start(self):
        """Start the heartbeat monitor."""
        if self._thread is not None:
            return

        self._start_time = time.time()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the heartbeat monitor."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1)

    def _monitor(self):
        """Monitor loop that emits heartbeats."""
        last_heartbeat = self._start_time
        while not self._stop_event.is_set():
            now = time.time()
            elapsed = now - self._start_time
            # Check if it's time to emit
            should_emit_now = (now - last_heartbeat >= self._interval)
            # If a condition was provided, apply it as an additional filter
            if should_emit_now and self._should_emit is not None:
                should_emit_now = self._should_emit()
            if should_emit_now:
                if self._on_heartbeat:
                    self._on_heartbeat(elapsed)
                last_heartbeat = now
            time.sleep(1)


@contextmanager
def suppress_heartbeat():
    enabled = os.getenv(_CONTROL_ENV_VAR) == '1'
    if enabled:
        print(_SUPPRESS_START_MARKER, flush=True)
    try:
        yield
    finally:
        if enabled:
            print(_SUPPRESS_END_MARKER, flush=True)


def make_heartbeat_callback(logger=None, log_file=None, label='HEARTBEAT'):
    """Create a heartbeat callback for Heartbeat class.
    Callback signature depends on how the Heartbeat will use it:
    - If Heartbeat emits with elapsed time, callback receives (elapsed) argument
    - If Heartbeat emits without time, callback receives no arguments
    Args:
        logger: Optional logger object (for logging via logger.info)
        log_file: Optional file handle (for writing log lines)
        label: Label to prepend in message (default: 'HEARTBEAT')
    Returns:
        Callback function that handles both (elapsed) and () signatures
    """
    def callback(elapsed=None):
        if elapsed is not None:
            msg = f'[{label}] Still running... ({elapsed:.0f}s elapsed)'
        else:
            msg = f'[{label}] Command still running...'

        if logger is not None:
            logger.info(msg)
        else:
            print(msg, flush=True)

        if log_file is not None:
            log_file.write(msg + '\n')
            log_file.flush()

    return callback


