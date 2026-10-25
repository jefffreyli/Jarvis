"""
Forwards the relay Lambdas' log records to the Jarvis backend, so they land in
jarvis.log next to the backend's own lines. An invocation calls begin() with the trace
for its request (like "alexa:3f9a1c2b", the same trace the backend logs under), and
every record from a "jarvis.*" logger until flush() is kept, besides going to CloudWatch
as usual. flush() sends the kept records down the relay websocket in one "relay.log"
message. Forwarding is best effort: when the backend is offline the records are only in
CloudWatch.
"""

import logging
import time
import traceback

import store

LOGGER_PREFIX = "jarvis."
# Keeps one relay.log message under API Gateway's 128 KB limit.
MAX_RECORDS = 40
MAX_MESSAGE_CHARS = 2000
MAX_SNIPPET_CHARS = 300


class _Collector(logging.Handler):
    """Keeps the records logged during one invocation."""

    def __init__(self):
        super().__init__()
        # None while no invocation is collecting.
        self.trace: str | None = None
        self.records: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        """
        Keep a record for forwarding, if an invocation is collecting.

        Args:
            record: The record being logged.
        """
        if self.trace is None or len(self.records) >= MAX_RECORDS:
            return

        message = record.getMessage()
        if record.exc_info is not None:
            message += "\n" + "".join(traceback.format_exception(*record.exc_info))
        self.records.append(
            {
                "time": record.created,
                "level": record.levelname,
                "logger": record.name.removeprefix(LOGGER_PREFIX),
                "trace": self.trace,
                "message": message[:MAX_MESSAGE_CHARS],
            }
        )


_collector = _Collector()
_root = logging.getLogger("jarvis")
_root.setLevel(logging.INFO)
_root.addHandler(_collector)


def begin(trace: str) -> None:
    """
    Start keeping records for an invocation.

    Args:
        trace: The request's trace, like "alexa:3f9a1c2b".
    """
    _collector.trace = trace
    _collector.records = []


def flush() -> None:
    """Send the records kept since begin() to the backend, and stop keeping records."""
    records = _collector.records
    _collector.trace = None
    _collector.records = []
    if len(records) == 0:
        return

    try:
        store.send_to_backend({"type": "relay.log", "records": records})
    except store.BackendOffline:
        pass
    except Exception:
        # Logged after collecting stopped, so this only reaches CloudWatch.
        _root.exception("Forwarding log records to the backend failed")


def gateway_delay_ms(event: dict) -> int | None:
    """
    Measure how long ago API Gateway received the request that started this invocation.

    Args:
        event: The event from API Gateway.
    Returns:
        Milliseconds since API Gateway received the request, or None if the event has no time.
    """
    context = event.get("requestContext") or {}
    # WebSocket APIs call it requestTimeEpoch, HTTP APIs timeEpoch.
    received = context.get("requestTimeEpoch")
    if received is None:
        received = context.get("timeEpoch")
    if received is None:
        return None
    return int(time.time() * 1000 - received)


def snippet(text: str) -> str:
    """
    Shorten text to one line for a log message.

    Args:
        text: The text to quote.
    Returns:
        The text on one line, cut to MAX_SNIPPET_CHARS.
    """
    one_line = " ".join(text.split())
    if len(one_line) <= MAX_SNIPPET_CHARS:
        return repr(one_line)
    return repr(one_line[:MAX_SNIPPET_CHARS]) + f" ... ({len(text)} chars)"
