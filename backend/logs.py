"""
Logging for the whole backend, written to one file (jarvis.log). Every line carries a
trace, like "alexa:3f9a1c2b", naming the interface a request came in on and a short id
for that request, so one request can be followed from the AWS Lambdas through the relay,
the agent loop, the model and the tools by searching for its id. A gateway starts a
trace with start_trace when a request arrives; asyncio tasks copy the current trace, so
everything done for the request is tagged with it. Lines logged outside any request show
"-" as the trace.
"""

import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

LINE_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s [%(trace)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
NO_TRACE = "-"
# Longest text quoted in one log line.
MAX_SNIPPET_CHARS = 300

_trace: ContextVar[str] = ContextVar("trace", default=NO_TRACE)


class TraceFilter(logging.Filter):
    """Tags each record with the current trace, unless it already carries one."""

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add the trace to a record.

        Args:
            record: The record being written.
        Returns:
            True, so the record is always written.
        """
        if not hasattr(record, "trace"):
            record.trace = _trace.get()
        return True


def setup_logging(path: Path, level: int = logging.INFO) -> None:
    """
    Send every log record to one file, tagged with its trace.

    Args:
        path: The log file, appended to.
        level: The lowest level written.
    """
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LINE_FORMAT, DATE_FORMAT))
    handler.addFilter(TraceFilter())

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)


def start_trace(source: str, trace_id: str | None = None) -> str:
    """
    Tag everything logged from here on in the current async context with a new trace.

    Args:
        source: The interface the request came in on, like "terminal" or "alexa".
        trace_id: An id the request already has, such as one the Lambda chose. A new one
            is made when None.
    Returns:
        The trace, like "alexa:3f9a1c2b".
    """
    if trace_id is None or trace_id == "":
        trace_id = new_trace_id()
    trace = f"{source}:{trace_id}"
    _trace.set(trace)
    return trace


@contextmanager
def traced(source: str) -> Iterator[str]:
    """
    Tag everything logged inside the block with a new trace, then restore the old one.

    Args:
        source: The interface the request came in on.
    Returns:
        A context manager that gives the trace.
    """
    trace = f"{source}:{new_trace_id()}"
    token = _trace.set(trace)
    try:
        yield trace
    finally:
        _trace.reset(token)


def new_trace_id() -> str:
    """
    Make a short random id for a request.

    Returns:
        Eight hex characters.
    """
    return uuid.uuid4().hex[:8]


def snippet(text: str) -> str:
    """
    Shorten text to one line for a log message.

    Args:
        text: The text to quote.
    Returns:
        The text on one line, cut to MAX_SNIPPET_CHARS with a note of the full length.
    """
    one_line = " ".join(text.split())
    if len(one_line) <= MAX_SNIPPET_CHARS:
        return repr(one_line)
    return repr(one_line[:MAX_SNIPPET_CHARS]) + f" ... ({len(text)} chars)"
