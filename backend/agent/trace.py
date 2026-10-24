"""
Harness events describe what the agent loop is doing while it answers one user
message: the turn starting, context being built, the model being called, tools running
(and waiting for approval), the answer, the memory save, and the turn finishing. The
loop creates a TurnTrace per turn and emits events through it; an observer (such as the
websocket that feeds the frontend) receives each event as it happens. Long text is
clipped so events stay small, and observer failures are logged rather than raised so
watching a turn can never break it. Storage reads and writes are reported through
report_storage, which attributes them to the turn whose trace is active in the current
async context.
"""

import logging
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

# Longest text carried in an event. The full text stays in the session.
MAX_TEXT_CHARS = 2_000


class EventType(StrEnum):
    TURN_STARTED = "turn_started"
    CONTEXT_STARTED = "context_started"
    CONTEXT_BUILT = "context_built"
    MODEL_STARTED = "model_started"
    MODEL_FINISHED = "model_finished"
    TOOL_STARTED = "tool_started"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RESOLVED = "approval_resolved"
    TOOL_FINISHED = "tool_finished"
    ANSWER = "answer"
    MEMORY_STARTED = "memory_started"
    MEMORY_FINISHED = "memory_finished"
    STORAGE_ACCESSED = "storage_accessed"
    TURN_FINISHED = "turn_finished"


class TurnOutcome(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class HarnessEvent:
    """One thing that happened while the agent handled a user message."""

    type: EventType
    turn_id: str
    session_id: str
    # The loop step (1-based) in progress when the event happened, or None before the first step.
    step: int | None
    # Seconds since the epoch.
    time: float
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the event to a JSON-ready dict.

        Returns:
            The event's fields, with the type as a plain string.
        """
        return {
            "type": str(self.type),
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "step": self.step,
            "time": self.time,
            "data": self.data,
        }


HarnessObserver = Callable[[HarnessEvent], None]

# The trace of the turn doing work in the current async context, if any.
_active_trace: ContextVar["TurnTrace | None"] = ContextVar("active_trace", default=None)


class TurnTrace:
    """Emits the events of one turn to an observer, tagging each with the current step."""

    def __init__(self, observer: HarnessObserver | None, session_id: str):
        """
        Args:
            observer: Receives each event. None means nobody is watching.
            session_id: The session the turn belongs to.
        """
        self.observer = observer
        self.session_id = session_id
        self.turn_id = uuid.uuid4().hex
        self.step: int | None = None

    def emit(self, event_type: EventType, data: dict[str, Any] | None = None) -> None:
        """
        Send an event to the observer, logging instead of raising if it fails.

        Args:
            event_type: What happened.
            data: Details of the event. Must be JSON serializable.
        """
        if self.observer is None:
            return
        if data is None:
            data = {}

        event = HarnessEvent(event_type, self.turn_id, self.session_id, self.step, time.time(), data)
        try:
            self.observer(event)
        except Exception:
            logger.exception("Harness observer failed on %s", event_type)

    @contextmanager
    def active(self) -> Iterator[None]:
        """
        Attribute storage accesses made inside the block to this turn. The block must not
        yield from the agent's generator, so the context is entered and left together.

        Returns:
            A context manager.
        """
        token = _active_trace.set(self)
        try:
            yield
        finally:
            _active_trace.reset(token)


def report_storage(operation: str) -> None:
    """
    Emit a storage event on the active turn. Accesses outside a turn are ignored.

    Args:
        operation: "read" or "write".
    """
    trace = _active_trace.get()
    if trace is not None:
        trace.emit(EventType.STORAGE_ACCESSED, {"operation": operation})


def clip(text: str) -> str:
    """
    Shorten text to MAX_TEXT_CHARS for an event.

    Args:
        text: The text to shorten.
    Returns:
        The text, cut with a note if it was too long.
    """
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return text[:MAX_TEXT_CHARS] + f" ... [{len(text) - MAX_TEXT_CHARS} more characters]"
