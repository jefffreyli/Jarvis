"""
A Session holds the message history for one conversation with the user. The agent
loop appends to it and the context builder reads from it. The full history is kept
append-only so that tool calls and their results always stay paired. When the context
fills up, older messages are compacted: a summary replaces them in the model's view,
but the original messages stay in `messages` for display and storage.
"""

import uuid
from datetime import datetime, timezone

from .models import Message


class Session:
    """An append-only conversation history with an optional summary of older messages."""

    def __init__(self, session_id: str | None = None, messages: list[Message] | None = None):
        """
        Create a session, optionally restoring existing history.

        Args:
            session_id: Stable id for the session. A new one is generated if omitted.
            messages: Prior history to restore.
        """
        if session_id is None:
            session_id = uuid.uuid4().hex
        if messages is None:
            messages = []

        self.id = session_id
        self.created_at = datetime.now(timezone.utc)
        self._messages: list[Message] = list(messages)
        self._summary: str | None = None
        # Index of the first message not covered by the summary.
        self._summary_end = 0

    @property
    def messages(self) -> tuple[Message, ...]:
        """
        Read-only view of the full history, including compacted messages.

        Returns:
            All messages in order.
        """
        return tuple(self._messages)

    @property
    def summary(self) -> str | None:
        """
        Summary of the compacted messages.

        Returns:
            The summary text, or None if nothing has been compacted yet.
        """
        return self._summary

    @property
    def recent_messages(self) -> tuple[Message, ...]:
        """
        Messages that are not covered by the summary.

        Returns:
            The uncompacted messages in order.
        """
        return tuple(self._messages[self._summary_end:])

    def add(self, *messages: Message) -> None:
        """
        Append messages to the history.

        Args:
            messages: Messages to append, in order.
        """
        for message in messages:
            self._messages.append(message)

    def compact(self, summary: str, count: int) -> None:
        """
        Replace the oldest recent messages with a summary in the model's view.

        Args:
            summary: Summary covering the previous summary plus the compacted messages.
            count: How many of the oldest recent messages the summary now covers.
        """
        recent_count = len(self._messages) - self._summary_end
        if count <= 0 or count > recent_count:
            raise ValueError(f"Cannot compact {count} of {recent_count} recent messages")

        self._summary = summary
        self._summary_end += count
