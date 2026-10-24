"""
Builds the message list sent to the LLM on each step: a system prompt (including
remembered facts, the skill index, channel instructions and the summary of compacted
history) followed by the
session's recent messages. When recent
messages exceed the budget, the oldest ones are summarized by the LLM and compacted
in the session, keeping roughly `keep_chars` of the newest messages verbatim. The
compaction boundary never lands on a tool result, so tool calls and their results are
always summarized or kept together.
"""

from llm.base import LLM
from memory.memory import Memory

from .models import Message, Role
from .prompts import COMPACTION_REQUEST, build_compaction_prompt, build_system_prompt
from .session import Session

# A resized image costs the model roughly as much as this many characters of text.
IMAGE_SIZE_CHARS = 4_000


class ContextBuilder:
    """Builds model input from a session, compacting old history when it gets too large."""

    def __init__(
        self,
        llm: LLM,
        memory: Memory | None = None,
        max_chars: int = 400_000,
        keep_chars: int | None = None,
    ):
        """
        Args:
            llm: Model used to write summaries.
            memory: Long-term memory whose facts and skill index go in the system prompt.
            max_chars: History size in characters (about 4 per token) that triggers compaction.
            keep_chars: Approximate size of the newest history kept verbatim after compacting.
                Defaults to a quarter of max_chars.
        """
        if keep_chars is None:
            keep_chars = max_chars // 4

        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if keep_chars <= 0 or keep_chars >= max_chars:
            raise ValueError("keep_chars must be positive and smaller than max_chars")

        self.llm = llm
        self.memory = memory
        self.max_chars = max_chars
        self.keep_chars = keep_chars

    async def build(self, session: Session, channel_instructions: str = "") -> list[Message]:
        """
        Build the model input, compacting the session first if it is over budget.

        Args:
            session: The conversation to build context from. May be compacted in place.
            channel_instructions: Instructions for the channel the user is on, if any.
        Returns:
            The system prompt followed by the recent history.
        """
        if _total_size(session.recent_messages) > self.max_chars:
            await self._compact(session)

        facts = []
        skills = []
        if self.memory is not None:
            facts = self.memory.facts.all()
            skills = self.memory.skills.all()

        system_prompt = build_system_prompt(session.summary, facts, skills, channel_instructions)
        return [Message.system(system_prompt)] + list(session.recent_messages)

    async def _compact(self, session: Session) -> None:
        """
        Summarize the oldest recent messages and mark them as compacted in the session.

        Args:
            session: The conversation to compact.
        """
        recent = session.recent_messages
        cut = self._find_cut(recent)
        if cut == 0:
            # Nothing can be compacted without splitting a tool call from its results.
            return

        old_messages = recent[:cut]
        summary = await self._summarize(session.summary, old_messages)
        session.compact(summary, cut)

    def _find_cut(self, messages: tuple[Message, ...]) -> int:
        """
        Choose where to split messages into a compacted part and a kept part.

        Args:
            messages: The recent messages, oldest first.
        Returns:
            Number of oldest messages to compact. Zero means nothing can be compacted.
        """
        # Walk backward from the newest message, keeping messages while they fit.
        cut = len(messages)
        kept_size = 0
        while cut > 0:
            size = _size(messages[cut - 1])
            if kept_size + size > self.keep_chars:
                break
            kept_size += size
            cut -= 1

        # Always keep at least the newest message.
        last_index = len(messages) - 1
        if cut > last_index:
            cut = last_index

        # The kept part must not start on a tool result, so keep its assistant message too.
        while cut > 0 and messages[cut].role == Role.TOOL:
            cut -= 1
        return cut

    async def _summarize(self, previous_summary: str | None, messages: tuple[Message, ...]) -> str:
        """
        Ask the LLM to summarize messages, merging in any previous summary.

        Args:
            previous_summary: The existing summary, if any.
            messages: The messages to summarize, oldest first.
        Returns:
            The new summary text.
        """
        system_message = Message.system(build_compaction_prompt(previous_summary))
        request_message = Message.user(COMPACTION_REQUEST)
        compaction_input = [system_message] + list(messages) + [request_message]

        no_tools = []
        reply = await self.llm.complete(compaction_input, no_tools)

        summary = reply.content.strip()
        if summary == "":
            raise RuntimeError("Compaction failed: the model returned an empty summary")
        return summary


def _total_size(messages: tuple[Message, ...]) -> int:
    """
    Approximate the combined size of several messages.

    Args:
        messages: The messages to measure.
    Returns:
        Total character count.
    """
    total = 0
    for message in messages:
        total += _size(message)
    return total


def _size(message: Message) -> int:
    """
    Approximate a message's size in characters.

    Args:
        message: The message to measure.
    Returns:
        Character count of its content and tool call payloads (hosted calls with their
        outputs included), with each image counted
        as IMAGE_SIZE_CHARS.
    """
    size = len(message.content)
    size += len(message.images) * IMAGE_SIZE_CHARS
    for call in message.tool_calls:
        size += len(call.name)
        size += len(call.arguments)
    for hosted in message.hosted_calls:
        size += len(hosted.name)
        size += len(hosted.arguments)
        size += len(hosted.output)
    return size
