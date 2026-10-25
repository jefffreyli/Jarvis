"""
The core agent loop. Given a user message, it repeatedly asks the LLM for a reply,
runs any tools the reply requests, and feeds the results back until the model answers
without tool calls or the step limit is reached. Each new message is yielded as it is
produced so callers (terminal, voice, text) can stream progress. Each run takes the
Channel the message came from, which says how to ask the user for approval and how
replies should be written. An assistant turn and its
tool results are committed to the session together, so a cancelled run never leaves
unanswered tool calls in history. With long-term memory, each finished turn (the user's
message and the final answer) is saved as an episode after the answer is yielded.
An optional observer receives a HarnessEvent for every stage of the turn, so a UI can
follow the run live, and each turn, model reply, tool call and approval is also logged.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import logs
from llm.base import LLM
from memory.memory import Memory
from tools.registry import ApprovalCallback, ToolRegistry

from .context import ContextBuilder
from .models import Message, ToolCall
from .session import Session
from .trace import EventType, HarnessObserver, TurnOutcome, TurnTrace, clip

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Channel:
    """Where a user message came from, such as the terminal, voice or a text message."""

    # Asks the user whether a tool call may run. None denies every call that needs approval.
    approve: ApprovalCallback | None = None
    # Extra system prompt instructions for this channel, such as how to format replies.
    instructions: str = ""
    # Short name shown when following the harness, such as "terminal" or "text".
    name: str = ""


class AgentLoop:
    """Drives the think, act, observe cycle for one session at a time."""

    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry,
        memory: Memory | None = None,
        context: ContextBuilder | None = None,
        max_steps: int = 20,
        observer: HarnessObserver | None = None,
    ):
        """
        Args:
            llm: Model used to generate replies.
            tools: Tools the LLM can call.
            memory: Long-term memory. Each finished turn is saved to it as an episode.
            context: Builds model input from the session. Defaults to one using memory.
            max_steps: Maximum LLM calls per user message.
            observer: Receives a HarnessEvent for each stage of every turn.
        """
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if context is None:
            context = ContextBuilder(llm, memory)

        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.context = context
        self.max_steps = max_steps
        self.observer = observer

    async def run(self, session: Session, user_input: str, channel: Channel) -> AsyncIterator[Message]:
        """
        Handle one user message, yielding each new message as it is produced.

        Args:
            session: Conversation to read from and append to.
            user_input: The user's message text.
            channel: Where the message came from.
        Returns:
            An async iterator of assistant and tool messages, ending with the final answer.
        """
        trace = TurnTrace(self.observer, session.id)
        started = time.monotonic()
        logger.info(
            "Turn %s started on %s in session %s: %s",
            trace.turn_id[:8],
            channel.name,
            session.id[:8],
            logs.snippet(user_input),
        )
        trace.emit(
            EventType.TURN_STARTED,
            {
                "input": clip(user_input),
                "channel": channel.name,
                "max_steps": self.max_steps,
                "memory": self.memory is not None,
            },
        )

        try:
            async for message in self._run_steps(session, user_input, channel, trace):
                yield message
        except (asyncio.CancelledError, GeneratorExit):
            logger.info("Turn %s cancelled after %.2fs", trace.turn_id[:8], time.monotonic() - started)
            trace.emit(EventType.TURN_FINISHED, {"outcome": TurnOutcome.CANCELLED})
            raise
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            logger.error("Turn %s failed after %.2fs: %s", trace.turn_id[:8], time.monotonic() - started, error)
            trace.emit(EventType.TURN_FINISHED, {"outcome": TurnOutcome.FAILED, "error": clip(error)})
            raise
        logger.info("Turn %s finished in %.2fs", trace.turn_id[:8], time.monotonic() - started)
        trace.emit(EventType.TURN_FINISHED, {"outcome": TurnOutcome.COMPLETED})

    async def _run_steps(
        self, session: Session, user_input: str, channel: Channel, trace: TurnTrace
    ) -> AsyncIterator[Message]:
        """
        Run the think, act, observe cycle for one user message.

        Args:
            session: Conversation to read from and append to.
            user_input: The user's message text.
            channel: Where the message came from.
            trace: Receives an event for each stage.
        Returns:
            An async iterator of assistant and tool messages, ending with the final answer.
        """
        session.add(Message.user(user_input))
        approve = self._traced_approval(channel, trace)

        for step in range(1, self.max_steps + 1):
            trace.step = step
            trace.emit(EventType.CONTEXT_STARTED)
            summary_before = session.summary
            with trace.active():
                model_input = await self.context.build(session, channel.instructions)
            tool_schemas = self.tools.schemas()
            trace.emit(
                EventType.CONTEXT_BUILT,
                {
                    "messages": len(model_input),
                    "system_prompt_chars": len(model_input[0].content),
                    "tools": len(tool_schemas),
                    "compacted": session.summary != summary_before,
                },
            )

            logger.info(
                "Step %d: sending %d messages and %d tools to the model (compacted: %s)",
                step,
                len(model_input),
                len(tool_schemas),
                session.summary != summary_before,
            )
            trace.emit(EventType.MODEL_STARTED)
            reply = await self.llm.complete(model_input, tool_schemas)
            _log_reply(step, reply)
            trace.emit(
                EventType.MODEL_FINISHED,
                {
                    "text": clip(reply.content),
                    "tool_calls": _describe_calls(reply),
                    "hosted_calls": _describe_hosted_calls(reply),
                },
            )

            # No tool calls means the model has given its final answer.
            if len(reply.tool_calls) == 0:
                session.add(reply)
                logger.info("Answer: %s", logs.snippet(reply.content))
                trace.emit(EventType.ANSWER, {"text": clip(reply.content), "step_limit": False})
                yield reply
                await self._remember_turn(session, user_input, reply, trace)
                return

            yield reply

            results = []
            for call in reply.tool_calls:
                with trace.active():
                    result = await self._execute(call, approve, trace)
                results.append(result)

            session.add(reply)
            for result in results:
                session.add(result)

            for result in results:
                yield result

        notice_text = f"I stopped after {self.max_steps} steps without finishing. Let me know how to continue."
        notice = Message.assistant(notice_text)
        session.add(notice)
        logger.warning("Stopped at the step limit of %d", self.max_steps)
        trace.emit(EventType.ANSWER, {"text": notice_text, "step_limit": True})
        yield notice
        await self._remember_turn(session, user_input, notice, trace)

    def _traced_approval(self, channel: Channel, trace: TurnTrace) -> ApprovalCallback | None:
        """
        Wrap the channel's approval callback so each request and answer is traced.

        Args:
            channel: Where the user's message came from.
            trace: Receives the approval events.
        Returns:
            The wrapped callback, or None if the channel cannot ask for approval.
        """
        if channel.approve is None:
            return None
        ask_user = channel.approve

        async def approve(call: ToolCall) -> bool:
            logger.info("Asking the user on %s to approve %s", channel.name, call.name)
            trace.emit(EventType.APPROVAL_REQUESTED, {"call_id": call.id})
            approved = await ask_user(call)
            if approved:
                logger.info("The user approved %s", call.name)
            else:
                logger.info("The user did not approve %s", call.name)
            trace.emit(EventType.APPROVAL_RESOLVED, {"call_id": call.id, "approved": approved})
            return approved

        return approve

    async def _remember_turn(self, session: Session, user_input: str, answer: Message, trace: TurnTrace) -> None:
        """
        Save a finished turn to long-term memory. A failure is logged rather than raised,
        since the user already has their answer.

        Args:
            session: The conversation the turn belongs to.
            user_input: The user's message.
            answer: The final assistant message.
            trace: Receives the memory events.
        """
        if self.memory is None:
            return

        trace.emit(EventType.MEMORY_STARTED)
        saved = True
        try:
            with trace.active():
                await self.memory.episodes.save(session.id, user_input, answer.content)
        except Exception:
            logger.exception("Saving the turn to memory failed")
            saved = False
        trace.emit(EventType.MEMORY_FINISHED, {"saved": saved})

    async def _execute(self, call: ToolCall, approve: ApprovalCallback | None, trace: TurnTrace) -> Message:
        """
        Run a tool call, converting failures into an error the model can read.

        Args:
            call: The tool call to run.
            approve: Asks the user whether the call may run.
            trace: Receives the tool's start and finish events.
        Returns:
            A tool result message answering the call.
        """
        logger.info("Tool %s called with %s", call.name, logs.snippet(call.arguments))
        trace.emit(EventType.TOOL_STARTED, {"call_id": call.id, "name": call.name})
        started = time.monotonic()
        failed = False
        try:
            output = await self.tools.execute(call, approve)
            result = Message.tool_result(call.id, output.text, output.images)
            logger.info(
                "Tool %s returned in %.2fs (%d images): %s",
                call.name,
                time.monotonic() - started,
                len(output.images),
                logs.snippet(output.text),
            )
        except Exception as e:
            logger.exception("Tool %s failed after %.2fs", call.name, time.monotonic() - started)
            error_type = type(e).__name__
            result = Message.tool_result(call.id, f"Error running tool '{call.name}': {error_type}: {e}")
            failed = True

        trace.emit(
            EventType.TOOL_FINISHED,
            {"call_id": call.id, "output": clip(result.content), "images": len(result.images), "failed": failed},
        )
        return result


def _log_reply(step: int, reply: Message) -> None:
    """
    Log what the model asked for in one reply.

    Args:
        step: The loop step the reply belongs to.
        reply: The assistant message.
    """
    for hosted in reply.hosted_calls:
        logger.info(
            "Step %d: %s ran %s with %s and got %s",
            step,
            hosted.server,
            hosted.name,
            logs.snippet(hosted.arguments),
            logs.snippet(hosted.output),
        )
    names = []
    for call in reply.tool_calls:
        names.append(call.name)
    if len(names) > 0:
        logger.info("Step %d: the model called %s", step, ", ".join(names))


def _describe_calls(reply: Message) -> list[dict[str, str]]:
    """
    Describe a reply's tool calls for a harness event.

    Args:
        reply: An assistant message.
    Returns:
        One {"id", "name", "arguments"} dict per tool call.
    """
    calls = []
    for call in reply.tool_calls:
        calls.append({"id": call.id, "name": call.name, "arguments": clip(call.arguments)})
    return calls


def _describe_hosted_calls(reply: Message) -> list[dict[str, str]]:
    """
    Describe the calls the provider ran itself during a reply, for a harness event.

    Args:
        reply: An assistant message.
    Returns:
        One {"id", "server", "name", "arguments", "output"} dict per hosted call.
    """
    calls = []
    for hosted in reply.hosted_calls:
        calls.append(
            {
                "id": hosted.id,
                "server": hosted.server,
                "name": hosted.name,
                "arguments": clip(hosted.arguments),
                "output": clip(hosted.output),
            }
        )
    return calls
