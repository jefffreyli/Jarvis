"""
Lets the user talk to Jarvis through an Amazon Echo. The Alexa skill's Lambda forwards
each utterance through the AWS relay as an "alexa.ask" message, and Jarvis sends a
reply for that request, which the Lambda speaks. Each Alexa user has one ongoing
conversation whose turns run one at a time. A tool that needs approval is asked aloud
as the reply, and the user's next utterance is taken as the answer, so the rest of the
turn replies to that newer request.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent.loop import AgentLoop, Channel
from agent.models import Role, ToolCall
from agent.prompts import VOICE_INSTRUCTIONS
from agent.session import Session

from .approval import describe_call, is_yes
from .relay import RelayClient, RelayError

logger = logging.getLogger(__name__)

ASK_TYPE = "alexa.ask"
APPROVAL_TIMEOUT_SECONDS = 300
# Alexa speaks at most 8000 characters, and the relay caps message size.
MAX_REPLY_CHARS = 6000


@dataclass
class VoiceConversation:
    """One Alexa user's conversation with Jarvis."""

    user_id: str
    session: Session = field(default_factory=Session)
    turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Set while Jarvis waits for the user to answer an approval question.
    pending_answer: asyncio.Future | None = None
    # The Alexa request waiting to hear what Jarvis says next.
    reply_to: str = ""


class AlexaGateway:
    """Answers the user's requests to the Jarvis Alexa skill."""

    def __init__(self, agent: AgentLoop, relay: RelayClient, notify: Callable[[str], None] = print):
        """
        Args:
            agent: The agent that answers.
            relay: The relay connection Alexa requests arrive on.
            notify: Shows a one-line note about Alexa activity, by default in the terminal.
        """
        self.agent = agent
        self.relay = relay
        self.notify = notify
        self._conversations: dict[str, VoiceConversation] = {}
        relay.on(ASK_TYPE, self._handle_ask)

    async def _handle_ask(self, message: dict[str, Any]) -> None:
        """
        Answer one utterance, or use it as the answer to a pending approval question.

        Args:
            message: The relay message with the request id, Alexa user id and text.
        """
        request_id = message.get("request_id")
        user_id = message.get("user_id")
        text = message.get("text")
        if not isinstance(request_id, str) or not isinstance(user_id, str) or not isinstance(text, str):
            logger.warning("Ignored a malformed Alexa request")
            return

        conversation = self._conversations.get(user_id)
        if conversation is None:
            conversation = VoiceConversation(user_id)
            self._conversations[user_id] = conversation

        pending = conversation.pending_answer
        if pending is not None and not pending.done():
            conversation.reply_to = request_id
            pending.set_result(text)
            return

        async with conversation.turn_lock:
            conversation.reply_to = request_id
            self.notify(f"[Alexa] {text}")
            try:
                reply = await self._run_turn(conversation, text)
            except Exception as e:
                logger.exception("Alexa turn failed")
                reply = f"Sorry, something went wrong: {type(e).__name__}."
            await self._reply(conversation, reply)
            self.notify(f"[Alexa reply] {reply}")

    async def _run_turn(self, conversation: VoiceConversation, text: str) -> str:
        """
        Run one utterance through the agent.

        Args:
            conversation: The user's conversation.
            text: What the user said.
        Returns:
            Jarvis's final answer.
        """

        async def approve(call: ToolCall) -> bool:
            return await self._approve(conversation, call)

        channel = Channel(approve=approve, instructions=VOICE_INSTRUCTIONS, name="alexa")
        reply = ""
        async for message in self.agent.run(conversation.session, text, channel):
            if message.role == Role.ASSISTANT and len(message.tool_calls) == 0:
                reply = message.content

        if reply.strip() == "":
            reply = "Done."
        return reply

    async def _approve(self, conversation: VoiceConversation, call: ToolCall) -> bool:
        """
        Ask the user aloud whether a tool call may run and wait for their answer.

        Args:
            conversation: The user's conversation.
            call: The tool call awaiting approval.
        Returns:
            True if the user said a clear yes in time.
        """
        # Created before asking, so a quick answer always finds it.
        conversation.pending_answer = asyncio.get_running_loop().create_future()
        try:
            await self._reply(conversation, f"Can I do this? {describe_call(call)}. Say yes to allow.")
            answer = await asyncio.wait_for(conversation.pending_answer, APPROVAL_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            return False
        finally:
            conversation.pending_answer = None
        return is_yes(answer)

    async def _reply(self, conversation: VoiceConversation, text: str) -> None:
        """
        Send what Jarvis says to the Alexa request that is waiting for it.

        Args:
            conversation: The user's conversation.
            text: What to say.
        """
        message = {"action": "reply", "request_id": conversation.reply_to, "text": text[:MAX_REPLY_CHARS]}
        try:
            await self.relay.send(message)
        except RelayError:
            logger.exception("Sending the Alexa reply failed")
