"""
Lets the user text Jarvis over iMessage through Linq. On start it subscribes the AWS
relay's public webhook URL to Linq's message.received webhook (removing subscriptions
left by earlier runs); the relay forwards each webhook down its websocket as a
"linq.webhook" message. Each webhook is checked against Linq's signature,
de-duplicated, and accepted only from the user's own numbers in one-on-one chats.
Every chat gets its own conversation; its messages run one turn at a time, with a
typing indicator while Jarvis works, and the final answer is texted back. A tool that
needs approval is described in a text, and the user's next text is taken as the answer.
"""

import asyncio
import base64
import binascii
import json
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent.loop import AgentLoop, Channel
from agent.models import Role, ToolCall
from agent.prompts import TEXT_INSTRUCTIONS
from agent.session import Session
from tools.base import ToolError
from tools.phone import to_e164

from .approval import describe_call, is_yes
from .linq import LinqClient, LinqError, parse_incoming_text, verify_signature
from .relay import RelayClient

logger = logging.getLogger(__name__)

WEBHOOK_TYPE = "linq.webhook"
APPROVAL_TIMEOUT_SECONDS = 300
TYPING_REFRESH_SECONDS = 60
REMEMBERED_EVENTS = 1000


@dataclass
class TextConversation:
    """One iMessage chat with the user."""

    chat_id: str
    session: Session = field(default_factory=Session)
    # Turns in one chat run one at a time, in the order the texts arrived.
    turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Set while Jarvis waits for the user to answer an approval question.
    pending_answer: asyncio.Future | None = None


class TextGateway:
    """Answers the user's texts to the Linq number."""

    def __init__(
        self,
        agent: AgentLoop,
        linq: LinqClient,
        allowed_numbers: list[str],
        relay: RelayClient,
        webhook_url: str,
        notify: Callable[[str], None] = print,
    ):
        """
        Args:
            agent: The agent that answers texts.
            linq: Client for the Linq API.
            allowed_numbers: Phone numbers allowed to text Jarvis. Others are ignored.
            relay: The relay connection that webhooks arrive on.
            webhook_url: The relay's public URL for Linq webhooks.
            notify: Shows a one-line note about texting activity, by default in the terminal.
        """
        self.agent = agent
        self.linq = linq
        self.webhook_url = webhook_url
        self.notify = notify

        self._allowed_numbers = set()
        for number in allowed_numbers:
            try:
                self._allowed_numbers.add(to_e164(number))
            except ToolError:
                raise RuntimeError(f"LINQ_ALLOWED_NUMBERS has an invalid phone number: {number}") from None

        self._subscription_id: str | None = None
        self._signing_secret: str | None = None
        self._conversations: dict[str, TextConversation] = {}
        self._seen_events: deque[str] = deque(maxlen=REMEMBERED_EVENTS)
        relay.on(WEBHOOK_TYPE, self._handle_webhook)

    async def start(self) -> None:
        """Subscribe the relay's webhook URL to Linq's incoming texts."""
        await self._remove_old_subscriptions()
        subscription, secret = await self.linq.subscribe(self.webhook_url)
        self._subscription_id = subscription.id
        self._signing_secret = secret
        logger.info("Texting gateway subscribed at %s", self.webhook_url)

    async def stop(self) -> None:
        """Unsubscribe from Linq."""
        if self._subscription_id is not None:
            try:
                await self.linq.unsubscribe(self._subscription_id)
            except LinqError:
                logger.exception("Removing the webhook subscription failed")
            self._subscription_id = None
        await self.linq.close()

    async def _remove_old_subscriptions(self) -> None:
        """Delete subscriptions to the webhook URL that earlier runs did not remove."""
        for subscription in await self.linq.subscriptions():
            if subscription.target_url == self.webhook_url:
                await self.linq.unsubscribe(subscription.id)
                logger.info("Removed old webhook subscription %s", subscription.id)

    async def _handle_webhook(self, message: dict[str, Any]) -> None:
        """
        Handle one webhook from Linq, forwarded by the relay.

        Args:
            message: The relay message with Linq's signature headers and the base64 raw body.
        """
        if self._signing_secret is None:
            logger.warning("Ignored a webhook that arrived before the subscription was ready")
            return

        headers = message.get("headers")
        try:
            body = base64.b64decode(message.get("body", ""), validate=True)
        except (binascii.Error, TypeError):
            logger.warning("Ignored a webhook with an unreadable body")
            return
        if not isinstance(headers, dict) or not verify_signature(self._signing_secret, headers, body):
            logger.warning("Rejected a webhook with an invalid signature")
            return

        try:
            payload = json.loads(body)
            incoming = parse_incoming_text(payload)
        except (ValueError, KeyError, TypeError):
            logger.exception("Could not read a webhook")
            return

        if incoming is None:
            return

        # Linq may deliver the same event more than once.
        if incoming.event_id in self._seen_events:
            return
        self._seen_events.append(incoming.event_id)

        if not self._is_allowed(incoming.sender):
            logger.warning("Ignored a text from %s, who is not in LINQ_ALLOWED_NUMBERS", incoming.sender)
            return
        if incoming.is_group:
            # Replies would reach everyone in the group, so group chats are ignored.
            logger.warning("Ignored a text in group chat %s", incoming.chat_id)
            return

        await self._handle_text(incoming.chat_id, incoming.sender, incoming.text)

    def _is_allowed(self, sender: str) -> bool:
        """
        Check whether a sender may use Jarvis.

        Args:
            sender: The sender's phone number or email address.
        Returns:
            True if the sender is one of the allowed numbers.
        """
        try:
            number = to_e164(sender)
        except ToolError:
            return False
        return number in self._allowed_numbers

    async def _handle_text(self, chat_id: str, sender: str, text: str) -> None:
        """
        Answer one text, or use it as the answer to a pending approval question.

        Args:
            chat_id: The chat it was sent in.
            sender: Who sent it.
            text: What it says.
        """
        conversation = self._conversations.get(chat_id)
        if conversation is None:
            conversation = TextConversation(chat_id)
            self._conversations[chat_id] = conversation

        pending = conversation.pending_answer
        if pending is not None and not pending.done():
            pending.set_result(text)
            return

        async with conversation.turn_lock:
            self.notify(f"[Text from {sender}] {text}")
            try:
                reply = await self._run_turn(conversation, text)
            except Exception as e:
                logger.exception("Text turn failed")
                reply = f"Sorry, something went wrong: {type(e).__name__}: {e}"

            try:
                await self.linq.send_text(chat_id, reply)
            except LinqError:
                logger.exception("Sending the reply failed")
                return
            self.notify(f"[Text to {sender}] {reply}")

    async def _run_turn(self, conversation: TextConversation, text: str) -> str:
        """
        Run one text through the agent, showing the typing indicator meanwhile.

        Args:
            conversation: The chat.
            text: The user's text.
        Returns:
            Jarvis's final answer.
        """

        async def approve(call: ToolCall) -> bool:
            return await self._approve(conversation, call)

        channel = Channel(approve=approve, instructions=TEXT_INSTRUCTIONS, name="text")
        await self._show_typing(conversation.chat_id)
        typing = asyncio.create_task(self._keep_typing(conversation.chat_id))
        reply = ""
        try:
            async for message in self.agent.run(conversation.session, text, channel):
                if message.role == Role.ASSISTANT and len(message.tool_calls) == 0:
                    reply = message.content
        finally:
            typing.cancel()
            try:
                await self.linq.stop_typing(conversation.chat_id)
            except LinqError:
                logger.warning("Hiding the typing indicator failed")

        # Linq rejects empty messages.
        if reply.strip() == "":
            reply = "Done."
        return reply

    async def _approve(self, conversation: TextConversation, call: ToolCall) -> bool:
        """
        Text the user a tool call and wait for their answer.

        Args:
            conversation: The chat to ask in.
            call: The tool call awaiting approval.
        Returns:
            True if the user texted back a clear yes in time.
        """
        question = f"Can I do this?\n\n{describe_call(call)}\n\nReply yes to allow."
        await self.linq.send_text(conversation.chat_id, question)

        conversation.pending_answer = asyncio.get_running_loop().create_future()
        try:
            answer = await asyncio.wait_for(conversation.pending_answer, APPROVAL_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            await self.linq.send_text(conversation.chat_id, "No answer, so I didn't do it.")
            return False
        finally:
            conversation.pending_answer = None
        return is_yes(answer)

    async def _show_typing(self, chat_id: str) -> None:
        """
        Show the typing indicator. A failure is only logged, since it is cosmetic.

        Args:
            chat_id: The chat.
        """
        try:
            await self.linq.start_typing(chat_id)
        except LinqError:
            logger.warning("Showing the typing indicator failed")

    async def _keep_typing(self, chat_id: str) -> None:
        """
        Keep the typing indicator showing until cancelled. Linq's indicator expires
        after about a minute, so it is refreshed.

        Args:
            chat_id: The chat.
        """
        while True:
            await asyncio.sleep(TYPING_REFRESH_SECONDS)
            await self._show_typing(chat_id)
