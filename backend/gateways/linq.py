"""
Client for Linq's Partner API (v3), which sends and receives iMessages from a phone
number Linq provisions. LinqClient sends texts, shows the typing indicator and manages
webhook subscriptions. Incoming texts arrive as signed webhooks: verify_signature checks
that a request really came from Linq (HMAC-SHA256 over the id, timestamp and raw body,
keyed by the subscription's signing secret), and parse_incoming_text pulls the parts
Jarvis needs out of a message.received event.
"""

import base64
import hashlib
import hmac
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

API_URL = "https://api.linqapp.com/api/partner/v3"
REQUEST_TIMEOUT_SECONDS = 30
# Pins the webhook payload format so a Linq update cannot change what we parse.
WEBHOOK_VERSION = "2026-02-03"
SIGNATURE_TOLERANCE_SECONDS = 300
MAX_TEXT_CHARS = 10_000


class LinqError(Exception):
    """Raised when a Linq API request fails."""


@dataclass(frozen=True)
class IncomingText:
    """A text message someone sent to the Linq number."""

    event_id: str
    chat_id: str
    sender: str
    text: str
    is_group: bool


@dataclass(frozen=True)
class WebhookSubscription:
    """A Linq webhook subscription."""

    id: str
    target_url: str


class LinqClient:
    """Sends texts and manages webhooks through Linq's API."""

    def __init__(self, api_key: str):
        """
        Args:
            api_key: Linq Partner API key.
        """
        self._client = httpx.AsyncClient(
            base_url=API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    async def close(self) -> None:
        """Close the HTTP connection pool."""
        await self._client.aclose()

    async def send_text(self, chat_id: str, text: str) -> None:
        """
        Send a text to an existing chat.

        Args:
            chat_id: The chat to send to.
            text: The message. Longer than MAX_TEXT_CHARS is cut off.
        """
        if len(text) > MAX_TEXT_CHARS:
            text = text[: MAX_TEXT_CHARS - 3] + "..."
        body = {"message": {"parts": [{"type": "text", "value": text}]}}
        await self._request("POST", f"/chats/{chat_id}/messages", body)

    async def start_typing(self, chat_id: str) -> None:
        """
        Show the typing indicator in a chat. It disappears on its own after about a minute.

        Args:
            chat_id: The chat.
        """
        await self._request("POST", f"/chats/{chat_id}/typing", None)

    async def stop_typing(self, chat_id: str) -> None:
        """
        Hide the typing indicator in a chat.

        Args:
            chat_id: The chat.
        """
        await self._request("DELETE", f"/chats/{chat_id}/typing", None)

    async def subscribe(self, target_url: str) -> tuple[WebhookSubscription, str]:
        """
        Ask Linq to deliver received messages to a URL.

        Args:
            target_url: The public webhook address.
        Returns:
            The subscription, and its signing secret (which Linq shows only once).
        """
        body = {
            "target_url": f"{target_url}?version={WEBHOOK_VERSION}",
            "subscribed_events": ["message.received"],
        }
        data = await self._request("POST", "/webhook-subscriptions", body)
        subscription = WebhookSubscription(id=data["id"], target_url=data["target_url"])
        return subscription, data["signing_secret"]

    async def subscriptions(self) -> list[WebhookSubscription]:
        """
        List the account's webhook subscriptions.

        Returns:
            Every subscription, active or not.
        """
        data = await self._request("GET", "/webhook-subscriptions", None)
        subscriptions = []
        for item in data["subscriptions"]:
            subscriptions.append(WebhookSubscription(id=item["id"], target_url=item["target_url"]))
        return subscriptions

    async def unsubscribe(self, subscription_id: str) -> None:
        """
        Delete a webhook subscription.

        Args:
            subscription_id: The subscription to delete.
        """
        await self._request("DELETE", f"/webhook-subscriptions/{subscription_id}", None)

    async def _request(self, method: str, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        """
        Send one API request.

        Args:
            method: The HTTP method.
            path: The path after API_URL.
            body: The JSON body, or None.
        Returns:
            The JSON response, or {} when there is no body.
        """
        try:
            response = await self._client.request(method, path, json=body)
        except httpx.HTTPError as e:
            raise LinqError(f"Could not reach Linq: {e}") from None

        if response.status_code >= 400:
            raise LinqError(f"Linq returned HTTP {response.status_code} for {method} {path}: {response.text[:300]}")
        if response.content == b"":
            return {}
        return response.json()


def verify_signature(secret: str, headers: Mapping[str, str], body: bytes, now: float | None = None) -> bool:
    """
    Check that a webhook request was signed by Linq and is recent.

    Args:
        secret: The subscription's signing secret, like "whsec_...".
        headers: The request headers.
        body: The raw request body, exactly as received.
        now: The current Unix time. Defaults to the clock.
    Returns:
        True if one of the request's signatures matches and its timestamp is recent.
    """
    if now is None:
        now = time.time()

    webhook_id = headers.get("webhook-id")
    timestamp = headers.get("webhook-timestamp")
    signatures = headers.get("webhook-signature")
    if webhook_id is None or timestamp is None or signatures is None:
        return False

    # Rejecting old timestamps stops someone from replaying a captured request.
    try:
        age = abs(now - int(timestamp))
    except ValueError:
        return False
    if age > SIGNATURE_TOLERANCE_SECONDS:
        return False

    key = base64.b64decode(secret.removeprefix("whsec_"))
    signed = f"{webhook_id}.{timestamp}.".encode() + body
    digest = hmac.new(key, signed, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode()

    # The header may hold several space-separated signatures, like "v1,abc= v1,def=".
    for signature in signatures.split(" "):
        version, _, value = signature.partition(",")
        if version == "v1" and hmac.compare_digest(value, expected):
            return True
    return False


def parse_incoming_text(payload: dict[str, Any]) -> IncomingText | None:
    """
    Read a webhook event as an incoming text.

    Args:
        payload: The parsed webhook body.
    Returns:
        The text, or None if the event is not a message someone else sent.
    """
    if payload.get("event_type") != "message.received":
        return None

    data = payload["data"]
    sender = data["sender_handle"]
    if data.get("direction") != "inbound" or sender.get("is_me"):
        return None

    texts = []
    for part in data["parts"]:
        if part.get("type") == "text":
            texts.append(part["value"])
        else:
            texts.append(f"[{part.get('type')} attachment, which you cannot open over text]")
    text = "\n".join(texts).strip()
    if text == "":
        return None

    return IncomingText(
        event_id=payload["event_id"],
        chat_id=data["chat"]["id"],
        sender=sender["handle"],
        text=text,
        is_group=data["chat"].get("is_group") is True,
    )
