"""
Tools for placing phone calls through a Smallest AI voice agent. The voice agent is set
up once in the Smallest AI dashboard with a prompt that contains {{objective}}; each call
fills that variable with what Jarvis wants the call to achieve. Before the first call,
the agent is switched to the user's cloned voice (config.CALL_VOICE_ID) if it uses a
different one. make_phone_call starts the call, then polls the call log until it ends
and returns the summary and transcript.
A call that is still going when the wait runs out can be checked later with
check_phone_call. Placing a call asks the user first.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx

import config

from .base import ToolError, tool

API_URL = "https://api.smallest.ai/atoms/v1"
REQUEST_TIMEOUT_SECONDS = 30
POLL_INTERVAL_SECONDS = 5
MAX_WAIT_SECONDS = 600
FINISHED_STATUSES = ("completed", "failed", "no_answer", "cancelled")
# Synthesizer settings kept as they are when the agent's voice is changed.
KEPT_SYNTHESIZER_FIELDS = ("speed", "consistency", "similarity", "enhancement", "sampleRate")


@dataclass(frozen=True)
class CallLog:
    """What happened on a call, as far as Smallest AI knows so far."""

    id: str
    status: str
    to: str
    duration_seconds: float
    summary: str
    failure_reason: str
    transcript: tuple[tuple[str, str], ...]

    @property
    def finished(self) -> bool:
        """True once the call has ended, successfully or not."""
        return self.status in FINISHED_STATUSES


class SmallestClient:
    """Starts calls with one Smallest AI voice agent and reads their call logs."""

    def __init__(self, api_key: str, agent_id: str, from_number: str, voice_id: str, voice_model: str):
        """
        Args:
            api_key: Smallest AI API key.
            agent_id: The voice agent that makes the calls.
            from_number: Caller ID to dial from, or "" for Smallest AI's default.
            voice_id: The voice the agent speaks with, such as a cloned voice.
            voice_model: The Waves text-to-speech model that voice is used with.
        """
        self.agent_id = agent_id
        self.from_number = from_number
        self.voice_id = voice_id
        self.voice_model = voice_model
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._voice_checked = False

    async def ensure_voice(self) -> None:
        """
        Switch the agent to the configured voice if it uses a different one. The agent
        is checked once per run, since nothing else should change its voice.
        """
        if self._voice_checked:
            return

        agent = await self._request("GET", f"/agent/{self.agent_id}", None)
        synthesizer = agent.get("synthesizer") or {}
        voice_config = synthesizer.get("voiceConfig") or {}
        if voice_config.get("voiceId") == self.voice_id:
            self._voice_checked = True
            return

        # A versioned agent only changes through a published version, which Jarvis
        # should not publish on the user's behalf.
        if agent.get("activeVersionId"):
            raise ToolError(
                f"The calling agent uses versioned configuration, so Jarvis cannot switch it to voice "
                f"{self.voice_id}. Set that voice on the agent in the Smallest AI dashboard."
            )

        new_voice_config: dict[str, Any] = {"model": self.voice_model, "voiceId": self.voice_id}
        if voice_config.get("gender") is not None:
            new_voice_config["gender"] = voice_config["gender"]

        new_synthesizer: dict[str, Any] = {"voiceConfig": new_voice_config}
        for field in KEPT_SYNTHESIZER_FIELDS:
            if synthesizer.get(field) is not None:
                new_synthesizer[field] = synthesizer[field]

        await self._request("PATCH", f"/agent/{self.agent_id}", {"synthesizer": new_synthesizer})
        self._voice_checked = True

    async def start_call(self, phone_number: str, objective: str) -> str:
        """
        Start an outbound call.

        Args:
            phone_number: The number to call, in E.164 form like "+14155551234".
            objective: What the voice agent should accomplish, filled into its prompt.
        Returns:
            The call id.
        """
        body: dict[str, Any] = {
            "agentId": self.agent_id,
            "phoneNumber": phone_number,
            "variables": {"objective": objective},
        }
        if self.from_number != "":
            body["fromNumber"] = self.from_number

        data = await self._request("POST", "/conversation/outbound", body)
        return data["conversationId"]

    async def call_log(self, call_id: str) -> CallLog:
        """
        Read a call's current status and transcript.

        Args:
            call_id: The id returned by start_call.
        Returns:
            The call log.
        """
        data = await self._request("GET", f"/conversation/{call_id}", None)

        transcript = []
        if data.get("transcript") is not None:
            for turn in data["transcript"]:
                transcript.append((turn["role"], turn["content"]))

        summary = ""
        analytics = data.get("postCallAnalytics")
        if analytics is not None:
            summary = _text(analytics, "summary")

        duration_seconds = data.get("duration")
        if duration_seconds is None:
            duration_seconds = 0

        return CallLog(
            id=call_id,
            status=data["status"],
            to=_text(data, "to"),
            duration_seconds=duration_seconds,
            summary=summary,
            failure_reason=_text(data, "callFailureReason"),
            transcript=tuple(transcript),
        )

    async def _request(self, method: str, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        """
        Send one API request and unwrap its {"status", "data"} response.

        Args:
            method: "GET", "POST" or "PATCH".
            path: Path after API_URL.
            body: JSON body, or None.
        Returns:
            The response's data object, or {} if it has none.
        """
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            try:
                response = await client.request(method, API_URL + path, headers=self._headers, json=body)
            except httpx.HTTPError as e:
                raise ToolError(f"Could not reach Smallest AI: {e}") from None

        try:
            payload = response.json()
        except ValueError:
            raise ToolError(f"Smallest AI returned HTTP {response.status_code}: {response.text[:300]}") from None

        if response.status_code != 200 or payload.get("status") is not True:
            raise ToolError(f"Smallest AI returned HTTP {response.status_code}: {_error_text(payload)}")

        data = payload.get("data")
        if data is None:
            return {}
        return data


def _error_text(payload: dict[str, Any]) -> str:
    """
    Pull a readable message out of an error response.

    Args:
        payload: The parsed JSON body.
    Returns:
        The listed errors, or the whole body if there are none.
    """
    errors = payload.get("errors")
    if not isinstance(errors, list) or len(errors) == 0:
        return str(payload)[:300]

    messages = []
    for error in errors:
        messages.append(str(error))
    return "; ".join(messages)


def _text(data: dict[str, Any], key: str) -> str:
    """
    Read a text field that the API may leave out or set to null.

    Args:
        data: A JSON object from the API.
        key: The field name.
    Returns:
        The text, or "" when it is missing.
    """
    value = data.get(key)
    if value is None:
        return ""
    return str(value)


_smallest_client: SmallestClient | None = None


def smallest_client() -> SmallestClient:
    """
    Get the shared Smallest AI client, created from .env settings on first use.

    Returns:
        The client.
    """
    global _smallest_client
    if _smallest_client is None:
        try:
            api_key = config.smallest_api_key()
            agent_id, from_number = config.smallest_call_settings()
        except RuntimeError as e:
            raise ToolError(str(e)) from None
        _smallest_client = SmallestClient(api_key, agent_id, from_number, config.CALL_VOICE_ID, config.CALL_VOICE_MODEL)
    return _smallest_client


def to_e164(phone_number: str) -> str:
    """
    Convert a phone number to E.164 form. Numbers without a country code are taken
    to be US numbers.

    Args:
        phone_number: A number like "(555) 123-4567", "1-555-123-4567" or "+44 20 7946 0958".
    Returns:
        The number like "+15551234567".
    """
    digits = ""
    for character in phone_number:
        if character.isdigit():
            digits += character

    if phone_number.strip().startswith("+"):
        result = "+" + digits
    elif len(digits) == 10:
        result = "+1" + digits
    elif len(digits) == 11 and digits.startswith("1"):
        result = "+" + digits
    else:
        raise ToolError(f"'{phone_number}' is not a full phone number. Include the country code, like +14155551234.")

    # E.164 allows at most 15 digits after the "+".
    if len(result) < 9 or len(result) > 16:
        raise ToolError(f"'{phone_number}' is not a valid phone number.")
    return result


# Tools


@tool
async def make_phone_call(phone_number: str, objective: str) -> str:
    """
    Phone someone with an AI voice agent that speaks on the user's behalf, and wait
    for the call to end (up to 10 minutes). The voice agent knows only the objective,
    so include everything it needs: who it is calling, who it represents, the goal,
    and any details like names, dates, times, preferences and limits.

    Args:
        phone_number: The number to call. Use lookup_contact to find a contact's number.
        objective: Instructions for the voice agent, such as "Call Luigi's Pizza for Jeffrey Li and book a table for 4 this Friday at 7pm. Any time 6-8pm is fine. Do not give payment details."
    Returns:
        How the call went: its status, summary and transcript, or the call id if it is still going.
    """
    if objective.strip() == "":
        raise ToolError("objective must not be empty")

    number = to_e164(phone_number)
    client = smallest_client()
    await client.ensure_voice()
    call_id = await client.start_call(number, objective)

    deadline = time.monotonic() + MAX_WAIT_SECONDS
    while time.monotonic() < deadline:
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        log = await client.call_log(call_id)
        if log.finished:
            return _describe_call(log)

    return (
        f"Call {call_id} is still going after {MAX_WAIT_SECONDS // 60} minutes. "
        "Check on it later with check_phone_call."
    )


@tool(requires_approval=False)
async def check_phone_call(call_id: str) -> str:
    """
    Check on a call placed with make_phone_call.

    Args:
        call_id: The call id, like "CALL-1737000000000-abc123".
    Returns:
        The call's status, plus its summary and transcript so far.
    """
    log = await smallest_client().call_log(call_id)
    return _describe_call(log)


def _describe_call(log: CallLog) -> str:
    """
    Format a call log for the model.

    Args:
        log: The call log.
    Returns:
        A readable description.
    """
    lines = [f"Call {log.id} to {log.to}: {log.status}, {round(log.duration_seconds)}s"]
    if log.failure_reason != "":
        lines.append(f"Failure reason: {log.failure_reason}")
    if log.summary != "":
        lines.append(f"Summary: {log.summary}")

    if len(log.transcript) == 0:
        lines.append("No transcript.")
    else:
        lines.append("Transcript:")
        for role, content in log.transcript:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)
