"""
Helpers every gateway uses to ask the user to approve a tool call: a readable
description of the call, and a strict reading of the user's answer, where anything
other than a clear yes counts as no.
"""

import json
import re

from agent.models import ToolCall

MAX_VALUE_CHARS = 300
YES_ANSWERS = ("y", "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "go ahead", "do it", "approve", "allow")


def describe_call(call: ToolCall) -> str:
    """
    Describe a tool call with one line per argument.

    Args:
        call: The tool call.
    Returns:
        Text like "send_email\n  to: sam@example.com\n  subject: Hi".
    """
    lines = [call.name]
    try:
        arguments = json.loads(call.arguments)
    except json.JSONDecodeError:
        arguments = None

    if not isinstance(arguments, dict):
        lines.append(f"  {call.arguments[:MAX_VALUE_CHARS]}")
        return "\n".join(lines)

    for name, value in arguments.items():
        text = str(value)
        if len(text) > MAX_VALUE_CHARS:
            text = text[:MAX_VALUE_CHARS] + "..."
        lines.append(f"  {name}: {text}")
    return "\n".join(lines)


def is_yes(answer: str) -> bool:
    """
    Check whether the user's answer clearly approves.

    Args:
        answer: What the user typed, texted or said.
    Returns:
        True only for answers like "yes" or "go ahead".
    """
    words = re.sub(r"[^a-z ]", "", answer.lower())
    words = " ".join(words.split())
    return words in YES_ANSWERS
