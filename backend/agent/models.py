"""
Provider-agnostic data types shared across the agent. A conversation is a list of
immutable Messages; assistant messages may carry ToolCalls, and each call is answered
by a tool Message that references the call's id. Tool messages can also carry Images
(such as a photo a tool loaded for the model to look at). Assistant messages may also
carry HostedCalls, which the provider ran itself (such as calls to a remote MCP server)
and need no answer from the agent. LLM adapters translate these
to and from their own wire formats.
"""

from dataclasses import dataclass, field
from enum import StrEnum


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class ToolCall:
    """A request from the model to run a tool. `arguments` is the raw JSON string."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Image:
    """An image for the model to look at, as encoded file bytes."""

    media_type: str
    # Excluded from repr so logs and errors never dump megabytes of image bytes.
    data: bytes = field(repr=False)


@dataclass(frozen=True)
class HostedCall:
    """
    A tool call the model provider ran itself while writing a reply, such as a call to a
    remote MCP server. It is kept so later requests still show the model what it found.
    """

    id: str
    # The server the call went to, like "notion".
    server: str
    name: str
    arguments: str
    # The tool's output, or a description of the error when it failed.
    output: str


@dataclass(frozen=True)
class Message:
    """A single conversation entry."""

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    images: tuple[Image, ...] = ()
    # Calls the provider already ran before this reply; they come before its text.
    hosted_calls: tuple[HostedCall, ...] = ()

    def __post_init__(self):
        if self.role == Role.TOOL and not self.tool_call_id:
            raise ValueError("Tool messages require a tool_call_id")
        if self.tool_calls and self.role != Role.ASSISTANT:
            raise ValueError("Only assistant messages can contain tool calls")
        if self.hosted_calls and self.role != Role.ASSISTANT:
            raise ValueError("Only assistant messages can contain hosted calls")
        if self.images and self.role != Role.TOOL:
            raise ValueError("Only tool messages can contain images")

    @classmethod
    def system(cls, content: str) -> "Message":
        """
        Build a system message.

        Args:
            content: The system prompt text.
        Returns:
            A system Message.
        """
        return cls(Role.SYSTEM, content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """
        Build a user message.

        Args:
            content: The user's text.
        Returns:
            A user Message.
        """
        return cls(Role.USER, content)

    @classmethod
    def assistant(
        cls,
        content: str = "",
        tool_calls: tuple[ToolCall, ...] = (),
        hosted_calls: tuple[HostedCall, ...] = (),
    ) -> "Message":
        """
        Build an assistant message.

        Args:
            content: The assistant's text reply.
            tool_calls: Tools the assistant wants to run.
            hosted_calls: Calls the provider already ran while writing this reply.
        Returns:
            An assistant Message.
        """
        return cls(Role.ASSISTANT, content, tool_calls, hosted_calls=hosted_calls)

    @classmethod
    def tool_result(cls, call_id: str, content: str, images: tuple[Image, ...] = ()) -> "Message":
        """
        Build a tool result message answering a ToolCall.

        Args:
            call_id: The id of the ToolCall being answered.
            content: The tool's text output.
            images: Images the tool produced for the model to see.
        Returns:
            A tool Message.
        """
        return cls(Role.TOOL, content, tool_call_id=call_id, images=images)
