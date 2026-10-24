"""
The ToolRegistry holds the tools available to the agent and runs tool calls. For each
call it looks up the tool, validates the arguments, asks the user for approval through
the caller's callback if the tool requires it, runs the tool, and truncates oversized
output so one call cannot flood the context. Unknown tools and invalid arguments raise ToolError, which the
agent loop reports back to the LLM.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from agent.models import ToolCall

from .base import Tool, ToolError, ToolOutput

ApprovalCallback = Callable[[ToolCall], Awaitable[bool]]

DENIED_MESSAGE = "The user did not approve this tool call, so it was not run."


class ToolRegistry:
    """The set of tools the agent can use."""

    def __init__(
        self,
        tools: list[Tool],
        max_output_chars: int = 20_000,
    ):
        """
        Args:
            tools: Tools to make available. Names must be unique.
            max_output_chars: Longest tool output passed back to the LLM.
        """
        if max_output_chars <= 0:
            raise ValueError("max_output_chars must be positive")

        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"Duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

        self.max_output_chars = max_output_chars

    def schemas(self) -> list[dict[str, Any]]:
        """
        Describe every tool for the LLM.

        Returns:
            One {"name", "description", "parameters"} dict per tool.
        """
        schemas = []
        for tool in self._tools.values():
            schemas.append(tool.schema())
        return schemas

    async def execute(self, call: ToolCall, approve: ApprovalCallback | None) -> ToolOutput:
        """
        Run a tool call, asking for approval first if the tool requires it.

        Args:
            call: The tool call from the LLM.
            approve: Asks the user whether the call may run. Without it, a call that
                requires approval is denied.
        Returns:
            The tool's output, or a note that the user denied the call.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            raise ToolError(f"Unknown tool '{call.name}'")

        # Validate before asking, so the user is never asked to approve a broken call.
        args = tool.parse_arguments(call.arguments)

        if tool.needs_approval(args):
            if approve is None:
                return ToolOutput(DENIED_MESSAGE)
            approved = await approve(call)
            if not approved:
                return ToolOutput(DENIED_MESSAGE)

        output = await tool.run(args)
        return ToolOutput(self._truncate(output.text), output.images)

    def _truncate(self, output: str) -> str:
        """
        Shorten output that exceeds the limit, noting how much was cut.

        Args:
            output: The tool's full output.
        Returns:
            The output, truncated if needed.
        """
        if len(output) <= self.max_output_chars:
            return output

        kept = output[: self.max_output_chars]
        notice = f"\n\n[Output truncated: showing {self.max_output_chars} of {len(output)} characters]"
        return kept + notice
