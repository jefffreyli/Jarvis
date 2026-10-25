"""
Tools from local MCP servers, such as canvas-mcp. Unlike the hosted servers in
llm/hosted_mcp.py, which OpenAI calls, Jarvis runs a local server itself as a
subprocess and talks to it over stdio for as long as Jarvis runs. On start it asks the
server for its tools and keeps only those the server marks read-only, each wrapped as an
McpTool so the ToolRegistry offers and runs it like any other tool. A call is forwarded
to the server, and its text result comes back as the tool's output. The server's own log
(its stderr) is read through a pipe and written to Jarvis's log, one record per line.
"""

import asyncio
import json
import logging
import os
import re
import threading
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any, TextIO

import anyio
from mcp import ClientSession, McpError, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as McpToolInfo

from .base import BaseTool, ToolError, ToolOutput

logger = logging.getLogger(__name__)

START_TIMEOUT_SECONDS = 30
CALL_TIMEOUT_SECONDS = 60
# A line in Python's common log format, like "2026-10-24 22:47:40 - canvas_mcp - INFO - Ready".
SERVER_LOG_LINE = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\S* - \S+ - (?P<level>[A-Z]+) - (?P<message>.*)$")


class LocalMcpServer:
    """A local MCP server that Jarvis runs as a subprocess and calls tools on over stdio."""

    def __init__(self, label: str, command: str, env: dict[str, str]):
        """
        Args:
            label: Short name for logs and errors, like "canvas".
            command: The executable that starts the server, like "canvas-mcp-server".
            env: Environment variables for the server. Jarvis's own environment, with its
                API keys, is not passed on.
        """
        self.label = label
        self.command = command
        self.env = env
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None
        # The server's read-only tools, filled in by start().
        self.tools: list[McpTool] = []

    async def start(self) -> None:
        """Start the server, complete the MCP handshake and list its read-only tools. Call stop() afterwards, even if this fails."""
        params = StdioServerParameters(command=self.command, env=self.env)
        errlog = self._stack.enter_context(_open_server_log(logging.getLogger(f"mcp.{self.label}")))

        read, write = await self._stack.enter_async_context(stdio_client(params, errlog=errlog))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await asyncio.wait_for(session.initialize(), START_TIMEOUT_SECONDS)
        listed = await asyncio.wait_for(session.list_tools(), START_TIMEOUT_SECONDS)

        self._session = session
        self.tools = self._read_only(listed.tools)
        logger.info("Started the %s MCP server with %d read-only tools", self.label, len(self.tools))

    def _read_only(self, listed: list[McpToolInfo]) -> list["McpTool"]:
        """
        Keep only the tools the server marks read-only. A tool without the mark is left out.

        Args:
            listed: The tools the server listed.
        Returns:
            One McpTool per read-only tool.
        """
        tools = []
        left_out = []
        for info in listed:
            if info.annotations is not None and info.annotations.readOnlyHint is True:
                tools.append(McpTool(self, info))
            else:
                left_out.append(info.name)

        if len(left_out) > 0:
            logger.info("Leaving out %s tools that are not read-only: %s", self.label, ", ".join(left_out))
        return tools

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolOutput:
        """
        Run one of the server's tools.

        Args:
            name: The tool's name on the server.
            arguments: The tool's arguments.
        Returns:
            The text the tool returned.
        """
        session = self._require_session()
        try:
            result = await session.call_tool(name, arguments, read_timeout_seconds=timedelta(seconds=CALL_TIMEOUT_SECONDS))
        except (McpError, anyio.ClosedResourceError, anyio.BrokenResourceError) as e:
            raise ToolError(f"The {self.label} MCP server could not run '{name}': {e}") from e

        text = _result_text(result)
        if result.isError:
            raise ToolError(text)
        return ToolOutput(text)

    async def stop(self) -> None:
        """Close the session and stop the server process."""
        self._session = None
        await self._stack.aclose()

    def _require_session(self) -> ClientSession:
        """
        Get the open session.

        Returns:
            The session.
        """
        if self._session is None:
            raise ToolError(f"The {self.label} MCP server is not running")
        return self._session


class McpTool(BaseTool):
    """One tool of a LocalMcpServer, offered to the LLM like a built-in tool."""

    def __init__(self, server: LocalMcpServer, info: McpToolInfo):
        """
        Args:
            server: The server that runs the tool.
            info: The tool's name, description and argument schema, as the server listed them.
        """
        self.name = info.name
        self.server = server
        self._info = info

    def schema(self) -> dict[str, Any]:
        """
        Describe the tool for the LLM, using the schema the server provided.

        Returns:
            A {"name", "description", "parameters"} dict.
        """
        description = self._info.description
        if description is None:
            description = ""
        return {
            "name": self.name,
            "description": description,
            "parameters": self._info.inputSchema,
        }

    def parse_arguments(self, arguments: str) -> dict[str, Any]:
        """
        Parse the LLM's JSON arguments. The server checks them against its schema.

        Args:
            arguments: JSON-encoded arguments from the tool call.
        Returns:
            The arguments as a dict.
        """
        # Models sometimes send an empty string for tools with no parameters.
        if arguments.strip() == "":
            return {}

        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as e:
            raise ToolError(f"Invalid JSON arguments for '{self.name}': {e}") from e
        if not isinstance(parsed, dict):
            raise ToolError(f"Arguments for '{self.name}' must be a JSON object")
        return parsed

    def needs_approval(self, args: dict[str, Any]) -> bool:
        """
        Read-only tools run without asking, like the other read tools.

        Args:
            args: Arguments returned by parse_arguments.
        Returns:
            False.
        """
        return False

    async def run(self, args: dict[str, Any]) -> ToolOutput:
        """
        Forward the call to the server.

        Args:
            args: Arguments returned by parse_arguments.
        Returns:
            The tool's text output.
        """
        return await self.server.call(self.name, args)


def _result_text(result: CallToolResult) -> str:
    """
    Join the text parts of a tool result. Other parts, such as images, are left out.

    Args:
        result: The server's result.
    Returns:
        The text, or a note when there was none.
    """
    parts = []
    for content in result.content:
        if isinstance(content, TextContent):
            parts.append(content.text)

    if len(parts) == 0:
        return "(The tool returned no text.)"
    return "\n".join(parts)


def _open_server_log(server_logger: logging.Logger) -> TextIO:
    """
    Make a file for a server's stderr whose lines are written to a logger. The server
    needs a real file descriptor, so this is the write end of a pipe, and a background
    thread reads the other end until the server and the file are both closed.

    Args:
        server_logger: Receives one record per line.
    Returns:
        The write end, to close once the server has stopped.
    """
    read_fd, write_fd = os.pipe()
    reader = threading.Thread(target=_forward_lines, args=(os.fdopen(read_fd), server_logger), daemon=True)
    reader.start()
    return os.fdopen(write_fd, "w")


def _forward_lines(pipe: TextIO, server_logger: logging.Logger) -> None:
    """
    Log each line a server writes, at the level the line names if it has one.

    Args:
        pipe: The read end of the server's stderr.
        server_logger: Receives one record per line.
    """
    levels = logging.getLevelNamesMapping()
    with pipe:
        for line in pipe:
            line = line.rstrip()
            if line == "":
                continue
            match = SERVER_LOG_LINE.match(line)
            if match is None:
                server_logger.info(line)
                continue
            level = levels.get(match.group("level"), logging.INFO)
            server_logger.log(level, match.group("message"))
