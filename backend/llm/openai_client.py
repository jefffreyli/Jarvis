"""
LLM implementation backed by OpenAI's Responses API. Each call converts our Messages
and tool schemas into Responses input items, sends one non-streaming request with
store=False (our Session stays the source of truth), and converts the output items
back into a Message. Tool results with images are sent as text plus base64 data URL
images. Reasoning items in the output are not kept between calls.
Remote MCP servers (such as Notion's) are offered as OpenAI's hosted mcp tool, so OpenAI
calls them during the reply without approval. Those calls come back as HostedCalls and
are replayed on later requests as function calls with their outputs, since nothing is
stored on OpenAI's side. If a server cannot be used (no token, or OpenAI cannot reach
it), the request goes ahead without it rather than failing.
Transient failures (rate limits, timeouts, 5xx) are retried by the OpenAI SDK; anything
still failing, or a response that did not complete, raises an error instead of
returning a partial answer.
"""

import base64
import logging
from typing import Any

from openai import APIStatusError, AsyncOpenAI
from openai.types.responses import Response

from agent.models import HostedCall, Image, Message, Role, ToolCall

from .base import LLM, Embedder, LLMError
from .hosted_mcp import HostedMcpServer, McpAuthError

logger = logging.getLogger(__name__)

# OpenAI answers 424 Failed Dependency when it cannot use a remote MCP server.
MCP_FAILED_STATUS = 424


class OpenAIClient(LLM):
    """Responses API client for a single OpenAI model."""

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: float,
        max_retries: int,
        mcp_servers: list[HostedMcpServer] | None = None,
    ):
        """
        Args:
            api_key: OpenAI API key.
            model: Model id to request, such as "gpt-6-sol".
            timeout: Seconds to wait for each request before giving up.
            max_retries: How many times the SDK retries transient failures.
            mcp_servers: Remote MCP servers OpenAI may call while answering, when tools are offered.
        """
        if mcp_servers is None:
            mcp_servers = []

        self.model = model
        self.mcp_servers = mcp_servers
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)

    async def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> Message:
        """
        Send the conversation to OpenAI and return the assistant's reply.

        Args:
            messages: Full model input, system prompt first.
            tools: Available tools as {"name", "description", "parameters"} dicts. When
                empty, no tools are offered, remote MCP servers included.
        Returns:
            The assistant's reply.
        """
        input_items = []
        for message in messages:
            for item in _to_input_items(message):
                input_items.append(item)

        openai_tools = []
        for tool in tools:
            openai_tools.append(_to_openai_tool(tool))

        mcp_tools = []
        if len(tools) > 0:
            mcp_tools = await self._mcp_tools()

        try:
            response = await self._create(input_items, openai_tools + mcp_tools)
        except APIStatusError as e:
            if e.status_code != MCP_FAILED_STATUS or len(mcp_tools) == 0:
                raise
            logger.warning("OpenAI could not use a remote MCP server, retrying without it: %s", e)
            response = await self._create(input_items, openai_tools)

        _check_completed(response)
        return _from_response(response)

    async def _create(self, input_items: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Response:
        """
        Send one Responses API request.

        Args:
            input_items: The conversation as input items.
            tools: Tools in OpenAI's format.
        Returns:
            The API response.
        """
        return await self._client.responses.create(
            model=self.model,
            input=input_items,
            tools=tools,
            store=False,
        )

    async def _mcp_tools(self) -> list[dict[str, Any]]:
        """
        Describe each remote MCP server as an OpenAI mcp tool with a current access token.
        A server whose token cannot be had is left out.

        Returns:
            The mcp tools.
        """
        mcp_tools = []
        for server in self.mcp_servers:
            try:
                token = await server.auth.access_token()
            except McpAuthError as e:
                logger.warning("Leaving out the %s MCP server: %s", server.label, e)
                continue
            mcp_tools.append(
                {
                    "type": "mcp",
                    "server_label": server.label,
                    "server_url": server.url,
                    "server_description": server.description,
                    "authorization": token,
                    "require_approval": "never",
                }
            )
        return mcp_tools


class OpenAIEmbedder(Embedder):
    """Embeddings API client for a single OpenAI embedding model."""

    def __init__(self, api_key: str, model: str, timeout: float, max_retries: int):
        """
        Args:
            api_key: OpenAI API key.
            model: Embedding model id, such as "text-embedding-3-small".
            timeout: Seconds to wait for each request before giving up.
            max_retries: How many times the SDK retries transient failures.
        """
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed several texts in one request.

        Args:
            texts: The texts to embed.
        Returns:
            One vector per text, in the same order.
        """
        response = await self._client.embeddings.create(model=self.model, input=texts)

        # The API labels each vector with the index of its input, so order by that.
        vectors_by_index = {}
        for item in response.data:
            vectors_by_index[item.index] = item.embedding

        vectors = []
        for index in range(len(texts)):
            if index not in vectors_by_index:
                raise LLMError(f"OpenAI returned no embedding for input {index}")
            vectors.append(vectors_by_index[index])
        return vectors


def _to_input_items(message: Message) -> list[dict[str, Any]]:
    """
    Convert one of our messages to Responses API input items. An assistant message
    becomes its hosted calls (each as a function call and its output), then its text
    (if any), then one item per tool call.

    Args:
        message: The message to convert.
    Returns:
        The equivalent input items, in order.
    """
    if message.role == Role.TOOL:
        return [{"type": "function_call_output", "call_id": message.tool_call_id, "output": _tool_output(message)}]

    items = []
    # Replayed as function calls, which the model reads reliably without stored state.
    for hosted in message.hosted_calls:
        items.append(
            {
                "type": "function_call",
                "call_id": hosted.id,
                "name": hosted.name,
                "arguments": hosted.arguments,
            }
        )
        items.append({"type": "function_call_output", "call_id": hosted.id, "output": hosted.output})

    if message.content != "":
        items.append({"role": message.role.value, "content": message.content})

    for call in message.tool_calls:
        items.append(
            {
                "type": "function_call",
                "call_id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            }
        )
    return items


def _tool_output(message: Message) -> str | list[dict[str, Any]]:
    """
    Build the output of a function_call_output item.

    Args:
        message: A tool result message.
    Returns:
        The text alone, or a list of the text and each image when there are images.
    """
    if len(message.images) == 0:
        return message.content

    parts: list[dict[str, Any]] = [{"type": "input_text", "text": message.content}]
    for image in message.images:
        parts.append({"type": "input_image", "image_url": _data_url(image)})
    return parts


def _data_url(image: Image) -> str:
    """
    Encode an image as a data URL, which the API accepts in place of a web address.

    Args:
        image: The image.
    Returns:
        Text like "data:image/jpeg;base64,...".
    """
    encoded = base64.b64encode(image.data).decode("ascii")
    return f"data:{image.media_type};base64,{encoded}"


def _to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a provider-agnostic tool schema to a Responses API function tool.

    Args:
        tool: A {"name", "description", "parameters"} dict.
    Returns:
        The tool in OpenAI's format.
    """
    return {
        "type": "function",
        "name": tool["name"],
        "description": tool["description"],
        "parameters": tool["parameters"],
        # Strict mode requires every parameter, which rules out optional ones. Our tool
        # registry validates arguments itself.
        "strict": False,
    }


def _check_completed(response: Response) -> None:
    """
    Raise if the response did not finish normally.

    Args:
        response: The API response.
    """
    if response.status == "completed":
        return

    if response.status == "incomplete" and response.incomplete_details is not None:
        reason = response.incomplete_details.reason
        if reason == "max_output_tokens":
            raise LLMError("The reply was cut off by the output token limit")
        if reason == "content_filter":
            raise LLMError("The reply was blocked by OpenAI's content filter")

    if response.error is not None:
        raise LLMError(f"OpenAI response failed: {response.error.message}")
    raise LLMError(f"OpenAI response ended with status '{response.status}'")


def _from_response(response: Response) -> Message:
    """
    Convert the response's output items to one assistant message.

    Args:
        response: A completed API response.
    Returns:
        The assistant's reply, with its text, any tool calls and any hosted calls.
    """
    text_parts = []
    tool_calls = []
    hosted_calls = []

    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "output_text":
                    text_parts.append(part.text)
                elif part.type == "refusal":
                    text_parts.append(part.refusal)
        elif item.type == "function_call":
            tool_calls.append(ToolCall(id=item.call_id, name=item.name, arguments=item.arguments))
        elif item.type == "mcp_call":
            hosted_calls.append(_to_hosted_call(item))
        elif item.type == "mcp_list_tools":
            if item.error:
                logger.warning("Listing the %s MCP server's tools failed: %s", item.server_label, item.error)
        elif item.type == "reasoning":
            continue  # The model's private reasoning; not part of the reply.
        else:
            raise LLMError(f"Unsupported output item type from OpenAI: {item.type}")

    content = "\n".join(text_parts)
    return Message.assistant(content, tuple(tool_calls), tuple(hosted_calls))


def _to_hosted_call(item: Any) -> HostedCall:
    """
    Convert an mcp_call output item, which OpenAI already ran, to a HostedCall.

    Args:
        item: The mcp_call item.
    Returns:
        The hosted call, with the error as its output if the call failed.
    """
    if item.error:
        output = f"Error: {item.error}"
    elif item.output is None:
        output = ""
    else:
        output = item.output
    return HostedCall(id=item.id, server=item.server_label, name=item.name, arguments=item.arguments, output=output)
