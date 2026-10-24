"""
The interfaces every model provider implements. The agent only talks to chat models
through LLM.complete, using the provider-agnostic Message type, and to embedding models
through Embedder.embed, so swapping providers means writing new subclasses without
touching the agent or memory.
"""

from abc import ABC, abstractmethod
from typing import Any

from agent.models import Message


class LLMError(Exception):
    """Raised when a model call fails in a way the agent should not continue past."""


class LLM(ABC):
    """A chat model that can produce the next assistant message."""

    @abstractmethod
    async def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> Message:
        """
        Generate the next assistant message.

        Args:
            messages: Full model input, system prompt first.
            tools: Available tools, each as {"name", "description", "parameters"} where
                parameters is a JSON schema. Empty when no tools should be offered.
        Returns:
            The assistant's reply, possibly containing tool calls.
        """


class Embedder(ABC):
    """A model that turns text into vectors whose distance reflects meaning."""

    model: str

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed several texts in one request.

        Args:
            texts: The texts to embed.
        Returns:
            One vector per text, in the same order.
        """
