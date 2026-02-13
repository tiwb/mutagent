"""mutagent.client -- LLM client declaration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mutagent

if TYPE_CHECKING:
    from mutagent.messages import Message, Response, ToolSchema


class LLMClient(mutagent.Object):
    """LLM client interface.

    Provides an async interface for sending messages to a language model.
    The default implementation (Claude) is registered via @impl in
    builtins/claude.impl.py.

    Attributes:
        model: Model identifier (e.g. "claude-sonnet-4-20250514").
        api_key: API key for authentication.
        base_url: Base URL for the API endpoint.
    """

    model: str
    api_key: str
    base_url: str

    async def send_message(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        system_prompt: str = "",
    ) -> Response:
        """Send messages to the LLM and return the response.

        Args:
            messages: Conversation history.
            tools: Available tool schemas for the LLM to use.
            system_prompt: System-level instruction for the LLM.

        Returns:
            The LLM response containing the assistant message,
            stop reason, and token usage.
        """
        ...
