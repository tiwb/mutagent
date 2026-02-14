"""mutagent.client -- LLM client declaration."""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator

import mutagent

if TYPE_CHECKING:
    from mutagent.messages import Message, StreamEvent, ToolSchema


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
        stream: bool = True,
    ) -> AsyncIterator[StreamEvent]:
        """Send messages to the LLM and yield streaming events.

        When stream=True, the underlying HTTP request uses SSE streaming
        and events are yielded incrementally as they arrive.

        When stream=False, a regular HTTP request is made and the complete
        response is wrapped into a small sequence of StreamEvents.

        In both cases, the final event is always a ``response_done``
        carrying the complete Response object.

        Args:
            messages: Conversation history.
            tools: Available tool schemas for the LLM to use.
            system_prompt: System-level instruction for the LLM.
            stream: Whether to use SSE streaming for the HTTP request.

        Yields:
            StreamEvent instances.
        """
        ...
