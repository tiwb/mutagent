"""mutagent.agent -- Agent declaration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mutagent

if TYPE_CHECKING:
    from mutagent.client import LLMClient
    from mutagent.messages import Message, Response, ToolCall, ToolResult
    from mutagent.selector import ToolSelector


class Agent(mutagent.Object):
    """Agent manages the conversation loop with an LLM.

    The agent sends messages to the LLM, handles tool calls by dispatching
    them through the ToolSelector, and continues until the LLM signals
    end_turn.

    Attributes:
        client: The LLM client for sending messages.
        tool_selector: The tool selector for tool discovery and dispatch.
        system_prompt: System prompt for the LLM.
        messages: Conversation history.
    """

    client: LLMClient
    tool_selector: ToolSelector
    system_prompt: str
    messages: list

    async def run(self, user_input: str) -> str:
        """Run the agent with user input and return the final response.

        This is the main entry point. It adds the user message, then
        loops step/handle_tool_calls until the LLM produces an end_turn.

        Args:
            user_input: The user's input message.

        Returns:
            The final text response from the LLM.
        """
        ...

    async def step(self) -> Response:
        """Execute a single LLM call with the current messages and tools.

        Returns:
            The LLM response.
        """
        ...

    async def handle_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls and return results.

        Args:
            tool_calls: List of tool calls from the LLM.

        Returns:
            List of tool results.
        """
        ...
