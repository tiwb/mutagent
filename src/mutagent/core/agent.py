"""mutagent.core.agent -- Agent declaration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import mutobj

if TYPE_CHECKING:
    from .context import AgentContext
    from .messages import StreamEvent, ToolResultBlock, ToolUseBlock
    from .llm import LLMApiClient
    from .tools import ToolSet


CancelFn = Callable[[], None]
"""取消订阅的回调类型。"""


class Agent(mutobj.Declaration):
    """Agent manages the conversation loop with an LLM.

    The agent sends messages to the LLM, handles tool calls by dispatching
    them through the ToolSet, and continues until the LLM signals
    end_turn.

    Attributes:
        llm: The LLM client for sending messages.
        model: The current model identifier.
        tools: The tool set for tool management and dispatch.
        context: Agent context managing prompts, messages, and token tracking.
    """

    llm: LLMApiClient
    model: str
    tools: ToolSet
    context: AgentContext

    async def handle_tool_calls(
        self, tool_calls: list[ToolUseBlock]
    ) -> list[ToolResultBlock]:
        """Execute tool calls and return their result blocks.

        Args:
            tool_calls: List of ToolUseBlock from the LLM response.
        """
        ...

    async def submit(self, text: str) -> None:
        """Submit one user turn for background processing."""
        ...

    def cancel(self) -> bool:
        """Cancel the currently running turn if there is one."""
        ...

    def subscribe(self, callback: Callable[[StreamEvent], Any]) -> CancelFn:
        """Subscribe to streaming events emitted by submit()."""
        ...

    def is_busy(self) -> bool:
        """Return True when a turn is currently running."""
        ...


from . import _agent_impl as _agent_impl  # noqa: F401, E402
