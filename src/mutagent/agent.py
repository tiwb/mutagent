"""mutagent.agent -- Agent declaration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

import mutagent

if TYPE_CHECKING:
    from mutagent.client import LLMClient
    from mutagent.config import Config, Disposable
    from mutagent.context import AgentContext
    from mutagent.messages import Message, StreamEvent, ToolUseBlock
    from mutagent.tools import ToolSet


class Agent(mutagent.Declaration):
    """Agent manages the conversation loop with an LLM.

    The agent sends messages to the LLM, handles tool calls by dispatching
    them through the ToolSet, and continues until the LLM signals
    end_turn.

    Attributes:
        llm: The LLM client for sending messages.
        tools: The tool set for tool management and dispatch.
        context: Agent context managing prompts, messages, and token tracking.
        config: The shared configuration instance.
    """

    llm: LLMClient
    tools: ToolSet
    context: AgentContext
    config: Config
    session: Any  # 运行时由上层（如 mutbot）注入

    async def run(
        self,
        input_stream: AsyncIterator[Message],
        stream: bool = True,
        check_pending: Callable[[], bool] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run the agent conversation loop, consuming input messages and yielding output events.

        Args:
            input_stream: AsyncIterator of user input Messages. Messages containing
                a TurnStartBlock trigger agent processing; others are stored only.
            stream: Whether to use SSE streaming for the HTTP request.
            check_pending: Optional callback that returns True if new input
                is available.

        Yields:
            StreamEvent instances for each piece of incremental output.
        """
        yield agent_impl.run  # type: ignore[reportReturnType]

    async def step(self, stream: bool = True) -> AsyncIterator[StreamEvent]:
        """Execute a single LLM call, yielding streaming events."""
        yield agent_impl.step  # type: ignore[reportReturnType]

    async def handle_tool_calls(self, tool_calls: list[ToolUseBlock]) -> None:
        """Execute tool calls, updating each ToolUseBlock in-place.

        Args:
            tool_calls: List of ToolUseBlock from the LLM response.
        """
        return await agent_impl.handle_tool_calls(self, tool_calls)

    async def submit(self, text: str) -> None:
        """Submit one user turn for background processing."""
        return await agent_impl.submit(self, text)

    def cancel(self) -> bool:
        """Cancel the currently running turn if there is one."""
        return agent_impl.cancel(self)

    def subscribe(self, callback: Callable[[StreamEvent], Any]) -> Disposable:
        """Subscribe to streaming events emitted by submit()."""
        return agent_impl.subscribe(self, callback)

    def select_model(self, name: str) -> None:
        """Switch the model used for the next turn."""
        return agent_impl.select_model(self, name)

    def list_models(self) -> list[dict[str, Any]]:
        """List configured models for UI selectors."""
        return agent_impl.list_models(self)

    def is_busy(self) -> bool:
        """Return True when a turn is currently running."""
        return agent_impl.is_busy(self)


from .builtins import agent_impl
mutagent.register_module_impls(agent_impl)
