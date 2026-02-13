"""mutagent.selector -- ToolSelector declaration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mutagent

if TYPE_CHECKING:
    from mutagent.essential_tools import EssentialTools
    from mutagent.messages import ToolCall, ToolResult, ToolSchema


class ToolSelector(mutagent.Object):
    """Tool selection and dispatch -- the sole bridge between Agent and tools.

    Responsibilities:
    1. Decide which tools are available in the current context (get_tools).
    2. Route LLM tool calls to concrete implementations (dispatch).

    Both responsibilities are fully determined by the @impl, allowing
    the Agent to evolve tool selection over time.

    Attributes:
        essential_tools: The EssentialTools instance providing tool methods.
    """

    essential_tools: EssentialTools

    async def get_tools(self, context: dict) -> list[ToolSchema]:
        """Return the list of tool schemas available in the current context.

        Args:
            context: Contextual information that may influence tool selection.

        Returns:
            List of ToolSchema objects describing available tools.
        """
        ...

    async def dispatch(self, tool_call: ToolCall) -> ToolResult:
        """Route a tool call to the appropriate implementation and execute it.

        Args:
            tool_call: The tool call from the LLM.

        Returns:
            The result of executing the tool.
        """
        ...
