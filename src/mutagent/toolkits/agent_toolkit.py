"""mutagent.agent_toolkit -- AgentToolkit declaration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mutagent.core.tools import Toolkit

if TYPE_CHECKING:
    from mutagent.core.agent import Agent


class AgentToolkit(Toolkit):
    """Tools for multi-agent delegation.

    Holds a set of pre-created Sub-Agent instances. The ``delegate``
    method dispatches a task to a named Sub-Agent, which runs to
    completion and returns its result.

    Each delegate call clears the Sub-Agent's message history first,
    so every call is an independent task.

    Attributes:
        agents: Dict of pre-created Sub-Agent instances keyed by name.
    """

    agents: dict[str, Agent]

    def delegate(self, agent_name: str, task: str) -> str:
        """Delegate a task to a named Sub-Agent.

        Args:
            agent_name: Name of the Sub-Agent to delegate to.
            task: Task description for the Sub-Agent.

        Returns:
            The Sub-Agent's execution result as text.
        """
        ...


from . import _agent_toolkit_impl as _agent_toolkit_impl  # noqa: F401, E402
