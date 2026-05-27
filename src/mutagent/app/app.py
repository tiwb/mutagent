"""mutagent.app.app -- Bootstrap and Main entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mutobj
from mutagent.app.config import Config

if TYPE_CHECKING:
    from mutagent.core.agent import Agent
    from mutagent.sandbox.app import SandboxApp


class App(mutobj.Declaration):
    """App entry point.

    Attributes:
        config: The loaded Config object.
        agent: The Agent for this session, set by ``setup_agent()``.
        sandbox: The SandboxApp for this session, set by ``setup_agent()``.
    """

    config: Config
    agent: Agent
    sandbox: SandboxApp

    def load_config(self, config_path: str = ".mutagent/config.json") -> None:
        """Load configuration from the given path and store in ``self.config``.

        Override if you want to control config loading (e.g. different path,
        different format, etc.).

        Args:
            config_path: Path to the config file.
        """
        ...

    def setup_agent(self, system_prompt: str = "") -> Agent:
        """Initialise the session Agent and store it in ``self.agent``.

        Also creates the UserIO instance (``self.userio``) and an empty
        SandboxApp (``self.sandbox``).  This method is **synchronous and
        does NOT connect to MCP/CLI sources**—call ``connect_sources()``
        in the appropriate event loop afterwards to populate sandbox
        namespaces.

        Override to customise component assembly (different tools,
        different LLMProvider, etc.).

        Args:
            system_prompt: System prompt for the agent.

        Returns:
            The created Agent instance (also stored as ``self.agent``).
        """
        ...

    async def connect_sources(self) -> None:
        """Connect ``mcp_sources`` / ``cli_sources`` and inject namespaces
        into ``self.sandbox``.

        Must be awaited in the event loop where the agent will run—MCP
        clients (httpx-based and stdio-based) bind to the loop captured
        at connection time.  Calling from a temporary loop (e.g.
        ``asyncio.run``) will deactivate clients once that loop exits.

        MCP source config supports two extra fields:

        - ``autostart`` (default ``true``): if true, kick off connection in
          the background; otherwise the connection is deferred until the
          first attribute access on the namespace (lazy).
        - ``retry_cooldown`` (seconds, default ``5``): after a failed
          connection attempt, calls within this window raise the cached
          error instead of re-attempting.  Set ``0`` to disable.

        Connection failures never drop the namespace—``help()`` will
        show the connection state and the next call attempt will retry.
        """
        ...


from . import _app_impl as _app_impl  # noqa: F401, E402