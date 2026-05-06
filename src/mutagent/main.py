"""mutagent.main -- Bootstrap and Main entry point."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mutagent
from mutagent.config import Config

if TYPE_CHECKING:
    from mutagent.agent import Agent
    from mutagent.userio import UserIO


class App(mutagent.Declaration):
    """App entry point.  Override via ``@impl`` for custom UI (e.g. TUI).

    Interaction with the user (input collection, output rendering) is
    delegated to the ``userio`` attribute, a :class:`UserIO` instance.

    Attributes:
        config: The loaded Config object.
        agent: The Agent for this session, set by ``setup_agent()``.
        userio: The UserIO instance handling user interaction.
    """

    config: Config
    config_path: Path
    agent: Agent
    userio: UserIO

    def load_config(self, config_path: str = ".mutagent/config.json") -> None:
        """Load configuration from the given path and store in ``self.config``.

        Override if you want to control config loading (e.g. different path,
        different format, etc.).

        Args:
            config_path: Path to the config file.
        """
        return main_impl.load_config(self, config_path)

    def setup_agent(self, system_prompt: str = "") -> Agent:
        """Initialise the session Agent and store it in ``self.agent``.

        Also creates the UserIO instance and stores it in ``self.userio``.
        Override to customise component assembly (different tools,
        different LLMClient, etc.).

        Args:
            system_prompt: System prompt for the agent.

        Returns:
            The created Agent instance (also stored as ``self.agent``).
        """
        return main_impl.setup_agent(self, system_prompt=system_prompt)

    def run(self) -> None:
        """Run the agent session loop.

        The default implementation calls ``setup_agent()`` then enters
        a terminal REPL.  Override for TUI, Web, or other interfaces.
        """
        return main_impl.run(self)

    def run_webui(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        open_browser: bool = True,
    ) -> None:
        """Run the built-in WebUI server."""
        return main_impl.run_webui(
            self,
            host=host,
            port=port,
            open_browser=open_browser,
        )


def main() -> None:
    """Bootstrap mutagent.  Not overridable.
    """
    import argparse
    from mutagent.webui.cli import add_webui_subcommand, dispatch_webui

    parser = argparse.ArgumentParser(description="mutagent — AI Agent Framework")
    parser.add_argument("-V", "--version", action="version", version=f"mutagent {mutagent.__version__}")
    parser.add_argument("--config", default=".mutagent/config.json",
                        help="Path to config file (default: .mutagent/config.json)")
    parser.add_argument("--headless", action="store_true",
                        help="Explicitly use the default terminal UI")
    subparsers = parser.add_subparsers(dest="command")
    add_webui_subcommand(subparsers)
    args = parser.parse_args()

    if args.command == "webui" and args.headless:
        parser.error("--headless cannot be used together with the webui subcommand")

    app = App()
    app.load_config(args.config)
    if args.command == "webui":
        dispatch_webui(app, args)
        return
    app.run()


from .builtins import main_impl
mutagent.register_module_impls(main_impl)
