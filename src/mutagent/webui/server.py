"""WebUI server declaration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mutio.net.server import Server

if TYPE_CHECKING:
    from mutagent.core.agent import Agent
    from mutagent.app.app import App
    from mutagent.webui.conversation import Conversation


class WebUIServer(Server):
    app: App
    agent: Agent
    conversation: Conversation

    def __init__(
        self,
        *,
        app: App,
        agent: Agent,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None: ...


from . import _server_impl as _server_impl  # noqa: E402,F401
