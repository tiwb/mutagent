"""Conversation root view and Agent ↔ View adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mutgui import View, ViewBlock

if TYPE_CHECKING:
    from mutagent.agent import Agent
    from mutagent.main import App


class Conversation(View):
    """Root conversation shell for the built-in WebUI."""

    def __init__(self, *, agent: Agent, app: App | None = None) -> None: ...

    def render(self) -> ViewBlock: ...


from . import _conversation_impl  # noqa: E402,F401
