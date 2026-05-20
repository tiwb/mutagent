"""Conversation root view and Agent ↔ View adapter."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from mutgui import View, ViewBlock

if TYPE_CHECKING:
    from mutgui.events import Event
    from mutagent.agent import Agent
    from mutagent.main import App


class Conversation(View):
    """Root conversation shell for the built-in WebUI.

    路由权威集中在此类。``current_route`` 是单一真相源：

    - ``""``                — 对话主页
    - ``"settings"``        — 设置页（默认 panel）
    - ``"settings/<id>"``   — 设置页的指定 panel

    URL hash 由 mutgui 的 ``mutgui.setHash`` 命令 + ``$hashchange`` 事件
    通道双向同步——本类既不直接读写 ``window.location.hash``，也不依赖
    任何防循环标记位（W3C 规定 ``pushState`` 不触发 ``hashchange``，
    天然无回环）。
    """

    current_route: str
    agent: Agent
    app: App | None
    models: list[dict[str, Any]]
    current_model: str
    status: str
    is_busy: bool
    refresh_models: Any

    def __init__(self, *, agent: Agent, app: App | None = None) -> None: ...

    def render(self) -> ViewBlock: ...

    async def navigate_to(self, route: str) -> None: ...
    async def on_hash_change(self, hash_value: str) -> None: ...
    async def on_event(self, event: Event) -> bool: ...


from . import _conversation_impl as _conversation_impl  # noqa: E402,F401
