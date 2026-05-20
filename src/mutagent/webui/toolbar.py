"""Toolbar widgets: status bar."""

from __future__ import annotations

from typing import Any

from mutgui import View, ViewBlock


class AgentStatusBar(View):
    status: str
    input_tokens: int
    output_tokens: int
    context_percent: float
    context_total: int
    context_used: int
    total_cost: float
    cache_read_tokens: int
    cache_write_tokens: int
    expanded: bool

    def __init__(
        self,
        *,
        status: str = "idle",
        input_tokens: int = 0,
        output_tokens: int = 0,
        context_percent: float = 0.0,
        context_total: int = 0,
        context_used: int = 0,
        total_cost: float = 0.0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        expanded: bool = False,
    ) -> None: ...

    def render(self) -> ViewBlock: ...


from . import _toolbar_impl as _toolbar_impl  # noqa: E402,F401
