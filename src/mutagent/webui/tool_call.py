"""Tool call card widget."""

from __future__ import annotations

from typing import Any

from mutgui import View, ViewBlock


class ToolCallCard(View):
    item: Any

    def __init__(self, *, item: Any) -> None: ...

    def render(self) -> ViewBlock: ...


from . import _tool_call_impl  # noqa: E402,F401
