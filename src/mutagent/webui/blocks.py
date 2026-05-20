"""Block renderer and specialized fenced-block widgets."""

from __future__ import annotations

from mutgui import View, ViewBlock


class BlockRenderer(View):
    text: str

    def __init__(self, *, text: str = "") -> None: ...

    def render(self) -> ViewBlock: ...


class ThinkingBlock(View):
    body: str
    expanded: bool

    def __init__(self, *, body: str = "", expanded: bool = False) -> None: ...

    def render(self) -> ViewBlock: ...


from . import _blocks_impl as _blocks_impl  # noqa: E402,F401
