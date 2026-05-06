"""Message list widgets and chat item models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mutgui import View, ViewBlock, VirtualList, VirtualListItemAdapter


@dataclass(slots=True)
class ChatItem:
    id: str
    kind: str


@dataclass(slots=True)
class UserTextItem(ChatItem):
    text: str
    timestamp: float = 0.0


@dataclass(slots=True)
class AssistantTextItem(ChatItem):
    text: str
    model: str = ""
    timestamp: float = 0.0
    duration: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(slots=True)
class AssistantErrorItem(ChatItem):
    error: str
    timestamp: float = 0.0


@dataclass(slots=True)
class TurnSeparatorItem(ChatItem):
    turn_id: str
    duration: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(slots=True)
class ToolCallItem(ChatItem):
    tool_id: str
    name: str
    input_text: str = ""
    result_text: str = ""
    status: str = "pending"
    is_error: bool = False
    duration: float = 0.0
    expanded: bool = True


class MessageList(View):
    items: list[ChatItem]

    def __init__(self, *, items: list[ChatItem] | None = None) -> None: ...

    def refresh(self) -> None: ...

    def invalidate_item(self, item_id: str) -> None: ...

    def render(self) -> ViewBlock: ...


class UserMessage(View):
    item: UserTextItem

    def __init__(self, *, item: UserTextItem) -> None: ...

    def render(self) -> ViewBlock: ...


class AssistantMessage(View):
    item: AssistantTextItem

    def __init__(self, *, item: AssistantTextItem) -> None: ...

    def render(self) -> ViewBlock: ...


class AssistantError(View):
    item: AssistantErrorItem

    def __init__(self, *, item: AssistantErrorItem) -> None: ...

    def render(self) -> ViewBlock: ...


class TurnSeparator(View):
    item: TurnSeparatorItem

    def __init__(self, *, item: TurnSeparatorItem) -> None: ...

    def render(self) -> ViewBlock: ...


from . import _messages_impl  # noqa: E402,F401
