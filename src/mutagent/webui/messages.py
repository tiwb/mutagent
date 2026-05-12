"""Message list widgets and chat item models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from mutgui import View, ViewBlock


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



class ChatItemView(View):
    """所有 ChatItem 渲染器的统一基类。

    子类声明 ``item_type`` 绑定到具体 ChatItem 数据类，
    ``ChatItemView.for_item()`` 据此自动分派 View 类。
    上层项目新增 ChatItem + ChatItemView 子类即可扩展消息类型，
    无需修改 mutagent 本身。
    """

    item: ChatItem
    item_type: ClassVar[type[ChatItem]] = ChatItem  # 子类必须覆盖

    def __init__(self, *, item: ChatItem) -> None: ...

    def render(self) -> ViewBlock: ...

    @classmethod
    def for_item(cls, item: ChatItem) -> "ChatItemView": ...


class MessageList(View):
    items: list[ChatItem]

    def __init__(self, *, items: list[ChatItem] | None = None) -> None: ...

    def refresh(self) -> None: ...

    def invalidate_item(self, item_id: str) -> None: ...

    def render(self) -> ViewBlock: ...


class UserMessage(ChatItemView):
    item_type: ClassVar[type[ChatItem]] = UserTextItem
    item: UserTextItem

    def __init__(self, *, item: UserTextItem) -> None: ...

    def render(self) -> ViewBlock: ...


class AssistantMessage(ChatItemView):
    item_type: ClassVar[type[ChatItem]] = AssistantTextItem
    item: AssistantTextItem

    def __init__(self, *, item: AssistantTextItem) -> None: ...

    def render(self) -> ViewBlock: ...


class AssistantError(ChatItemView):
    item_type: ClassVar[type[ChatItem]] = AssistantErrorItem
    item: AssistantErrorItem

    def __init__(self, *, item: AssistantErrorItem) -> None: ...

    def render(self) -> ViewBlock: ...


class TurnSeparator(ChatItemView):
    item_type: ClassVar[type[ChatItem]] = TurnSeparatorItem
    item: TurnSeparatorItem

    def __init__(self, *, item: TurnSeparatorItem) -> None: ...

    def render(self) -> ViewBlock: ...


class ToolCallCard(ChatItemView):
    item_type: ClassVar[type[ChatItem]] = ToolCallItem
    item: ToolCallItem

    def __init__(self, *, item: ToolCallItem) -> None: ...

    def render(self) -> ViewBlock: ...


from . import _messages_impl  # noqa: E402,F401
