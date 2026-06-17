"""Message list widgets and chat item models — Declaration + Implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

import mutobj
from ._blocks import BlockRenderer
from mutgui import Callback, View, ViewBlock, VirtualList, VirtualListItemAdapter
from mutio.codec.json import JsonObject

_T = TypeVar("_T", bound="ChatItem")


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
    input_kwargs: JsonObject = mutobj.field(default_factory=dict)
    result_text: str = ""
    status: str = "pending"
    is_error: bool = False
    duration: float = 0.0
    expanded: bool = True


@dataclass(slots=True)
class ThinkingBlockItem(ChatItem):
    thinking: str
    signature: str = ""
    data: str = ""
    expanded: bool = False


class ChatItemView(View, Generic[_T]):
    """所有 ChatItem 渲染器的统一基类。

    泛型参数 ``_T`` 绑定具体 ChatItem 子类，子类继承时指定：
    ``UserMessage(ChatItemView[UserTextItem])``。

    ``item_type`` ClassVar 用于运行时自动发现，
    ``ChatItemView.for_item()`` 据此自动分派 View 类。
    上层项目新增 ChatItem + ChatItemView 子类即可扩展消息类型，
    无需修改 mutagent 本身。
    """

    item: _T
    item_type: ClassVar[type[ChatItem]] = ChatItem  # 子类必须覆盖

    def render(self) -> ViewBlock: ...

    @classmethod
    def for_item(cls, item: ChatItem) -> "ChatItemView[ChatItem]": ...


class MessageList(View):
    items: list[ChatItem] = mutobj.field(default_factory=list)
    adapter: _MessageListAdapter
    virtual_list: VirtualList

    def __init__(self) -> None: ...

    def append_item(self, item: ChatItem) -> None: ...

    def find_item(self, item_id: str) -> ChatItem | None: ...

    def replace_items(self, items: list[ChatItem]) -> None: ...

    def refresh(self) -> None: ...

    def invalidate_item(self, item_id: str) -> None: ...

    def render(self) -> ViewBlock: ...


class UserMessage(ChatItemView[UserTextItem]):
    item_type: ClassVar[type[ChatItem]] = UserTextItem
    item: UserTextItem

    def render(self) -> ViewBlock: ...


class AssistantMessage(ChatItemView[AssistantTextItem]):
    item_type: ClassVar[type[ChatItem]] = AssistantTextItem
    item: AssistantTextItem
    renderer: BlockRenderer

    def __init__(self, *, item: AssistantTextItem) -> None: ...

    def render(self) -> ViewBlock: ...


class AssistantError(ChatItemView[AssistantErrorItem]):
    item_type: ClassVar[type[ChatItem]] = AssistantErrorItem
    item: AssistantErrorItem

    def render(self) -> ViewBlock: ...


class TurnSeparator(ChatItemView[TurnSeparatorItem]):
    item_type: ClassVar[type[ChatItem]] = TurnSeparatorItem
    item: TurnSeparatorItem

    def render(self) -> ViewBlock: ...


class ToolCallCard(ChatItemView[ToolCallItem]):
    item_type: ClassVar[type[ChatItem]] = ToolCallItem
    item: ToolCallItem

    def render(self) -> ViewBlock: ...


class ThinkingBlockView(ChatItemView[ThinkingBlockItem]):
    item_type: ClassVar[type[ChatItem]] = ThinkingBlockItem
    item: ThinkingBlockItem

    def render(self) -> ViewBlock: ...


# ---------------------------------------------------------------------------
# ChatItemView 基类实现
# ---------------------------------------------------------------------------

# 类型驱动的 ChatItem -> ChatItemView 映射缓存。
# 按 mutobj 注册表 generation 失效，首次/变更时重建一次，其余 O(1)。
_view_map_cache: dict[type[ChatItem], type[ChatItemView[Any]]] = {}
_view_map_generation: int = -1


def _resolve_view_class(item_type: type[ChatItem]) -> type[ChatItemView[Any]]:
    """按 ChatItem 类型查 ChatItemView 子类。"""
    global _view_map_generation
    gen = mutobj.get_registry_generation()
    if gen != _view_map_generation:
        _view_map_cache.clear()
        for view_cls in mutobj.discover_subclasses(ChatItemView):
            registered = view_cls.item_type
            if registered is ChatItem:
                continue  # 未覆盖默认值的子类跳过
            if registered in _view_map_cache:
                raise RuntimeError(
                    f"Duplicate ChatItemView for {registered.__name__}: "
                    f"{_view_map_cache[registered].__name__} vs {view_cls.__name__}"
                )
            _view_map_cache[registered] = view_cls
        _view_map_generation = gen
    try:
        return _view_map_cache[item_type]
    except KeyError:
        raise TypeError(
            f"No ChatItemView registered for {item_type.__name__}"
        ) from None


@mutobj.impl(ChatItemView[ChatItem].render)
def chat_item_view_render(self: ChatItemView[ChatItem]) -> ViewBlock:
    raise NotImplementedError


@mutobj.impl(ChatItemView.for_item)
def chat_item_view_for_item(cls: type[ChatItemView[ChatItem]], item: ChatItem) -> ChatItemView[ChatItem]:
    view_cls = _resolve_view_class(type(item))
    return view_cls(item=item)


# ---------------------------------------------------------------------------
# MessageList
# ---------------------------------------------------------------------------


class _MessageListAdapter(VirtualListItemAdapter):
    message_list: MessageList

    @property
    def item_count(self) -> int:
        return len(self.message_list.items)

    def item_id(self, index: int) -> str:
        return self.message_list.items[index].id

    def create_item_view(self, index: int) -> View:
        return ChatItemView.for_item(self.message_list.items[index])

    def invalidate_existing_item(self, item_id: str) -> None:
        for virtual_list in self.virtual_lists:
            item_view = virtual_list.get_item_view(item_id)
            if item_view is not None:
                item_view.invalidate()
                return
        self.invalidate()


@mutobj.impl(MessageList.__init__)
def message_list_init(self: MessageList) -> None:
    super(MessageList, self).__init__()
    self.id = "message-list"
    self.adapter = _MessageListAdapter(message_list=self)
    self.virtual_list = VirtualList(
        id="chat-list",
        adapter=self.adapter,
        stick_to_bottom=True,
        estimated_item_height=128,
    )


@mutobj.impl(MessageList.refresh)
def message_list_refresh(self: MessageList) -> None:
    self.adapter.invalidate()
    self.invalidate()


@mutobj.impl(MessageList.invalidate_item)
def message_list_invalidate_item(self: MessageList, item_id: str) -> None:
    self.adapter.invalidate_existing_item(item_id)


@mutobj.impl(MessageList.render)
def message_list_render(self: MessageList) -> ViewBlock:
    return ViewBlock([
        {
            "$component": "html.div",
            "$id": "message-list-shell",
            "style": {
                "height": "100%",
                "minHeight": 0,
                "display": "flex",
                "flexDirection": "column",
                "overflow": "hidden",
            },
            "$children": [self.virtual_list],
        }
    ])


@mutobj.impl(MessageList.append_item)
def message_list_append_item(self: MessageList, item: ChatItem) -> None:
    self.items.append(item)
    self.refresh()


@mutobj.impl(MessageList.find_item)
def message_list_find_item(self: MessageList, item_id: str) -> ChatItem | None:
    for item in self.items:
        if item.id == item_id:
            return item
    return None


@mutobj.impl(MessageList.replace_items)
def message_list_replace_items(self: MessageList, items: list[ChatItem]) -> None:
    self.items[:] = items
    self.refresh()

# ---------------------------------------------------------------------------
# UserMessage
# ---------------------------------------------------------------------------


@mutobj.impl(UserMessage.render)
def user_message_render(self: UserMessage) -> ViewBlock:
    return ViewBlock([
        {
            "$component": "mutagent.UserMessage",
            "$id": self.item.id,
            "role": "你",
            "timestamp": self.item.timestamp,
            "text": self.item.text,
        }
    ])


# ---------------------------------------------------------------------------
# AssistantMessage
# ---------------------------------------------------------------------------


@mutobj.impl(AssistantMessage.__init__)
def assistant_message_init(
    self: AssistantMessage, *, item: AssistantTextItem
) -> None:
    super(AssistantMessage, self).__init__(item=item)
    self.renderer = BlockRenderer(text=item.text)
    self.renderer.id = f"block-renderer-{item.id}"


@mutobj.impl(AssistantMessage.render)
def assistant_message_render(self: AssistantMessage) -> ViewBlock:
    if self.renderer.text != self.item.text:
        self.renderer = BlockRenderer(text=self.item.text)
        self.renderer.id = f"block-renderer-{self.item.id}"
    return ViewBlock([
        {
            "$component": "mutagent.AssistantMessage",
            "$id": self.item.id,
            "role": "助手",
            "model": self.item.model,
            "timestamp": self.item.timestamp,
            "$children": [self.renderer],
        }
    ])


# ---------------------------------------------------------------------------
# AssistantError
# ---------------------------------------------------------------------------


@mutobj.impl(AssistantError.render)
def assistant_error_render(self: AssistantError) -> ViewBlock:
    return ViewBlock([
        {
            "$component": "mutagent.AssistantError",
            "$id": self.item.id,
            "role": "错误",
            "timestamp": self.item.timestamp,
            "error": self.item.error,
        }
    ])


# ---------------------------------------------------------------------------
# TurnSeparator
# ---------------------------------------------------------------------------


@mutobj.impl(TurnSeparator.render)
def turn_separator_render(self: TurnSeparator) -> ViewBlock:
    return ViewBlock([
        {
            "$component": "mutagent.TurnSeparator",
            "$id": self.item.id,
            "duration": self.item.duration,
            "inputTokens": self.item.input_tokens,
            "outputTokens": self.item.output_tokens,
        }
    ])


# ---------------------------------------------------------------------------
# ThinkingBlockView
# ---------------------------------------------------------------------------


def _toggle_thinking(*, view: ThinkingBlockView) -> None:
    view.item.expanded = not view.item.expanded
    view.invalidate()


@mutobj.impl(ThinkingBlockView.render)
def thinking_block_view_render(self: ThinkingBlockView) -> ViewBlock:
    return ViewBlock([
        {
            "$component": "mutagent.ThinkingBlock",
            "$id": self.item.id,
            "thinking": self.item.thinking,
            "expanded": self.item.expanded,
            "$children": [
                {
                    "$component": "antd.Button",
                    "size": "small",
                    "children": "展开" if not self.item.expanded else "收起",
                    "onClick": Callback(_toggle_thinking, view=self),
                }
            ],
        }
    ])


# ---------------------------------------------------------------------------
# ToolCallCard
# ---------------------------------------------------------------------------


@mutobj.impl(ToolCallCard.render)
def tool_call_card_render(self: ToolCallCard) -> ViewBlock:
    return ViewBlock([
        {
            "$component": "mutagent.ToolCallCard",
            "$id": self.item.id,
            "name": self.item.name,
            "status": self.item.status,
            "input": self.item.input_kwargs,
            "resultText": self.item.result_text or None,
            "isError": self.item.is_error,
        }
    ])
