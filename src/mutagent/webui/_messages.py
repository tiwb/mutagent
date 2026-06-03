"""Message list widgets and chat item models — Declaration + Implementation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, ClassVar, Generic, TypeVar

import mutobj
from ._blocks import BlockRenderer
from mutgui import Callback, View, ViewBlock, VirtualList, VirtualListItemAdapter

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
    input_text: str = ""
    result_text: str = ""
    status: str = "pending"
    is_error: bool = False
    duration: float = 0.0
    expanded: bool = True


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


# ── 工具函数 ──────────────────────────────────────────────────


def _format_clock(timestamp: float) -> str:
    if not timestamp:
        return ""
    return time.strftime("%H:%M:%S", time.localtime(timestamp))


def _role_meta(role: str, model: str = "", timestamp: float = 0.0) -> str:
    parts = [role]
    if model:
        parts.append(model)
    clock = _format_clock(timestamp)
    if clock:
        parts.append(clock)
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# ChatItemView 基类实现
# ---------------------------------------------------------------------------

# 类型驱动的 ChatItem -> ChatItemView 映射缓存。
# 按 mutobj 注册表 generation 失效，首次/变更时重建一次，其余 O(1)。
_view_map_cache: dict[type[ChatItem], type[ChatItemView]] = {}
_view_map_generation: int = -1


def _resolve_view_class(item_type: type[ChatItem]) -> type[ChatItemView]:
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


@mutobj.impl(ChatItemView.for_item)
def chat_item_view_for_item(cls: type[ChatItemView], item: ChatItem) -> ChatItemView:
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
            "$component": "div",
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
# 气泡通用样式
# ---------------------------------------------------------------------------


def _bubble_shell(is_user: bool) -> dict[str, Any]:
    return {
        "display": "flex",
        "justifyContent": "flex-end" if is_user else "flex-start",
        "padding": "6px 0",
    }


def _bubble_style(is_user: bool) -> dict[str, Any]:
    return {
        "maxWidth": "80%",
        "borderRadius": 14,
        "padding": "10px 14px",
        "background": "var(--mutgui-accent)" if is_user else "var(--mutgui-surface)",
        "color": "#ffffff" if is_user else "var(--mutgui-text)",
        "boxShadow": "0 1px 3px rgba(0, 0, 0, 0.08)",
        "border": "none" if is_user else "1px solid var(--mutgui-border)",
    }


def _meta_style() -> dict[str, Any]:
    return {
        "fontSize": "var(--mutagent-font-size-meta)",
        "color": "var(--mutgui-text-dim)",
        "marginBottom": 6,
    }


# ---------------------------------------------------------------------------
# UserMessage
# ---------------------------------------------------------------------------


@mutobj.impl(UserMessage.render)
def user_message_render(self: UserMessage) -> ViewBlock:
    return ViewBlock([
        {
            "$component": "div",
            "$id": "user-row",
            "style": _bubble_shell(True),
            "$children": [
                {
                    "$component": "div",
                    "$id": "user-bubble",
                    "style": _bubble_style(True),
                    "$children": [
                        {
                            "$component": "div",
                            "$id": "user-meta",
                            "style": _meta_style(),
                            "children": _role_meta("你", timestamp=self.item.timestamp),
                        },
                        {
                            "$component": "div",
                            "$id": "user-text",
                            "style": {
                                "whiteSpace": "pre-wrap",
                                "wordBreak": "break-word",
                                "lineHeight": 1.65,
                                "fontSize": "var(--mutagent-font-size-base)",
                            },
                            "children": self.item.text,
                        },
                    ],
                }
            ],
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
            "$component": "div",
            "$id": "assistant-row",
            "style": _bubble_shell(False),
            "$children": [
                {
                    "$component": "div",
                    "$id": "assistant-bubble",
                    "style": _bubble_style(False),
                    "$children": [
                        {
                            "$component": "div",
                            "$id": "assistant-meta",
                            "style": _meta_style(),
                            "children": _role_meta(
                                "助手",
                                model=self.item.model,
                                timestamp=self.item.timestamp,
                            ),
                        },
                        self.renderer,
                    ],
                }
            ],
        }
    ])


# ---------------------------------------------------------------------------
# AssistantError
# ---------------------------------------------------------------------------


@mutobj.impl(AssistantError.render)
def assistant_error_render(self: AssistantError) -> ViewBlock:
    return ViewBlock([
        {
            "$component": "div",
            "$id": "error-row",
            "style": _bubble_shell(False),
            "$children": [
                {
                    "$component": "div",
                    "$id": "error-bubble",
                    "style": {
                        **_bubble_style(False),
                        "border": "1px solid #e5534b",
                        "background": "rgba(229, 83, 75, 0.08)",
                    },
                    "$children": [
                        {
                            "$component": "div",
                            "$id": "error-meta",
                            "style": _meta_style(),
                            "children": _role_meta("错误", timestamp=self.item.timestamp),
                        },
                        {
                            "$component": "pre",
                            "$id": "error-text",
                            "style": {
                                "margin": 0,
                                "whiteSpace": "pre-wrap",
                                "wordBreak": "break-word",
                                "fontSize": "var(--mutagent-font-size-base)",
                                "fontFamily": "var(--mutgui-font-mono, monospace)",
                            },
                            "children": self.item.error,
                        },
                    ],
                }
            ],
        }
    ])


# ---------------------------------------------------------------------------
# TurnSeparator
# ---------------------------------------------------------------------------


@mutobj.impl(TurnSeparator.render)
def turn_separator_render(self: TurnSeparator) -> ViewBlock:
    detail = (
        f"{self.item.duration:.1f}s · in {self.item.input_tokens} · out {self.item.output_tokens}"
        if self.item.duration or self.item.input_tokens or self.item.output_tokens
        else "turn done"
    )
    return ViewBlock([
        {
            "$component": "div",
            "$id": "turn-separator",
            "style": {
                "display": "flex",
                "alignItems": "center",
                "gap": "12px",
                "padding": "10px 0",
            },
            "$children": [
                {
                    "$component": "div",
                    "$id": "line-left",
                    "style": {
                        "flex": 1,
                        "height": "1px",
                        "background": "var(--mutgui-border)",
                    },
                },
                {
                    "$component": "div",
                    "$id": "turn-detail",
                    "style": {
                        "fontSize": "var(--mutagent-font-size-meta)",
                        "color": "var(--mutgui-text-dim)",
                    },
                    "children": detail,
                },
                {
                    "$component": "div",
                    "$id": "line-right",
                    "style": {
                        "flex": 1,
                        "height": "1px",
                        "background": "var(--mutgui-border)",
                    },
                },
            ],
        }
    ])


# ---------------------------------------------------------------------------
# ToolCallCard
# ---------------------------------------------------------------------------


def _toggle_tool_card(*, view: ToolCallCard) -> None:
    view.item.expanded = not view.item.expanded
    view.invalidate()


def _pretty_json(text: str) -> str:
    if not text:
        return ""
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except Exception:
        return text


@mutobj.impl(ToolCallCard.render)
def tool_call_card_render(self: ToolCallCard) -> ViewBlock:
    status = self.item.status
    status_text = {
        "pending": "pending",
        "success": "success",
        "error": "error",
        "cancelled": "cancelled",
    }.get(status, status)
    status_color = {
        "pending": "#d4a72c",
        "success": "#2fb171",
        "error": "#e5534b",
        "cancelled": "#8b949e",
    }.get(status, "#8b949e")
    input_text = _pretty_json(self.item.input_text)
    result_text = _pretty_json(self.item.result_text)
    children: list[Any] = [
        {
            "$component": "div",
            "$id": "header",
            "style": {
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
                "gap": "12px",
            },
            "$children": [
                {
                    "$component": "div",
                    "$id": "tool-title",
                    "style": {"fontWeight": 600},
                    "children": self.item.name,
                },
                {
                    "$component": "div",
                    "$id": "status",
                    "style": {
                        "fontSize": "var(--mutagent-font-size-meta)",
                        "color": status_color,
                    },
                    "children": status_text,
                },
            ],
        },
        {
            "$component": "antd.Button",
            "$id": "toggle",
            "size": "small",
            "children": "展开" if not self.item.expanded else "收起",
            "onClick": Callback(_toggle_tool_card, view=self),
        },
    ]
    if self.item.expanded:
        if input_text:
            children.append(
                {
                    "$component": "div",
                    "$id": "input",
                    "style": {"marginTop": "10px"},
                    "$children": [
                        {
                            "$component": "div",
                            "$id": "input-label",
                            "style": {
                                "fontSize": "var(--mutagent-font-size-meta)",
                                "color": "var(--mutgui-text-dim)",
                                "marginBottom": "6px",
                            },
                            "children": "Input",
                        },
                        {
                            "$component": "pre",
                            "$id": "input-pre",
                            "style": {
                                "margin": 0,
                                "padding": "10px 12px",
                                "borderRadius": 12,
                                "overflowX": "auto",
                                "whiteSpace": "pre-wrap",
                                "background": "rgba(255,255,255,0.04)",
                                "fontSize": "var(--mutagent-font-size-base)",
                                "fontFamily": "var(--mutgui-font-mono, monospace)",
                            },
                            "children": input_text,
                        },
                    ],
                }
            )
        if result_text:
            children.append(
                {
                    "$component": "div",
                    "$id": "result",
                    "style": {"marginTop": "10px"},
                    "$children": [
                        {
                            "$component": "div",
                            "$id": "result-label",
                            "style": {
                                "fontSize": "var(--mutagent-font-size-meta)",
                                "color": "var(--mutgui-text-dim)",
                                "marginBottom": "6px",
                            },
                            "children": "Result",
                        },
                        {
                            "$component": "pre",
                            "$id": "result-pre",
                            "style": {
                                "margin": 0,
                                "padding": "10px 12px",
                                "borderRadius": 12,
                                "overflowX": "auto",
                                "whiteSpace": "pre-wrap",
                                "background": "rgba(255,255,255,0.04)",
                                "fontSize": "var(--mutagent-font-size-base)",
                                "fontFamily": "var(--mutgui-font-mono, monospace)",
                            },
                            "children": result_text,
                        },
                    ],
                }
            )
    return ViewBlock([
        {
            "$component": "div",
            "$id": "tool-card",
            "style": {
                "margin": "6px 0 10px 0",
                "padding": "12px 14px",
                "borderRadius": 14,
                "border": f"1px solid {status_color}",
                "background": "rgba(255,255,255,0.02)",
                "display": "flex",
                "flexDirection": "column",
                "gap": "8px",
            },
            "$children": children,
        }
    ])
