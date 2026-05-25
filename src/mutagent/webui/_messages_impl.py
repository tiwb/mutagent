"""Default MessageList / ChatItemView implementations."""

from __future__ import annotations

import json
import time
from typing import Any

import mutagent
import mutobj
from mutagent.webui.blocks import BlockRenderer
from mutagent.webui.messages import (
    AssistantError,
    AssistantErrorItem,
    AssistantMessage,
    AssistantTextItem,
    ChatItem,
    ChatItemView,
    MessageList,
    ToolCallCard,
    ToolCallItem,
    TurnSeparator,
    TurnSeparatorItem,
    UserMessage,
    UserTextItem,
)
from mutgui import Callback, View, ViewBlock, VirtualList, VirtualListItemAdapter


# ── Extensions ─────────────────────────────────────────────────

class MessageListExt(mutobj.Extension[MessageList]):
    """MessageList 的运行时私有状态。"""
    adapter: Any = None
    virtual_list: Any = None


class AssistantMessageExt(mutobj.Extension[AssistantMessage]):
    """AssistantMessage 的运行时私有状态。"""
    renderer: Any = None


def _mext(self: MessageList) -> MessageListExt:
    return MessageListExt.get_or_create(self)


def _aext(self: AssistantMessage) -> AssistantMessageExt:
    return AssistantMessageExt.get_or_create(self)


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


@mutagent.impl(ChatItemView.__init__)
def chat_item_view_init(self: ChatItemView, *, item: ChatItem) -> None:
    super(ChatItemView, self).__init__()
    self.item = item


@mutagent.impl(ChatItemView.for_item)
def chat_item_view_for_item(cls: type[ChatItemView], item: ChatItem) -> ChatItemView:
    view_cls = _resolve_view_class(type(item))
    return view_cls(item=item)


# ---------------------------------------------------------------------------
# MessageList
# ---------------------------------------------------------------------------


class _MessageListAdapter(VirtualListItemAdapter):
    items: list[ChatItem] = mutobj.field(default_factory=list)

    @property
    def item_count(self) -> int:
        return len(self.items)

    def item_id(self, index: int) -> str:
        return self.items[index].id

    def create_item_view(self, index: int) -> View:
        return ChatItemView.for_item(self.items[index])

    def invalidate_existing_item(self, item_id: str) -> None:
        for virtual_list in self.virtual_lists:
            item_view = virtual_list.get_item_view(item_id)
            if item_view is not None:
                item_view.invalidate()
                return
        self.invalidate()


@mutagent.impl(MessageList.__init__)
def message_list_init(
    self: MessageList, *, items: list[Any] | None = None
) -> None:
    super(MessageList, self).__init__()
    ext = _mext(self)
    self.id = "message-list"
    self.items = items if items is not None else []
    ext.adapter = _MessageListAdapter(items=self.items)
    ext.virtual_list = VirtualList(
        id="chat-list",
        adapter=ext.adapter,
        stick_to_bottom=True,
        estimated_item_height=128,
    )


@mutagent.impl(MessageList.refresh)
def refresh(self: MessageList) -> None:
    ext = _mext(self)
    ext.adapter.invalidate()
    self.invalidate()


@mutagent.impl(MessageList.invalidate_item)
def invalidate_item(self: MessageList, item_id: str) -> None:
    ext = _mext(self)
    ext.adapter.invalidate_existing_item(item_id)


@mutagent.impl(MessageList.render)
def message_list_render(self: MessageList) -> ViewBlock:
    ext = _mext(self)
    ext.adapter.items = self.items
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
            "$children": [ext.virtual_list],
        }
    ])


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


@mutagent.impl(UserMessage.__init__)
def user_message_init(self: UserMessage, *, item: UserTextItem) -> None:
    super(UserMessage, self).__init__(item=item)


@mutagent.impl(UserMessage.render)
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


@mutagent.impl(AssistantMessage.__init__)
def assistant_message_init(
    self: AssistantMessage, *, item: AssistantTextItem
) -> None:
    super(AssistantMessage, self).__init__(item=item)
    ext = _aext(self)
    ext.renderer = BlockRenderer(text=item.text)
    ext.renderer.id = f"block-renderer-{item.id}"


@mutagent.impl(AssistantMessage.render)
def assistant_message_render(self: AssistantMessage) -> ViewBlock:
    ext = _aext(self)
    if ext.renderer.text != self.item.text:
        ext.renderer = BlockRenderer(text=self.item.text)
        ext.renderer.id = f"block-renderer-{self.item.id}"
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
                        ext.renderer,
                    ],
                }
            ],
        }
    ])


# ---------------------------------------------------------------------------
# AssistantError
# ---------------------------------------------------------------------------


@mutagent.impl(AssistantError.__init__)
def assistant_error_init(
    self: AssistantError, *, item: AssistantErrorItem
) -> None:
    super(AssistantError, self).__init__(item=item)


@mutagent.impl(AssistantError.render)
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


@mutagent.impl(TurnSeparator.__init__)
def turn_separator_init(
    self: TurnSeparator, *, item: TurnSeparatorItem
) -> None:
    super(TurnSeparator, self).__init__(item=item)


@mutagent.impl(TurnSeparator.render)
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


@mutagent.impl(ToolCallCard.__init__)
def tool_call_card_init(self: ToolCallCard, *, item: ToolCallItem) -> None:
    super(ToolCallCard, self).__init__(item=item)


@mutagent.impl(ToolCallCard.render)
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
