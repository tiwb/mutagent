"""Default MessageList implementation."""

from __future__ import annotations

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
    MessageList,
    ToolCallItem,
    TurnSeparator,
    TurnSeparatorItem,
    UserMessage,
    UserTextItem,
)
from mutagent.webui.tool_call import ToolCallCard
from mutgui import View, ViewBlock, VirtualList, VirtualListItemAdapter


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


class _MessageListAdapter(VirtualListItemAdapter):
    items: list[Any] = mutobj.field(default_factory=list)

    @property
    def item_count(self) -> int:
        return len(self.items)

    def item_id(self, index: int) -> str:
        return self.items[index].id

    def create_item_view(self, index: int) -> View:
        item = self.items[index]
        if isinstance(item, UserTextItem):
            return UserMessage(item=item)
        if isinstance(item, AssistantTextItem):
            return AssistantMessage(item=item)
        if isinstance(item, AssistantErrorItem):
            return AssistantError(item=item)
        if isinstance(item, TurnSeparatorItem):
            return TurnSeparator(item=item)
        if isinstance(item, ToolCallItem):
            return ToolCallCard(item=item)
        raise TypeError(f"Unsupported chat item: {type(item)!r}")

    def invalidate_existing_item(self, item_id: str) -> None:
        for virtual_list in self.virtual_lists:
            item_view = virtual_list.item_views.get(item_id)
            if item_view is not None:
                item_view.invalidate()
                return
        self.invalidate()


@mutagent.impl(MessageList.__init__)
def message_list_init(
    self: MessageList, *, items: list[Any] | None = None
) -> None:
    super(MessageList, self).__init__()
    self.id = "message-list"
    self.items = items if items is not None else []
    self._adapter = _MessageListAdapter(items=self.items)
    self._virtual_list = VirtualList(
        id="chat-list",
        adapter=self._adapter,
        stick_to_bottom=True,
        estimated_item_height=128,
    )


@mutagent.impl(MessageList.refresh)
def refresh(self: MessageList) -> None:
    self._adapter.invalidate()
    self.invalidate()


@mutagent.impl(MessageList.invalidate_item)
def invalidate_item(self: MessageList, item_id: str) -> None:
    self._adapter.invalidate_existing_item(item_id)


@mutagent.impl(MessageList.render)
def message_list_render(self: MessageList) -> ViewBlock:
    self._adapter.items = self.items
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
            "$children": [self._virtual_list],
        }
    ])


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


@mutagent.impl(UserMessage.__init__)
def user_message_init(self: UserMessage, *, item: UserTextItem) -> None:
    super(UserMessage, self).__init__()
    self.item = item


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


@mutagent.impl(AssistantMessage.__init__)
def assistant_message_init(
    self: AssistantMessage, *, item: AssistantTextItem
) -> None:
    super(AssistantMessage, self).__init__()
    self.item = item
    self._renderer = BlockRenderer(text=item.text)
    self._renderer.id = f"block-renderer-{item.id}"


@mutagent.impl(AssistantMessage.render)
def assistant_message_render(self: AssistantMessage) -> ViewBlock:
    if self._renderer.text != self.item.text:
        self._renderer = BlockRenderer(text=self.item.text)
        self._renderer.id = f"block-renderer-{self.item.id}"
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
                        self._renderer,
                    ],
                }
            ],
        }
    ])


@mutagent.impl(AssistantError.__init__)
def assistant_error_init(
    self: AssistantError, *, item: AssistantErrorItem
) -> None:
    super(AssistantError, self).__init__()
    self.item = item


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


@mutagent.impl(TurnSeparator.__init__)
def turn_separator_init(
    self: TurnSeparator, *, item: TurnSeparatorItem
) -> None:
    super(TurnSeparator, self).__init__()
    self.item = item


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
