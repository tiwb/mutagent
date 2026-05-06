"""Default ToolCallCard implementation."""

from __future__ import annotations

import json

import mutagent
from mutagent.webui.tool_call import ToolCallCard
from mutgui import Callback, ViewBlock


@mutagent.impl(ToolCallCard.__init__)
def tool_call_card_init(self: ToolCallCard, *, item: object) -> None:
    super(ToolCallCard, self).__init__()
    self.item = item


def _toggle(*, view: ToolCallCard) -> None:
    view.item.expanded = not view.item.expanded
    view.invalidate()


def _pretty(text: str) -> str:
    if not text:
        return ""
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except Exception:
        return text


@mutagent.impl(ToolCallCard.render)
def tool_call_card_render(self: ToolCallCard) -> ViewBlock:
    status = getattr(self.item, "status", "pending")
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
    input_text = _pretty(getattr(self.item, "input_text", ""))
    result_text = _pretty(getattr(self.item, "result_text", ""))
    children = [
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
                    "children": getattr(self.item, "name", "tool"),
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
            "children": "展开" if not getattr(self.item, "expanded", True) else "收起",
            "onClick": Callback(_toggle, view="@view"),
        },
    ]
    if getattr(self.item, "expanded", True):
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
