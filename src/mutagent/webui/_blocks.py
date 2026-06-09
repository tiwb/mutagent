"""Block renderer and specialized fenced-block widgets — Declaration + Implementation."""

from __future__ import annotations

import re

import mutobj
from mutgui import Callback, View, ViewBlock
from mutgui.view import RenderComponent, RenderNode, RenderTree


class BlockRenderer(View):
    text: str = ""

    def render(self) -> ViewBlock: ...


class ThinkingBlock(View):
    body: str = ""
    expanded: bool = False

    def render(self) -> ViewBlock: ...


_FENCE_RE = re.compile(r"^```([^\n]*)$")


def _toggle_thinking(*, view: ThinkingBlock) -> None:
    view.expanded = not view.expanded
    view.invalidate()


def _render_text_block(text: str, *, kind: str = "paragraph") -> RenderComponent:
    style = {
        "whiteSpace": "pre-wrap",
        "wordBreak": "break-word",
        "lineHeight": 1.7,
        "fontSize": "var(--mutagent-font-size-base)",
    }
    if kind == "heading":
        style["fontWeight"] = 700
        style["margin"] = "10px 0 8px 0"
    elif kind == "quote":
        style.update(
            {
                "margin": "10px 0",
                "paddingLeft": "12px",
                "borderLeft": "3px solid var(--mutgui-border)",
                "color": "var(--mutgui-text-dim)",
            }
        )
    else:
        style["margin"] = "8px 0"
    return {"$component": "div", "$id": f"{kind}-{abs(hash(text))}", "style": style, "children": text}


def _render_code_block(code: str, language: str) -> RenderComponent:
    header = language or "code"
    return {
        "$component": "div",
        "$id": f"code-{abs(hash(code + language))}",
        "style": {
            "margin": "10px 0",
            "borderRadius": 12,
            "overflow": "hidden",
            "border": "1px solid var(--mutgui-border)",
            "background": "rgba(255,255,255,0.03)",
        },
        "$children": [
            {
                "$component": "div",
                "$id": "code-header",
                "style": {
                    "padding": "6px 10px",
                    "fontSize": "var(--mutagent-font-size-meta)",
                    "color": "var(--mutgui-text-dim)",
                    "borderBottom": "1px solid var(--mutgui-border)",
                },
                "children": header,
            },
            {
                "$component": "pre",
                "$id": "code-pre",
                "style": {
                    "margin": 0,
                    "padding": "12px 14px",
                    "overflowX": "auto",
                    "whiteSpace": "pre-wrap",
                    "fontSize": "var(--mutagent-font-size-base)",
                    "fontFamily": "var(--mutgui-font-mono, monospace)",
                },
                "children": code,
            },
        ],
    }


def _render_mutagent_block(block_type: str, body: str) -> RenderNode:
    if block_type == "thinking":
        return ThinkingBlock(body=body)
    return {
        "$component": "div",
        "$id": f"mutagent-{block_type}-{abs(hash(body))}",
        "style": {
            "margin": "10px 0",
            "padding": "10px 12px",
            "borderRadius": 12,
            "border": "1px solid var(--mutgui-border)",
            "background": "rgba(255,255,255,0.03)",
        },
        "$children": [
            {
                "$component": "div",
                "$id": "mutagent-label",
                "style": {
                    "fontSize": "var(--mutagent-font-size-meta)",
                    "color": "var(--mutgui-text-dim)",
                    "marginBottom": "6px",
                },
                "children": f"mutagent:{block_type}",
            },
            {
                "$component": "pre",
                "$id": "mutagent-body",
                "style": {
                    "margin": 0,
                    "whiteSpace": "pre-wrap",
                    "fontSize": "var(--mutagent-font-size-base)",
                    "fontFamily": "var(--mutgui-font-mono, monospace)",
                },
                "children": body,
            },
        ],
    }


def _render_markdown_chunks(text: str) -> list[RenderComponent]:
    parts: list[RenderComponent] = []
    for raw in text.split("\n\n"):
        chunk = raw.strip("\n")
        if not chunk:
            continue
        lines = chunk.splitlines()
        first = lines[0].lstrip()
        if first.startswith(">"):
            parts.append(_render_text_block("\n".join(lines), kind="quote"))
        elif first.startswith("#"):
            parts.append(_render_text_block("\n".join(lines), kind="heading"))
        else:
            parts.append(_render_text_block("\n".join(lines), kind="paragraph"))
    return parts


def _render_segments(text: str) -> RenderTree:
    lines = text.splitlines()
    parts: list[RenderNode] = []
    markdown_buffer: list[str] = []
    in_fence = False
    fence_lang = ""
    fence_lines: list[str] = []

    def flush_markdown() -> None:
        if markdown_buffer:
            parts.extend(_render_markdown_chunks("\n".join(markdown_buffer)))
            markdown_buffer.clear()

    for line in lines:
        match = _FENCE_RE.match(line)
        if match:
            if not in_fence:
                flush_markdown()
                in_fence = True
                fence_lang = match.group(1).strip()
                fence_lines = []
            else:
                body = "\n".join(fence_lines)
                if fence_lang.startswith("mutagent:"):
                    block_type = fence_lang[len("mutagent:") :].strip() or "code"
                    parts.append(_render_mutagent_block(block_type, body))
                else:
                    parts.append(_render_code_block(body, fence_lang))
                in_fence = False
                fence_lang = ""
                fence_lines = []
            continue
        if in_fence:
            fence_lines.append(line)
        else:
            markdown_buffer.append(line)

    if in_fence:
        markdown_buffer.append(f"```{fence_lang}")
        markdown_buffer.extend(fence_lines)
    flush_markdown()
    return parts


@mutobj.impl(BlockRenderer.render)
def block_renderer_render(self: BlockRenderer) -> ViewBlock:
    parts = _render_segments(self.text)
    return ViewBlock([
        {
            "$component": "div",
            "$id": "block-renderer",
            "style": {"display": "flex", "flexDirection": "column"},
            "$children": parts,
        }
    ])


@mutobj.impl(ThinkingBlock.render)
def thinking_block_render(self: ThinkingBlock) -> ViewBlock:
    children: list[RenderNode] = [
        {
            "$component": "div",
            "$id": "thinking-header",
            "style": {
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
            },
            "$children": [
                {
                    "$component": "div",
                    "$id": "thinking-title",
                    "style": {
                        "fontSize": "var(--mutagent-font-size-meta)",
                        "color": "var(--mutgui-text-dim)",
                    },
                    "children": "thinking",
                },
                {
                    "$component": "antd.Button",
                    "$id": "thinking-toggle",
                    "size": "small",
                    "children": "展开" if not self.expanded else "收起",
                    "onClick": Callback(_toggle_thinking, view=self),
                },
            ],
        }
    ]
    if self.expanded:
        children.append(
            {
                "$component": "pre",
                "$id": "thinking-body",
                "style": {
                    "margin": "8px 0 0 0",
                    "whiteSpace": "pre-wrap",
                    "fontSize": "var(--mutagent-font-size-base)",
                    "fontFamily": "var(--mutgui-font-mono, monospace)",
                    "color": "var(--mutgui-text-dim)",
                },
                "children": self.body,
            }
        )
    return ViewBlock([
        {
            "$component": "div",
            "$id": "thinking-shell",
            "style": {
                "margin": "10px 0",
                "padding": "10px 12px",
                "borderRadius": 12,
                "border": "1px solid var(--mutgui-border)",
                "background": "rgba(255,255,255,0.03)",
            },
            "$children": children,
        }
    ])
