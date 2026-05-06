"""Default toolbar widget implementations."""

from __future__ import annotations

from typing import Any

import mutagent
from mutagent.webui.toolbar import AgentStatusBar
from mutgui import Callback, ViewBlock


@mutagent.impl(AgentStatusBar.__init__)
def status_bar_init(
    self: AgentStatusBar,
    *,
    status: str = "idle",
    input_tokens: int = 0,
    output_tokens: int = 0,
    context_percent: float = 0.0,
    context_total: int = 0,
    context_used: int = 0,
    total_cost: float = 0.0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    expanded: bool = False,
) -> None:
    super(AgentStatusBar, self).__init__()
    self.id = "agent-status-bar"
    self.status = status
    self.input_tokens = input_tokens
    self.output_tokens = output_tokens
    self.context_percent = context_percent
    self.context_total = context_total
    self.context_used = context_used
    self.total_cost = total_cost
    self.cache_read_tokens = cache_read_tokens
    self.cache_write_tokens = cache_write_tokens
    self.expanded = expanded


def _format_count(n: int) -> str:
    """简略显示数字：1234567 → 1.2M, 1234 → 1.2k。"""
    if n >= 1_000_000:
        v = n / 1_000_000
        if v >= 100:
            return f"{round(v)}M"
        if v >= 10:
            return f"{v:.1f}M"
        return f"{v:.2f}M"
    if n >= 1000:
        v = n / 1000
        if v >= 100:
            return f"{round(v)}k"
        if v >= 10:
            return f"{v:.1f}k"
        return f"{v:.2f}k"
    return str(n)


@mutagent.impl(AgentStatusBar.render)
def status_bar_render(self: AgentStatusBar) -> ViewBlock:
    # 紧凑行内容
    segments: list[dict[str, Any]] = []

    # Cost
    cost_text = f"${self.total_cost:.3f}" if self.total_cost else "$0"
    segments.append({
        "$component": "div",
        "$id": "status-cost",
        "style": {
            "fontSize": "var(--mutagent-font-size-meta)",
            "color": "var(--mutgui-text-dim)",
            "fontVariantNumeric": "tabular-nums",
        },
        "children": cost_text,
    })

    # Context used
    ctx_used_str = _format_count(self.context_used) if self.context_used else "0"
    if self.context_total:
        ctx_str = f"{ctx_used_str}/{_format_count(self.context_total)}"
    else:
        ctx_str = f"{ctx_used_str}/?"
    segments.append({
        "$component": "div",
        "$id": "status-context",
        "style": {
            "fontSize": "var(--mutagent-font-size-meta)",
            "color": "var(--mutgui-text-dim)",
            "fontVariantNumeric": "tabular-nums",
        },
        "children": ctx_str,
    })

    # 紧凑行
    compact_row = {
        "$component": "div",
        "$id": "status-compact",
        "style": {
            "display": "flex",
            "alignItems": "center",
            "gap": "12px",
            "fontSize": "var(--mutagent-font-size-meta)",
            "color": "var(--mutgui-text-dim)",
            "cursor": "pointer",
            "userSelect": "none",
        },
        "$children": segments,
    }

    # 详情面板
    detail_lines: list[dict[str, Any]] = []

    def _add_row(label: str, value: str) -> None:
        detail_lines.append({
            "$component": "div",
            "style": {
                "display": "flex",
                "justifyContent": "space-between",
                "gap": "16px",
                "fontSize": "var(--mutagent-font-size-meta)",
                "padding": "2px 0",
            },
            "$children": [
                {"$component": "div", "style": {"color": "var(--mutgui-text-dim)"}, "children": label},
                {"$component": "div", "style": {"color": "var(--mutgui-text)", "fontVariantNumeric": "tabular-nums"}, "children": value},
            ],
        })

    # Context row
    if self.context_total:
        ctx_used_full = f"{self.context_used:,} / {self.context_total:,}"
    else:
        ctx_used_full = f"{self.context_used:,} (window unknown)"
    _add_row("Context", ctx_used_full)

    # Context progress bar
    if self.context_total:
        ctx_pct = self.context_percent
        bar_color = (
            "var(--mutgui-accent)" if ctx_pct < 0.75
            else "var(--mutgui-warning)" if ctx_pct < 0.9
            else "var(--mutgui-danger, #ff4d4f)"
        )
        detail_lines.append({
            "$component": "div",
            "style": {
                "marginBottom": "8px",
                "height": "4px",
                "borderRadius": "2px",
                "background": "var(--mutgui-border)",
                "overflow": "hidden",
            },
            "$children": [{
                "$component": "div",
                "style": {
                    "height": "100%",
                    "width": f"{min(ctx_pct * 100, 100):.1f}%",
                    "background": bar_color,
                    "borderRadius": "2px",
                },
            }],
        })

    _add_row("Input", f"{self.input_tokens:,}")
    _add_row("Output", f"{self.output_tokens:,}")

    if self.cache_read_tokens:
        _add_row("Cache read", f"{self.cache_read_tokens:,}")
    if self.cache_write_tokens:
        _add_row("Cache write", f"{self.cache_write_tokens:,}")

    _add_row("Cost", f"${self.total_cost:.6f}")

    detail_panel = {
        "$component": "div",
        "$id": "status-detail",
        "style": {
            "minWidth": "240px",
            "padding": "4px 0",
        },
        "$children": detail_lines,
    }

    return ViewBlock([{
        "$component": "antd.Popover",
        "$id": "status-popover",
        "trigger": "click",
        "placement": "bottomRight",
        "arrow": True,
        "open": self.expanded,
        "onOpenChange": Callback(
            lambda visible, *, view: setattr(view, 'expanded', visible) or view.invalidate(),
            "$0",
            view="@view",
        ),
        "content": detail_panel,
        "$children": [compact_row],
    }])
