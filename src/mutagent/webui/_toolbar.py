"""Toolbar widgets: status bar — Declaration + Implementation, plus toolbar-domain Actions."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import mutobj
from mutgui import Action, ActionContext, ActionRef, ActionToolbar, Callback, Expr, View, ViewBlock

if TYPE_CHECKING:
    from ._conversation import Conversation
    from mutgui.view import ViewId


class AgentToolbar(ActionToolbar):
    """Conversation 顶栏，封装 ActionToolbar 配置和 toolbar 域 Actions 上下文。"""
    id: ViewId = "conversation-toolbar"
    conversation: Conversation

    def __init__(self, *, conversation: Conversation) -> None:
        super(AgentToolbar, self).__init__()
        self.categories = ["mutagent.conversation.toolbar"]
        self.context = ActionContext(data={"conversation": conversation})
        self.conversation = conversation


class AgentStatusBar(View):
    id: ViewId = "agent-status-bar"
    status: str = "idle"
    input_tokens: int = 0
    output_tokens: int = 0
    context_percent: float = 0.0
    context_total: int = 0
    context_used: int = 0
    total_cost: float = 0.0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    expanded: bool = False

    def render(self) -> ViewBlock: ...


# ── 私有辅助：从 ActionContext 提取 Conversation；通用 async 调用包装 ─────
def _conversation(context: ActionContext) -> Conversation:
    ret = context.get("conversation")
    if not ret:
        raise ValueError("Conversation not found in ActionContext")
    return ret


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


def _on_expand_change(self: AgentStatusBar, visible: bool) -> None:
    self.expanded = visible
    self.invalidate()


@mutobj.impl(AgentStatusBar.render)
def agent_status_bar_render(self: AgentStatusBar) -> ViewBlock:
    # 紧凑行内容
    segments: list[dict[str, Any]] = []

    # Status
    if self.status:
        status_color = {
            "idle": "var(--mutgui-text-dim)",
            "busy": "var(--mutgui-accent)",
            "error": "var(--mutgui-danger, #ff4d4f)",
        }.get(self.status, "var(--mutgui-text-dim)")
        segments.append({
            "$component": "div",
            "$id": "status-text",
            "style": {
                "fontSize": "var(--mutagent-font-size-meta)",
                "color": status_color,
                "fontVariantNumeric": "tabular-nums",
            },
            "children": self.status,
        })

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
        "onOpenChange": Callback(_on_expand_change, self, Expr.wire("$0")),
        "content": detail_panel,
        "$children": [compact_row],
    }])


# ── Toolbar 域 Actions ──────────────────────────────────────────────


class ModelSelectorAction(Action):
    action_id = "mutagent.toolbar.model_selector"
    categories = ("mutagent.conversation.toolbar",)
    label = "Model"
    position = "start"
    placement = "primary:10/10"
    variant = "dropdown"
    menu_placement = "bottom-start"

    def resolved_label(self, context: ActionContext | None = None) -> str:
        assert context is not None
        return _conversation(context).current_model

    def check_enabled(self, context: ActionContext) -> bool:
        conv = _conversation(context)
        return not conv.is_busy

    def menu_actions(self, context: ActionContext) -> list[ActionRef]:
        conv = _conversation(context)
        app = conv.app
        models = app.config.list_models() if app else []
        return [
            ActionRef(action=SelectModelAction(str(m.get("name", ""))))
            for m in models
            if m.get("name")
        ]


class SelectModelAction(Action):
    """菜单内单个模型选项 — 动态 label + checked 态。"""
    variant = "button"

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self._model_name = model_name
        self.label = model_name

    def resolved_action_id(self) -> str:
        return f"mutagent.model.select.{self._model_name}"

    def check_checked(self, context: ActionContext) -> bool:
        conv = _conversation(context)
        return conv.current_model == self._model_name

    async def execute(self, context: ActionContext) -> None:
        conv = _conversation(context)
        conv.change_model(self._model_name)


class AgentStatusAction(Action):
    action_id = "mutagent.toolbar.status"
    categories = ("mutagent.conversation.toolbar",)
    label = "Status"
    position = "start"
    placement = "primary:10/20"
    variant = "widget"

    def toolbar_view(self, context: ActionContext) -> Any:
        conversation = _conversation(context)
        return conversation.status_bar


class MainMenuAction(Action):
    action_id = "mutagent.toolbar.main_menu"
    categories = ("mutagent.conversation.toolbar",)
    label = "☰"
    tooltip = "Settings"
    position = "end"
    placement = "menu:20/10"
    variant = "dropdown"

    def menu_actions(self, context: ActionContext) -> list[ActionRef]:
        return [ActionRef(category="mutagent.main_menu")]
