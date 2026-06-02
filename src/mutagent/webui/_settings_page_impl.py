"""Default SettingsPage implementation + settings-domain Actions.

全页面式 SettingsPage —— 替代旧的 SettingsDrawer 浮层。

通过 ``self.conversation`` 级联引用直接访问 Conversation 的路由/模型刷新能力，
不再需要回调注入。
"""

from __future__ import annotations

import inspect
from typing import Any, TYPE_CHECKING

import mutobj
from mutobj import AttributeDescriptor
from mutagent.webui.settings import SettingsPage, SettingsPanel
from mutgui import Action, ActionContext, Callback, Expr, ViewBlock

if TYPE_CHECKING:
    from mutagent.webui.conversation import Conversation


# ── Extension ─────────────────────────────────────────────────

class SettingsPageExt(mutobj.Extension[SettingsPage]):
    """SettingsPage 的运行时私有状态。"""
    conversation: Conversation | None = None
    panels: dict[str, SettingsPanel] = mutobj.field(default_factory=dict)
    ordered_panel_ids: list[str] = mutobj.field(default_factory=list)
    active: bool = False


def _spext(self: SettingsPage) -> SettingsPageExt:
    return SettingsPageExt.get_or_create(self)


# ── helpers ──────────────────────────────────────────────────


def _resolve_panel_attr(cls, attr_name: str, default: str = "") -> str:
    """Extract a string class-attribute from a mutobj Declaration subclass.

    Class-level attrs on Declaration subclasses are wrapped in AttributeDescriptor
    objects.  This helper unwraps them so cls.panel_id / cls.panel_title etc.
    are used as plain strings at class-discovery time.
    """
    desc = cls.__dict__.get(attr_name)
    if isinstance(desc, AttributeDescriptor):
        val = desc.default
        return str(val) if val is not ... else default
    return str(desc) if desc else default


async def _call_maybe_async(handler: Any, *args: Any) -> None:
    """通用调用：handler 为 None 跳过；同步/异步均支持。"""
    if handler is None:
        return
    result = handler(*args)
    if inspect.isawaitable(result):
        await result


# ── @impl: __init__ ──────────────────────────────────────────


@mutobj.impl(SettingsPage.__init__)
def settings_page_init__(
    self: SettingsPage,
    *,
    conversation: Conversation | None = None,
) -> None:
    super(SettingsPage, self).__init__()
    ext = _spext(self)
    self.id = "settings-page"
    self.conversation = conversation
    ext.conversation = conversation
    self.active_panel_id = ""

    # ── 发现并实例化全部 SettingsPanel 子类 ─────
    panel_classes = mutobj.discover_subclasses(SettingsPanel)
    ext.panels = {}
    ext.ordered_panel_ids = []

    for cls in panel_classes:
        panel_id = _resolve_panel_attr(cls, "panel_id")
        if not panel_id:
            continue
        panel = cls(conversation=conversation)
        setattr(panel, "page", self)
        ext.panels[panel_id] = panel

    def _placement_key(panel_id: str) -> str:
        panel = ext.panels[panel_id]
        placement = _resolve_panel_attr(type(panel), "panel_placement")
        return placement or f"zzzz:{panel_id}"

    ext.ordered_panel_ids = sorted(ext.panels.keys(), key=_placement_key)
    if ext.ordered_panel_ids:
        self.active_panel_id = ext.ordered_panel_ids[0]
    # 是否处于"已激活"状态。初始 False：``active_panel_id`` 只是默认占位，
    # panel 未触发过 ``on_open``。activate / deactivate 均仅在 active=True 时
    # 才 close 旧 panel，避免首次进入 settings 时误发 close 事件。
    ext.active = False


# ── @impl: render ────────────────────────────────────────────


@mutobj.impl(SettingsPage.render)
def settings_page_render(self: SettingsPage) -> ViewBlock:
    ext = _spext(self)
    active = ext.panels.get(self.active_panel_id)

    menu_items: list[dict[str, Any]] = []
    for panel_id in ext.ordered_panel_ids:
        panel = ext.panels[panel_id]
        title = _resolve_panel_attr(type(panel), "panel_title") or panel_id
        menu_items.append({"key": panel_id, "label": title})

    sider: dict[str, Any] = {
        "$component": "antd.Menu",
        "$id": "settings-sider-menu",
        "mode": "inline",
        "selectedKeys": [self.active_panel_id] if self.active_panel_id else [],
        "items": menu_items,
        "onClick": Callback(_on_menu_click, self, Expr.wire("$0.key")),
        "style": {
            "height": "100%",
        },
    }

    back_btn: dict[str, Any] = {
        "$component": "div",
        "$id": "settings-back-btn",
        "children": "← 返回对话",
        "onClick": Callback(_on_back_click, self),
        "style": {
            "cursor": "pointer",
            "userSelect": "none",
            "fontSize": "var(--mutagent-font-size-base)",
            "color": "var(--mutgui-text-secondary)",
        },
    }

    sider_header: dict[str, Any] = {
        "$component": "div",
        "$id": "settings-sider-header",
        "style": {
            "display": "flex",
            "alignItems": "center",
            "padding": "12px 16px",
            "borderBottom": "1px solid var(--mutgui-border)",
            "flex": "0 0 auto",
        },
        "$children": [back_btn],
    }

    sider_wrap: dict[str, Any] = {
        "$component": "div",
        "$id": "settings-sider",
        "style": {
            "width": "220px",
            "flex": "0 0 220px",
            "minHeight": 0,
            "display": "flex",
            "flexDirection": "column",
            "background": "var(--mutgui-surface, transparent)",
            "borderRight": "1px solid var(--mutgui-border)",
            "overflow": "auto",
        },
        "$children": [sider_header, sider],
    }

    title_text = active.panel_title if active else ""
    header: dict[str, Any] = {
        "$component": "div",
        "$id": "settings-content-header",
        "style": {
            "display": "flex",
            "alignItems": "center",
            "padding": "12px 16px",
            "borderBottom": "1px solid var(--mutgui-border)",
            "flex": "0 0 auto",
        },
        "$children": [
            {
                "$component": "div",
                "$id": "settings-content-title",
                "style": {
                    "fontSize": "var(--mutagent-font-size-base)",
                    "fontWeight": 600,
                    "color": "var(--mutgui-text)",
                },
                "children": title_text,
            },
        ],
    }

    body_children: list[Any] = [active] if active is not None else []
    body: dict[str, Any] = {
        "$component": "div",
        "$id": "settings-content-body",
        "style": {
            "flex": 1,
            "minHeight": 0,
            "overflow": "auto",
            "padding": "16px",
        },
        "$children": body_children,
    }

    content: dict[str, Any] = {
        "$component": "div",
        "$id": "settings-content",
        "style": {
            "flex": 1,
            "minWidth": 0,
            "minHeight": 0,
            "display": "flex",
            "flexDirection": "column",
        },
        "$children": [header, body],
    }

    root: dict[str, Any] = {
        "$component": "div",
        "$id": "settings-page-root",
        "style": {
            "display": "flex",
            "flexDirection": "row",
            "flex": 1,
            "minHeight": 0,
            "height": "100%",
            "color": "var(--mutgui-text)",
        },
        "$children": [sider_wrap, content],
    }

    return ViewBlock([root])


# ── @impl: activate / deactivate / close / list / notify ─────


@mutobj.impl(SettingsPage.activate)
async def settings_page_activate(self: SettingsPage, panel_id: str) -> None:
    ext = _spext(self)
    target = panel_id or (ext.ordered_panel_ids[0] if ext.ordered_panel_ids else "")
    if not target or target not in ext.panels:
        return

    if ext.active:
        prev = ext.panels.get(self.active_panel_id)
        if prev is not None and prev is not ext.panels.get(target):
            on_close = getattr(prev, "on_close", None)
            if callable(on_close):
                await _call_maybe_async(on_close)

    self.active_panel_id = target
    ext.active = True
    new_panel = ext.panels[target]
    on_open = getattr(new_panel, "on_open", None)
    if callable(on_open):
        await _call_maybe_async(on_open)
    new_panel.invalidate()
    self.invalidate()


@mutobj.impl(SettingsPage.deactivate)
async def settings_page_deactivate(self: SettingsPage) -> None:
    ext = _spext(self)
    if ext.active:
        prev = ext.panels.get(self.active_panel_id)
        if prev is not None:
            on_close = getattr(prev, "on_close", None)
            if callable(on_close):
                await _call_maybe_async(on_close)
    ext.active = False
    self.invalidate()


@mutobj.impl(SettingsPage.close)
async def settings_page_close(self: SettingsPage) -> None:
    if self.conversation is not None:
        await self.conversation.navigate_to("")


@mutobj.impl(SettingsPage.list_panels)
def settings_page_list_panels(self: SettingsPage) -> list[SettingsPanel]:
    ext = _spext(self)
    return [ext.panels[pid] for pid in ext.ordered_panel_ids]



# ── Callback handlers ─────────────────────────────────────────


async def _on_menu_click(view: SettingsPage, panel_id: str = "") -> None:
    if not panel_id or view.conversation is None:
        return
    await view.conversation.navigate_to(f"settings/{panel_id}")


async def _on_back_click(view: SettingsPage, *_: Any) -> None:
    if view.conversation is not None:
        await view.conversation.navigate_to("")


# ── Settings 域 Actions ────────────────────────────


class OpenSettingsAction(Action):
    action_id = "mutagent.menu.settings"
    categories = ("mutagent.main_menu",)
    label = "Settings"
    placement = "settings:10/10"

    async def execute(self, context: ActionContext) -> None:
        conv = context.get("conversation")
        if conv is not None:
            await conv.navigate_to("settings")


