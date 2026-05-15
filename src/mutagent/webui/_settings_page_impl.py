"""Default SettingsPage implementation + settings-domain Actions.

全页面式 SettingsPage —— 替代旧的 SettingsDrawer 浮层。

路由由 Conversation 驱动：
- Conversation.navigate_to / on_hash_change → SettingsPage.activate / deactivate
- 左侧菜单点击 → on_request_navigate(f"settings/{panel_id}") → Conversation 决定走 navigate_to
- 「← 返回对话」按钮 → on_request_close() → Conversation.navigate_to("")
"""

from __future__ import annotations

import inspect
from typing import Any

import mutagent
import mutobj
from mutobj.core import AttributeDescriptor
from mutagent.webui.settings import SettingsPage, SettingsPanel
from mutgui import Action, ActionContext, Callback, Expr, ViewBlock


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


@mutagent.impl(SettingsPage.__init__)
def __init__(
    self: SettingsPage,
    *,
    app: Any,
    agent: Any,
    on_models_changed: Any = None,
    on_request_close: Any = None,
    on_request_navigate: Any = None,
) -> None:
    super(SettingsPage, self).__init__()
    self.id = "settings-page"
    self._app = app
    self._agent = agent
    self._on_models_changed = on_models_changed
    self._on_request_close = on_request_close
    self._on_request_navigate = on_request_navigate
    self.active_panel_id = ""

    # ── 发现并实例化全部 SettingsPanel 子类 ─────
    panel_classes = mutobj.discover_subclasses(SettingsPanel)
    self._panels: dict[str, SettingsPanel] = {}
    self._ordered_panel_ids: list[str] = []

    for cls in panel_classes:
        panel_id = _resolve_panel_attr(cls, "panel_id")
        if not panel_id:
            continue
        panel = cls(app=app, agent=agent)
        # 字段名 `drawer` 沿用历史（panel 文件零改动），实际指向 SettingsPage 实例
        setattr(panel, "drawer", self)
        self._panels[panel_id] = panel

    def _placement_key(panel_id: str) -> str:
        panel = self._panels[panel_id]
        placement = _resolve_panel_attr(type(panel), "panel_placement")
        return placement or f"zzzz:{panel_id}"

    self._ordered_panel_ids = sorted(self._panels.keys(), key=_placement_key)
    if self._ordered_panel_ids:
        self.active_panel_id = self._ordered_panel_ids[0]
    # 是否处于"已激活"状态。初始 False：``active_panel_id`` 只是默认占位，
    # panel 未触发过 ``on_open``。activate / deactivate 均仅在 _active=True 时
    # 才 close 旧 panel，避免首次进入 settings 时误发 close 事件。
    self._active = False


@mutagent.impl(SettingsPage.render)
def render(self: SettingsPage) -> ViewBlock:
    active = self._panels.get(self.active_panel_id)

    # 左侧菜单 items
    menu_items: list[dict[str, Any]] = []
    for panel_id in self._ordered_panel_ids:
        panel = self._panels[panel_id]
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

    # 左侧标题栏 —— 与右侧 header 完全对称
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


@mutagent.impl(SettingsPage.activate)
async def activate(self: SettingsPage, panel_id: str) -> None:
    """切到指定 panel；仅在"已激活"状态下才 close 旧 panel。

    panel_id 为空时使用默认（首个）。调用者保证 route 已变（
    ``Conversation.navigate_to`` 在 same-route 时已早返回），所以进到这里时
    要么 panel 变了，要么从未激活过。
    """
    target = panel_id or (self._ordered_panel_ids[0] if self._ordered_panel_ids else "")
    if not target or target not in self._panels:
        return

    # 仅当之前已激活且 target 不是当前 panel 时才 close（避免首次进入误发）
    if self._active:
        prev = self._panels.get(self.active_panel_id)
        if prev is not None and prev is not self._panels.get(target):
            on_close = getattr(prev, "on_close", None)
            if callable(on_close):
                await _call_maybe_async(on_close)

    self.active_panel_id = target
    self._active = True
    new_panel = self._panels[target]
    on_open = getattr(new_panel, "on_open", None)
    if callable(on_open):
        await _call_maybe_async(on_open)
    new_panel.invalidate()
    self.invalidate()


@mutagent.impl(SettingsPage.deactivate)
async def deactivate(self: SettingsPage) -> None:
    """离开 settings 视图：触发当前 panel.on_close（仅在已激活时）。

    保留 ``active_panel_id`` 不清空——作为"上次激活"记忆（当前不依赖该
    记忆，只为未来保留可能性）。再次进入 settings 时由 Conversation 显式
    ``activate(panel_id)``，panel_id 来自 hash 解析，状态权威。
    """
    if self._active:
        prev = self._panels.get(self.active_panel_id)
        if prev is not None:
            on_close = getattr(prev, "on_close", None)
            if callable(on_close):
                await _call_maybe_async(on_close)
    self._active = False
    self.invalidate()


@mutagent.impl(SettingsPage.close)
async def close(self: SettingsPage) -> None:
    """兼容方法 — 转发给 on_request_close 回调。

    保留以维持 ``LLMSettingsPanel._save_all_settings`` 等既有代码：
    ``await view.drawer.close()`` 不需修改。
    """
    if self._on_request_close is not None:
        await _call_maybe_async(self._on_request_close)


@mutagent.impl(SettingsPage.list_panels)
def list_panels(self: SettingsPage) -> list[SettingsPanel]:
    return [self._panels[pid] for pid in self._ordered_panel_ids]


@mutagent.impl(SettingsPage.notify_models_changed)
async def notify_models_changed(self: SettingsPage, preferred_model: str = "") -> None:
    cb = self._on_models_changed
    if cb is not None:
        result = cb(preferred_model)
        if inspect.isawaitable(result):
            await result


# ── Callback handlers ─────────────────────────────────────────


async def _on_menu_click(view: SettingsPage, panel_id: str = "") -> None:
    """antd.Menu onClick 回调 — 前端提取 ``$0.key`` 只发 panel_id 字符串。

    必须走 on_request_navigate 让 Conversation 同步 URL；
    不直接修改 active_panel_id，否则 URL 与 state 会失同步。
    """
    if not panel_id:
        return
    if view._on_request_navigate is not None:
        await _call_maybe_async(view._on_request_navigate, f"settings/{panel_id}")


async def _on_back_click(view: SettingsPage, *_: Any) -> None:
    """← 返回对话 — 走 on_request_close 让 Conversation 决定怎么退。"""
    if view._on_request_close is not None:
        await _call_maybe_async(view._on_request_close)


# ── Settings 域 Actions ────────────────────────────


class OpenSettingsAction(Action):
    """通用设置面板入口 — 一个类支撑所有 SettingsPanel 子类。

    新版：不再读 ``settings_drawer``，改为通过 ``conversation.navigate_to``
    让路由 + URL 一起切换。
    """

    def __init__(self, panel_id: str, label: str, placement: str) -> None:
        super().__init__()
        self._panel_id = panel_id
        self.label = label
        self.placement = placement

    def resolved_action_id(self) -> str:
        return f"mutagent.menu.settings.{self._panel_id}" if self._panel_id else "mutagent.menu.settings"

    async def execute(self, context: ActionContext) -> None:
        conv = context.get("conversation")
        if conv is not None:
            route = f"settings/{self._panel_id}" if self._panel_id else "settings"
            await conv.navigate_to(route)


class RefreshModelsAction(Action):
    action_id = "mutagent.menu.refresh_models"
    label = "Refresh Models"
    placement = "settings:10/20"

    async def execute(self, context: ActionContext) -> None:
        conv = context.get("conversation")
        handler = getattr(conv, "refresh_models", None) if conv is not None else None
        if handler is None:
            return
        result = handler()
        if inspect.isawaitable(result):
            await result
