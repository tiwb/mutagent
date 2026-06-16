"""Settings subsystem 全页面式 SettingsPage 。 """

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING, ClassVar

import mutobj
from mutgui import Action, ActionContext, Callback, Expr, View, ViewBlock

if TYPE_CHECKING:
    from ._conversation import Conversation


class SettingsPage(View):
    """全页面设置容器（替代旧的 SettingsDrawer 浮层）。

    路由权威由 Conversation 持有；本类不持有 ``is_open``/打开状态字段。
    panel 切换通过 ``activate(panel_id)`` / ``deactivate()`` 由
    Conversation 显式驱动。

    通过与 Conversation 的级联引用直接完成导航：
    ``self.conversation.navigate_to(route)``。
    """

    active_panel_id: str
    conversation: Conversation
    panels: dict[str, SettingPanel] = mutobj.field(default_factory=dict)
    ordered_panel_ids: list[str] = mutobj.field(default_factory=list)
    active: bool = False

    def __init__(self, *, conversation: Conversation) -> None: ...

    def render(self) -> ViewBlock: ...

    async def activate(self, panel_id: str) -> None: ...
    async def deactivate(self) -> None: ...
    async def close(self) -> None: ...

    def list_panels(self) -> list[SettingPanel]: ...


class SettingPanel(View):
    """所有设置面板基类。子类声明 panel_id / panel_title / panel_placement。

    SettingsPage 通过 discover_subclasses 自动发现所有子类，
    分配到对应 panel_id 路由。每个子类独占一个 _settings_<name>.py 文件。
    """

    panel_id: ClassVar[str] = ""
    panel_title: ClassVar[str] = ""
    panel_placement: ClassVar[str] = ""
    panel_width: ClassVar[int] = 560
    page: SettingsPage

    def render(self) -> ViewBlock: ...

    on_open: Callable[[], None] = mutobj.field(default=lambda: None)
    on_close: Callable[[], None] = mutobj.field(default=lambda: None)


# ── @impl: SettingPanel ─────────────────────────────────────


@mutobj.impl(SettingPanel.render)
def setting_panel_render(self: SettingPanel) -> ViewBlock:
    raise NotImplementedError


# ── @impl: __init__ ──────────────────────────────────────────


@mutobj.impl(SettingsPage.__init__)
def settings_page_init__(
    self: SettingsPage,
    *,
    conversation: Conversation,
) -> None:
    super(SettingsPage, self).__init__()
    self.id = "settings-page"
    self.conversation = conversation
    self.active_panel_id = ""

    # ── 发现并实例化全部 SettingsPanel 子类 ─────
    panel_classes = mutobj.discover_subclasses(SettingPanel)
    self.panels = {}
    self.ordered_panel_ids = []

    for cls in panel_classes:
        panel_id = cls.panel_id
        if not panel_id:
            continue
        panel = cls(page=self)
        self.panels[panel_id] = panel

    def _placement_key(panel_id: str) -> str:
        panel = self.panels[panel_id]
        placement = type(panel).panel_placement
        return placement or f"zzzz:{panel_id}"

    self.ordered_panel_ids = sorted(self.panels.keys(), key=_placement_key)
    if self.ordered_panel_ids:
        self.active_panel_id = self.ordered_panel_ids[0]
    # 是否处于"已激活"状态。初始 False：``active_panel_id`` 只是默认占位，
    # panel 未触发过 ``on_open``。activate / deactivate 均仅在 active=True 时
    # 才 close 旧 panel，避免首次进入 settings 时误发 close 事件。
    self.active = False


@mutobj.impl(SettingsPage.render)
def settings_page_render(self: SettingsPage) -> ViewBlock:
    active = self.panels.get(self.active_panel_id)

    menu_items: list[dict[str, Any]] = []
    for panel_id in self.ordered_panel_ids:
        panel = self.panels[panel_id]
        title = type(panel).panel_title or panel_id
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


@mutobj.impl(SettingsPage.activate)
async def settings_page_activate(self: SettingsPage, panel_id: str) -> None:
    target = panel_id or (self.ordered_panel_ids[0] if self.ordered_panel_ids else "")
    if not target or target not in self.panels:
        return

    if self.active:
        prev = self.panels.get(self.active_panel_id)
        if prev is not None and prev is not self.panels.get(target):
            prev.on_close()

    self.active_panel_id = target
    self.active = True
    new_panel = self.panels[target]
    new_panel.on_open()
    new_panel.invalidate()
    self.invalidate()


@mutobj.impl(SettingsPage.deactivate)
async def settings_page_deactivate(self: SettingsPage) -> None:
    if self.active:
        prev = self.panels.get(self.active_panel_id)
        if prev is not None:
            prev.on_close()
    self.active = False
    self.invalidate()


@mutobj.impl(SettingsPage.close)
async def settings_page_close(self: SettingsPage) -> None:
    await self.conversation.navigate_to("")


@mutobj.impl(SettingsPage.list_panels)
def settings_page_list_panels(self: SettingsPage) -> list[SettingPanel]:
    return [self.panels[pid] for pid in self.ordered_panel_ids]



# ── Callback handlers ─────────────────────────────────────────


async def _on_menu_click(view: SettingsPage, panel_id: str = "") -> None:
    if not panel_id:
        return
    await view.conversation.navigate_to(f"settings/{panel_id}")


async def _on_back_click(view: SettingsPage, *_: Any) -> None:
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


# 导入 settings panel 子类以触发 mutobj 注册（SettingsPage 发现子类前需要）
from . import _settings_llm as _settings_llm # noqa: E402,F401
from . import _settings_mcp as _settings_mcp  # noqa: E402,F401
