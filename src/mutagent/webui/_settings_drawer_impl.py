"""Default SettingsDrawer implementation + settings-domain Actions."""

from __future__ import annotations

import inspect
from typing import Any

from functools import partial

import mutagent
import mutobj
from mutobj.core import AttributeDescriptor
from mutagent.webui.settings import SettingsDrawer, SettingsPanel
from mutgui import Action, ActionContext, Callback, ViewBlock


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


@mutagent.impl(SettingsDrawer.__init__)
def __init__(
    self: SettingsDrawer,
    *,
    app: Any,
    agent: Any,
    on_models_changed: Any = None,
) -> None:
    super(SettingsDrawer, self).__init__()
    self.id = "settings-drawer"
    self._app = app
    self._agent = agent
    self._on_models_changed = on_models_changed
    self.is_open = False
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
        setattr(panel, "drawer", self)
        self._panels[panel_id] = panel

    def _placement_key(panel_id: str) -> str:
        panel = self._panels[panel_id]
        placement = _resolve_panel_attr(type(panel), "panel_placement")
        return placement or f"zzzz:{panel_id}"

    self._ordered_panel_ids = sorted(self._panels.keys(), key=_placement_key)
    if self._ordered_panel_ids:
        self.active_panel_id = self._ordered_panel_ids[0]


@mutagent.impl(SettingsDrawer.render)
def render(self: SettingsDrawer) -> ViewBlock:
    active = self._panels.get(self.active_panel_id) if self.is_open else None
    drawer_props: dict[str, Any] = {
        "$component": "antd.Drawer",
        "$id": "settings-drawer",
        "placement": "right",
        "open": self.is_open,
        "width": getattr(active, "panel_width", 560) if active else 560,
        "destroyOnHidden": False,
        "onClose": Callback(partial(_close_handler, self)),
    }
    if active:
        drawer_props["title"] = active.panel_title
        drawer_props["$children"] = [active]
    else:
        drawer_props["$children"] = []
    return ViewBlock([drawer_props])


@mutagent.impl(SettingsDrawer.open)
async def open(self: SettingsDrawer, panel_id: str) -> None:
    panel = self._panels.get(panel_id)
    if panel is None:
        return
    self.active_panel_id = panel_id
    self.is_open = True
    on_open = getattr(panel, "on_open", None)
    if callable(on_open):
        result = on_open()
        if inspect.isawaitable(result):
            await result
    panel.invalidate()
    self.invalidate()


@mutagent.impl(SettingsDrawer.close)
async def close(self: SettingsDrawer) -> None:
    active = self._panels.get(self.active_panel_id)
    if active is not None:
        on_close = getattr(active, "on_close", None)
        if callable(on_close):
            result = on_close()
            if inspect.isawaitable(result):
                await result
    self.is_open = False
    self.invalidate()


@mutagent.impl(SettingsDrawer.switch_to)
async def switch_to(self: SettingsDrawer, panel_id: str) -> None:
    if panel_id not in self._panels:
        return
    prev = self._panels.get(self.active_panel_id)
    if prev is not None:
        on_close = getattr(prev, "on_close", None)
        if callable(on_close):
            result = on_close()
            if inspect.isawaitable(result):
                await result
    self.active_panel_id = panel_id
    next_panel = self._panels[panel_id]
    on_open = getattr(next_panel, "on_open", None)
    if callable(on_open):
        result = on_open()
        if inspect.isawaitable(result):
            await result
    next_panel.invalidate()
    self.invalidate()


@mutagent.impl(SettingsDrawer.list_panels)
def list_panels(self: SettingsDrawer) -> list[SettingsPanel]:
    return [self._panels[pid] for pid in self._ordered_panel_ids]


@mutagent.impl(SettingsDrawer.notify_models_changed)
async def notify_models_changed(self: SettingsDrawer, preferred_model: str = "") -> None:
    cb = self._on_models_changed
    if cb is not None:
        result = cb(preferred_model)
        if inspect.isawaitable(result):
            await result


async def _close_handler(view: SettingsDrawer) -> None:
    await view.close()


# ── 私有辅助 ─────────────────────────────────────────
def _conversation(context: ActionContext) -> Any | None:
    return context.get("conversation")


async def _call_action(handler: Any, *args: Any) -> None:
    if handler is None:
        return
    result = handler(*args)
    if inspect.isawaitable(result):
        await result


# ── Settings 域 Actions ────────────────────────────


class OpenSettingsAction(Action):
    """通用设置面板入口 — 一个类支撑所有 SettingsPanel 子类。"""

    def __init__(self, panel_id: str, label: str, placement: str) -> None:
        super().__init__()
        self._panel_id = panel_id
        self.label = label
        self.placement = placement

    def resolved_action_id(self) -> str:
        return f"mutagent.menu.settings.{self._panel_id}"

    async def execute(self, context: ActionContext) -> None:
        drawer = context.get("settings_drawer")
        if drawer is not None:
            await drawer.open(self._panel_id)


class RefreshModelsAction(Action):
    action_id = "mutagent.menu.refresh_models"
    label = "Refresh Models"
    placement = "settings:10/20"

    async def execute(self, context: ActionContext) -> None:
        conv = _conversation(context)
        await _call_action(getattr(conv, "refresh_models", None))
