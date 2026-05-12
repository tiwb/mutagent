"""Action definitions for mutagent WebUI conversation shell."""

from __future__ import annotations

import inspect
from typing import Any

from mutgui import Action, ActionContext, ActionRef


def _conversation(context: ActionContext) -> Any | None:
    return context.get("conversation")


def _chat_input(context: ActionContext) -> Any | None:
    return context.get("chat_input")


async def _call_action(handler: Any, *args: Any) -> None:
    if handler is None:
        return
    result = handler(*args)
    if inspect.isawaitable(result):
        await result


class ModelSelectorAction(Action):
    action_id = "mutagent.toolbar.model_selector"
    categories = ("mutagent.conversation.toolbar",)
    label = "Model"
    position = "start"
    placement = "primary:10/10"
    variant = "dropdown"
    menu_placement = "bottom-start"

    def resolved_label(self, context: ActionContext | None = None) -> str:
        if context:
            conv = context.get("conversation")
            if conv and getattr(conv, "current_model", ""):
                return conv.current_model
        return self.label or "Model"

    def check_enabled(self, context: ActionContext) -> bool:
        conv = _conversation(context)
        return not getattr(conv, "is_busy", False)

    def menu_actions(self, context: ActionContext) -> list[ActionRef]:
        conv = _conversation(context)
        models = getattr(conv, "models", [])
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
        return getattr(conv, "current_model", "") == self._model_name

    async def execute(self, context: ActionContext) -> None:
        conv = _conversation(context)
        await _call_action(getattr(conv, "_handle_model_change", None), self._model_name)


class AgentStatusAction(Action):
    action_id = "mutagent.toolbar.status"
    categories = ("mutagent.conversation.toolbar",)
    label = "Status"
    position = "start"
    placement = "primary:10/20"
    variant = "widget"

    def toolbar_view(self, context: ActionContext) -> Any:
        conversation = _conversation(context)
        return getattr(conversation, "status_bar", None)


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


class MainMenuAction(Action):
    action_id = "mutagent.toolbar.main_menu"
    categories = ("mutagent.conversation.toolbar",)
    label = "☰"
    tooltip = "Settings"
    position = "end"
    placement = "menu:20/10"
    variant = "dropdown"

    def menu_actions(self, context: ActionContext) -> list[ActionRef]:
        drawer = context.get("settings_drawer")
        items: list[ActionRef] = []
        if drawer is not None:
            for panel in drawer.list_panels():
                items.append(ActionRef(action=OpenSettingsAction(
                    panel_id=panel.panel_id,
                    label=panel.panel_title,
                    placement=getattr(panel, "panel_placement", ""),
                )))
        items.append(ActionRef(action=RefreshModelsAction))
        return items


class SendMessageAction(Action):
    action_id = "mutagent.chat_input.send"
    categories = ("mutagent.chat_input.toolbar",)
    label = "Send"
    position = "end"
    placement = "submit:10/10"
    variant = "split"
    menu_placement = "top-start"

    def check_enabled(self, context: ActionContext) -> bool:
        chat_input = _chat_input(context)
        if chat_input is None or getattr(chat_input, "disabled", False):
            return False
        return bool(str(getattr(chat_input, "text", "")).strip())

    async def execute(self, context: ActionContext) -> None:
        chat_input = _chat_input(context)
        await _call_action(getattr(chat_input, "_submit_action", None))

    def menu_actions(self, context: ActionContext) -> list[ActionRef]:
        return [
            ActionRef(action=SetSendModeChoiceAction("enter", "Send with Enter")),
            ActionRef(action=SetSendModeChoiceAction("ctrl-enter", "Send with Ctrl+Enter")),
        ]


class CancelMessageAction(Action):
    action_id = "mutagent.chat_input.cancel"
    categories = ("mutagent.chat_input.toolbar",)
    label = "Stop"
    position = "end"
    placement = "submit:10/20"

    def check_visible(self, context: ActionContext) -> bool:
        chat_input = _chat_input(context)
        return bool(getattr(chat_input, "is_busy", False))

    async def execute(self, context: ActionContext) -> None:
        chat_input = _chat_input(context)
        await _call_action(getattr(chat_input, "_cancel_action", None))


class SetSendModeChoiceAction(Action):
    variant = "button"

    def __init__(self, mode: str, label: str) -> None:
        super().__init__()
        self.mode = mode
        self.label = label

    def resolved_action_id(self) -> str:
        return f"mutagent.chat_input.send_mode.{self.mode}"

    def check_checked(self, context: ActionContext) -> bool:
        chat_input = _chat_input(context)
        return getattr(chat_input, "send_mode", "enter") == self.mode

    async def execute(self, context: ActionContext) -> None:
        chat_input = _chat_input(context)
        await _call_action(getattr(chat_input, "_set_send_mode_action", None), self.mode)
