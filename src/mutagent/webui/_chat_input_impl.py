"""Default ChatInput implementation + chat-input-domain Actions."""

from __future__ import annotations

import mutobj

import inspect
import logging
from functools import partial
from typing import Any, Callable

from mutagent.webui.chat_input import ChatInput
from mutgui import Action, ActionContext, ActionRef, ActionToolbar, Bind, Callback, ViewBlock

logger = logging.getLogger(__name__)


# ── Extension ──────────────────────────────────────────────

class ChatInputExt(mutobj.Extension[ChatInput]):
    """ChatInput 的运行时私有状态。"""
    on_send: Any = None
    on_cancel: Any = None
    submit_action: Any = None
    cancel_action: Any = None
    set_send_mode_action: Any = None


def _ciext(self: ChatInput) -> ChatInputExt:
    return ChatInputExt.get_or_create(self)


@mutobj.impl(ChatInput.__init__)
def chat_input_init__(
    self: ChatInput,
    *,
    on_send: Callable[[str], Any],
    on_cancel: Callable[[], Any] | None = None,
) -> None:
    ext = _ciext(self)
    super(ChatInput, self).__init__()
    self.id = "chat-input"
    self.text = ""
    self.send_mode = "enter"
    self.disabled = False
    self.is_busy = False
    self.toolbar = ActionToolbar(
        id="chat-input-toolbar",
        categories=["mutagent.chat_input.toolbar"],
        context=ActionContext(data={"chat_input": self}),
        label_mode="auto",
    )
    self.conversation = None
    ext.on_send = on_send
    ext.on_cancel = on_cancel
    ext.submit_action = partial(_submit, view=self)
    ext.cancel_action = partial(_cancel, view=self)
    ext.set_send_mode_action = partial(_set_send_mode, view=self)


async def _submit(*, view: ChatInput) -> None:
    if view.disabled:
        logger.info("ChatInput submit ignored: disabled")
        return
    text = view.text.strip()
    if not text:
        logger.info("ChatInput submit ignored: empty text")
        return
    logger.info("ChatInput submit triggered (%d chars)", len(text))
    ext = _ciext(view)
    result = ext.on_send(text)
    if hasattr(result, "__await__"):
        await result
    view.text = ""
    view.invalidate()


async def _cancel(*, view: ChatInput) -> None:
    ext = _ciext(view)
    if ext.on_cancel is None:
        return
    logger.info("ChatInput cancel triggered")
    result = ext.on_cancel()
    if hasattr(result, "__await__"):
        await result


def _set_send_mode(value: str, *, view: ChatInput) -> None:
    view.send_mode = value or "enter"
    view.invalidate()


@mutobj.impl(ChatInput.render)
def chat_input_render(self: ChatInput) -> ViewBlock:
    placeholder = (
        "Type a message… (Shift+Enter for newline)"
        if self.send_mode == "enter"
        else "Type a message… (Ctrl+Enter to send)"
    )
    self.toolbar.context = ActionContext(
        data={
            "chat_input": self,
            "conversation": self.conversation,
        },
    )
    self.toolbar.invalidate()
    return ViewBlock([
        {
            "$component": "mutagent.ChatInput",
            "$id": "chat-input-shell",
            "value": self.text,
            "sendMode": self.send_mode,
            "disabled": self.disabled,
            "placeholder": placeholder,
            "$children": [self.toolbar],
            "onChange": Bind(self, "text", "$0"),
            "onSubmit": Callback(_submit, view=self),
        }
    ])


# ── 私有辅助 ─────────────────────────────────────────
def _chat_input(context: ActionContext) -> Any | None:
    return context.get("chat_input")


async def _call_action(handler: Any, *args: Any) -> None:
    if handler is None:
        return
    result = handler(*args)
    if inspect.isawaitable(result):
        await result


# ── ChatInput 域 Actions ────────────────────────────


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
        if chat_input is None:
            return
        ext = _ciext(chat_input)
        await _call_action(ext.submit_action)

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
        if chat_input is None:
            return
        ext = _ciext(chat_input)
        await _call_action(ext.cancel_action)


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
        if chat_input is None:
            return
        ext = _ciext(chat_input)
        await _call_action(ext.set_send_mode_action, self.mode)
