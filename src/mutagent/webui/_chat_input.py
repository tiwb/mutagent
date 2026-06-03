"""Chat input widget — Declaration + Implementation, plus chat-input-domain Actions."""

from __future__ import annotations

import mutobj

import logging
from typing import Any, TYPE_CHECKING

from mutgui import Action, ActionContext, ActionRef, ActionToolbar, Bind, Callback, View, ViewBlock

if TYPE_CHECKING:
    from ._conversation import Conversation
    from mutgui.view import ViewId


logger = logging.getLogger(__name__)


class ChatInput(View):
    id: ViewId = "chat-input"
    text: str = ""
    send_mode: str = "enter"
    disabled: bool = False
    toolbar: ActionToolbar
    conversation: Conversation

    def __init__(self, *, conversation: Conversation) -> None: ...

    def render(self) -> ViewBlock: ...


@mutobj.impl(ChatInput.__init__)
def chat_input_init__(
    self: ChatInput,
    conversation: Conversation,
) -> None:
    super(ChatInput, self).__init__()
    self.toolbar = ActionToolbar(
        id="chat-input-toolbar",
        categories=["mutagent.chat_input.toolbar"],
        context=ActionContext(data={"chat_input": self}),
        label_mode="auto",
    )
    self.conversation = conversation


async def _submit(*, view: ChatInput) -> None:
    if view.disabled:
        logger.info("ChatInput submit ignored: disabled")
        return
    text = view.text.strip()
    if not text:
        logger.info("ChatInput submit ignored: empty text")
        return
    logger.info("ChatInput submit triggered (%d chars)", len(text))
    await view.conversation.send(text)
    view.text = ""
    view.invalidate()


async def _cancel(*, view: ChatInput) -> None:
    logger.info("ChatInput cancel triggered")
    await view.conversation.cancel()


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
def _chat_input(context: ActionContext) -> ChatInput | None:
    return context.get("chat_input")


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
        if chat_input is None or chat_input.disabled:
            return False
        return bool(str(chat_input.text).strip())

    async def execute(self, context: ActionContext) -> None:
        chat_input = _chat_input(context)
        if chat_input is None:
            return
        await _submit(view=chat_input)

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
        if chat_input is None:
            return False
        return bool(chat_input.conversation.is_busy)

    async def execute(self, context: ActionContext) -> None:
        chat_input = _chat_input(context)
        if chat_input is None:
            return
        await _cancel(view=chat_input)


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
        if chat_input is None:
            return False
        return chat_input.send_mode == self.mode

    async def execute(self, context: ActionContext) -> None:
        chat_input = _chat_input(context)
        if chat_input is None:
            return
        _set_send_mode(self.mode, view=chat_input)
