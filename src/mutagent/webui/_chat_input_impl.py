"""Default ChatInput implementation."""

from __future__ import annotations

import logging
from functools import partial
from typing import Callable

import mutagent
from mutagent.webui.chat_input import ChatInput
from mutgui import ActionContext, ActionToolbar, Bind, Callback, ViewBlock

logger = logging.getLogger(__name__)


@mutagent.impl(ChatInput.__init__)
def __init__(
    self: ChatInput,
    *,
    on_send: Callable[[str], Any],
    on_cancel: Callable[[], Any] | None = None,
) -> None:
    super(ChatInput, self).__init__()
    self.id = "chat-input"
    self.text = ""
    self.send_mode = "enter"
    self.disabled = False
    self.is_busy = False
    self.toolbar = ActionToolbar(
        id="chat-input-toolbar",
        categories=["mutagent.chat_input.toolbar"],
        context=ActionContext(owner=self, data={"chat_input": self}),
        label_mode="auto",
    )
    self.conversation = None
    self._on_send = on_send
    self._on_cancel = on_cancel
    self._submit_action = partial(_submit, view=self)
    self._cancel_action = partial(_cancel, view=self)
    self._set_send_mode_action = partial(_set_send_mode, view=self)


async def _submit(*, view: ChatInput) -> None:
    if view.disabled:
        logger.info("ChatInput submit ignored: disabled")
        return
    text = view.text.strip()
    if not text:
        logger.info("ChatInput submit ignored: empty text")
        return
    logger.info("ChatInput submit triggered (%d chars)", len(text))
    result = view._on_send(text)
    if hasattr(result, "__await__"):
        await result
    view.text = ""
    view.invalidate()


async def _cancel(*, view: ChatInput) -> None:
    if view._on_cancel is None:
        return
    logger.info("ChatInput cancel triggered")
    result = view._on_cancel()
    if hasattr(result, "__await__"):
        await result


def _set_send_mode(value: str, *, view: ChatInput) -> None:
    view.send_mode = value or "enter"
    view.invalidate()


@mutagent.impl(ChatInput.render)
def render(self: ChatInput) -> ViewBlock:
    placeholder = (
        "Type a message… (Shift+Enter for newline)"
        if self.send_mode == "enter"
        else "Type a message… (Ctrl+Enter to send)"
    )
    self.toolbar.context = ActionContext(
        owner=self,
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
            "onSubmit": Callback(_submit, view="@view"),
        }
    ])
