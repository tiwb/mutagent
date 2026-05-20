"""Chat input widget."""

from __future__ import annotations

from typing import Any, Callable

from mutgui import ActionToolbar, View, ViewBlock


class ChatInput(View):
    text: str
    send_mode: str
    disabled: bool
    is_busy: bool
    toolbar: ActionToolbar
    conversation: Any | None

    def __init__(
        self,
        *,
        on_send: Callable[[str], Any],
        on_cancel: Callable[[], Any] | None = None,
    ) -> None: ...

    def render(self) -> ViewBlock: ...


from . import _chat_input_impl as _chat_input_impl  # noqa: E402,F401
