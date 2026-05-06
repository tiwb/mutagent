"""LLM settings panel for mutagent WebUI."""

from __future__ import annotations

from typing import Any, Callable

from mutgui import View, ViewBlock


class LLMSettingsPanel(View):
    current_step: str
    editing_key: str
    editing_is_new: bool
    provider_name: str
    provider_type: str
    provider_type_label: str
    base_url: str
    auth_token: str
    models: list[str]
    discovered_models: list[str]
    default_model: str
    error: str
    notice: str

    def __init__(
        self,
        *,
        app: Any,
        agent: Any,
        on_close: Callable[[], Any] | None = None,
        on_saved: Callable[[str], Any] | None = None,
    ) -> None: ...

    def render(self) -> ViewBlock: ...


from . import _settings_impl  # noqa: E402,F401
