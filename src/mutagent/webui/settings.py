"""Settings subsystem public contracts — SettingsPanel base + SettingsDrawer Declaration."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mutgui import View, ViewBlock

if TYPE_CHECKING:
    from mutagent.agent import Agent
    from mutagent.main import App


class SettingsDrawer(View):
    """Drawer 容器，按 active_panel_id 路由到 SettingsPanel 子类。

    通过构造参数 on_models_changed 接收外部回调，由具体面板通过
    notify_models_changed() 触发（解耦面板与 Conversation 的相互引用）。
    """

    is_open: bool
    active_panel_id: str

    def __init__(
        self,
        *,
        app: App,
        agent: Agent,
        on_models_changed=None,
    ) -> None: ...

    def render(self) -> ViewBlock: ...

    async def open(self, panel_id: str) -> None: ...
    async def close(self) -> None: ...
    async def switch_to(self, panel_id: str) -> None: ...

    def list_panels(self) -> list[SettingsPanel]: ...
    async def notify_models_changed(self, preferred_model: str = "") -> None: ...


class SettingsPanel(View):
    """所有设置面板基类。子类声明 panel_id / panel_title / panel_placement。

    SettingsDrawer 通过 discover_subclasses 自动发现所有子类，
    分配到对应 panel_id 路由。每个子类独占一个 _settings_<name>.py 文件。
    """

    panel_id: ClassVar[str] = ""
    panel_title: ClassVar[str] = ""
    panel_placement: ClassVar[str] = ""
    panel_width: ClassVar[int] = 560

    def render(self) -> ViewBlock: ...

    def on_open(self) -> None: ...
    def on_close(self) -> None: ...


from . import _settings_drawer_impl  # noqa: E402,F401
from . import _settings_llm  # noqa: E402,F401
from . import _settings_mcp  # noqa: E402,F401
