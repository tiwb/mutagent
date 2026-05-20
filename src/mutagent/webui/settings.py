"""Settings subsystem public contracts — SettingsPanel base + SettingsPage Declaration."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, ClassVar

from mutgui import View, ViewBlock

if TYPE_CHECKING:
    from mutagent.agent import Agent
    from mutagent.main import App


class SettingsPage(View):
    """全页面设置容器（替代旧的 SettingsDrawer 浮层）。

    路由权威由 Conversation 持有；本类不持有 ``is_open``/打开状态字段。
    panel 切换通过 ``activate(panel_id)`` / ``deactivate()`` 由
    Conversation 显式驱动。

    构造时通过两个回调把"想要切换路由"的请求委派回上层：

    - ``on_request_close`` — 用户点「← 返回对话」或保存后等场景，等价于
      ``Conversation.navigate_to("")``。
    - ``on_request_navigate(route)`` — 左侧菜单切换 panel，等价于
      ``Conversation.navigate_to(f"settings/{panel_id}")``。


    """

    active_panel_id: str

    def __init__(
        self,
        *,
        app: App,
        agent: Agent,
        on_models_changed=None,
        on_request_close=None,
        on_request_navigate=None,
    ) -> None: ...

    def render(self) -> ViewBlock: ...

    async def activate(self, panel_id: str) -> None: ...
    async def deactivate(self) -> None: ...
    async def close(self) -> None: ...

    def list_panels(self) -> list[SettingsPanel]: ...
    async def notify_models_changed(self, preferred_model: str = "") -> None: ...


class SettingsPanel(View):
    """所有设置面板基类。子类声明 panel_id / panel_title / panel_placement。

    SettingsPage 通过 discover_subclasses 自动发现所有子类，
    分配到对应 panel_id 路由。每个子类独占一个 _settings_<name>.py 文件。
    """

    panel_id: ClassVar[str] = ""
    panel_title: ClassVar[str] = ""
    panel_placement: ClassVar[str] = ""
    panel_width: ClassVar[int] = 560
    page: Any  # SettingsPage 实例，由 SettingsPage.__init__ 注入

    def render(self) -> ViewBlock: ...

    def on_open(self) -> None: ...
    def on_close(self) -> None: ...


from . import _settings_page_impl  # noqa: E402,F401
from . import _settings_llm  # noqa: E402,F401
from . import _settings_mcp  # noqa: E402,F401
