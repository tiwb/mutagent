"""WebUI server — Declaration + Implementation."""

from __future__ import annotations

import logging
import json
from typing import Any, TYPE_CHECKING

import mutagent
import mutobj
from mutagent.sandbox.entry_mcp import PySandboxTools
from ._conversation import Conversation
from mutgui import Channel, ModuleRegistry, ViewPort
from mutio.mcp.view import MCPView
from mutio.codec.json import JsonObject, get_field, narrow_value
from mutio.net.server import HTMLResponse, Server, StaticView, WebSocketConnection, WebSocketDisconnect

if TYPE_CHECKING:
    from mutagent.core.agent import Agent
    from mutagent.app.app import App


logger = logging.getLogger(__name__)


class WebUIServer(Server):
    app: App
    agent: Agent
    conversation: Conversation

    def __init__(
        self,
        *,
        app: App,
        agent: Agent,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None: ...


@mutobj.impl(WebUIServer.on_startup)
async def web_ui_server_on_startup(self: WebUIServer) -> None:
    """在 server 自己的 event loop 上连接 mcp_sources / cli_sources。

    必须在这里 await（而不是 setup_agent 后马上），因为 MCP client 会绑定到
    调用时的 event loop，后续 agent.run 也跑在此 loop。
    """
    try:
        await self.app.connect_sources()
        # bind_main_loop 注入 _async_loop，供 MCPSettingsPanel 的
        # _submit_async 跨线程投递协程使用（Connect/Disconnect/Reconnect 等按钮）
        sandbox = self.app.sandbox
        sandbox.bind_main_loop()
    except Exception:
        logger.exception("connect_sources failed during WebUI startup")


def _module_registry() -> ModuleRegistry:
    registry = ModuleRegistry()
    registry.add_from_package("mutgui")
    registry.add_from_package("mutagent")
    return registry


class WebSocketChannel(Channel):
    """mutgui Channel backed by mutio WebSocketConnection."""
    _ws: WebSocketConnection

    def __init__(self, ws: WebSocketConnection) -> None:
        super().__init__()
        self._ws = ws

    async def send(self, message: dict[str, Any]) -> None:
        await self._ws.send_json(message)


def _json_script(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _mutgui_runtime_assets(registry: ModuleRegistry) -> str:
    runtime_manifest = registry.runtime_manifest()
    import_map = {"imports": runtime_manifest["importMap"]}
    return f'  <script type="importmap">{_json_script(import_map)}</script>'


def _mutgui_boot_script(registry: ModuleRegistry) -> str:
    return f'  <script src="{registry.url_for("mutgui", "boot.js")}"></script>'


def _render_root_html(registry: ModuleRegistry | None = None) -> str:
    registry = registry or _module_registry()
    return """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>mutagent webui</title>
  <style>
    html, body, #app { height: 100%; margin: 0; }
  </style>
</head>
<body>
  <div data-mutgui-app data-ws-url="/ws" id="app"></div>
""" + _mutgui_runtime_assets(registry) + """
""" + _mutgui_boot_script(registry) + """
</body>
</html>
"""


def _runtime_messages(registry: ModuleRegistry) -> list[dict[str, str]]:
    runtime_manifest = registry.runtime_manifest()
    messages: list[dict[str, str]] = []
    for href in runtime_manifest["css"]:
        messages.append({"type": "runtime.css", "href": href})
    messages.extend([
        {"type": "runtime.import", "module": "@mutgui/antd"},
        {"type": "runtime.import", "module": "@mutagent/ui"},
        {"type": "runtime.install", "module": "@mutgui/theme-dark"},
        {"type": "runtime.mount"},
    ])
    return messages


@mutobj.impl(WebUIServer.__init__)
def web_ui_server_init__(
    self: WebUIServer,
    *,
    app: Any,
    agent: Any,
    host: str = "127.0.0.1",
    port: int = 0,
) -> None:
    super(WebUIServer, self).__init__(host=host, port=port)
    self.app = app
    self.agent = agent
    self.conversation = Conversation(agent=agent, app=app)
    registry = _module_registry()

    conversation = self.conversation
    from mutio.net.server import View as HttpView, WebSocketView

    class _HTTPRoot(HttpView):  # pyright: ignore[reportUnusedClass]
        path = "/"

        async def get(self, request: Any) -> HTMLResponse:
            logger.info("Serving WebUI root HTML")
            return HTMLResponse(_render_root_html(registry))

    class _WSView(WebSocketView):  # pyright: ignore[reportUnusedClass]
        path = "/ws"

        async def connect(self, ws: WebSocketConnection) -> None:
            logger.info("WebUI WebSocket connected")
            await ws.accept()
            first_message = narrow_value(await ws.receive_json(), JsonObject)
            if get_field(first_message, "type", str, default="") != "mount.attach":
                await ws.close(code=4400, reason="expected mount.attach")
                return
            for message in _runtime_messages(registry):
                await ws.send_json(message)
            channel = WebSocketChannel(ws)
            viewport = ViewPort(conversation, channel, _client=get_field(first_message, "client", JsonObject, default=None, fallback=None))
            await viewport.initialize()
            await conversation.rendered()
            try:
                while True:
                    event = narrow_value(await ws.receive_json(), JsonObject)
                    logger.debug("WebUI received frontend event: %s", event)
                    await viewport.handle_event(event)
            except WebSocketDisconnect:
                logger.info("WebUI WebSocket disconnected")
                pass
            finally:
                viewport.detach()

    static_views: list[type[StaticView]] = []
    for index, (path, directory) in enumerate(registry.static_mounts()):
        static_views.append(type(
            f"_StaticFiles{index}",
            (StaticView,),
            {"path": path, "directory": str(directory)},
        ))

    # PySandbox MCP endpoint —— 注入 sandbox 后注册 MCPView，
    # PySandboxTools.path == "/mcp" 会自动挂接到该 view。
    PySandboxTools.env = self.app.sandbox

    class _MCPView(MCPView):  # pyright: ignore[reportUnusedClass]
        path = "/mcp"
        name = "mutagent-webui"
        version = mutagent.__version__


