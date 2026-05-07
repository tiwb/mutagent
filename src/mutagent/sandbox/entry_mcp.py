"""PySandbox MCP tool — 通过 MCPToolSet 暴露给外部消费者（entry_mcp）。"""

from __future__ import annotations

import asyncio
from typing import Any

from mutagent.net.mcp import MCPToolSet
from mutagent.net._mcp_proto import ToolResult
from mutagent.sandbox.app import SandboxApp, PYSANDBOX_DOC


class PySandboxTools(MCPToolSet):
    """PySandbox tool — 在沙箱中执行 Python 代码。

    需要在 server 启动时设置 _app 引用。
    使用 path 路由到 MCPView（由消费者定义 view）。
    """
    path = "/mcp"
    _app: SandboxApp | None = None

    async def pysandbox(self, code: str) -> str | ToolResult:
        """%s
NOT supported (will raise): import, class, eval, exec, open, getattr,
globals, dir, __builtins__ access.
""" % PYSANDBOX_DOC
        if self._app is None:
            return ToolResult.error("Sandbox not initialized")
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self._app.exec_code, code)

        text, is_error = self._app.format_result(result)
        if is_error:
            return ToolResult.error(text)
        return text
