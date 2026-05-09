"""PySandbox MCP tool — 通过 MCPToolSet 暴露给外部消费者（entry_mcp）。"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from mutio.mcp.toolset import MCPToolSet
from mutio.mcp.protocol import ToolResult
from mutagent.sandbox.app import SandboxApp, PYSANDBOX_DOC


class PySandboxTools(MCPToolSet):
    """PySandbox tool — 在沙箱中执行 Python 代码。

    需要在 server 启动时设置 _app 引用。
    使用 path 路由到 MCPView（由消费者定义 view）。
    """
    path = "/mcp"
    # 启动期注入的类级单例（ClassVar 避免被 mutobj 包成 per-instance AttributeDescriptor）
    _app: ClassVar[SandboxApp | None] = None

    async def pysandbox(self, code: str) -> str | ToolResult:
        if self._app is None:
            return ToolResult.error("Sandbox not initialized")
        loop = asyncio.get_running_loop()
        # 注入主 loop 供 _wrap_async 投递 async NamespaceTools 方法
        self._app.bind_main_loop()
        result = await loop.run_in_executor(None, self._app.exec_code, code)

        text, is_error = self._app.format_result(result)
        if is_error:
            return ToolResult.error(text)
        return text


PySandboxTools.pysandbox.__doc__ = PYSANDBOX_DOC
