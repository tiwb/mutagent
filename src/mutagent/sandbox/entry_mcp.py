"""PySandbox MCP tool — 通过 MCPToolSet 暴露给外部消费者（entry_mcp）。"""

from __future__ import annotations

import asyncio
from typing import ClassVar

from mutio.mcp.toolset import MCPToolSet
from mutio.mcp.protocol import ToolResult
from mutagent.sandbox.env import SandboxEnv, PYSANDBOX_DOC


class PySandboxTools(MCPToolSet):
    """PySandbox tool — 在沙箱中执行 Python 代码。

    需要在 server 启动时设置 sandbox 引用。
    使用 path 路由到 MCPView（由消费者定义 view）。
    """
    path = "/mcp"
    # 启动期注入的类级单例
    env: ClassVar[SandboxEnv | None] = None

    async def pysandbox(self, code: str) -> str | ToolResult:
        if self.env is None:
            return ToolResult.error("Sandbox not initialized")
        loop = asyncio.get_running_loop()
        # 注入主 loop 供 _wrap_async 投递 async NamespaceTools 方法
        self.env.bind_main_loop()
        result = await loop.run_in_executor(None, self.env.exec_code, code)

        text, is_error = self.env.format_result(result)
        if is_error:
            return ToolResult.error(text)
        return text


PySandboxTools.pysandbox.__doc__ = PYSANDBOX_DOC
