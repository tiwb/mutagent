"""PySandbox MCP tool — 通过 MCPToolSet 暴露给外部消费者。"""

import asyncio

from mutagent.net.mcp import MCPToolSet
from mutagent.net._mcp_proto import ToolResult


class PySandboxTools(MCPToolSet):
    """PySandbox tool — 在沙箱中执行 Python 代码。

    需要在 server 启动时设置 _app 引用。
    使用 path 路由到 MCPView（由消费者定义 view）。
    """
    path = "/mcp"
    _app = None

    async def pysandbox(self, code: str) -> str | ToolResult:
        """Execute Python code in a sandboxed environment.

All available functions are pre-injected as namespace objects.
Use help(func) for detailed documentation.
import is not supported.
"""
        if self._app is None:
            return ToolResult.error("Sandbox not initialized")
        # 在线程池执行，避免阻塞 event loop
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self._app.exec_code, code)

        if "error" in result:
            text = result["error"]
            if result.get("traceback"):
                text += "\n" + result["traceback"]
            return ToolResult.error(text)

        parts = []
        if result.get("stdout"):
            parts.append(result["stdout"])
        if result.get("result") is not None:
            parts.append(repr(result["result"]))
        return '\n'.join(parts) if parts else "(no output)"
