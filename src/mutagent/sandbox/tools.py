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

External MCP servers, CLI tools, and in-process capabilities are pre-injected
as namespace objects (e.g. `web`, `mutbot`, `playwright`). Discover what's
available:

    help()                       -> list all namespaces
    help(web)                    -> list functions in `web` namespace
    help(web.fetch)              -> show docstring + signature

Calling convention — all namespace functions are keyword-only:

    web.fetch(url="https://example.com")           # correct
    web.fetch("https://example.com")               # WRONG — TypeError

Supported Python: variables, if/for/while, try/except, function/lambda,
f-string, comprehensions, print(), common built-ins (len, range, sorted,
str, int, list, dict, ...).

NOT supported (will raise): import, class, eval, exec, open, getattr,
globals, dir, __builtins__ access.

Multi-step example:

    results = web.search(query="Python PEP 8")
    for r in results[:3]:
        print(web.fetch(url=r["url"]))
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
