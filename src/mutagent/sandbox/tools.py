"""PySandbox MCP tool — 通过 MCPToolSet 暴露给外部消费者。"""

import asyncio
from typing import Any

from mutagent.net.mcp import MCPToolSet
from mutagent.net._mcp_proto import ToolResult


def format_exec_result(result: dict[str, Any]) -> tuple[str, bool]:
    """把 SandboxApp.exec_code 的返回拍成文本。返回 (text, is_error)。

    字符串类型的 result 直接原文输出(多行原文可读);其他类型用 repr()(dict/list
    的 repr ≈ str,自定义对象保留结构)。pysandbox 不是 REPL,不需要 REPL 式区分
    "a"/'a' 的 repr 包装。
    """
    if "error" in result:
        text = result["error"]
        if result.get("traceback"):
            text += "\n" + result["traceback"]
        return text, True

    parts: list[str] = []
    if result.get("stdout"):
        parts.append(result["stdout"])
    value = result.get("result")
    if value is not None:
        if isinstance(value, str):
            parts.append(value)
        else:
            parts.append(repr(value))
    return ("\n".join(parts) if parts else "(no output)"), False


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

        text, is_error = format_exec_result(result)
        if is_error:
            return ToolResult.error(text)
        return text
