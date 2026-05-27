"""mutagent.sandbox.env -- SandboxEnv Declaration。

纯 sync 的 namespace registry + 受限代码执行引擎。
不读 config，不管理 MCP 连接，不创建 event loop。

MCP 连接通过 ``connect_source()`` / ``disconnect_source()`` 注入和摘除，
Namespace 构造和 on_remove 回调等内部细节不暴露。
"""

from __future__ import annotations

from typing import Any, Iterator, TYPE_CHECKING

import mutobj

if TYPE_CHECKING:
    from mutagent.sandbox.mcp import MCPConnection


PYSANDBOX_DOC = """\
Execute Python code in a sandboxed environment.

Supported Python: variables, if/for/while, try/except, function/lambda,
f-string, comprehensions, print(), common built-ins (len, range, sorted,
str, int, list, dict, ...).

NOT supported (will raise): import, eval, exec, compile, __import__,
open, breakpoint, input.

External MCP servers, CLI tools, and in-process capabilities are pre-injected
as namespace objects. Discover what's available:

    help()                       -> list all namespaces
    help(web)                    -> describe `web` namespace + list functions
    help(web.fetch)              -> show function signature + full docstring

Calling convention — all namespace functions are keyword-only:

    web.fetch(url="https://example.com")           # correct
    web.fetch("https://example.com")               # WRONG — TypeError

Multi-step example:

    results = web.search(query="Python PEP 8")
    for r in results[:3]:
        print(web.fetch(url=r["url"]))
"""


class SandboxEnv(mutobj.Declaration):
    """Python 沙箱 — namespace registry + 受限代码执行环境。

    职责边界：
    - **持有**：注入的 Namespace 集合、清理回调、REPL 缓存
    - **不持有**：config、MCP client、event loop 引用

    能力来源：
    - MCP server 桥接（通过 ``connect_source()`` 注入）
    - CLI 白名单（内部构造，不暴露）
    - NamespaceTools（本进程 Declaration 自动发现，懒加载）
    """

    def connect_source(self, conn: "MCPConnection") -> None:
        """注入一个 MCP 连接源。幂等。

        内部完成：登记 connection、挂 sandbox 回引、注册 namespace
        （绑定 ``conn.close`` 作为 on_remove 回调）。

        用于：启动期 ``connect_sources`` 和 Settings Panel 的 Connect 按钮。
        """
        ...

    def disconnect_source(self, name: str) -> None:
        """移除一个 MCP 连接源。幂等。

        内部完成：摘除 namespace（触发 ``conn.close()`` 清理）、
        取消 connection 登记。

        用于：Settings Panel 的 Rename / Delete 操作。
        """
        ...

    def list_sources(self) -> dict[str, "MCPConnection"]:
        """返回当前所有已注册的 MCP 连接。

        Returns:
            ``{source_name: MCPConnection}`` 的浅拷贝。
            无已注册连接时返回空 dict。
        """
        ...

    def exec_code(
        self, code: str, state: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """在沙箱中同步执行 Python 代码。

        Args:
            code: Python 源代码。
            state: 跨步骤共享的 REPL 变量。None 时每次执行独立。

        Returns:
            ``{"result": Any, "stdout": str, "stderr": str}`` 或
            ``{"error": str, "traceback": str}``。
        """
        ...

    def format_result(self, result: dict[str, Any]) -> tuple[str, bool]:
        """把 exec_code 的返回拍成文本。返回 (text, is_error)。"""
        ...

    async def close(self) -> None:
        """批量调用所有 on_remove 回调，清空 registry。

        供调用方在 shutdown 时使用，确保 MCP 子进程 / HTTP client 被关闭。
        多次调用幂等。
        """
        ...

    def bind_main_loop(self) -> None:
        """注入当前 event loop，供 async NamespaceTools 跨线程投递。

        必须在主 loop 线程调用，重复调用幂等。
        """
        ...


from . import _env_impl as _env_impl  # noqa: E402,F401
