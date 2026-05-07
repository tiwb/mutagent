"""mutagent.sandbox.app -- SandboxApp Declaration。

纯 sync 的 namespace registry + 受限代码执行引擎。
不读 config，不管理 MCP 连接，不创建 event loop。
namespace 由外部通过 ``add_namespace()`` 注入，
MCP/CLI 生命周期由调用方（``connect_sources`` / 应用层）管理。
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, TYPE_CHECKING

import mutagent

if TYPE_CHECKING:
    from mutagent.sandbox._namespace import Namespace


# 类型别名：on_remove 回调可以是 sync 或 async
CleanupCallback = Callable[[], Any]


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


class SandboxApp(mutagent.Declaration):
    """Python 沙箱 — namespace registry + 受限代码执行环境。

    职责边界：
    - **持有**：注入的 Namespace 集合、对应的 on_remove 清理回调、REPL 缓存
    - **不持有**：config、MCP client、event loop 引用

    能力来源（外部注入）：
    - MCP server 桥接（外部 ``bridge_mcp_server`` + ``add_namespace``）
    - CLI 白名单（外部 ``build_cli_namespace`` + ``add_namespace``）
    - NamespaceTools（本进程 Declaration 自动发现，懒加载）
    """

    def add_namespace(
        self,
        ns: Namespace,
        on_remove: CleanupCallback | None = None,
    ) -> None:
        """注入 namespace。

        Args:
            ns: Namespace 实例，由外部构造（``bridge_mcp_server`` /
                ``build_cli_namespace`` / 业务代码）。
            on_remove: 可选清理回调。``remove_namespace`` 或 ``close()``
                时被调用。可为 sync 或 async。

                典型用法：``add_namespace(ns, on_remove=mcp_client.close)``
                让 sandbox 在清理时关掉对应的 MCP 子进程或 HTTP 连接。
        """
        ...

    def remove_namespace(self, name: str) -> None:
        """移除 namespace。

        若注册时提供了 on_remove，调度执行（sync 直接调，
        async 通过当前 event loop 调度，没有 loop 则用 ``asyncio.run``）。
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


from mutagent.sandbox import _app_impl  # noqa: E402
mutagent.register_module_impls(_app_impl)
