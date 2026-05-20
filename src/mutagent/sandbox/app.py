"""mutagent.sandbox.app -- SandboxApp Declaration。

纯 sync 的 namespace registry + 受限代码执行引擎。
不读 config，不管理 MCP 连接，不创建 event loop。
namespace 由外部通过 ``add_namespace()`` 注入，
MCP/CLI 生命周期由调用方（``connect_sources`` / 应用层）管理。
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Iterator, TYPE_CHECKING

import mutagent

if TYPE_CHECKING:
    from mutagent.sandbox._namespace import (
        MergedNamespaceView,
        Namespace,
    )


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

    def bind_main_loop(self) -> None:
        """注入当前 event loop，供 async NamespaceTools 跨线程投递。

        必须在主 loop 线程调用，重复调用幂等。
        """
        ...

    def register_mcp_connection(self, name: str, conn: Any) -> None:
        """登记一个 MCPConnection 供 panel 反查。

        Args:
            name: 原始 source 名（config dict key，未 sanitize）
            conn: ``MCPConnection`` 实例

        启动期 ``connect_sources`` 创建每个 conn 后调用一次。同名重复调用
        会覆盖（rename 场景）。panel 通过 ``mcp_connections()`` 读取。

        独立于 ``add_namespace`` 是为了覆盖 ``autostart=false``、namespace 暂不
        入 registry 但仍需 panel 能看到的场景（设计决策 D3）。
        """
        ...

    def unregister_mcp_connection(self, name: str) -> None:
        """从 conn 登记表中移除（panel 删除 / rename 后调用）。幂等。"""
        ...

    def mcp_connections(self) -> dict[str, Any]:
        """返回当前已登记的所有 MCPConnection。

        Returns:
            ``{name: MCPConnection}`` 的浅拷贝。key 是原始 source 名。
            从未调用 ``register_mcp_connection`` 时返回空 dict。
        """
        ...

    def iter_namespaces(self) -> Iterator["Namespace | MergedNamespaceView"]:
        """按名排序遍历 sandbox 当前可见的全部 namespace。

        返回的集合包含外部注入（``add_namespace``）与 NamespaceTools
        Declaration 自动发现的合并结果，与 ``exec_code`` / ``help()``
        路径可见集严格一致。

        同名 2+ providers 时返回 :class:`MergedNamespaceView`，单 provider
        时返回 :class:`Namespace`。这两类对象接口等价（同名 property），
        消费者通常无需按类型分支。
        """
        ...

    def get_namespace(
        self, name: str
    ) -> "Namespace | MergedNamespaceView | None":
        """按名获取一个 namespace。

        多 provider 时返回合并视图；不存在时返回 ``None``。
        与 :meth:`iter_namespaces` 来自同一可见集。
        """
        ...


from . import _app_impl as _app_impl  # noqa: E402,F401
