"""mutagent.sandbox.app -- SandboxApp Declaration。

管理执行引擎、命名空间和外部能力源连接。
上层消费者（skill daemon / mutbot）通过此类使用 sandbox 能力。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mutagent

if TYPE_CHECKING:
    from mutagent.config import Config


class SandboxApp(mutagent.Declaration):
    """Python 沙箱 — 聚合能力源，提供受限代码执行环境。

    能力源包括：
    - MCP server 桥接（config ``mcp_sources``）
    - CLI 白名单（config ``cli_sources``）
    - NamespaceTools（本进程 Declaration 自动发现）

    Attributes:
        config: 配置容器，读取 ``mcp_sources`` / ``cli_sources``。
    """

    config: Config

    async def setup(self) -> None:
        """根据 self.config 初始化能力源连接（MCP/CLI）。"""
        ...

    def exec_code(self, code: str, state: dict[str, Any] | None = None) -> dict[str, Any]:
        """在沙箱中执行 Python 代码。

        Args:
            code: Python 源代码。
            state: 跨步骤共享的 REPL 变量。None 时每次执行独立。

        Returns:
            {"result": Any, "stdout": str} 或 {"error": str, "traceback": str}。
        """
        ...

    async def reload(self) -> dict[str, Any]:
        """从 self.config 重载，重连所有能力源。"""
        ...

    async def shutdown(self) -> None:
        """关闭所有连接。"""
        ...


from mutagent.sandbox import _app_impl
mutagent.register_module_impls(_app_impl)
