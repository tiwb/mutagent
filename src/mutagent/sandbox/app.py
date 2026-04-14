"""SandboxApp — sandbox 核心 API。

管理执行引擎、命名空间注册表和外部能力源连接。
上层消费者（skill daemon / mutbot）通过此类使用 sandbox 能力。
"""

import sys
import time
from typing import Any

from mutagent.sandbox.engine import execute
from mutagent.sandbox.namespace import Namespace, NamespaceRegistry
from mutagent.sandbox.adapters.mcp_bridge import bridge_mcp_server, StdioMCPClient
from mutagent.sandbox.adapters.cli import build_cli_namespace


class SandboxApp:
    """Sandbox 应用，管理执行引擎、命名空间和能力源连接。

    Args:
        mcp_servers: MCP 桥接配置 {ns_name: {"command": ..., "args": [...]}}
        cli_commands: CLI 白名单配置 {func_name: {"command": ..., "args": [...]}}
    """

    def __init__(self, mcp_servers: dict[str, dict[str, Any]] | None = None,
                 cli_commands: dict[str, dict[str, Any]] | None = None):
        self._mcp_config = mcp_servers or {}
        self._cli_config = cli_commands or {}
        self._registry = NamespaceRegistry()
        self._mcp_clients: dict[str, StdioMCPClient] = {}
        self._state: dict[str, Any] = {}
        self._start_time = time.time()

    async def setup(self) -> None:
        """初始化所有能力源。"""
        for ns_name, server_config in self._mcp_config.items():
            await self._connect_mcp(ns_name, server_config)

        if self._cli_config:
            cli_ns = build_cli_namespace(self._cli_config)
            self._registry.add(cli_ns)

    async def _connect_mcp(self, ns_name: str,
                           server_config: dict[str, Any]) -> None:
        """连接单个 MCP server。"""
        command = server_config.get("command", "")
        args = server_config.get("args", [])
        shell = server_config.get("shell", False)
        try:
            ns, client = await bridge_mcp_server(
                ns_name, command, args, shell=shell)
            self._registry.add(ns)
            self._mcp_clients[ns_name] = client
        except Exception as e:
            print(f"Warning: Failed to connect MCP '{ns_name}': {e}",
                  file=sys.stderr)

    async def shutdown(self) -> None:
        """关闭所有连接。"""
        for client in self._mcp_clients.values():
            try:
                await client.close()
            except Exception:
                pass
        self._mcp_clients.clear()

    async def reload(self, mcp_servers: dict[str, dict[str, Any]] | None = None,
                     cli_commands: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
        """重新加载配置并重连所有能力源。保留 REPL 状态。"""
        await self.shutdown()
        self._registry = NamespaceRegistry()
        if mcp_servers is not None:
            self._mcp_config = mcp_servers
        if cli_commands is not None:
            self._cli_config = cli_commands
        await self.setup()
        ns_count = len(self._registry._namespaces)
        return {"namespaces": ns_count}

    async def reconnect_mcp(self, ns_name: str,
                            server_config: dict[str, Any] | None = None) -> dict[str, Any]:
        """重连单个 MCP namespace。"""
        config = server_config or self._mcp_config.get(ns_name)
        if not config:
            return {"error": f"'{ns_name}' not in config"}

        # 关闭旧连接
        old_client = self._mcp_clients.pop(ns_name, None)
        if old_client:
            try:
                await old_client.close()
            except Exception:
                pass
        self._registry.remove(ns_name)

        # 重连
        await self._connect_mcp(ns_name, config)
        ns = self._registry.get(ns_name)
        func_count = len(ns._functions) if ns else 0
        return {"namespace": ns_name, "functions": func_count}

    def exec_code(self, code: str) -> dict[str, Any]:
        """执行 Python 代码。"""
        namespace = self._registry.build_namespace_dict()
        return execute(code, namespace, self._state)

    def register_namespace(self, ns: Namespace) -> None:
        """手动注册一个命名空间。"""
        self._registry.add(ns)

    @property
    def registry(self) -> NamespaceRegistry:
        return self._registry

    @property
    def uptime(self) -> float:
        return time.time() - self._start_time
