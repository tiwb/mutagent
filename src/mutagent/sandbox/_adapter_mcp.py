"""MCP 桥接 — 连接外部 MCP server，自动生成命名空间函数。

支持两种 transport:
- stdio: 通过 subprocess + JSON-RPC over stdin/stdout（`StdioMCPClient`）
- http : 通过 `mutio.mcp.client.MCPClient` 连接 Streamable HTTP server（`HTTPMCPClient`）

两个 client 接口对齐（duck typing）:
``connect()`` / ``list_tools()`` / ``call_tool(name, arguments)`` / ``close()``。

连接管理由 `MCPConnection` 统一负责（懒连接、自动重连、状态跟踪），
对外暴露的 `Namespace` 全程长寿命，启动失败 / 运行时断连都不会丢命名空间。
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
from typing import Any, Optional, Union

import httpx

from mutagent.sandbox._namespace import Namespace
from mutio.mcp.client import MCPClient
from mutio.mcp.protocol import PROTOCOL_VERSION

logger = logging.getLogger(__name__)

# Windows: 抑制子进程弹出控制台窗口
_POPEN_KWARGS: dict[str, Any] = {}
if sys.platform == "win32":
    _POPEN_KWARGS["creationflags"] = subprocess.CREATE_NO_WINDOW


# ---------------------------------------------------------------------------
# 异常体系
# ---------------------------------------------------------------------------

class MCPTransportError(Exception):
    """传输层错误 — 触发重连。

    包括：子进程退出 / pipe broken / EOF / httpx 连接类异常 /
    HTTP session 失效（404/410）。
    """


class MCPToolError(Exception):
    """业务层错误（tool 返回 isError=True） — 直接抛给用户，不重连。"""


# httpx 中应当视为「传输错误」的异常类型
_TRANSPORT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.WriteError,
    httpx.PoolTimeout,
    BrokenPipeError,
    ConnectionResetError,
    EOFError,
)


def _is_transport_error(exc: BaseException) -> bool:
    """判定异常是否属于「传输错误」。

    传输错误意味着连接本身已不可用，应触发重连。
    业务错误（MCPToolError / 普通 ValueError 等）保持原状抛给用户。
    """
    if isinstance(exc, MCPToolError):
        return False
    if isinstance(exc, MCPTransportError):
        return True
    if isinstance(exc, _TRANSPORT_EXCEPTIONS):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        # 404 / 410：HTTP MCP session 失效（server 重启，旧 Mcp-Session-Id 作废）
        if exc.response.status_code in (404, 410):
            return True
    if isinstance(exc, RuntimeError) and "closed unexpectedly" in str(exc):
        return True
    return False


# ---------------------------------------------------------------------------
# 内容提取
# ---------------------------------------------------------------------------

def _extract_content(result: dict[str, Any]) -> Any:
    """从 MCP tool 调用结果中提取内容。

    - isError=True 抛 :class:`MCPToolError`（业务错误，不触发重连）
    - 单个 text: 尝试 JSON 解析，失败返回原字符串
    - 多个 text: 换行拼接
    - 没有 text: 返回 raw content 列表
    """
    content = result.get("content", [])
    if result.get("isError"):
        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
        raise MCPToolError('\n'.join(texts) if texts else "MCP tool call failed")

    texts = [c.get("text", "") for c in content if c.get("type") == "text"]
    if len(texts) == 1:
        try:
            return json.loads(texts[0])
        except (json.JSONDecodeError, ValueError):
            return texts[0]
    elif texts:
        return '\n'.join(texts)
    return content


# ---------------------------------------------------------------------------
# Stdio client
# ---------------------------------------------------------------------------

class StdioMCPClient:
    """Stdio MCP client — 通过 subprocess 连接 MCP server。"""

    def __init__(self, command: str, args: list[str] | None = None,
                 shell: bool = False, env: dict[str, str] | None = None):
        self._command = command
        self._args = args or []
        self._shell = shell
        # env 语义：None / {} → 直接继承父进程（Popen env=None）；
        # 非空 → 用 ``os.environ | env`` 合并下发，避免完全覆盖丢 PATH 等系统变量
        self._env = dict(env) if env else None
        self._process: subprocess.Popen | None = None
        self._request_id = 0

    def _merged_env(self) -> dict[str, str] | None:
        if not self._env:
            return None
        return {**os.environ, **self._env}

    async def connect(self) -> dict[str, Any]:
        """启动 MCP server 子进程并完成 initialize 握手。"""
        merged_env = self._merged_env()
        try:
            if self._shell:
                cmd = self._command + ' ' + ' '.join(self._args)
                self._process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    shell=True,
                    env=merged_env,
                    **_POPEN_KWARGS,
                )
            else:
                self._process = subprocess.Popen(
                    [self._command] + self._args,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=merged_env,
                    **_POPEN_KWARGS,
                )
        except (OSError, FileNotFoundError) as exc:
            # 启动失败按传输错误处理（命令找不到 / 权限不足等）
            raise MCPTransportError(
                f"failed to start MCP subprocess: {exc}") from exc

        # MCP initialize 握手 — 协议版本来自 mutio.mcp.protocol，避免硬编码漂移
        result = await self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "mutagent-sandbox", "version": "0.1.0"},
        })
        # 发送 initialized 通知
        self._send_notification("notifications/initialized", {})
        return result

    async def list_tools(self) -> list[dict[str, Any]]:
        """获取 server 的 tool 列表。"""
        result = await self._request("tools/list", {})
        return result.get("tools", [])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """调用 tool 并返回结果。"""
        result = await self._request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        return _extract_content(result)

    async def close(self) -> None:
        """关闭连接。"""
        if self._process:
            if self._process.stdin is not None:
                try:
                    self._process.stdin.close()
                except Exception:
                    pass
            try:
                self._process.terminate()
            except Exception:
                pass
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

    async def _request(self, method: str, params: dict) -> dict:
        """发送 JSON-RPC 请求并等待响应。"""
        self._request_id += 1
        msg = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        line = json.dumps(msg) + '\n'

        loop = asyncio.get_event_loop()
        # 在线程中执行阻塞 IO
        response = await loop.run_in_executor(None, self._send_and_receive, line)
        return response

    def _send_and_receive(self, line: str) -> dict:
        """同步发送请求并读取响应。

        进程退出 / pipe 断 / stdout EOF 全部 wrap 为 :class:`MCPTransportError`，
        触发上层 `MCPConnection.reconnect`。
        """
        if not self._process or not self._process.stdin or not self._process.stdout:
            raise MCPTransportError("MCP server process not running")

        try:
            self._process.stdin.write(line)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise MCPTransportError(
                f"MCP stdin write failed: {exc}") from exc

        # 读取响应行（跳过通知等非响应消息）
        while True:
            try:
                resp_line = self._process.stdout.readline()
            except (OSError, ValueError) as exc:
                raise MCPTransportError(
                    f"MCP stdout read failed: {exc}") from exc
            if not resp_line:
                raise MCPTransportError("MCP server closed unexpectedly")
            try:
                resp = json.loads(resp_line)
            except json.JSONDecodeError:
                # 非 JSON 行通常是 server 把 logger 误写到 stdout — 跳过
                continue
            # 跳过通知（没有 id 的消息）
            if "id" in resp:
                if "error" in resp:
                    err = resp["error"]
                    # JSON-RPC 协议层错误：业务错而非传输断 — 仍按非传输处理
                    raise RuntimeError(
                        f"MCP error {err.get('code')}: {err.get('message')}")
                return resp.get("result", {})

    def _send_notification(self, method: str, params: dict) -> None:
        """发送 JSON-RPC 通知（不期望响应）。"""
        msg = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        if self._process and self._process.stdin:
            try:
                self._process.stdin.write(json.dumps(msg) + '\n')
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise MCPTransportError(
                    f"MCP stdin write failed: {exc}") from exc


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

class HTTPMCPClient:
    """HTTP MCP client — 薄包 `mutio.mcp.client.MCPClient`，对齐 StdioMCPClient 接口。

    在 client 边界统一捕获 httpx 传输异常 / session 失效，
    wrap 为 :class:`MCPTransportError` 让上层判定重连。
    """

    def __init__(self, url: str, timeout: float = 30.0):
        self._mcp = MCPClient(url=url, timeout=timeout)

    async def connect(self) -> dict[str, Any]:
        """连接并完成 initialize 握手。"""
        try:
            await self._mcp.connect()
        except Exception as exc:
            if _is_transport_error(exc):
                raise MCPTransportError(f"MCP connect failed: {exc}") from exc
            raise
        return {
            "serverInfo": self._mcp.server_info,
            "capabilities": self._mcp.server_capabilities,
            "instructions": getattr(self._mcp, "server_instructions", ""),
        }

    async def list_tools(self) -> list[dict[str, Any]]:
        """获取 server 的 tool 列表。"""
        try:
            return await self._mcp.list_tools()
        except Exception as exc:
            if _is_transport_error(exc):
                raise MCPTransportError(f"list_tools failed: {exc}") from exc
            raise

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """调用 tool 并返回结果。"""
        try:
            result = await self._mcp.call_tool(name, **arguments)
        except MCPToolError:
            raise
        except Exception as exc:
            if _is_transport_error(exc):
                raise MCPTransportError(f"call_tool failed: {exc}") from exc
            raise
        return _extract_content(result)

    async def close(self) -> None:
        """关闭连接。"""
        try:
            await self._mcp.close()
        except Exception as exc:  # 关闭路径上的传输错可忽略
            logger.debug("HTTPMCPClient.close swallowed: %s", exc)


AnyMCPClient = Union[StdioMCPClient, HTTPMCPClient]


# ---------------------------------------------------------------------------
# Client 工厂
# ---------------------------------------------------------------------------

def make_client(ns_name: str, server_config: dict[str, Any]) -> AnyMCPClient:
    """根据 config 构造对应 transport 的 client（不连接）。

    Args:
        ns_name: 命名空间名（仅用于错误信息）
        server_config: server 配置
            - stdio（默认）: ``command``（必需）, ``args``, ``shell``
            - http: ``url``（必需）, ``timeout``

    Raises:
        ValueError: 配置缺失或 transport 未知
    """
    transport = server_config.get("transport", "stdio")
    if transport == "stdio":
        command = server_config.get("command", "")
        if not command:
            raise ValueError(
                f"MCP source '{ns_name}': stdio transport requires 'command'")
        return StdioMCPClient(
            command,
            server_config.get("args", []),
            shell=server_config.get("shell", False),
            env=server_config.get("env"),
        )
    if transport == "http":
        url = server_config.get("url", "")
        if not url:
            raise ValueError(
                f"MCP source '{ns_name}': http transport requires 'url'")
        return HTTPMCPClient(
            url,
            timeout=server_config.get("timeout", 30.0),
        )
    raise ValueError(
        f"MCP source '{ns_name}': unknown transport {transport!r}")


# ---------------------------------------------------------------------------
# MCPConnection — 长生命周期连接代理
# ---------------------------------------------------------------------------

# 有效状态：
#   "disconnected" — 从未连过 / 已主动断开
#   "connecting"   — 正在连接（reconnect 进行中）
#   "connected"    — 当前可用
#   "failed"       — 上次连接失败，处于冷却期 / 等下次触发
ConnectionState = str  # Literal["disconnected", "connecting", "connected", "failed"]


def _sanitize_ns_name(name: str) -> str:
    """将 MCP namespace 名转换为合法 Python 标识符。

    规则：
    - 字母、数字、下划线保留
    - 其他字符替换为 ``_``
    - 连续 ``_`` 折叠为一个
    - 首尾 ``_`` 去掉
    - 以数字开头时前补 ``_``
    - 全特殊字符映射后为空时返回 ``_``

    >>> _sanitize_ns_name("My MCP")
    'My_MCP'
    >>> _sanitize_ns_name("my-srv")
    'my_srv'
    >>> _sanitize_ns_name("!@#$")
    '_'
    """
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_')
    if not sanitized:
        return '_'
    if sanitized[0].isdigit():
        sanitized = '_' + sanitized
    return sanitized


class MCPConnection:
    """一个 MCP source 的长生命周期代理。

    职责：
    - 统一持有 namespace（无论连接成功与否，namespace 始终存在）
    - 管理 client 的生命周期（创建 / 重建 / close）
    - 状态机 + cooldown + 并发锁
    - 根据连接结果增删 namespace 上的函数
    """

    def __init__(self, ns_name: str, server_config: dict[str, Any],
                 main_loop: asyncio.AbstractEventLoop,
                 retry_cooldown: float = 5.0):
        self.ns_name = ns_name  # 原始名，用于日志
        self.config = server_config
        self.main_loop = main_loop
        self.retry_cooldown = max(0.0, float(retry_cooldown))

        self.client: Optional[AnyMCPClient] = None
        self.state: ConnectionState = "disconnected"
        self.last_error: Optional[str] = None
        self.last_attempt_at: Optional[float] = None
        self._lock = asyncio.Lock()

        # 始终存在的 namespace；失败 / 未连状态下函数表为空
        # namespace 名用 sanitized 版本，确保可作为 Python 标识符访问
        safe_name = _sanitize_ns_name(ns_name)
        # provider_kind="tool"：本 conn 主 namespace 由 MCP tools 列表驱动
        self.namespace = Namespace(safe_name, description="",
                                   provider_kind="tool")
        self.namespace._connection = self  # type: ignore[attr-defined]
        self.namespace.connection_state = self.state  # type: ignore[attr-defined]
        self.namespace.connection_error = None  # type: ignore[attr-defined]

        # 通过 pysandbox/namespaces.* 扩展协议从对端融合进来的 peer
        # namespaces。每次 _do_rebuild 重建；与 self.namespace 一样
        # 共享本 conn 的连接状态。详见
        # ``mutagent/docs/specifications/feature-pysandbox-namespace-sharing.md``。
        self.peer_namespaces: list[Namespace] = []

        # SandboxApp 回引：由调用方（connect_sources / pysandbox._build_sandbox）
        # 在 ``sandbox.add_namespace(conn.namespace)`` 之后立即赋值。
        # 用于 _do_rebuild 中把 peer namespaces 同步注册到 sandbox registry，
        # 以及 close 时摘除。允许为 None：单元测试或裸 conn 自测场景下
        # 不挂 sandbox，peer 同步逻辑自动 no-op。详见
        # ``mutagent/docs/specifications/feature-namespace-multi-provider.md``。
        self._sandbox: Any | None = None

    # -- 状态变更 helper（保证 namespace 状态字段同步）-------------------

    def _set_state(self, state: ConnectionState,
                   error: str | None = None) -> None:
        self.state = state
        self.last_error = error
        self.namespace.connection_state = state  # type: ignore[attr-defined]
        self.namespace.connection_error = error  # type: ignore[attr-defined]
        # peer namespaces 共享同一连接状态（D6）
        for peer in self.peer_namespaces:
            peer.connection_state = state  # type: ignore[attr-defined]
            peer.connection_error = error  # type: ignore[attr-defined]

    def mark_disconnected(self, reason: str) -> None:
        """tool 调用发现传输错时，标记当前 client 已不可用。

        不主动 close（reconnect 会处理旧 client），只翻状态。
        """
        if self.state == "connected":
            self._set_state("failed", reason)
            logger.info("MCP '%s' marked disconnected: %s",
                        self.ns_name, reason)

    # -- 公开接口 ---------------------------------------------------------

    async def ensure_connected(self) -> None:
        """幂等：保证 self.client 可用。

        - 已 connected：立即返回
        - failed 且在冷却期内：抛上次的错
        - 其他状态：在 lock 下重检，避免并发重建
        """
        if self.state == "connected" and self.client is not None:
            return
        if self.state == "failed" and self._in_cooldown():
            raise MCPTransportError(
                f"MCP '{self.ns_name}' in cooldown after failure: "
                f"{self.last_error}")
        async with self._lock:
            # 拿锁后重检：别人可能已经重连完
            if self.state == "connected" and self.client is not None:
                return
            if self.state == "failed" and self._in_cooldown():
                raise MCPTransportError(
                    f"MCP '{self.ns_name}' in cooldown after failure: "
                    f"{self.last_error}")
            await self._do_rebuild()

    async def reconnect(self) -> None:
        """显式重连 — 始终全量重建（不看当前状态）。

        用于 tool 调用传输错后的强制重建，或者用户主动调用刷新 tool 列表。
        失败时抛 :class:`MCPTransportError`，并把 state 置为 "failed"。
        """
        async with self._lock:
            await self._do_rebuild()

    async def _do_rebuild(self) -> None:
        """实际重建逻辑 — 调用者需持有 self._lock。

        D11：入口设 connecting 后，**所有出口要么 connected 要么 failed**。
        peer 构建 / 冲突检测等任何后置逻辑都必须在统一 try 范围内，
        否则 state 卡 connecting + cooldown 失效，autostart 会静默吞错。
        """
        self._set_state("connecting", None)
        self.last_attempt_at = time.time()

        # 关闭旧 client（如果有）— 失败不影响重建流程
        old_client = self.client
        self.client = None
        if old_client is not None:
            try:
                await old_client.close()
            except Exception as exc:
                logger.debug("MCP '%s' old client close failed: %s",
                             self.ns_name, exc)

        # 配置错误（make_client 抛 ValueError）走单独分支：
        # 不 wrap 为 MCPTransportError，但依然记为 failed 状态供 help 展示
        try:
            new_client = make_client(self.ns_name, self.config)
        except ValueError as exc:
            self._set_state("failed", str(exc))
            self.last_attempt_at = time.time()
            raise

        # ------------------------------------------------------------
        # 主重建段：connect / list_tools / refresh_namespace /
        # build_peer_namespaces / _check_peer_name_conflicts / set connected
        # 任意一步异常 → failed（D11）
        # ------------------------------------------------------------
        try:
            init_result = await new_client.connect()
            tools = await new_client.list_tools()

            # 检测 pysandbox capability（D3）— 决定是否过滤对端 pysandbox tool
            # 自身（D2）以及是否融合 peer namespaces（D4 Eager 拉取）
            from mutagent.sandbox._adapter_pysandbox import (
                build_peer_namespaces,
                has_pysandbox_capability,
            )
            is_peer = (
                isinstance(new_client, HTTPMCPClient)
                and has_pysandbox_capability(init_result)
            )
            if is_peer:
                # D2: 隐藏对端 pysandbox tool 自身，避免递归调用语义混乱
                tools = [t for t in tools if t.get("name") != "pysandbox"]

            self.client = new_client
            self._refresh_namespace(init_result, tools)

            # 融合 peer namespaces（multi-provider 模型下不再做 namespace-级
            # 全局冲突检测；只查同 conn 内 peer 互撞 — 见 _check_peer_name_conflicts）
            new_peer_namespaces: list[Namespace] = []
            if is_peer:
                assert isinstance(new_client, HTTPMCPClient)
                new_peer_namespaces = await build_peer_namespaces(
                    self, init_result, new_client)
                self._check_peer_name_conflicts(new_peer_namespaces)
            # multi-provider 同步：把 new 注册到 sandbox，把 old 中不在 new
            # 的从 sandbox registry 摘掉。事务式更新，按实例 id 区分。
            self._sync_peer_providers(self.peer_namespaces,
                                      new_peer_namespaces)
            self.peer_namespaces = new_peer_namespaces

            self._set_state("connected", None)
            if new_peer_namespaces:
                logger.info(
                    "MCP '%s' connected (%d functions, merged %d namespaces from %s)",
                    self.ns_name, len(self.namespace._functions),
                    len(new_peer_namespaces), self.ns_name)
            else:
                logger.info("MCP '%s' connected (%d functions)",
                            self.ns_name, len(self.namespace._functions))
        except MCPTransportError as exc:
            reason = str(exc) or exc.__class__.__name__
            self._set_state("failed", reason)
            self.last_attempt_at = time.time()
            logger.warning("MCP '%s' rebuild failed (transport): %s",
                           self.ns_name, reason)
            # 清空 client，避免 failed 状态下还残留旧引用
            self.client = None
            raise
        except Exception as exc:
            # D11 兜底：peer 构建 / 冲突检测 / 任何编程错都进 failed，
            # 不允许 state 留在 connecting
            reason = str(exc) or exc.__class__.__name__
            self._set_state("failed", reason)
            self.last_attempt_at = time.time()
            logger.warning("MCP '%s' rebuild failed: %s",
                           self.ns_name, reason)
            # 清空 client / peer 列表，避免 failed 状态残留。
            # 把已注册到 sandbox 的旧 peer providers 全部摘掉，与 D11
            # 「出口要么 connected 要么 failed，状态绝对一致」对齐。
            self.client = None
            self._sync_peer_providers(self.peer_namespaces, [])
            self.peer_namespaces = []
            # 包成 MCPTransportError 让上层 cooldown 生效
            raise MCPTransportError(
                f"MCP '{self.ns_name}' rebuild failed: {reason}"
            ) from exc

    async def close(self) -> None:
        """彻底关闭 — sandbox cleanup 入口。多次调用幂等。"""
        async with self._lock:
            # 摘掉本 conn 注册到 sandbox 的全部 peer providers。
            # conn.namespace 自己由 SandboxApp 用 on_remove → conn.close
            # 持有，不在这里清（会循环）。
            self._sync_peer_providers(self.peer_namespaces, [])
            self.peer_namespaces = []

            client = self.client
            self.client = None
            self._set_state("disconnected", None)
            if client is not None:
                try:
                    await client.close()
                except Exception as exc:
                    logger.debug("MCP '%s' close failed: %s",
                                 self.ns_name, exc)

    # -- 外部查询 ----------------------------------------------------

    def list_tools_metadata(self) -> list[dict[str, Any]]:
        """返回当前 conn 可见的所有 tool 元数据。

        覆盖本 conn 的主 namespace 与所有 peer namespaces。未连接 / 连失败
        时返回空列表（不报错，panel 以状态 tag 交代）。

        返回元素形式::

            {
                "name":             str,    # tool 名
                "description":      str,
                "input_schema":     dict,   # 原始 JSON Schema（properties / required 等）
                "source_namespace": str,    # 隔属哪个 namespace（主 ns 或 peer ns 名）
            }
        """
        result: list[dict[str, Any]] = []
        seen: list[Namespace] = [self.namespace, *self.peer_namespaces]
        for ns in seen:
            for fn_name, fn in ns._functions.items():
                schema = getattr(fn, '_mcp_input_schema', None) or {}
                desc = (
                    getattr(fn, '_mcp_description', None)
                    or ns._descriptions.get(fn_name, '')
                    or ''
                )
                result.append({
                    "name": fn_name,
                    "description": desc,
                    "input_schema": schema,
                    "source_namespace": ns.name,
                })
        return result

    # -- 内部 helper -----------------------------------------------------

    def _sync_peer_providers(
        self,
        old_peers: list[Namespace],
        new_peers: list[Namespace],
    ) -> None:
        """把 peer namespaces 的注册状态同步到 SandboxApp registry。

        按实例 id 做 diff：

        - 在 ``old_peers`` 但不在 ``new_peers`` 的 → 从 sandbox 摘除
        - 在 ``new_peers`` 但不在 ``old_peers`` 的 → 注册到 sandbox

        ``self._sandbox`` 为 None（未挂 sandbox 的纯 conn / 单元测试）时
        no-op。peer 注册时不传 ``on_remove``——peer 是 conn 的从属，移除
        时只需从 registry 摘掉，不应反向触发 ``conn.close``（会循环）。
        """
        sandbox = self._sandbox
        if sandbox is None:
            return
        new_ids = {id(p) for p in new_peers}
        old_ids = {id(p) for p in old_peers}
        for old in old_peers:
            if id(old) not in new_ids:
                sandbox.remove_provider(old)
        for new in new_peers:
            if id(new) not in old_ids:
                sandbox.add_namespace(new)

    def _check_peer_name_conflicts(
        self, peer_namespaces: list[Namespace]) -> None:
        """D1 (multi-provider 重写)：只检查 peer 之间是否重名。

        旧逻辑还检查 peer vs 本 conn 的 tool ns 同名 → 阻塞注册。
        新模型下「source 名 = peer ns 名」是常态，撞名由 SandboxApp 走
        :class:`MergedNamespaceView` 在调用/help 级处理，启动期不阻塞。

        但同一 server 自我 export 两个同名 peer namespace 必然是 server bug，
        仍然抛 RuntimeError。
        """
        seen: set[str] = set()
        for peer in peer_namespaces:
            if peer.name in seen:
                raise RuntimeError(
                    f"Pysandbox peer-namespace duplicate on source '{self.ns_name}': "
                    f"server exported namespace '{peer.name}' more than once"
                )
            seen.add(peer.name)

    def _in_cooldown(self) -> bool:
        if self.retry_cooldown <= 0:
            return False
        if self.last_attempt_at is None:
            return False
        return (time.time() - self.last_attempt_at) < self.retry_cooldown

    def _refresh_namespace(self, init_result: dict[str, Any],
                           tools: list[dict[str, Any]]) -> None:
        """根据最新握手 / tool 列表，刷新 namespace 的描述与函数表。

        - 删除当前不存在的旧 tool
        - 注册 / 覆盖最新 tool 的 wrapper
        """
        ns = self.namespace
        # 描述：优先 instructions，退化 serverInfo.title
        ns_desc = (
            (init_result.get("instructions") or "").strip()
            or (init_result.get("serverInfo") or {}).get("title", "")
            or ""
        )
        ns._description = ns_desc

        new_names = {t["name"] for t in tools}
        # 删除消失的 tool
        for old in list(ns._functions.keys()):
            if old not in new_names:
                ns._functions.pop(old, None)
                ns._descriptions.pop(old, None)

        # 注册 / 覆盖
        for tool in tools:
            tool_name = tool["name"]
            tool_desc = tool.get("description", "")
            input_schema = tool.get("inputSchema", {})
            fn = _make_tool_func(self, tool_name, tool_desc, input_schema)
            ns.register(tool_name, fn, tool_desc)

        # 函数表已变更，通知所属 view 失效缓存。
        # MergedNamespaceView._resolved_cache_key = tuple(id(p) for p in providers)，
        # 只在 providers 列表变化时失效；本函数直改 ns._functions（id 不变），
        # 导致 view.displayed / primary / _description 拿到旧结果。
        sandbox = getattr(self, "_sandbox", None)
        if sandbox is not None:
            registry = getattr(sandbox, "_registry", None)
            if registry is not None:
                view = registry._views.get(ns.name)
                if view is not None:
                    view.invalidate()


# ---------------------------------------------------------------------------
# tool wrapper
# ---------------------------------------------------------------------------

def _make_tool_func(conn: MCPConnection, tool_name: str,
                    description: str,
                    input_schema: dict) -> Any:
    """为一个 MCP tool 生成 Python 函数。

    闭包持有 :class:`MCPConnection` 而非裸 client：
    - 调用时先 ``ensure_connected``（懒触发）
    - 传输错时 ``mark_disconnected`` + ``reconnect`` 后重试一次
    - 第二次仍失败抛 :class:`MCPTransportError` 给用户

    所有 IO 走 ``run_coroutine_threadsafe`` 回到 setup 时捕获的 main_loop
    （httpx.AsyncClient 资源绑定约束）。

    签名层：根据 ``input_schema`` 构造真签名并挂到 ``__signature__``（见
    refactor-wrapper-faithful-signature.md）；wrapper 内用 ``sig.bind`` 把
    位置调用规范化为 kwargs 后再走 RPC。构造失败时回落为
    ``(**kwargs)`` wrapper。

    docstring 层：按 feature-mcp-schema-help-display.iter2.md 渲染为
    「头部 description + Args 段 + Annotations 段」三段式。Args 段仅
    写 ``name: description.``，所有约束字段原词进入 ``Annotations:`` 段
    以 JSON 透传。
    """
    from mutagent.sandbox._signature import format_annotations_section

    properties = input_schema.get("properties", {}) or {}

    sections: list[str] = []
    if description:
        sections.append(description)

    if isinstance(properties, dict) and properties:
        args_lines = ["Args:"]
        for pname, pinfo in properties.items():
            pdesc = (
                pinfo.get("description", "")
                if isinstance(pinfo, dict) else ""
            )
            if pdesc:
                args_lines.append(f"    {pname}: {pdesc}")
            else:
                # 无 description 仍保留 "name:"，与有描述行视觉一致
                args_lines.append(f"    {pname}:")
        sections.append("\n".join(args_lines))

        annotations_section = format_annotations_section(properties)
        if annotations_section:
            sections.append(annotations_section)

    doc = "\n\n".join(sections)

    async def call_with_retry(kwargs: dict[str, Any]) -> Any:
        from mutagent.sandbox._signature import _MISSING

        kwargs = {k: v for k, v in kwargs.items() if v is not _MISSING}
        await conn.ensure_connected()
        assert conn.client is not None  # ensure_connected 成功后必非 None
        try:
            return await conn.client.call_tool(tool_name, kwargs)
        except Exception as exc:
            if not _is_transport_error(exc):
                raise
            # 传输错 → 重连一次，再试
            conn.mark_disconnected(str(exc) or exc.__class__.__name__)
            await conn.reconnect()
            assert conn.client is not None
            return await conn.client.call_tool(tool_name, kwargs)

    # 先构真签名（失败回落 → 旧形态 ``(**kwargs)``）
    from mutagent.sandbox._signature import (
        mcp_schema_to_specs,
        try_build_signature,
    )
    sig = try_build_signature(
        mcp_schema_to_specs(input_schema),
        context=f"MCP tool {tool_name!r}")

    # _async_original: 供 share.py:_handle_call 等在事件循环线程上的
    # 调用方直接 await，避免 sync wrapper 的 run_coroutine_threadsafe
    # + future.result() 同线程死锁（与 _wrap_async 的 _async_original 模式一致）。
    async def _tool_async(**kwargs: Any) -> Any:
        return await call_with_retry(kwargs)

    if sig is not None:
        _bind_sig = sig

        def tool_func(*args: Any, **kwargs: Any) -> Any:  # type: ignore[reportRedeclaration]
            bound = _bind_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            future = asyncio.run_coroutine_threadsafe(
                call_with_retry(dict(bound.arguments)), conn.main_loop)
            return future.result(timeout=120)

        tool_func.__signature__ = sig  # type: ignore[attr-defined]
    else:
        def tool_func(**kwargs: Any) -> Any:  # type: ignore[misc, reportRedeclaration]
            future = asyncio.run_coroutine_threadsafe(
                call_with_retry(kwargs), conn.main_loop)
            return future.result(timeout=120)

    tool_func.__name__ = tool_name
    tool_func.__doc__ = doc
    tool_func._async_original = _tool_async  # type: ignore[attr-defined]
    # 保留 input_schema 供 panel / list_tools_metadata 展示使用
    tool_func._mcp_input_schema = input_schema  # type: ignore[attr-defined]
    tool_func._mcp_description = description  # type: ignore[attr-defined]
    return tool_func


# ---------------------------------------------------------------------------
# 兼容入口 — 旧 API
# ---------------------------------------------------------------------------

async def bridge_mcp_server(ns_name: str,
                            server_config: dict[str, Any]
                            ) -> tuple[Namespace, AnyMCPClient]:
    """[Legacy] 同步桥接 — 一次性创建 connection、立即连、返回 (namespace, client)。

    新代码请直接构造 :class:`MCPConnection`。该入口保留是为了避免破坏
    旧调用方（pysandbox CLI 等），其语义改为：内部仍走 MCPConnection，
    但启动期失败保持原行为（向上抛异常）。
    """
    main_loop = asyncio.get_running_loop()
    conn = MCPConnection(ns_name, server_config, main_loop)
    await conn.reconnect()
    assert conn.client is not None
    return conn.namespace, conn.client
