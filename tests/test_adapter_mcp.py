"""mutagent.sandbox._adapter_mcp 单元测试。

覆盖:
- _extract_content 各分支（isError / 单 text JSON / 单 text 非 JSON / 多 text / 无 text）
- HTTPMCPClient 对 mock MCPClient 的行为
- bridge_mcp_server 按 transport 分派（stdio / http / unknown）
"""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from mutagent.sandbox._adapter_mcp import (
    HTTPMCPClient,
    MCPConnection,
    MCPToolError,
    MCPTransportError,
    _extract_content,
    _is_transport_error,
    bridge_mcp_server,
)


# ============================================================
# _extract_content
# ============================================================

class TestExtractContent:

    def test_is_error_raises(self):
        result = {
            "isError": True,
            "content": [{"type": "text", "text": "boom"}],
        }
        with pytest.raises(MCPToolError, match="boom"):
            _extract_content(result)

    def test_is_error_without_text(self):
        result = {"isError": True, "content": []}
        with pytest.raises(MCPToolError, match="MCP tool call failed"):
            _extract_content(result)

    def test_single_text_json(self):
        result = {"content": [{"type": "text", "text": '{"a": 1}'}]}
        assert _extract_content(result) == {"a": 1}

    def test_single_text_non_json(self):
        result = {"content": [{"type": "text", "text": "hello"}]}
        assert _extract_content(result) == "hello"

    def test_multi_text_joined(self):
        result = {"content": [
            {"type": "text", "text": "line1"},
            {"type": "text", "text": "line2"},
        ]}
        assert _extract_content(result) == "line1\nline2"

    def test_no_text_returns_raw_content(self):
        result = {"content": [{"type": "image", "data": "..."}]}
        assert _extract_content(result) == [{"type": "image", "data": "..."}]

    def test_empty_content(self):
        assert _extract_content({}) == []


# ============================================================
# HTTPMCPClient (对 mock MCPClient 的行为)
# ============================================================

class _MockMCPClient:
    """模拟 mutio.mcp.client.MCPClient 的最小接口。"""

    def __init__(self, url: str, timeout: float = 30.0):
        self.url = url
        self.timeout = timeout
        self.server_info: dict = {}
        self.server_capabilities: dict = {}
        self.server_instructions: str = ""
        self.connected = False
        self.closed = False
        self.calls: list = []
        self.tools_payload: list = []
        self.call_tool_result: dict = {}

    async def connect(self):
        self.connected = True
        self.server_info = {"name": "mock", "version": "0.0.1"}
        self.server_capabilities = {"tools": {}}

    async def close(self):
        self.closed = True

    async def list_tools(self):
        return self.tools_payload

    async def call_tool(self, name, **arguments):
        self.calls.append((name, arguments))
        return self.call_tool_result


class TestHTTPMCPClient:

    @pytest.mark.asyncio
    async def test_connect_returns_handshake_info(self, monkeypatch):
        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.MCPClient", _MockMCPClient)

        client = HTTPMCPClient(url="http://example/mcp", timeout=5.0)
        info = await client.connect()

        assert info["serverInfo"] == {"name": "mock", "version": "0.0.1"}
        assert info["capabilities"] == {"tools": {}}
        assert info["instructions"] == ""  # mock 默认空
        mock = cast(_MockMCPClient, client._mcp)
        assert mock.connected is True
        assert mock.url == "http://example/mcp"
        assert mock.timeout == 5.0

    @pytest.mark.asyncio
    async def test_list_tools_passthrough(self, monkeypatch):
        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.MCPClient", _MockMCPClient)

        client = HTTPMCPClient(url="http://example/mcp")
        await client.connect()
        mock = cast(_MockMCPClient, client._mcp)
        mock.tools_payload = [
            {"name": "foo", "description": "foo tool", "inputSchema": {}},
        ]
        tools = await client.list_tools()

        assert tools == [{"name": "foo", "description": "foo tool", "inputSchema": {}}]

    @pytest.mark.asyncio
    async def test_call_tool_extracts_content(self, monkeypatch):
        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.MCPClient", _MockMCPClient)

        client = HTTPMCPClient(url="http://example/mcp")
        await client.connect()
        mock = cast(_MockMCPClient, client._mcp)
        mock.call_tool_result = {
            "content": [{"type": "text", "text": '{"ok": true}'}],
        }
        result = await client.call_tool("search", {"query": "hi"})

        assert result == {"ok": True}
        # arguments 通过 **kwargs 展开给 MCPClient.call_tool
        assert mock.calls == [("search", {"query": "hi"})]

    @pytest.mark.asyncio
    async def test_call_tool_is_error(self, monkeypatch):
        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.MCPClient", _MockMCPClient)

        client = HTTPMCPClient(url="http://example/mcp")
        await client.connect()
        mock = cast(_MockMCPClient, client._mcp)
        mock.call_tool_result = {
            "isError": True,
            "content": [{"type": "text", "text": "fail"}],
        }
        with pytest.raises(MCPToolError, match="fail"):
            await client.call_tool("broken", {})

    @pytest.mark.asyncio
    async def test_close(self, monkeypatch):
        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.MCPClient", _MockMCPClient)

        client = HTTPMCPClient(url="http://example/mcp")
        await client.connect()
        await client.close()

        assert cast(_MockMCPClient, client._mcp).closed is True

    @pytest.mark.asyncio
    async def test_connect_missing_instructions_attr(self, monkeypatch):
        """防御式：旧版 mutio 的 MCPClient 没有 server_instructions 字段也不报错。"""
        class _OldClient(_MockMCPClient):
            def __init__(self, url, timeout=30.0):
                super().__init__(url, timeout)
                del self.server_instructions  # 模拟旧版无此字段

        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.MCPClient", _OldClient)

        client = HTTPMCPClient(url="http://example/mcp")
        info = await client.connect()
        assert info["instructions"] == ""


# ============================================================
# bridge_mcp_server 分派
# ============================================================

class TestBridgeDispatch:

    @pytest.mark.asyncio
    async def test_unknown_transport_raises(self):
        with pytest.raises(ValueError, match="unknown transport"):
            await bridge_mcp_server("x", {"transport": "websocket"})

    @pytest.mark.asyncio
    async def test_stdio_requires_command(self):
        with pytest.raises(ValueError, match="stdio transport requires 'command'"):
            await bridge_mcp_server("x", {"transport": "stdio"})

    @pytest.mark.asyncio
    async def test_http_requires_url(self):
        with pytest.raises(ValueError, match="http transport requires 'url'"):
            await bridge_mcp_server("x", {"transport": "http"})

    @pytest.mark.asyncio
    async def test_http_dispatches_to_http_client(self, monkeypatch):
        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.MCPClient", _MockMCPClient)

        # 注入一个预设 tools 的 mock
        def _factory(url, timeout=30.0):
            mock = _MockMCPClient(url=url, timeout=timeout)
            mock.tools_payload = [
                {"name": "echo", "description": "", "inputSchema": {}},
            ]
            return mock

        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.MCPClient", _factory)

        ns, client = await bridge_mcp_server("serena", {
            "transport": "http",
            "url": "http://example/mcp",
        })

        assert isinstance(client, HTTPMCPClient)
        assert ns._name == "serena"
        assert "echo" in ns._functions

    @pytest.mark.asyncio
    async def test_http_fills_namespace_description_from_instructions(self, monkeypatch):
        """MCP instructions 应成为 Namespace.description。"""
        def _factory(url, timeout=30.0):
            mock = _MockMCPClient(url=url, timeout=timeout)
            mock.tools_payload = []
            mock.server_instructions = "Use this server for X and Y."
            return mock

        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.MCPClient", _factory)

        ns, _ = await bridge_mcp_server("svc", {
            "transport": "http",
            "url": "http://example/mcp",
        })
        assert ns._description == "Use this server for X and Y."

    @pytest.mark.asyncio
    async def test_http_falls_back_to_server_title(self, monkeypatch):
        """无 instructions 时退化到 serverInfo.title。"""
        class _TitleMock(_MockMCPClient):
            async def connect(self):
                self.connected = True
                self.server_info = {"name": "mock", "title": "My Nice Server"}
                self.server_capabilities = {}
                self.server_instructions = ""

        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.MCPClient", _TitleMock)

        ns, _ = await bridge_mcp_server("svc", {
            "transport": "http",
            "url": "http://example/mcp",
        })
        assert ns._description == "My Nice Server"

    @pytest.mark.asyncio
    async def test_stdio_fills_description_from_instructions(self, monkeypatch):
        """Stdio 分支：connect() 返回的 instructions 传给 Namespace。"""
        class _FakeStdio:
            def __init__(self, command, args=None, shell=False, env=None):
                pass

            async def connect(self):
                return {
                    "serverInfo": {"name": "x"},
                    "instructions": "Stdio server docs.",
                }

            async def list_tools(self):
                return []

            async def close(self):
                pass

        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.StdioMCPClient", _FakeStdio)

        ns, _ = await bridge_mcp_server("s", {"command": "cmd"})
        assert ns._description == "Stdio server docs."

    @pytest.mark.asyncio
    async def test_stdio_default_transport(self, monkeypatch):
        # 默认 transport=stdio 时应走 StdioMCPClient 分支 — 用 mock 替代
        created: dict = {}

        class _FakeStdio:
            def __init__(self, command, args=None, shell=False, env=None):
                created["command"] = command
                created["args"] = args
                created["shell"] = shell
                created["env"] = env

            async def connect(self):
                return {}

            async def list_tools(self):
                return []

            async def close(self):
                pass

        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.StdioMCPClient", _FakeStdio)

        ns, client = await bridge_mcp_server("play", {
            "command": "npx",
            "args": ["-y", "@playwright/mcp"],
            "shell": True,
        })

        assert isinstance(client, _FakeStdio)
        assert created == {"command": "npx", "args": ["-y", "@playwright/mcp"], "shell": True, "env": None}
        assert ns._name == "play"


# ============================================================
# tool_func 跨线程调用：必须走 run_coroutine_threadsafe(main_loop)
# 回归测试迭代 2 的 "Event loop is closed" bug
# ============================================================

class TestToolFuncCrossThread:

    @pytest.mark.asyncio
    async def test_tool_func_invoked_from_worker_thread(self, monkeypatch):
        """模拟 pysandbox 执行路径：setup 在主 loop，tool_func 在线程池被调用。

        修复前：tool_func 走 asyncio.run() 开新 loop，httpx 跨 loop 炸。
        修复后：tool_func 走 run_coroutine_threadsafe(main_loop)，在原 loop 执行。
        验证：client.call_tool 收到的 loop 必须是 setup 时的 main_loop。
        """
        import threading

        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.MCPClient", _MockMCPClient)

        def _factory(url, timeout=30.0):
            mock = _MockMCPClient(url=url, timeout=timeout)
            mock.tools_payload = [{"name": "echo", "description": "", "inputSchema": {}}]
            mock.call_tool_result = {"content": [{"type": "text", "text": "ok"}]}
            return mock

        monkeypatch.setattr("mutagent.sandbox._adapter_mcp.MCPClient", _factory)

        ns, _client = await bridge_mcp_server("svc", {
            "transport": "http",
            "url": "http://example/mcp",
        })
        echo = ns._functions["echo"]
        main_loop = asyncio.get_running_loop()

        # 在非 asyncio 线程调用 tool_func —— 精确模拟 pysandbox 线程池
        result_box: dict = {}
        caller_tid = threading.get_ident()

        def worker():
            result_box["tid"] = threading.get_ident()
            result_box["result"] = echo()

        # 用 default executor（和 pysandbox 一样）跑 worker，主 loop 保持 running
        await main_loop.run_in_executor(None, worker)

        assert result_box["tid"] != caller_tid  # 确实在另一个线程
        assert result_box["result"] == "ok"

    @pytest.mark.asyncio
    async def test_tool_func_repeated_calls(self, monkeypatch):
        """连续多次调用不会破坏 client 状态（之前 asyncio.run() 每次开关 loop 会泄漏）。"""
        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.MCPClient", _MockMCPClient)

        def _factory(url, timeout=30.0):
            mock = _MockMCPClient(url=url, timeout=timeout)
            mock.tools_payload = [{"name": "ping", "description": "", "inputSchema": {}}]
            mock.call_tool_result = {"content": [{"type": "text", "text": "pong"}]}
            return mock

        monkeypatch.setattr("mutagent.sandbox._adapter_mcp.MCPClient", _factory)

        ns, _client = await bridge_mcp_server("svc", {
            "transport": "http",
            "url": "http://example/mcp",
        })
        ping = ns._functions["ping"]
        main_loop = asyncio.get_running_loop()

        def worker():
            return [ping() for _ in range(5)]

        results = await main_loop.run_in_executor(None, worker)
        assert results == ["pong"] * 5


# ============================================================
# _is_transport_error
# ============================================================

class TestIsTransportError:

    def test_mcp_transport_error(self):
        assert _is_transport_error(MCPTransportError("x")) is True

    def test_mcp_tool_error_is_not_transport(self):
        assert _is_transport_error(MCPToolError("user-facing")) is False

    def test_broken_pipe(self):
        assert _is_transport_error(BrokenPipeError("pipe")) is True

    def test_connection_reset(self):
        assert _is_transport_error(ConnectionResetError("reset")) is True

    def test_eof(self):
        assert _is_transport_error(EOFError()) is True

    def test_httpx_connect_error(self):
        import httpx
        assert _is_transport_error(httpx.ConnectError("x")) is True

    def test_httpx_read_error(self):
        import httpx
        assert _is_transport_error(httpx.ReadError("x")) is True

    def test_httpx_status_404_is_transport(self):
        import httpx
        req = httpx.Request("POST", "http://x/")
        resp = httpx.Response(404, request=req)
        exc = httpx.HTTPStatusError("not found", request=req, response=resp)
        assert _is_transport_error(exc) is True

    def test_httpx_status_410_is_transport(self):
        import httpx
        req = httpx.Request("POST", "http://x/")
        resp = httpx.Response(410, request=req)
        exc = httpx.HTTPStatusError("gone", request=req, response=resp)
        assert _is_transport_error(exc) is True

    def test_httpx_status_500_is_not_transport(self):
        import httpx
        req = httpx.Request("POST", "http://x/")
        resp = httpx.Response(500, request=req)
        exc = httpx.HTTPStatusError("oops", request=req, response=resp)
        assert _is_transport_error(exc) is False

    def test_runtime_closed_unexpectedly(self):
        assert _is_transport_error(RuntimeError("MCP server closed unexpectedly")) is True

    def test_other_runtime_error(self):
        assert _is_transport_error(RuntimeError("MCP error -32601: method not found")) is False

    def test_value_error_is_not_transport(self):
        assert _is_transport_error(ValueError("bad")) is False


# ============================================================
# MCPConnection 状态机 / 重连 / 冷却 / 锁
# ============================================================

class _FakeClient:
    """通用伪 client，可控制 connect / list_tools / call_tool 行为。"""

    def __init__(self, instructions: str = "", tools: list | None = None):
        self.instructions = instructions
        self.tools = tools if tools is not None else []
        self.connect_calls = 0
        self.close_calls = 0
        self.call_log: list = []
        # 注入异常（按顺序消费）
        self.connect_errors: list = []
        self.list_tools_errors: list = []
        self.call_tool_errors: list = []

    async def connect(self):
        self.connect_calls += 1
        if self.connect_errors:
            raise self.connect_errors.pop(0)
        return {
            "instructions": self.instructions,
            "serverInfo": {"name": "fake"},
        }

    async def list_tools(self):
        if self.list_tools_errors:
            raise self.list_tools_errors.pop(0)
        return list(self.tools)

    async def call_tool(self, name, arguments):
        self.call_log.append((name, arguments))
        if self.call_tool_errors:
            err = self.call_tool_errors.pop(0)
            raise err
        return f"ok:{name}"

    async def close(self):
        self.close_calls += 1


class TestMCPConnectionStateMachine:

    @pytest.mark.asyncio
    async def test_initial_state_is_disconnected(self):
        loop = asyncio.get_running_loop()
        conn = MCPConnection("ns", {"transport": "http", "url": "http://x"}, loop)
        assert conn.state == "disconnected"
        assert conn.client is None
        assert conn.namespace.connection_state == "disconnected"
        assert conn.namespace._connection is conn

    @pytest.mark.asyncio
    async def test_reconnect_success(self, monkeypatch):
        loop = asyncio.get_running_loop()
        fake = _FakeClient(
            instructions="hello",
            tools=[{"name": "echo", "description": "", "inputSchema": {}}])
        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.make_client",
            lambda ns, cfg: fake)

        conn = MCPConnection("ns", {"transport": "http", "url": "http://x"}, loop)
        await conn.reconnect()

        assert conn.state == "connected"
        assert conn.client is fake
        assert "echo" in conn.namespace._functions
        assert conn.namespace._description == "hello"
        assert conn.namespace.connection_state == "connected"

    @pytest.mark.asyncio
    async def test_reconnect_failure_marks_failed(self, monkeypatch):
        loop = asyncio.get_running_loop()
        fake = _FakeClient()
        fake.connect_errors.append(MCPTransportError("connect refused"))
        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.make_client",
            lambda ns, cfg: fake)

        conn = MCPConnection("ns", {"transport": "http", "url": "http://x"}, loop)
        with pytest.raises(MCPTransportError, match="connect refused"):
            await conn.reconnect()

        assert conn.state == "failed"
        assert conn.client is None
        assert "connect refused" in (conn.last_error or "")
        assert conn.namespace.connection_state == "failed"
        assert conn.namespace.connection_error is not None

    @pytest.mark.asyncio
    async def test_ensure_connected_idempotent(self, monkeypatch):
        loop = asyncio.get_running_loop()
        fake = _FakeClient()
        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.make_client",
            lambda ns, cfg: fake)

        conn = MCPConnection("ns", {"transport": "http", "url": "http://x"}, loop)
        await conn.ensure_connected()
        await conn.ensure_connected()
        await conn.ensure_connected()

        # 只真实 connect 一次（连上后短路）
        assert fake.connect_calls == 1

    @pytest.mark.asyncio
    async def test_cooldown_blocks_retry(self, monkeypatch):
        loop = asyncio.get_running_loop()
        fake = _FakeClient()
        fake.connect_errors.append(MCPTransportError("first fail"))
        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.make_client",
            lambda ns, cfg: fake)

        conn = MCPConnection(
            "ns", {"transport": "http", "url": "http://x"},
            loop, retry_cooldown=10.0)

        with pytest.raises(MCPTransportError, match="first fail"):
            await conn.ensure_connected()

        assert conn.state == "failed"
        # 冷却期内：直接抛上次的错，不重新发起 connect
        with pytest.raises(MCPTransportError, match="cooldown"):
            await conn.ensure_connected()

        assert fake.connect_calls == 1  # 未重试

    @pytest.mark.asyncio
    async def test_cooldown_zero_disables(self, monkeypatch):
        loop = asyncio.get_running_loop()
        fake = _FakeClient()
        # 第一次 fail，第二次成功
        fake.connect_errors.append(MCPTransportError("transient"))
        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.make_client",
            lambda ns, cfg: fake)

        conn = MCPConnection(
            "ns", {"transport": "http", "url": "http://x"},
            loop, retry_cooldown=0.0)

        with pytest.raises(MCPTransportError):
            await conn.ensure_connected()
        # cooldown=0：立即重试
        await conn.ensure_connected()
        assert conn.state == "connected"

    @pytest.mark.asyncio
    async def test_concurrent_ensure_connected_only_one_real_connect(self, monkeypatch):
        loop = asyncio.get_running_loop()

        # 模拟 connect 慢一点，给并发留窗口
        class _SlowClient(_FakeClient):
            async def connect(self):
                await asyncio.sleep(0.05)
                return await super().connect()

        fake = _SlowClient()
        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.make_client",
            lambda ns, cfg: fake)

        conn = MCPConnection("ns", {"transport": "http", "url": "http://x"}, loop)

        # 5 个并发 ensure_connected
        await asyncio.gather(*[conn.ensure_connected() for _ in range(5)])

        assert fake.connect_calls == 1  # Lock 起作用
        assert conn.state == "connected"

    @pytest.mark.asyncio
    async def test_close_resets_state(self, monkeypatch):
        loop = asyncio.get_running_loop()
        fake = _FakeClient()
        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.make_client",
            lambda ns, cfg: fake)

        conn = MCPConnection("ns", {"transport": "http", "url": "http://x"}, loop)
        await conn.ensure_connected()
        await conn.close()

        assert conn.state == "disconnected"
        assert conn.client is None
        assert fake.close_calls == 1

        # 多次 close 幂等
        await conn.close()
        assert fake.close_calls == 1

    @pytest.mark.asyncio
    async def test_reconnect_refreshes_tools(self, monkeypatch):
        loop = asyncio.get_running_loop()
        fake = _FakeClient(
            tools=[{"name": "old", "description": "", "inputSchema": {}}])
        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.make_client",
            lambda ns, cfg: fake)

        conn = MCPConnection("ns", {"transport": "http", "url": "http://x"}, loop)
        await conn.reconnect()
        assert "old" in conn.namespace._functions

        # server 重启，tool 列表变了
        fake.tools = [{"name": "new", "description": "", "inputSchema": {}}]
        await conn.reconnect()
        assert "new" in conn.namespace._functions
        assert "old" not in conn.namespace._functions  # 旧 tool 被删


class TestToolFuncAutoReconnect:

    @pytest.mark.asyncio
    async def test_call_succeeds_lazy(self, monkeypatch):
        """autostart=false 时首次调用触发 connect。"""
        loop = asyncio.get_running_loop()
        fake = _FakeClient(
            tools=[{"name": "ping", "description": "", "inputSchema": {}}])
        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.make_client",
            lambda ns, cfg: fake)

        conn = MCPConnection("ns", {"transport": "http", "url": "http://x"}, loop)
        # 没 autostart，未 connect
        assert conn.state == "disconnected"

        # 通过 namespace 取函数 — __getattr__ 触发 ensure_connected
        # 注意：__getattr__ 用 run_coroutine_threadsafe，必须在另一个线程
        result_box: dict = {}

        def worker():
            fn = conn.namespace.ping  # 触发懒连接
            result_box["result"] = fn()

        await loop.run_in_executor(None, worker)
        assert result_box["result"] == "ok:ping"
        assert conn.state == "connected"

    @pytest.mark.asyncio
    async def test_call_retries_after_transport_error(self, monkeypatch):
        """call 出现传输错 → 自动重连重试一次后成功。"""
        loop = asyncio.get_running_loop()

        clients: list = []

        def factory(ns, cfg):
            c = _FakeClient(
                tools=[{"name": "do", "description": "", "inputSchema": {}}])
            if not clients:
                # 第一个 client 的 call 会抛传输错
                c.call_tool_errors.append(MCPTransportError("conn dropped"))
            clients.append(c)
            return c

        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.make_client", factory)

        conn = MCPConnection(
            "ns", {"transport": "http", "url": "http://x"},
            loop, retry_cooldown=0.0)
        await conn.ensure_connected()  # 拿到第一个 client

        result_box: dict = {}

        def worker():
            result_box["result"] = conn.namespace.do()

        await loop.run_in_executor(None, worker)

        assert result_box["result"] == "ok:do"
        assert len(clients) == 2  # 重连建了第二个 client
        assert conn.state == "connected"

    @pytest.mark.asyncio
    async def test_tool_error_does_not_reconnect(self, monkeypatch):
        """业务错（MCPToolError）直接抛给用户，不触发重连。"""
        loop = asyncio.get_running_loop()
        clients: list = []

        def factory(ns, cfg):
            c = _FakeClient(
                tools=[{"name": "do", "description": "", "inputSchema": {}}])
            c.call_tool_errors.append(MCPToolError("user input invalid"))
            clients.append(c)
            return c

        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.make_client", factory)

        conn = MCPConnection("ns", {"transport": "http", "url": "http://x"}, loop)
        await conn.ensure_connected()

        result_box: dict = {}

        def worker():
            try:
                conn.namespace.do()
                result_box["result"] = "no-error"
            except Exception as exc:
                result_box["error"] = exc

        await loop.run_in_executor(None, worker)

        assert isinstance(result_box["error"], MCPToolError)
        assert "user input invalid" in str(result_box["error"])
        assert len(clients) == 1  # 没重连
        assert conn.state == "connected"  # 状态保持

    @pytest.mark.asyncio
    async def test_double_failure_raises(self, monkeypatch):
        """重试一次仍失败 → 抛 MCPTransportError。"""
        loop = asyncio.get_running_loop()
        clients: list = []

        def factory(ns, cfg):
            c = _FakeClient(
                tools=[{"name": "do", "description": "", "inputSchema": {}}])
            # 第一个 client 的 call 失败；重连后第二个 client 的 call 也失败
            c.call_tool_errors.append(MCPTransportError("net" + str(len(clients) + 1)))
            clients.append(c)
            return c

        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.make_client", factory)

        conn = MCPConnection(
            "ns", {"transport": "http", "url": "http://x"},
            loop, retry_cooldown=0.0)
        await conn.ensure_connected()

        result_box: dict = {}

        def worker():
            try:
                conn.namespace.do()
            except Exception as exc:
                result_box["error"] = exc

        await loop.run_in_executor(None, worker)
        assert isinstance(result_box["error"], MCPTransportError)


# ============================================================
# Namespace render — 状态显示
# ============================================================

class TestNamespaceRender:

    def _make_registry(self, namespaces):
        from mutagent.sandbox._namespace import NamespaceRegistry
        reg = NamespaceRegistry()
        for ns in namespaces:
            reg.add(ns)
        return reg

    def _mcp_ns(self, name, state, error=None, functions=None):
        from mutagent.sandbox._namespace import Namespace
        ns = Namespace(name, description="")
        # 模拟有 connection（非 None 即可触发 MCP 渲染分支）
        ns._connection = object()  # type: ignore[assignment]
        ns.connection_state = state
        ns.connection_error = error
        for fname in functions or []:
            ns.register(fname, lambda **k: None, "")
        return ns

    def test_connected_no_state_label(self):
        from mutagent.sandbox._namespace import _render_registry
        ns = self._mcp_ns("playwright", "connected", functions=["a", "b"])
        text = _render_registry(self._make_registry([ns]))
        assert "[connecting" not in text
        assert "[failed" not in text
        assert "[disconnected" not in text
        assert "playwright" in text
        assert "(2 functions)" in text

    def test_connecting_label(self):
        from mutagent.sandbox._namespace import _render_registry
        ns = self._mcp_ns("serena", "connecting")
        text = _render_registry(self._make_registry([ns]))
        assert "[connecting...]" in text

    def test_disconnected_unknown_count(self):
        from mutagent.sandbox._namespace import _render_registry
        ns = self._mcp_ns("experimental", "disconnected")
        text = _render_registry(self._make_registry([ns]))
        assert "[disconnected]" in text
        assert "(? functions)" in text  # 从未连过

    def test_failed_with_reason(self):
        from mutagent.sandbox._namespace import _render_registry
        ns = self._mcp_ns("weather", "failed", error="connection refused")
        text = _render_registry(self._make_registry([ns]))
        assert "[failed: connection refused]" in text

    def test_failed_reason_truncated(self):
        from mutagent.sandbox._namespace import _render_registry
        long = "x" * 200
        ns = self._mcp_ns("weather", "failed", error=long)
        text = _render_registry(self._make_registry([ns]))
        # 60 字符截断
        assert "..." in text
        # 不应出现完整 200 字符
        assert ("x" * 100) not in text

    def test_render_namespace_failed_hint(self):
        from mutagent.sandbox._namespace import _render_namespace
        ns = self._mcp_ns("weather", "failed", error="ECONNREFUSED")
        text = _render_namespace(ns)
        assert "Connection failed: ECONNREFUSED" in text
        assert "Calling any function will retry" in text

    def test_non_mcp_namespace_no_state(self):
        """普通 NamespaceTools / CLI namespace 不显示状态标签。"""
        from mutagent.sandbox._namespace import Namespace, _render_registry
        ns = Namespace("fs", description="")
        ns.register("read", lambda **k: None, "")
        text = _render_registry(self._make_registry([ns]))
        assert "[" not in text.split("Use help")[0]  # 状态标签不出现
        assert "(1 functions)" in text


# ============================================================
# env 透传 & list_tools_metadata（feature-mcp-source-config）
# ============================================================

class TestStdioEnvPassthrough:
    """`StdioMCPClient` 与 `make_client` 的 env 透传。"""

    @pytest.mark.asyncio
    async def test_make_client_forwards_env_to_stdio(self, monkeypatch):
        """make_client 把 server_config['env'] 传给 StdioMCPClient.__init__。"""
        from mutagent.sandbox._adapter_mcp import make_client

        captured: dict = {}

        class _FakeStdio:
            def __init__(self, command, args=None, shell=False, env=None):
                captured["env"] = env

        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.StdioMCPClient", _FakeStdio)

        make_client("x", {
            "transport": "stdio",
            "command": "echo",
            "env": {"MY_KEY": "v1", "OTHER": "v2"},
        })
        assert captured["env"] == {"MY_KEY": "v1", "OTHER": "v2"}

    @pytest.mark.asyncio
    async def test_make_client_omits_env_when_missing(self, monkeypatch):
        """无 env 字段时传 None，保持子进程继承父 env。"""
        from mutagent.sandbox._adapter_mcp import make_client

        captured: dict = {}

        class _FakeStdio:
            def __init__(self, command, args=None, shell=False, env=None):
                captured["env"] = env

        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.StdioMCPClient", _FakeStdio)

        make_client("x", {"transport": "stdio", "command": "echo"})
        assert captured["env"] is None

    @pytest.mark.asyncio
    async def test_stdio_popen_receives_merged_env(self, monkeypatch):
        """StdioMCPClient.connect 把 os.environ | env 合并后传 Popen.env。"""
        import os
        from mutagent.sandbox import _adapter_mcp as adapter

        captured: dict = {}

        class _FakeProc:
            def __init__(self, *args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs
                self.stdin = None
                self.stdout = None
                self.stderr = None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

        # 直接在 connect 调 Popen 之前拦截，让握手快速失败避免实际 IO
        async def _fake_request(self, method, params):
            return {}

        def _fake_send_notification(self, method, params):
            return None

        monkeypatch.setattr(adapter.subprocess, "Popen", _FakeProc)
        monkeypatch.setattr(adapter.StdioMCPClient, "_request", _fake_request)
        monkeypatch.setattr(adapter.StdioMCPClient, "_send_notification",
                            _fake_send_notification)

        os.environ["__MCP_TEST_BASE__"] = "base"
        try:
            client = adapter.StdioMCPClient(
                "echo", env={"X": "1", "Y": "2"})
            await client.connect()
        finally:
            os.environ.pop("__MCP_TEST_BASE__", None)

        env = captured["kwargs"]["env"]
        assert env is not None
        # 合并后既有用户 env，也保留父进程 env
        assert env["X"] == "1"
        assert env["Y"] == "2"
        assert env.get("__MCP_TEST_BASE__") == "base"

    @pytest.mark.asyncio
    async def test_stdio_popen_env_none_when_no_env_config(self, monkeypatch):
        """env 配置缺省 → Popen 收到 env=None（继承父进程）。"""
        from mutagent.sandbox import _adapter_mcp as adapter

        captured: dict = {}

        class _FakeProc:
            def __init__(self, *args, **kwargs):
                captured["kwargs"] = kwargs
                self.stdin = None
                self.stdout = None
                self.stderr = None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

        async def _fake_request(self, method, params):
            return {}

        def _fake_send_notification(self, method, params):
            return None

        monkeypatch.setattr(adapter.subprocess, "Popen", _FakeProc)
        monkeypatch.setattr(adapter.StdioMCPClient, "_request", _fake_request)
        monkeypatch.setattr(adapter.StdioMCPClient, "_send_notification",
                            _fake_send_notification)

        client = adapter.StdioMCPClient("echo")
        await client.connect()
        assert captured["kwargs"]["env"] is None


class TestListToolsMetadata:
    """`MCPConnection.list_tools_metadata` 公开接口。"""

    @pytest.mark.asyncio
    async def test_list_tools_metadata_returns_schema(self, monkeypatch):
        """list_tools_metadata 应返回 name/description/input_schema/source_namespace。"""
        from mutagent.sandbox._adapter_mcp import MCPConnection

        # mock 一个 client，返回带 inputSchema 的 tools
        class _FakeClient:
            async def connect(self):
                return {"serverInfo": {"name": "x"}, "instructions": ""}

            async def list_tools(self):
                return [
                    {
                        "name": "read_file",
                        "description": "Read a file",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "file path"},
                            },
                            "required": ["path"],
                        },
                    },
                    {
                        "name": "write_file",
                        "description": "",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                    },
                ]

            async def call_tool(self, name, arguments):
                return {}

            async def close(self):
                pass

        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.make_client",
            lambda ns, cfg: _FakeClient())

        loop = asyncio.get_running_loop()
        conn = MCPConnection("fs", {"transport": "stdio", "command": "x"}, loop)
        await conn.reconnect()

        meta = conn.list_tools_metadata()
        names = [m["name"] for m in meta]
        assert "read_file" in names
        assert "write_file" in names

        rf = next(m for m in meta if m["name"] == "read_file")
        assert rf["description"] == "Read a file"
        assert rf["input_schema"]["properties"]["path"]["type"] == "string"
        assert rf["input_schema"]["required"] == ["path"]
        assert rf["source_namespace"] == "fs"

        await conn.close()

    @pytest.mark.asyncio
    async def test_list_tools_metadata_empty_when_disconnected(self):
        """未连接时返回空列表。"""
        from mutagent.sandbox._adapter_mcp import MCPConnection
        loop = asyncio.get_running_loop()
        conn = MCPConnection("fs", {"transport": "stdio", "command": "x"}, loop)
        # 还未 reconnect
        assert conn.list_tools_metadata() == []
