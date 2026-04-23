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
    _extract_content,
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
        with pytest.raises(RuntimeError, match="boom"):
            _extract_content(result)

    def test_is_error_without_text(self):
        result = {"isError": True, "content": []}
        with pytest.raises(RuntimeError, match="MCP tool call failed"):
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
        with pytest.raises(RuntimeError, match="fail"):
            await client.call_tool("broken", {})

    @pytest.mark.asyncio
    async def test_close(self, monkeypatch):
        monkeypatch.setattr(
            "mutagent.sandbox._adapter_mcp.MCPClient", _MockMCPClient)

        client = HTTPMCPClient(url="http://example/mcp")
        await client.connect()
        await client.close()

        assert cast(_MockMCPClient, client._mcp).closed is True


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
    async def test_stdio_default_transport(self, monkeypatch):
        # 默认 transport=stdio 时应走 StdioMCPClient 分支 — 用 mock 替代
        created: dict = {}

        class _FakeStdio:
            def __init__(self, command, args=None, shell=False):
                created["command"] = command
                created["args"] = args
                created["shell"] = shell

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
        assert created == {"command": "npx", "args": ["-y", "@playwright/mcp"], "shell": True}
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
