"""Pysandbox namespace sharing — server + client 端到端集成测试。

绕过 HTTP，让 client 端的 ``HTTPMCPClient.request`` 直接走到 server 端
``register_pysandbox_methods`` 注册的 dispatcher，验证：

- ``pysandbox/namespaces.list/describe/call`` 协议契约
- ``MCPConnection._do_rebuild`` 检测 capability 并融合 peer namespaces
- ``D2`` 过滤对端 ``pysandbox`` tool
- ``D1`` 重名冲突直接 RuntimeError
- ``D6`` peer namespace 的描述带 ``(shared from <source>)`` 标记
- 状态同步：source 失败时 peer namespaces 一起翻为 failed
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from mutagent.sandbox._namespace import Namespace, NamespaceRegistry
from mutagent.sandbox.share import (
    PYSANDBOX_CAPABILITY,
    register_pysandbox_methods,
)
from mutio.mcp.protocol import JsonRpcDispatcher


# ---------------------------------------------------------------------------
# Server 侧：一个最小 SandboxApp-like 对象 + dispatcher
# ---------------------------------------------------------------------------


class _FakeSandbox:
    """最小 SandboxApp 替身 —— share.py 只摸 ``_registry._namespaces``。

    不走完整 SandboxApp Declaration 路径，避免引入 NamespaceTools 自动发现
    干扰；只验证 share 协议本身的行为。
    """

    def __init__(self) -> None:
        self._registry = NamespaceRegistry()


def _make_server_dispatch(sandbox: _FakeSandbox) -> JsonRpcDispatcher:
    dispatch = JsonRpcDispatcher()
    register_pysandbox_methods(dispatch, sandbox)  # type: ignore[arg-type]
    return dispatch


def _build_mutbot_namespace() -> Namespace:
    """模拟 mutbot 侧的 ``mutbot`` namespace（status / logs）。"""
    ns = Namespace("mutbot", description="mutbot runtime introspection")

    def status() -> dict[str, Any]:
        """Return server status snapshot."""
        return {"pid": 12345, "uptime": 60}

    def logs(level: str = "INFO", last_n: int = 10) -> list[str]:
        """Query recent log lines."""
        return [f"[{level}] line {i}" for i in range(last_n)]

    def boom() -> None:
        """Always raises — used to test error path."""
        raise ValueError("intentional explosion")

    ns.register("status", status, status.__doc__ or "")
    ns.register("logs", logs, logs.__doc__ or "")
    ns.register("boom", boom, boom.__doc__ or "")
    return ns


# ---------------------------------------------------------------------------
# Client 侧：把 dispatcher 包成假 ``HTTPMCPClient``
# ---------------------------------------------------------------------------


class _FakeMCPClient:
    """假 ``MCPClient`` — 把 ``request`` 转给本地 dispatcher。"""

    def __init__(self, dispatch: JsonRpcDispatcher) -> None:
        self._dispatch = dispatch
        self._id = 0

    async def request(self, method: str, params: dict | None = None) -> Any:
        self._id += 1
        msg = {
            "jsonrpc": "2.0", "id": self._id,
            "method": method, "params": params or {},
        }
        resp = await self._dispatch.handle(msg)
        assert resp is not None
        if "error" in resp:
            err = resp["error"]
            raise RuntimeError(
                f"JSON-RPC error {err.get('code')}: {err.get('message')}")
        return resp["result"]


class _FakeHTTPClient:
    """假 ``HTTPMCPClient`` — 仅暴露 ``_mcp`` 属性，被 PysandboxPeerClient 用。"""

    def __init__(self, dispatch: JsonRpcDispatcher) -> None:
        self._mcp = _FakeMCPClient(dispatch)


# ---------------------------------------------------------------------------
# 协议层：list / describe / call
# ---------------------------------------------------------------------------


class TestServerProtocol:

    def setup_method(self) -> None:
        self.sandbox = _FakeSandbox()
        self.sandbox._registry.add(_build_mutbot_namespace())
        self.dispatch = _make_server_dispatch(self.sandbox)

    def _call(self, method: str, params: dict | None = None) -> Any:
        client = _FakeMCPClient(self.dispatch)
        return asyncio.run(client.request(method, params))

    def test_list_returns_namespaces(self) -> None:
        result = self._call("pysandbox/namespaces.list", {})
        assert "namespaces" in result
        names = [item["name"] for item in result["namespaces"]]
        assert "mutbot" in names
        # function_count 字段存在
        mutbot_item = next(i for i in result["namespaces"] if i["name"] == "mutbot")
        assert mutbot_item["function_count"] == 3

    def test_describe_returns_signatures(self) -> None:
        result = self._call(
            "pysandbox/namespaces.describe", {"namespace": "mutbot"})
        assert result["name"] == "mutbot"
        functions = result["functions"]
        assert "status" in functions and "logs" in functions
        # signature 是字符串形式（D5：kwargs only 协议层）
        assert functions["logs"]["signature"].startswith("(")
        assert "level" in functions["logs"]["signature"]
        # kwargs_schema v1 留空 dict
        assert functions["logs"]["kwargs_schema"] == {}

    def test_describe_unknown_namespace_errors(self) -> None:
        with pytest.raises(RuntimeError, match="Namespace not found"):
            self._call("pysandbox/namespaces.describe", {"namespace": "nope"})

    def test_call_executes_function(self) -> None:
        result = self._call(
            "pysandbox/namespaces.call",
            {"namespace": "mutbot", "name": "status", "arguments": {}})
        assert result == {"pid": 12345, "uptime": 60}

    def test_call_with_kwargs(self) -> None:
        result = self._call(
            "pysandbox/namespaces.call",
            {"namespace": "mutbot", "name": "logs",
             "arguments": {"level": "ERROR", "last_n": 3}})
        assert result == ["[ERROR] line 0", "[ERROR] line 1", "[ERROR] line 2"]

    def test_call_business_error_becomes_jsonrpc_error(self) -> None:
        # 业务异常 → INTERNAL_ERROR，但不触发 client 侧重连（client 端通过
        # _is_transport_error 区分；这里只验证 server 不把它当传输错抛裸异常）
        with pytest.raises(RuntimeError, match="intentional explosion"):
            self._call(
                "pysandbox/namespaces.call",
                {"namespace": "mutbot", "name": "boom", "arguments": {}})

    def test_call_typeerror_becomes_invalid_params(self) -> None:
        # 不存在的 kwarg → TypeError → INVALID_PARAMS
        with pytest.raises(RuntimeError, match="JSON-RPC error -32602"):
            self._call(
                "pysandbox/namespaces.call",
                {"namespace": "mutbot", "name": "logs",
                 "arguments": {"bogus_kwarg": 1}})


# ---------------------------------------------------------------------------
# Client 侧：build_peer_namespaces + MCPConnection 融合
# ---------------------------------------------------------------------------


class TestPeerBuild:
    """build_peer_namespaces 单元行为。"""

    def test_builds_namespace_with_callable_functions(self) -> None:
        from mutagent.sandbox._adapter_pysandbox import build_peer_namespaces

        sandbox = _FakeSandbox()
        sandbox._registry.add(_build_mutbot_namespace())
        dispatch = _make_server_dispatch(sandbox)

        # 构造一个最小 conn 占位 —— build_peer_namespaces 只读 ns_name / state /
        # last_error，不调用 reconnect
        class _FakeConn:
            ns_name = "mutbot_local"
            state = "connected"
            last_error = None
            main_loop = None  # ns_func 不会被调用就不需要

        conn = _FakeConn()
        client = _FakeHTTPClient(dispatch)
        init_result = {"capabilities": PYSANDBOX_CAPABILITY}

        namespaces = asyncio.run(
            build_peer_namespaces(conn, init_result, client))  # type: ignore[arg-type]
        assert len(namespaces) == 1
        ns = namespaces[0]
        assert ns.name == "mutbot"
        # D6: 描述带来源标记
        assert "(shared from mutbot_local)" in ns._description
        # 函数集齐
        assert set(ns._functions.keys()) == {"status", "logs", "boom"}
        # 与 conn 共享状态
        assert ns._connection is conn
        assert ns.connection_state == "connected"


# ---------------------------------------------------------------------------
# MCPConnection 集成：peer 融合 + D2 tool 过滤 + D1 冲突 + 状态同步
# ---------------------------------------------------------------------------


class _FakeHTTPClientForConn:
    """假 HTTPMCPClient — 实现 connect / list_tools / close 接口供 MCPConnection 用。

    比 HTTPMCPClient 简化：connect 返回固定 init_result；list_tools 返回固定
    tool 列表（含一个名为 ``pysandbox`` 的 tool 用于验证 D2 过滤）。
    """

    def __init__(self, dispatch: JsonRpcDispatcher,
                 capabilities: dict, tools: list[dict]) -> None:
        self._mcp = _FakeMCPClient(dispatch)
        self._init_result = {
            "serverInfo": {"name": "fake", "title": "Fake Server"},
            "capabilities": capabilities,
            "instructions": "fake instructions",
        }
        self._tools = tools

    async def connect(self) -> dict:
        return self._init_result

    async def list_tools(self) -> list[dict]:
        return list(self._tools)

    async def call_tool(self, name: str, arguments: dict) -> Any:
        # 不会被本测试触发（peer client 走 namespace 路径）
        raise NotImplementedError

    async def close(self) -> None:
        pass


class TestMCPConnectionPeerIntegration:
    """通过 monkeypatch make_client，让 MCPConnection 用我们的假 client。"""

    def _setup_conn(self, capabilities: dict, tools: list[dict],
                    monkeypatch: pytest.MonkeyPatch):
        from mutagent.sandbox import _adapter_mcp

        sandbox = _FakeSandbox()
        sandbox._registry.add(_build_mutbot_namespace())
        dispatch = _make_server_dispatch(sandbox)

        fake_client = _FakeHTTPClientForConn(dispatch, capabilities, tools)

        def _fake_make_client(ns_name: str, cfg: dict):
            return fake_client

        monkeypatch.setattr(_adapter_mcp, "make_client", _fake_make_client)

        # _do_rebuild 用 isinstance(client, HTTPMCPClient) 判定 peer 路径，
        # 临时把 HTTPMCPClient 也指向 _FakeHTTPClientForConn 的基类（用 monkeypatch
        # 的方式不行 —— isinstance 检查走 import 时的真名）。改用 patch 替换。
        monkeypatch.setattr(
            _adapter_mcp, "HTTPMCPClient", _FakeHTTPClientForConn)
        # _adapter_pysandbox 也在内部 import 了 HTTPMCPClient
        from mutagent.sandbox import _adapter_pysandbox as _ap
        monkeypatch.setattr(_ap, "HTTPMCPClient", _FakeHTTPClientForConn,
                            raising=False)

        loop = asyncio.new_event_loop()
        try:
            conn = _adapter_mcp.MCPConnection(
                "mutbot_remote", {"url": "http://x"}, loop)
            loop.run_until_complete(conn.reconnect())
            return conn, loop
        except Exception:
            loop.close()
            raise

    def test_peer_namespace_merged(self, monkeypatch: pytest.MonkeyPatch):
        conn, loop = self._setup_conn(
            PYSANDBOX_CAPABILITY,
            tools=[{"name": "echo",
                    "description": "echo",
                    "inputSchema": {}}],
            monkeypatch=monkeypatch)
        try:
            # tool 路径：echo 注册到主 namespace
            assert "echo" in conn.namespace._functions
            # peer 路径：mutbot namespace 融合进来
            assert len(conn.peer_namespaces) == 1
            assert conn.peer_namespaces[0].name == "mutbot"
        finally:
            loop.run_until_complete(conn.close())
            loop.close()

    def test_d2_filters_pysandbox_tool(self, monkeypatch: pytest.MonkeyPatch):
        conn, loop = self._setup_conn(
            PYSANDBOX_CAPABILITY,
            tools=[
                {"name": "pysandbox", "description": "should be hidden",
                 "inputSchema": {}},
                {"name": "other", "description": "kept", "inputSchema": {}},
            ],
            monkeypatch=monkeypatch)
        try:
            # D2: pysandbox tool 被过滤
            assert "pysandbox" not in conn.namespace._functions
            assert "other" in conn.namespace._functions
        finally:
            loop.run_until_complete(conn.close())
            loop.close()

    def test_no_capability_no_peer_namespaces(self,
                                              monkeypatch: pytest.MonkeyPatch):
        conn, loop = self._setup_conn(
            capabilities={},  # 不声明 pysandbox cap
            tools=[{"name": "pysandbox", "description": "kept",
                    "inputSchema": {}}],
            monkeypatch=monkeypatch)
        try:
            assert conn.peer_namespaces == []
            # 没有 capability 时不过滤 pysandbox tool
            assert "pysandbox" in conn.namespace._functions
        finally:
            loop.run_until_complete(conn.close())
            loop.close()

    def test_state_propagates_to_peer_namespaces(
            self, monkeypatch: pytest.MonkeyPatch):
        conn, loop = self._setup_conn(
            PYSANDBOX_CAPABILITY,
            tools=[],
            monkeypatch=monkeypatch)
        try:
            peer = conn.peer_namespaces[0]
            assert peer.connection_state == "connected"
            # 模拟传输错
            conn.mark_disconnected("network down")
            assert peer.connection_state == "failed"
            assert peer.connection_error == "network down"
        finally:
            loop.run_until_complete(conn.close())
            loop.close()

    def test_d1_peer_name_conflict_no_longer_raises(
            self, monkeypatch: pytest.MonkeyPatch):
        """D1 (multi-provider 重写)：peer namespace 与本 conn 的 tool ns 同名
        不再阻塞。两者作为同名 namespace 的不同 provider 并存，
        冲突在调用/help 级走 :class:`MergedNamespaceView` 处理。详
        ``mutagent/docs/specifications/feature-namespace-multi-provider.md``。
        """
        from mutagent.sandbox import _adapter_mcp

        sandbox = _FakeSandbox()
        clash_ns = Namespace("mutbot_remote", description="clash")
        clash_ns.register("noop", lambda: None, "")
        sandbox._registry.add(clash_ns)
        dispatch = _make_server_dispatch(sandbox)
        fake_client = _FakeHTTPClientForConn(
            dispatch, PYSANDBOX_CAPABILITY, [])

        monkeypatch.setattr(
            _adapter_mcp, "make_client", lambda *a, **kw: fake_client)
        monkeypatch.setattr(
            _adapter_mcp, "HTTPMCPClient", _FakeHTTPClientForConn)
        from mutagent.sandbox import _adapter_pysandbox as _ap
        monkeypatch.setattr(_ap, "HTTPMCPClient", _FakeHTTPClientForConn,
                            raising=False)

        loop = asyncio.new_event_loop()
        try:
            conn = _adapter_mcp.MCPConnection(
                "mutbot_remote", {"url": "http://x"}, loop)
            # 不再抛错：conn 成功 connected，peer 列表含同名 ns
            loop.run_until_complete(conn.reconnect())
            assert conn.state == "connected"
            peer_names = [p.name for p in conn.peer_namespaces]
            assert "mutbot_remote" in peer_names
        finally:
            loop.run_until_complete(conn.close())
            loop.close()

    def test_d1_peer_self_duplicate_still_raises(
            self, monkeypatch: pytest.MonkeyPatch):
        """D1：同一 server 自我 export 两个同名 peer namespace 仍是 server bug，
        仍然报错。"""
        from mutagent.sandbox import _adapter_mcp, _adapter_pysandbox

        async def fake_build(conn, init_result, client):
            # 模拟 server 返回两个同名 peer ns
            return [Namespace("dup"), Namespace("dup")]

        sandbox = _FakeSandbox()
        dispatch = _make_server_dispatch(sandbox)
        fake_client = _FakeHTTPClientForConn(
            dispatch, PYSANDBOX_CAPABILITY, [])

        monkeypatch.setattr(
            _adapter_mcp, "make_client", lambda *a, **kw: fake_client)
        monkeypatch.setattr(
            _adapter_mcp, "HTTPMCPClient", _FakeHTTPClientForConn)
        monkeypatch.setattr(_adapter_pysandbox, "HTTPMCPClient",
                            _FakeHTTPClientForConn, raising=False)
        monkeypatch.setattr(_adapter_pysandbox, "build_peer_namespaces",
                            fake_build)

        loop = asyncio.new_event_loop()
        try:
            conn = _adapter_mcp.MCPConnection(
                "buggy_server", {"url": "http://x"}, loop)
            with pytest.raises(_adapter_mcp.MCPTransportError,
                               match="peer-namespace duplicate"):
                loop.run_until_complete(conn.reconnect())
            # D11：异常后 state 必须是 failed，不能卡 connecting
            assert conn.state == "failed"
        finally:
            loop.run_until_complete(conn.close())
            loop.close()
