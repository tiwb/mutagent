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
import inspect
from datetime import datetime
from typing import Any, cast

import mutobj
import pytest

from mutagent.sandbox import SandboxEnv, _mcp_impl, _mcp_impl_sandbox
from mutagent.sandbox._env_impl import SandboxEnvRuntime, _get_registry, _wrap_async
from mutagent.sandbox._mcp_impl import MCPConnectionImpl
from mutagent.sandbox._mcp_impl_sandbox import (
    _make_namespace_func,
    build_peer_namespaces,
)
from mutagent.sandbox._namespace_impl import (
    Namespace,
    NamespaceRegistry,
    render_function,
)
from mutagent.sandbox._signature import MISSING, _MissingSentinel
from mutagent.sandbox._mcp_share import (
    PYSANDBOX_CAPABILITY,
    _describe_function,
    register_pysandbox_methods,
)
from mutio.mcp.protocol import JsonRpcDispatcher


def _impl(conn):
    """Resolve MCPConnection → implementation."""
    impl = mutobj.implementation_of(conn, MCPConnectionImpl)
    return impl


# ---------------------------------------------------------------------------
# Server 侧：一个最小 SandboxEnv-like 对象 + dispatcher
# ---------------------------------------------------------------------------


class _FakeSandbox:
    """最小 SandboxEnv 替身 —— _mcp_share.py 只需要 runtime registry。

    不走完整 SandboxEnv Declaration 路径，避免引入 NamespaceTools 自动发现
    干扰；只验证 share 协议本身的行为。
    """

    def __init__(self) -> None:
        self.__mutobj_storage__ = {}
        SandboxEnvRuntime.get_or_create(self).registry = NamespaceRegistry()  # type: ignore[arg-type]


class _LoopSandbox:
    """最小 loop carrier，用于挂载 SandboxEnvRuntime。"""

    def __init__(self) -> None:
        self.__mutobj_storage__ = {}


def _sandbox_with_loop(loop: asyncio.AbstractEventLoop | None) -> _LoopSandbox:
    sandbox = _LoopSandbox()
    SandboxEnvRuntime.get_or_create(sandbox).async_loop = loop  # type: ignore[arg-type]
    return sandbox


def _make_server_dispatch(sandbox: SandboxEnv) -> JsonRpcDispatcher:
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
    """假 ``HTTPMCPClient`` — 仅暴露 ``mcp`` 属性，被 PysandboxPeerClient 用。"""

    def __init__(self, dispatch: JsonRpcDispatcher) -> None:
        self.mcp = _FakeMCPClient(dispatch)


# ---------------------------------------------------------------------------
# 协议层：list / describe / call
# ---------------------------------------------------------------------------


class TestServerProtocol:

    def setup_method(self) -> None:
        self.sandbox = cast(SandboxEnv, _FakeSandbox())
        _get_registry(self.sandbox).add(_build_mutbot_namespace())
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

        sandbox = cast(SandboxEnv, _FakeSandbox())
        _get_registry(sandbox).add(_build_mutbot_namespace())
        dispatch = _make_server_dispatch(sandbox)

        # 构造一个最小 conn 占位 —— build_peer_namespaces 只读 ns_name / state /
        # last_error，不调用 reconnect
        class _FakeConn:
            name = "mutbot_local"
            state = "connected"
            last_error = None
            sandbox = _sandbox_with_loop(
                cast(asyncio.AbstractEventLoop, object()),
            )  # ns_func 不会被调用就不需要

        conn = _FakeConn()
        client = _FakeHTTPClient(dispatch)
        init_result = {"capabilities": PYSANDBOX_CAPABILITY}

        namespaces = asyncio.run(
            build_peer_namespaces(conn, init_result, client))  # type: ignore[arg-type]
        assert len(namespaces) == 1
        ns = namespaces[0]
        assert ns.name == "mutbot"
        # D6: 描述带来源标记
        assert "(shared from mutbot_local)" in ns.description
        # 函数集齐
        assert set(ns.functions.keys()) == {"status", "logs", "boom"}
        # 与 conn 共享状态
        assert ns.connection is conn
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
        self.mcp = _FakeMCPClient(dispatch)
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

        sandbox = cast(SandboxEnv, _FakeSandbox())
        _get_registry(sandbox).add(_build_mutbot_namespace())
        dispatch = _make_server_dispatch(sandbox)

        fake_client = _FakeHTTPClientForConn(dispatch, capabilities, tools)

        def _fake_make_client(ns_name: str, cfg: dict):
            return fake_client

        monkeypatch.setattr(_mcp_impl, "make_client", _fake_make_client)

        # _do_rebuild 用 isinstance(client, HTTPMCPClient) 判定 peer 路径，
        # 临时把 HTTPMCPClient 也指向 _FakeHTTPClientForConn 的基类（用 monkeypatch
        # 的方式不行 —— isinstance 检查走 import 时的真名）。改用 patch 替换。
        monkeypatch.setattr(
            _mcp_impl, "HTTPMCPClient", _FakeHTTPClientForConn)
        # _mcp_impl_sandbox 也在内部 import 了 HTTPMCPClient
        monkeypatch.setattr(_mcp_impl_sandbox, "HTTPMCPClient", _FakeHTTPClientForConn,
                            raising=False)

        loop = asyncio.new_event_loop()
        try:
            conn = _mcp_impl.MCPConnection(
                "mutbot_remote", {"url": "http://x"})
            _impl(conn).sandbox = _sandbox_with_loop(loop)
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
            assert "echo" in conn.namespace.functions
            # peer 路径：mutbot namespace 融合进来
            assert len(_impl(conn).peer_namespaces) == 1
            assert _impl(conn).peer_namespaces[0].name == "mutbot"
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
            assert "pysandbox" not in conn.namespace.functions
            assert "other" in conn.namespace.functions
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
            assert _impl(conn).peer_namespaces == []
            # 没有 capability 时不过滤 pysandbox tool
            assert "pysandbox" in conn.namespace.functions
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
            peer = _impl(conn).peer_namespaces[0]
            assert peer.connection_state == "connected"
            # 模拟传输错
            _impl(conn).mark_disconnected("network down")
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

        sandbox = cast(SandboxEnv, _FakeSandbox())
        clash_ns = Namespace("mutbot_remote", description="clash")
        clash_ns.register("noop", lambda: None, "")
        _get_registry(sandbox).add(clash_ns)
        dispatch = _make_server_dispatch(sandbox)
        fake_client = _FakeHTTPClientForConn(
            dispatch, PYSANDBOX_CAPABILITY, [])

        monkeypatch.setattr(
            _mcp_impl, "make_client", lambda *a, **kw: fake_client)
        monkeypatch.setattr(
            _mcp_impl, "HTTPMCPClient", _FakeHTTPClientForConn)
        monkeypatch.setattr(_mcp_impl_sandbox, "HTTPMCPClient", _FakeHTTPClientForConn,
                            raising=False)

        loop = asyncio.new_event_loop()
        try:
            conn = _mcp_impl.MCPConnection(
                "mutbot_remote", {"url": "http://x"})
            _impl(conn).sandbox = _sandbox_with_loop(loop)
            # 不再抛错：conn 成功 connected，peer 列表含同名 ns
            loop.run_until_complete(conn.reconnect())
            assert conn.state == "connected"
            peer_names = [p.name for p in _impl(conn).peer_namespaces]
            assert "mutbot_remote" in peer_names
        finally:
            loop.run_until_complete(conn.close())
            loop.close()

    def test_d1_peer_self_duplicate_still_raises(
            self, monkeypatch: pytest.MonkeyPatch):
        """D1：同一 server 自我 export 两个同名 peer namespace 仍是 server bug，
        仍然报错。"""

        async def fake_build(conn, init_result, client):
            # 模拟 server 返回两个同名 peer ns
            return [Namespace("dup"), Namespace("dup")]

        sandbox = cast(SandboxEnv, _FakeSandbox())
        dispatch = _make_server_dispatch(sandbox)
        fake_client = _FakeHTTPClientForConn(
            dispatch, PYSANDBOX_CAPABILITY, [])

        monkeypatch.setattr(
            _mcp_impl, "make_client", lambda *a, **kw: fake_client)
        monkeypatch.setattr(
            _mcp_impl, "HTTPMCPClient", _FakeHTTPClientForConn)
        monkeypatch.setattr(_mcp_impl_sandbox, "HTTPMCPClient",
                            _FakeHTTPClientForConn, raising=False)
        monkeypatch.setattr(_mcp_impl_sandbox, "build_peer_namespaces",
                            fake_build)

        loop = asyncio.new_event_loop()
        try:
            conn = _mcp_impl.MCPConnection(
                "buggy_server", {"url": "http://x"})
            with pytest.raises(_mcp_impl.MCPTransportError,
                               match="peer-namespace duplicate"):
                loop.run_until_complete(conn.reconnect())
            # D11：异常后 state 必须是 failed，不能卡 connecting
            assert conn.state == "failed"
        finally:
            loop.run_until_complete(conn.close())
            loop.close()


# ============================================================
# _describe_function — params 字段（Phase 3）
# ============================================================

class TestDescribeFunctionParams:
    """验证 _mcp_share._describe_function 按设计方案输出 params 结构。"""

    def test_typical_function(self) -> None:

        def logs(level: str = "INFO", last_n: int = 10) -> list[str]:
            """Query logs."""
            return []

        entry = _describe_function(logs)
        assert entry["signature"] == "(level: str = 'INFO', last_n: int = 10) -> list[str]"
        assert "params" in entry
        params = entry["params"]
        assert [p["name"] for p in params] == ["level", "last_n"]
        assert params[0]["kind"] == "POSITIONAL_OR_KEYWORD"
        assert params[0]["default"] == "INFO"
        assert params[0]["annotation"] == "str"
        assert params[1]["default"] == 10
        assert params[1]["annotation"] == "int"

    def test_no_defaults(self) -> None:

        def add(a: int, b: int) -> int:
            return a + b

        entry = _describe_function(add)
        params = entry["params"]
        assert all("default" not in p for p in params)
        assert all(p["annotation"] == "int" for p in params)

    def test_no_annotations(self) -> None:

        def probe(x, y=1):
            return x

        entry = _describe_function(probe)
        params = entry["params"]
        assert all("annotation" not in p for p in params)
        assert "default" not in params[0]
        assert params[1]["default"] == 1

    def test_skips_var_positional_and_var_keyword(self) -> None:

        def variadic(a: int, *args, **kwargs) -> None:
            return None

        entry = _describe_function(variadic)
        names = [p["name"] for p in entry["params"]]
        assert names == ["a"]  # *args / **kwargs 被跳过

    def test_non_json_default_becomes_default_repr(self) -> None:


        sentinel = datetime(2020, 1, 1)

        def now_like(ts: datetime = sentinel) -> datetime:
            return ts

        entry = _describe_function(now_like)
        p = entry["params"][0]
        assert "default" not in p  # 非 JSON 原生 → 不回传真值
        assert "default_repr" in p
        assert "2020" in p["default_repr"]

    def test_missing_sentinel_becomes_default_missing_flag(self) -> None:

        def upload(paths=MISSING):
            return paths

        entry = _describe_function(upload)
        p = entry["params"][0]
        assert "default" not in p
        assert p["default_missing"] is True

    def test_keyword_only(self) -> None:

        def f(a: int, *, b: str = "x") -> None:
            return None

        entry = _describe_function(f)
        params = entry["params"]
        assert params[0]["kind"] == "POSITIONAL_OR_KEYWORD"
        assert params[1]["kind"] == "KEYWORD_ONLY"

    def test_unparseable_signature(self) -> None:

        # 某些 builtin 如 object.__init__ signature 不可解析
        entry = _describe_function(lambda: None)  # lambda 可解析
        assert "params" in entry

        # 用 builtin 触发 ValueError / TypeError
        try:
            entry2 = _describe_function(dict.fromkeys)
        except Exception:
            pytest.skip("dict.fromkeys signature parseable in this Python")
        # 如果解析失败，params 字段被省略
        if "params" not in entry2:
            assert entry2["signature"] == "(...)"

    def test_signature_field_preserved(self) -> None:
        """旧字段 signature 继续保留，供展示兜底。"""

        def f(a: int = 1) -> None:
            return None

        entry = _describe_function(f)
        assert entry["signature"].startswith("(")
        assert "a" in entry["signature"]
        assert entry["kwargs_schema"] == {}


# ============================================================
# _make_namespace_func — params 接入（Phase 4）
# ============================================================

class TestMakeNamespaceFuncSignature:
    """验证客户端 wrapper 按 describe 返回的 params 构造真签名，并兼容老 server。"""

    def _fake_conn(self):
        class _FakeConn:
            name = "peer"
            state = "connected"
            last_error = None
            sandbox = _sandbox_with_loop(
                cast(asyncio.AbstractEventLoop, object()),
            )  # 本组测试不触发调用
        return _FakeConn()

    def test_with_params_builds_signature(self) -> None:


        params = [
            {"name": "level", "kind": "POSITIONAL_OR_KEYWORD",
             "default": "INFO", "annotation": "str"},
            {"name": "last_n", "kind": "POSITIONAL_OR_KEYWORD",
             "default": 10, "annotation": "int"},
        ]
        # iter3: signature_str 形参已删除
        fn = _make_namespace_func(
            self._fake_conn(), "mutbot", "logs",  # type: ignore[arg-type]
            "Query logs.", params)
        sig = inspect.signature(fn)
        assert list(sig.parameters) == ["level", "last_n"]
        assert sig.parameters["level"].default == "INFO"
        assert sig.parameters["last_n"].default == 10
        # doc 不再被污染（没拼签名首行）
        assert fn.__doc__ == "Query logs."
        assert not fn.__doc__.startswith("logs(")

    def test_without_params_falls_back_to_kwargs(self) -> None:
        """老 server 不返回 params → 客户端保持 (**kwargs) 形态。"""


        fn = _make_namespace_func(
            self._fake_conn(), "mutbot", "logs",  # type: ignore[arg-type]
            "Query logs.", None)
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        assert len(params) == 1
        assert params[0].kind is inspect.Parameter.VAR_KEYWORD
        # doc 保持原样
        assert fn.__doc__ == "Query logs."

    def test_malformed_params_falls_back_to_kwargs(self) -> None:
        """params 畸形 → try_build_signature 返回 None → 回落 (**kwargs)。"""


        bad_params = [{"default": 1}]  # 缺 name
        fn = _make_namespace_func(
            self._fake_conn(), "mutbot", "weird",  # type: ignore[arg-type]
            "", bad_params)
        sig = inspect.signature(fn)
        assert len(list(sig.parameters)) == 1
        assert list(sig.parameters.values())[0].kind is inspect.Parameter.VAR_KEYWORD

    def test_async_original_preserved(self) -> None:

        params = [{"name": "x", "required": True, "annotation": "int"}]
        fn = _make_namespace_func(
            self._fake_conn(), "ns", "f",  # type: ignore[arg-type]
            "doc", params)
        assert callable(getattr(fn, "_async_original"))
        # iter3: _pysandbox_signature_str 不再写入
        assert not hasattr(fn, "_pysandbox_signature_str")


# ============================================================
# 端到端：describe 含 params → build_peer_namespaces → __signature__
# ============================================================


class TestPeerBuildWithParams:

    def test_peer_namespace_functions_carry_signature(self) -> None:
        """服务端真函数签名 → describe.params → 客户端 __signature__ 一致。"""


        sandbox = cast(SandboxEnv, _FakeSandbox())
        _get_registry(sandbox).add(_build_mutbot_namespace())
        dispatch = _make_server_dispatch(sandbox)

        class _FakeConn:
            name = "mutbot_local"
            state = "connected"
            last_error = None
            sandbox = _sandbox_with_loop(cast(asyncio.AbstractEventLoop, object()))

        conn = _FakeConn()
        client = _FakeHTTPClient(dispatch)
        init_result = {"capabilities": PYSANDBOX_CAPABILITY}

        namespaces = asyncio.run(
            build_peer_namespaces(conn, init_result, client))  # type: ignore[arg-type]
        ns = namespaces[0]

        # logs(level: str = "INFO", last_n: int = 10)
        logs = ns.functions["logs"]
        sig = inspect.signature(logs)
        assert list(sig.parameters) == ["level", "last_n"]
        assert sig.parameters["level"].default == "INFO"
        assert sig.parameters["last_n"].default == 10

        # status() — 无参
        status = ns.functions["status"]
        assert list(inspect.signature(status).parameters) == []

        # 原 bug 断言：help 形式（func.__doc__）不再以 "logs(" 开头
        assert not (logs.__doc__ or "").lstrip().startswith("logs(")

    def test_peer_namespace_optional_no_default_keeps_omit_signature(self) -> None:


        sandbox = cast(SandboxEnv, _FakeSandbox())
        ns = Namespace("mutbot", description="mutbot runtime introspection")

        def browser_file_upload(paths=MISSING) -> dict[str, Any]:
            return {"paths": paths}

        ns.register(
            "browser_file_upload",
            browser_file_upload,
            browser_file_upload.__doc__ or "",
        )
        _get_registry(sandbox).add(ns)
        dispatch = _make_server_dispatch(sandbox)

        class _FakeConn:
            name = "mutbot_local"
            state = "connected"
            last_error = None
            sandbox = _sandbox_with_loop(cast(asyncio.AbstractEventLoop, object()))

        conn = _FakeConn()
        client = _FakeHTTPClient(dispatch)
        init_result = {"capabilities": PYSANDBOX_CAPABILITY}

        namespaces = asyncio.run(
            build_peer_namespaces(conn, init_result, client))  # type: ignore[arg-type]
        peer_ns = namespaces[0]
        upload = peer_ns.functions["browser_file_upload"]
        sig = inspect.signature(upload)
        assert sig.parameters["paths"].default is MISSING
        # iter3: _MISSING.__repr__ → <omit>
        assert str(sig) == "(paths=<omit>)"


# ---------------------------------------------------------------------------
# _wrap_async 真签名 + 位置调用 + help() 单一签名
# ---------------------------------------------------------------------------


class TestWrapAsyncSignature:
    """验证 _wrap_async wrapper 携带正确 __signature__ 并支持位置调用。

    覆盖 SDD ``refactor-wrapper-faithful-signature.md`` Phase 3 缺失步骤：
    - wrapper.__signature__ 持有去除 self 的真签名
    - 位置调用通过 sig.bind().apply_defaults() 规范化
    - render_function 输出中签名仅出现一次（原 bug 终结）
    """

    def test_signature_with_positional_params(self) -> None:
        """带位置参数的 async 方法 → wrapper.__signature__ 去掉 self。"""

        app = SandboxEnv()

        async def method(self, a: str, b: int = 10) -> str:
            return f"{a}{b}"

        wrapper = _wrap_async(app, method)
        sig = inspect.signature(wrapper)
        params = list(sig.parameters.values())

        assert [p.name for p in params] == ["a", "b"]
        assert params[0].default is inspect.Parameter.empty  # a 必填
        assert params[1].default == 10
        assert sig.return_annotation == "str"
        assert wrapper._async_original is method

    def test_no_self_function(self) -> None:
        """无 self 的 async 函数 → 签名原样保留。"""

        app = SandboxEnv()

        async def standalone(x: int, *, y: str = "") -> None:
            ...

        wrapper = _wrap_async(app, standalone)
        sig = inspect.signature(wrapper)
        params = list(sig.parameters.values())

        assert [p.name for p in params] == ["x", "y"]
        assert params[0].default is inspect.Parameter.empty
        assert params[1].default == ""
        assert params[1].kind is inspect.Parameter.KEYWORD_ONLY

    def test_no_params_method(self) -> None:
        """无参 async 方法 → 空签名 ()"""

        app = SandboxEnv()

        async def method(self) -> str:
            return "ok"

        wrapper = _wrap_async(app, method)
        sig = inspect.signature(wrapper)

        assert list(sig.parameters) == []
        assert sig.return_annotation == "str"

    def test_unparseable_signature_no_signature_set(self) -> None:
        """签名不可解析 → __signature__ 不挂，回落 (*args, **kwargs)。"""

        app = SandboxEnv()

        # 构造 inspect.signature 抛 ValueError 的对象：挂无效 __signature__
        def fake_fn(a, b):
            pass
        # 设置无效的 __signature__，inspect.signature 会读它并抛 ValueError
        fake_fn.__signature__ = "not_a_signature"  # type: ignore[attr-defined]
        with pytest.raises((ValueError, TypeError)):
            inspect.signature(fake_fn)  # 确认确实不可解析

        wrapper = _wrap_async(app, fake_fn)
        sig = inspect.signature(wrapper)

        # 无真签名 → wrapper 回落为本身的 (*args, **kwargs)
        params = list(sig.parameters.values())
        assert params[0].kind is inspect.Parameter.VAR_POSITIONAL
        assert params[1].kind is inspect.Parameter.VAR_KEYWORD

    def test_positional_call(self) -> None:
        """位置调用 → sig.bind 规范化后正确传递给 coro_fn。"""

        app = SandboxEnv()
        # bind_main_loop 只在真执行时需要，这里只测 wrapper 参数规范化
        # 不实际执行 coroutine，用 mock 验证参数传递

        captured: list[str] = []

        async def method(self, a: str, b: int, c: bool = True):
            captured.append(f"{a}:{b}:{c}")

        wrapper = _wrap_async(app, method)

        # 触发 wrapper 内部逻辑：先用 sig.bind 验证参数是否可以正确 bind
        sig = inspect.signature(wrapper)
        bound = sig.bind("hello", 42)
        bound.apply_defaults()
        assert dict(bound.arguments) == {"a": "hello", "b": 42, "c": True}

        # 未知参数 TypeError
        with pytest.raises(TypeError, match="unexpected keyword"):
            sig.bind("hello", 42, unknown=True)

        # 缺少必填参数 TypeError
        with pytest.raises(TypeError, match="missing a required"):
            sig.bind()

    def test_render_function_single_signature(self) -> None:
        """render_function 对 wrapper 输出中签名字符串仅出现一次。

        原 bug：_make_namespace_func 把签名拼进 __doc__ 首行 + inspect.signature
        又展示一份 → 双签名。验证 render_function 只输出一份。
        """

        app = SandboxEnv()

        async def method(self, level: str = "INFO", last_n: int = 50) -> str:
            """查询日志。

            Args:
                level: 日志级别。
                last_n: 返回条数。

            Returns:
                格式化文本。
            """
            return "ok"

        wrapper = _wrap_async(app, method)
        wrapper.__name__ = "logs"
        output = render_function(wrapper, ns_name="test", fn_name="logs")

        # 签名字符串 "(level: " 应该恰好出现一次
        assert output.count("(level:") == 1, (
            f"signature should appear exactly once, got {output.count('(level:')}:"
            f"\n{output}"
        )
        # __doc__ 不应该包含以 "logs(" 开头的行
        doc = wrapper.__doc__ or ""
        assert not doc.lstrip().startswith("logs("), (
            f"__doc__ should not start with signature line, got: {doc!r}"
        )
        assert "test.logs(level: str = 'INFO', last_n: int = 50) -> str" in output
        assert "'str'" not in output
