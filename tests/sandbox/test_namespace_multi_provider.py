"""Multi-provider namespace 测试 — feature-namespace-multi-provider。

覆盖：
- ``NamespaceRegistry`` 多 provider 共存：add 不替换，get 返回 view
- ``MergedNamespaceView`` 函数解析「先注册先赢」+ shadowed 列表
- 冲突 WARNING 在 providers 签名内只触发一次，签名变化后重新触发一次
- ``_do_rebuild`` D11：peer-namespace duplicate / build 异常都把 state 翻 failed
- ``SandboxEnv`` add_namespace 同名不互相覆盖；remove_provider 按实例移除
- ``_render_namespace`` 多 provider 显示 Providers + 函数归属 + shadowed
"""

import asyncio
import logging
from typing import Any

import mutobj
import pytest

from mutagent.sandbox import SandboxEnv, _mcp_impl, _mcp_impl_sandbox
from mutagent.sandbox._mcp_share import _all_namespaces
from mutagent.sandbox._mcp_impl import MCPConnectionImpl
from mutagent.sandbox._env_impl import (
    SandboxEnvRuntime,
    _get_registry,
    sandbox_env_add_namespace,
    sandbox_env_remove_provider,
)
from mutagent.sandbox._namespace_impl import (
    MergedNamespaceView,
    Namespace,
    NamespaceRegistry,
    _render_namespace,
    _render_registry,
    displayed_of,
    flatten_view,
    primary_of,
)


# ---------------------------------------------------------------------------
# NamespaceRegistry multi-provider
# ---------------------------------------------------------------------------


class TestRegistryMultiProvider:

    def test_single_provider_returns_namespace(self):
        reg = NamespaceRegistry()
        ns = Namespace("foo")
        reg.add(ns)
        got = reg.get("foo")
        assert isinstance(got, Namespace)
        assert got is ns

    def test_two_providers_returns_view(self):
        reg = NamespaceRegistry()
        ns1 = Namespace("foo", provider_kind="tool")
        ns2 = Namespace("foo", provider_kind="peer")
        reg.add(ns1)
        reg.add(ns2)
        got = reg.get("foo")
        assert isinstance(got, MergedNamespaceView)
        assert got.providers == [ns1, ns2]

    def test_view_instance_stable_across_get(self):
        """view 必须 stable —— WARN-once 状态需要跨调用保留。"""
        reg = NamespaceRegistry()
        reg.add(Namespace("foo"))
        reg.add(Namespace("foo"))
        v1 = reg.get("foo")
        v2 = reg.get("foo")
        assert v1 is v2

    def test_remove_provider_by_instance(self):
        reg = NamespaceRegistry()
        ns1 = Namespace("foo")
        ns2 = Namespace("foo")
        reg.add(ns1)
        reg.add(ns2)
        assert reg.remove_provider(ns1) is True
        # 剩 1 个 provider，get 回到单 Namespace
        got = reg.get("foo")
        assert got is ns2

    def test_remove_provider_drops_key_when_empty(self):
        reg = NamespaceRegistry()
        ns = Namespace("foo")
        reg.add(ns)
        reg.remove_provider(ns)
        assert reg.get("foo") is None
        assert "foo" not in reg._namespaces

    def test_remove_by_name_drops_all_providers(self):
        reg = NamespaceRegistry()
        reg.add(Namespace("foo"))
        reg.add(Namespace("foo"))
        reg.remove("foo")
        assert reg.get("foo") is None


# ---------------------------------------------------------------------------
# MergedNamespaceView resolution
# ---------------------------------------------------------------------------


class TestMergedView:

    def _make(self, *kinds: str) -> tuple[NamespaceRegistry, list[Namespace]]:
        reg = NamespaceRegistry()
        nss: list[Namespace] = []
        for i, k in enumerate(kinds):
            ns = Namespace("foo", provider_kind=k)  # type: ignore[arg-type]
            ns.register(f"unique_{i}", lambda i=i: i, f"fn from {k}")
            nss.append(ns)
            reg.add(ns)
        return reg, nss

    def test_first_wins_on_conflict(self, caplog: pytest.LogCaptureFixture):
        reg = NamespaceRegistry()
        ns_a = Namespace("foo", provider_kind="tool")
        ns_a.register("logs", lambda: "A", "")
        ns_b = Namespace("foo", provider_kind="peer")
        ns_b.register("logs", lambda: "B", "")
        reg.add(ns_a)
        reg.add(ns_b)
        view = reg.get("foo")
        assert isinstance(view, MergedNamespaceView)

        with caplog.at_level(logging.WARNING):
            fn = view.logs
        # active = ns_a，函数返回 A
        assert fn() == "A"
        # WARN 含 active=tool 标签
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("logs" in r.message and "active=tool" in r.message
                   for r in warns)

    def test_warn_only_once_within_signature(
            self, caplog: pytest.LogCaptureFixture):
        reg = NamespaceRegistry()
        a = Namespace("foo")
        a.register("x", lambda: 1, "")
        b = Namespace("foo")
        b.register("x", lambda: 2, "")
        reg.add(a)
        reg.add(b)
        view = reg.get("foo")
        with caplog.at_level(logging.WARNING):
            view._resolved_functions()
            view._resolved_functions()
            view._resolved_functions()
        warns = [r for r in caplog.records
                 if r.levelno == logging.WARNING and "function 'x'" in r.message]
        assert len(warns) == 1

    def test_warn_re_emit_after_provider_change(
            self, caplog: pytest.LogCaptureFixture):
        reg = NamespaceRegistry()
        a = Namespace("foo")
        a.register("x", lambda: 1, "")
        b = Namespace("foo")
        b.register("x", lambda: 2, "")
        reg.add(a)
        reg.add(b)
        view = reg.get("foo")
        with caplog.at_level(logging.WARNING):
            view._resolved_functions()
        # 加第三个 provider 触发签名变化
        c = Namespace("foo")
        c.register("x", lambda: 3, "")
        reg.add(c)
        with caplog.at_level(logging.WARNING):
            view._resolved_functions()
        warns = [r for r in caplog.records
                 if r.levelno == logging.WARNING and "function 'x'" in r.message]
        assert len(warns) == 2

    def test_no_conflict_no_warning(self, caplog: pytest.LogCaptureFixture):
        reg = NamespaceRegistry()
        a = Namespace("foo")
        a.register("a", lambda: 1, "")
        b = Namespace("foo")
        b.register("b", lambda: 2, "")
        reg.add(a)
        reg.add(b)
        view = reg.get("foo")
        with caplog.at_level(logging.WARNING):
            view._resolved_functions()
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warns == []

    def test_view_functions_attribute_returns_unique_funcs(self):
        reg = NamespaceRegistry()
        a = Namespace("foo")
        a.register("x", lambda: 1, "")
        a.register("y", lambda: 2, "")
        b = Namespace("foo")
        b.register("y", lambda: 3, "")  # shadow
        b.register("z", lambda: 4, "")
        reg.add(a)
        reg.add(b)
        view = reg.get("foo")
        funcs = view._functions
        assert set(funcs.keys()) == {"x", "y", "z"}
        # y 的 active = a
        assert funcs["y"]() == 2

    def test_view_state_aggregation(self):
        reg = NamespaceRegistry()
        a = Namespace("foo")
        a.connection_state = "failed"
        a.connection_error = "boom"
        b = Namespace("foo")
        b.connection_state = "connected"
        reg.add(a)
        reg.add(b)
        view = reg.get("foo")
        # any connected -> connected
        assert view.connection_state == "connected"

    def test_view_state_connecting_when_no_connected(self):
        reg = NamespaceRegistry()
        a = Namespace("foo")
        a.connection_state = "failed"
        b = Namespace("foo")
        b.connection_state = "connecting"
        reg.add(a)
        reg.add(b)
        view = reg.get("foo")
        assert view.connection_state == "connecting"


# ---------------------------------------------------------------------------
# render multi-provider
# ---------------------------------------------------------------------------


class TestRenderMultiProvider:

    def test_render_registry_shows_provider_count_badge(self):
        reg = NamespaceRegistry()
        a = Namespace("foo", provider_kind="tool")
        a.register("x", lambda: 1, "x")
        b = Namespace("foo", provider_kind="peer")
        b.register("y", lambda: 2, "y")
        reg.add(a)
        reg.add(b)
        text = _render_registry(reg)
        assert "[2 providers]" in text

    def test_render_registry_filters_empty_providers(self):
        """空壳 provider（functions=0）被过滤，不进 badge 计数。"""
        reg = NamespaceRegistry()
        a = Namespace("foo", provider_kind="tool")  # 空壳
        b = Namespace("foo", provider_kind="peer")
        b.register("y", lambda: 2, "y")
        reg.add(a)
        reg.add(b)
        text = _render_registry(reg)
        # 只有 1 个 displayed provider → 不出 badge
        assert "providers]" not in text

    def test_render_namespace_lists_providers_and_origins(self):
        reg = NamespaceRegistry()
        a = Namespace("foo", provider_kind="tool")
        a.register("logs", lambda: "A", "tool's logs")
        b = Namespace("foo", provider_kind="peer")
        b.register("logs", lambda: "B", "peer's logs")
        b.register("status", lambda: "S", "status")
        reg.add(a)
        reg.add(b)
        view = reg.get("foo")
        text = _render_namespace(view)
        assert "Providers (2):" in text
        assert "kind=tool" in text and "kind=peer" in text
        # 归属用 #N 编号（引用 Providers 段）而非 kind#hex
        # tool 先注册 → #1；peer 后注册 → #2
        # logs：先注册先赢，active = tool(#1)，shadowed peer(#2)
        assert "[from #1]" in text
        assert "(shadowed: #2)" in text
        # status 只来自 peer(#2)，无 shadowed
        assert "status" in text
        # 不再出现旧的 kind#hex 样式
        assert "from tool#" not in text
        assert "from peer#" not in text

    def test_render_namespace_filters_empty_provider_falls_back_to_single(self):
        """一个空壳 + 一个有函数 → 退化为单 provider 路径，无 [from ...] 标签。

        类似 mutbot 场景：self tool ns 被 D2 过滤后为空壳，peer ns 提供全部函数。
        """
        reg = NamespaceRegistry()
        empty = Namespace("mutbot", provider_kind="tool")  # 空壳
        peer = Namespace("mutbot", provider_kind="peer")
        peer.register("logs", lambda: 1, "logs doc")
        peer.register("status", lambda: 2, "status doc")
        reg.add(empty)
        reg.add(peer)
        text = _render_namespace(reg.get("mutbot"))
        assert "Providers" not in text   # 不走 multi-provider 分支
        assert "[from" not in text       # 函数行不带归属标签
        assert "functions=0" not in text  # 空壳不出现
        assert "logs" in text and "status" in text

    def test_render_single_provider_unchanged(self):
        """单 provider 路径必须与改造前一致 — 不出现 Providers 段。"""
        reg = NamespaceRegistry()
        ns = Namespace("foo", description="desc")
        ns.register("x", lambda: 1, "do x")
        reg.add(ns)
        text = _render_namespace(reg.get("foo"))
        assert "Providers" not in text
        assert "do x" in text


# ---------------------------------------------------------------------------
# SandboxEnv 集成（add/remove provider）
# ---------------------------------------------------------------------------


class TestSandboxEnvMultiProvider:

    def test_same_name_add_does_not_replace(self):
        sandbox = SandboxEnv()
        a = Namespace("foo", provider_kind="tool")
        a.register("x", lambda: "A", "")
        b = Namespace("foo", provider_kind="peer")
        b.register("x", lambda: "B", "")
        sandbox_env_add_namespace(sandbox, a)
        sandbox_env_add_namespace(sandbox, b)
        # _registry 下应有 2 providers
        assert len(_get_registry(sandbox)._namespaces["foo"]) == 2

    def test_remove_provider_by_instance(self):
        sandbox = SandboxEnv()
        a = Namespace("foo")
        b = Namespace("foo")
        cleanups = []
        sandbox_env_add_namespace(sandbox, a, on_remove=lambda: cleanups.append("a"))
        sandbox_env_add_namespace(sandbox, b, on_remove=lambda: cleanups.append("b"))
        # 仅移除 a
        ok = sandbox_env_remove_provider(sandbox, a)
        assert ok is True
        assert _get_registry(sandbox)._namespaces["foo"] == [b]
        # a 的 cleanup 被调用，b 未被触发
        assert cleanups == ["a"]

    def test_exec_code_sees_view_for_multi_provider(self):
        sandbox = SandboxEnv()
        a = Namespace("foo")
        a.register("aa", lambda: "from-a", "")
        b = Namespace("foo")
        b.register("bb", lambda: "from-b", "")
        sandbox_env_add_namespace(sandbox, a)
        sandbox_env_add_namespace(sandbox, b)
        result = sandbox.exec_code("foo.aa() + '|' + foo.bb()")
        assert result.get("result") == "from-a|from-b"


# ---------------------------------------------------------------------------
# D11 — _do_rebuild 异常路径完整性
# ---------------------------------------------------------------------------


class TestDoRebuildExceptionCompleteness:

    def _setup_loop(self):
        return asyncio.new_event_loop()

    def test_peer_duplicate_lands_in_failed_state(
            self, monkeypatch: pytest.MonkeyPatch):
        """peer-namespace duplicate 异常发生在握手后，state 必须是 failed 而非 connecting。"""

        class _FakeClient:
            async def connect(self):
                # 声明 pysandbox capability，让 build_peer_namespaces 被触发
                return {"capabilities": {"pysandbox": {"version": "1"}}}

            async def list_tools(self):
                return []

            async def close(self):
                pass

        async def fake_build(conn, init_result, client):
            return [Namespace("dup"), Namespace("dup")]

        fake = _FakeClient()
        monkeypatch.setattr(
            _mcp_impl, "make_client", lambda *a, **kw: fake)
        monkeypatch.setattr(
            _mcp_impl, "HTTPMCPClient", _FakeClient)
        monkeypatch.setattr(_mcp_impl_sandbox, "HTTPMCPClient",
                            _FakeClient, raising=False)
        monkeypatch.setattr(_mcp_impl_sandbox, "build_peer_namespaces",
                            fake_build)

        loop = self._setup_loop()
        try:
            conn = _mcp_impl.MCPConnection(
                "x", {"url": "http://x"})
            with pytest.raises(_mcp_impl.MCPTransportError):
                loop.run_until_complete(conn.reconnect())
            assert conn.state == "failed", \
                f"state stuck at {conn.state!r}, should be 'failed'"
            assert conn.last_error is not None
        finally:
            loop.run_until_complete(conn.close())
            loop.close()

    def test_arbitrary_runtime_error_lands_in_failed_state(
            self, monkeypatch: pytest.MonkeyPatch):
        """任意 RuntimeError（非 transport）也必须翻 failed，不能卡 connecting。"""

        class _FakeClient:
            async def connect(self):
                return {"capabilities": {"pysandbox": {"version": "1"}}}

            async def list_tools(self):
                return []

            async def close(self):
                pass

        async def boom(*a, **kw):
            raise RuntimeError("simulated bug in peer build")

        fake = _FakeClient()
        monkeypatch.setattr(
            _mcp_impl, "make_client", lambda *a, **kw: fake)
        monkeypatch.setattr(
            _mcp_impl, "HTTPMCPClient", _FakeClient)
        monkeypatch.setattr(_mcp_impl_sandbox, "HTTPMCPClient",
                            _FakeClient, raising=False)
        monkeypatch.setattr(_mcp_impl_sandbox, "build_peer_namespaces", boom)

        loop = self._setup_loop()
        try:
            conn = _mcp_impl.MCPConnection(
                "x", {"url": "http://x"})
            with pytest.raises(_mcp_impl.MCPTransportError):
                loop.run_until_complete(conn.reconnect())
            assert conn.state == "failed"
            assert "simulated bug" in (conn.last_error or "")
        finally:
            loop.run_until_complete(conn.close())
            loop.close()


# ---------------------------------------------------------------------------
# refactor-namespace-provider-selection — displayed / primary / flatten
# ---------------------------------------------------------------------------


class TestDisplayedAndPrimary:
    """验证 ``MergedNamespaceView.displayed`` / ``primary`` 三种形态。"""

    def test_normal_view_primary_is_first_displayed(self):
        reg = NamespaceRegistry()
        a = Namespace("foo", provider_kind="tool", description="A")
        a.register("x", lambda: 1, "")
        b = Namespace("foo", provider_kind="peer", description="B")
        b.register("y", lambda: 2, "")
        reg.add(a)
        reg.add(b)
        view = reg.get("foo")
        assert isinstance(view, MergedNamespaceView)
        assert view.displayed == [a, b]
        assert view.primary is a
        assert view._description == "A"

    def test_empty_shell_view_primary_falls_back(self):
        """全是空壳 provider 时，displayed=[]，primary 退化为 _providers[0]。"""
        reg = NamespaceRegistry()
        a = Namespace("foo", description="A-desc")  # 空壳
        b = Namespace("foo", description="B-desc")  # 空壳
        reg.add(a)
        reg.add(b)
        view = reg.get("foo")
        assert view.displayed == []
        assert view.primary is a              # 首个 provider
        assert view._description == "A-desc"  # primary._description

    def test_shell_skipped_in_displayed_and_primary(self):
        """一个空壳 + 一个有函数 → displayed 只含后者，primary 是后者。"""
        reg = NamespaceRegistry()
        empty = Namespace("foo", provider_kind="tool", description="empty")
        full = Namespace("foo", provider_kind="peer", description="full")
        full.register("x", lambda: 1, "")
        reg.add(empty)
        reg.add(full)
        view = reg.get("foo")
        assert view.displayed == [full]
        assert view.primary is full
        assert view._description == "full"

    def test_fully_shadowed_provider_not_in_displayed(self):
        """全被 shadow 的 provider 不进 displayed，primary 是赢家。"""
        reg = NamespaceRegistry()
        a = Namespace("foo", description="A")
        a.register("only", lambda: "a", "")
        b = Namespace("foo", description="B")
        b.register("only", lambda: "b", "")  # 被 a shadow
        reg.add(a)
        reg.add(b)
        view = reg.get("foo")
        # b 的唯一函数被 a shadow，b 不在 displayed
        assert view.displayed == [a]
        assert view.primary is a

    def test_primary_of_returns_namespace_for_plain_ns(self):
        ns = Namespace("foo", description="plain")
        assert primary_of(ns) is ns
        assert displayed_of(ns) == []

    def test_primary_of_returns_view_primary(self):
        reg = NamespaceRegistry()
        a = Namespace("foo", description="A")
        a.register("x", lambda: 1, "")
        b = Namespace("foo", description="B")
        b.register("y", lambda: 2, "")
        reg.add(a)
        reg.add(b)
        view = reg.get("foo")
        assert primary_of(view) is a
        assert displayed_of(view) == [a, b]


class TestFlattenView:
    """验证 ``flatten_view`` 拍平后不丢函数。"""

    def test_flatten_merges_active_function_set(self):
        reg = NamespaceRegistry()
        a = Namespace("foo", provider_kind="tool", description="A-desc")
        a.register("shared", lambda: "a-wins", "a doc")
        a.register("only_a", lambda: "a-only", "")
        b = Namespace("foo", provider_kind="peer", description="B-desc")
        b.register("shared", lambda: "b-loses", "b doc")  # shadowed
        b.register("only_b", lambda: "b-only", "")
        reg.add(a)
        reg.add(b)
        view = reg.get("foo")
        flat = flatten_view(view)
        assert isinstance(flat, Namespace)
        # description / kind 走 primary (= a)
        assert flat._description == "A-desc"
        assert flat.provider_kind == "tool"
        # 函数集 = active 合并：shared (a) + only_a + only_b
        assert set(flat._functions.keys()) == {"shared", "only_a", "only_b"}
        assert flat._functions["shared"]() == "a-wins"
        assert flat._functions["only_b"]() == "b-only"
        # description 也拿到了 active provider 的
        assert flat._descriptions["shared"] == "a doc"

    def test_flatten_does_not_carry_connection(self):
        """拍平后的临时 Namespace 不挂 ``_connection``/state，对端看不到连接状态。"""
        reg = NamespaceRegistry()
        a = Namespace("foo")
        a.register("x", lambda: 1, "")
        a.connection_state = "connected"
        b = Namespace("foo")
        b.register("y", lambda: 2, "")
        reg.add(a)
        reg.add(b)
        flat = flatten_view(reg.get("foo"))
        assert flat._connection is None
        assert flat.connection_state is None


class TestAllNamespacesFlatten:
    """验证 ``_mcp_share._all_namespaces`` 拍平后与 exec_code 函数集一致。"""

    def test_multi_provider_export_function_set_matches_exec_code(self):
        """同名多 provider 时，_all_namespaces 拍平后的函数集 == exec_code 看到的。

        这是原设计文档针对的 bug 修复：旧实现用 dict.update 让 decl 整个
        覆盖 external，导致 export 丢了 external 的非冲突函数。
        """

        sandbox = SandboxEnv()
        # external provider（模拟 peer）
        ext = Namespace("foo", provider_kind="peer", description="ext-desc")
        ext.register("shared", lambda: "ext", "")
        ext.register("only_ext", lambda: "ext-only", "")
        sandbox_env_add_namespace(sandbox, ext)
        # 再加一个同名 external（模拟 decl-like 的优先 provider）
        local = Namespace("foo", provider_kind="builtin", description="local-desc")
        local.register("shared", lambda: "local", "local doc")
        local.register("only_local", lambda: "local-only", "")
        sandbox_env_add_namespace(sandbox, local)

        result = _all_namespaces(sandbox)
        assert "foo" in result
        flat = result["foo"]
        # 三个函数都在：shared (先注册先赢 = ext)、only_ext、only_local
        assert set(flat._functions.keys()) == {"shared", "only_ext", "only_local"}
        # 先注册先赢：ext 先 add，赢下 shared
        assert flat._functions["shared"]() == "ext"
        # description 走 primary（= ext）
        assert flat._description == "ext-desc"

        # 同一 SandboxEnv 走 exec_code 能看到同一集（验证不丢函数）
        r = sandbox.exec_code("sorted(foo._functions.keys())")
        assert r.get("result") == ["only_ext", "only_local", "shared"]

    def test_single_provider_returned_as_is(self):

        sandbox = SandboxEnv()
        ns = Namespace("foo", description="single")
        ns.register("x", lambda: 1, "")
        sandbox_env_add_namespace(sandbox, ns)
        result = _all_namespaces(sandbox)
        # 单 provider 名直接返回原 Namespace（不拍平）
        assert result["foo"] is ns


def _impl(conn):
    """Resolve MCPConnection → implementation."""
    impl = mutobj.implementation_of(conn, MCPConnectionImpl)
    return impl


class TestRefreshNamespaceInvalidatesView:
    """验证 ``_mcp_impl._refresh_namespace`` 后 view cache 失效。

    原隐藏 bug：provider 列表 id 不变但 ns._functions 变动，
    view.displayed / primary / _description 会拿到旧结果。
    """

    def test_refresh_namespace_invalidates_view_cache(self):

        sandbox = SandboxEnv()
        loop = asyncio.new_event_loop()
        try:
            SandboxEnvRuntime.get_or_create(sandbox).async_loop = loop
            conn = _mcp_impl.MCPConnection(
                "mysrv", {"transport": "http", "url": "http://x"})
            # 添加两个同名 provider，打出 MergedNamespaceView
            other = Namespace("mysrv", provider_kind="peer",
                              description="peer-desc")
            # 必须先加一个函数让 other 变为 displayed，否则空壳 view 退化路径
            # 会掩盖 cache 失效问题。
            other.register("placeholder", lambda: 0, "")
            sandbox_env_add_namespace(sandbox, conn.namespace)
            _impl(conn)._sandbox = sandbox
            sandbox_env_add_namespace(sandbox, other)
            view = _get_registry(sandbox).get("mysrv")
            assert isinstance(view, MergedNamespaceView)
            # 初始：conn.namespace 空壳 → displayed = [other]，primary = other
            assert view.displayed == [other]
            assert view.primary is other
            assert view._description == "peer-desc"

            # 模拟 _refresh_namespace：给 conn.namespace 填函数 + 描述
            init_result = {
                "instructions": "tool-desc",
                "serverInfo": {"title": "t"},
            }
            tools = [
                {"name": "do_something", "description": "do it",
                 "inputSchema": {}},
            ]
            _impl(conn)._refresh_namespace(init_result, tools)

            # 修正后：view.displayed 应立即看到 impl namespace 加入
            # （ns 先注册，位于 providers[0]，应为 primary）
            assert conn.namespace in view.displayed
            assert view.primary is conn.namespace
            # _description 走 primary = ns.description = "tool-desc"
            assert view._description == "tool-desc"
        finally:
            loop.close()
