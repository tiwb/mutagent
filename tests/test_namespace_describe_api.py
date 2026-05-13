"""SandboxApp.iter_namespaces / get_namespace + _collect_namespaces 单测。

覆盖 `refactor-namespace-describe-api.md` R1 / R2 的行为：
- `_collect_namespaces` 的 decl + external 合并策略、单/多 provider 返回类型、空 registry
- `iter_namespaces` 顺序稳定、返回类型随 provider 数切换
- `get_namespace` 存在/不存在路径
- exec_code 与 share 协议的 namespace 可见集严格一致
"""
from __future__ import annotations

from typing import Any

import pytest

from mutagent.sandbox._app_impl import _collect_namespaces
from mutagent.sandbox._namespace import (
    MergedNamespaceView,
    Namespace,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_ns(name: str, *, kind: str = "tool",
             functions: dict[str, Any] | None = None,
             desc: str = "") -> Namespace:
    ns = Namespace(name, description=desc, provider_kind=kind)  # type: ignore[arg-type]
    for fn_name, fn in (functions or {}).items():
        ns.register(fn_name, fn, fn.__doc__ or "")
    return ns


def _external_names(sandbox: Any) -> set[str]:
    """抽出 sandbox 上「非 NamespaceTools decl 发现」的外部注入 namespace 名。"""
    return set(sandbox._registry._namespaces.keys())


# ---------------------------------------------------------------------------
# _collect_namespaces
# ---------------------------------------------------------------------------

class TestCollectNamespaces:
    """`_collect_namespaces` 合并策略 / 返回类型 / 边界。"""

    def test_empty_registry_returns_only_decl(self):
        """空 registry：返回仅包含 NamespaceTools decl 发现的 namespace。"""
        from mutagent.sandbox.app import SandboxApp
        app = SandboxApp()

        result = _collect_namespaces(app)

        # 所有返回值必须是 Namespace 或 MergedNamespaceView
        assert all(isinstance(v, (Namespace, MergedNamespaceView))
                   for v in result.values())
        # 不含外部注入（因为从未 add_namespace）
        assert _external_names(app) == set()

    def test_single_external_provider_returns_namespace(self):
        """单 provider 名：返回原 Namespace 实例（非 view）。"""
        from mutagent.sandbox.app import SandboxApp
        app = SandboxApp()
        ns = _make_ns("uniq_single", functions={"ping": lambda: "ok"})
        app.add_namespace(ns)

        result = _collect_namespaces(app)

        assert "uniq_single" in result
        # 单 provider → 原 Namespace 实例，不经 view 包装
        assert result["uniq_single"] is ns
        assert isinstance(result["uniq_single"], Namespace)

    def test_multi_provider_returns_merged_view(self):
        """同名 2 providers：返回 MergedNamespaceView。"""
        from mutagent.sandbox.app import SandboxApp
        app = SandboxApp()
        a = _make_ns("uniq_multi", functions={"aa": lambda: "A"})
        b = _make_ns("uniq_multi", functions={"bb": lambda: "B"})
        app.add_namespace(a)
        app.add_namespace(b)

        result = _collect_namespaces(app)

        assert isinstance(result["uniq_multi"], MergedNamespaceView)
        # 合并后两个函数都可见
        funcs = result["uniq_multi"]._functions
        assert "aa" in funcs and "bb" in funcs

    def test_multi_provider_first_wins(self):
        """同名 provider 函数冲突：先注册先赢。"""
        from mutagent.sandbox.app import SandboxApp
        app = SandboxApp()
        first = _make_ns("uniq_conflict",
                         functions={"shared": lambda: "first"})
        second = _make_ns("uniq_conflict",
                          functions={"shared": lambda: "second"})
        app.add_namespace(first)
        app.add_namespace(second)

        view = _collect_namespaces(app)["uniq_conflict"]
        assert isinstance(view, MergedNamespaceView)
        # first provider 胜出
        assert view._functions["shared"]() == "first"

    def test_exec_and_share_see_same_names(self):
        """exec_code 路径（_build_namespace_dict）与 share 路径（_all_namespaces）
        看到的 namespace name 集合必须严格一致。"""
        from mutagent.sandbox._app_impl import _build_namespace_dict
        from mutagent.sandbox.app import SandboxApp
        from mutagent.sandbox.share import _all_namespaces
        app = SandboxApp()
        app.add_namespace(_make_ns("uniq_alpha",
                                   functions={"f": lambda: 1}))
        app.add_namespace(_make_ns("uniq_beta",
                                   functions={"g": lambda: 2}))
        # 两端看到的 name 集合（排除 exec 路径的 help 键）
        exec_names = {k for k in _build_namespace_dict(app) if k != "help"}
        share_names = set(_all_namespaces(app).keys())
        assert exec_names == share_names
        # 且必然包含外部注入的两个
        assert {"uniq_alpha", "uniq_beta"} <= exec_names


# ---------------------------------------------------------------------------
# SandboxApp.iter_namespaces / get_namespace
# ---------------------------------------------------------------------------

class TestIterNamespaces:

    def test_iter_sorted_by_name(self):
        from mutagent.sandbox.app import SandboxApp
        app = SandboxApp()
        app.add_namespace(_make_ns("zzz_last",
                                   functions={"f": lambda: 1}))
        app.add_namespace(_make_ns("aaa_first",
                                   functions={"f": lambda: 1}))
        app.add_namespace(_make_ns("mmm_mid",
                                   functions={"f": lambda: 1}))
        names = [ns._name for ns in app.iter_namespaces()]
        # 提取我们添加的三个，验证相对顺序（其他 decl namespace 可能夹杂）
        ours = [n for n in names if n in ("aaa_first", "mmm_mid", "zzz_last")]
        assert ours == ["aaa_first", "mmm_mid", "zzz_last"]

    def test_iter_returns_namespace_for_single_provider(self):
        from mutagent.sandbox.app import SandboxApp
        app = SandboxApp()
        ns = _make_ns("uniq_single_iter", functions={"f": lambda: 1})
        app.add_namespace(ns)
        matched = [x for x in app.iter_namespaces()
                   if x._name == "uniq_single_iter"]
        assert len(matched) == 1
        assert isinstance(matched[0], Namespace)
        assert matched[0] is ns

    def test_iter_returns_merged_view_for_multi_provider(self):
        from mutagent.sandbox.app import SandboxApp
        app = SandboxApp()
        app.add_namespace(_make_ns("uniq_multi_iter",
                                   functions={"a": lambda: 1}))
        app.add_namespace(_make_ns("uniq_multi_iter",
                                   functions={"b": lambda: 2}))
        matched = [x for x in app.iter_namespaces()
                   if x._name == "uniq_multi_iter"]
        assert len(matched) == 1
        assert isinstance(matched[0], MergedNamespaceView)


class TestGetNamespace:

    def test_returns_none_for_missing_name(self):
        from mutagent.sandbox.app import SandboxApp
        app = SandboxApp()
        assert app.get_namespace("__definitely_not_registered__") is None

    def test_returns_none_for_help_key(self):
        """'help' 是注入沙箱的特殊键，不应被 get_namespace 暴露。"""
        from mutagent.sandbox.app import SandboxApp
        app = SandboxApp()
        assert app.get_namespace("help") is None

    def test_returns_namespace_for_single_provider(self):
        from mutagent.sandbox.app import SandboxApp
        app = SandboxApp()
        ns = _make_ns("uniq_get_single", functions={"f": lambda: 1})
        app.add_namespace(ns)
        got = app.get_namespace("uniq_get_single")
        assert isinstance(got, Namespace)
        assert got is ns

    def test_returns_merged_view_for_multi_provider(self):
        from mutagent.sandbox.app import SandboxApp
        app = SandboxApp()
        app.add_namespace(_make_ns("uniq_get_multi",
                                   functions={"a": lambda: 1}))
        app.add_namespace(_make_ns("uniq_get_multi",
                                   functions={"b": lambda: 2}))
        got = app.get_namespace("uniq_get_multi")
        assert isinstance(got, MergedNamespaceView)

    def test_consistent_with_iter(self):
        """get_namespace 与 iter_namespaces 来自同一可见集。"""
        from mutagent.sandbox.app import SandboxApp
        app = SandboxApp()
        app.add_namespace(_make_ns("uniq_consistency",
                                   functions={"f": lambda: 1}))
        name_list = [ns._name for ns in app.iter_namespaces()]
        for name in name_list:
            got = app.get_namespace(name)
            assert got is not None
            assert got._name == name


# ---------------------------------------------------------------------------
# sandbox-bound help()
# ---------------------------------------------------------------------------

class TestSandboxBoundHelp:
    """验证 help() 数据源切换到 sandbox 视角后仍保持既有文本格式。"""

    def test_help_no_arg_lists_external(self):
        from mutagent.sandbox.app import SandboxApp
        app = SandboxApp()
        app.add_namespace(_make_ns("uniq_help_list",
                                   desc="demo",
                                   functions={"do": lambda: 1}))
        help_fn = app.exec_code("_r = help()")["state"]["_r"] if False else None  # noqa
        # 用 _build_namespace_dict 直接拿 help
        from mutagent.sandbox._app_impl import _build_namespace_dict
        ns_dict = _build_namespace_dict(app)
        help_fn = ns_dict["help"]
        text = help_fn()
        assert "Available namespaces" in text
        assert "uniq_help_list" in text
        assert "demo" in text
        assert "(1 functions)" in text

    def test_help_string_lookup(self):
        from mutagent.sandbox._app_impl import _build_namespace_dict
        from mutagent.sandbox.app import SandboxApp

        def do() -> str:
            """demo doc line."""
            return "ok"

        app = SandboxApp()
        app.add_namespace(_make_ns("uniq_help_string",
                                   functions={"do": do}))
        help_fn = _build_namespace_dict(app)["help"]
        text = help_fn("uniq_help_string.do")
        assert "uniq_help_string.do" in text
        assert "demo doc line" in text

    def test_help_missing_string_returns_no_documentation(self):
        from mutagent.sandbox._app_impl import _build_namespace_dict
        from mutagent.sandbox.app import SandboxApp
        app = SandboxApp()
        help_fn = _build_namespace_dict(app)["help"]
        assert "no documentation" in help_fn("nonexistent.func")

    def test_help_multi_provider_shows_merged_view(self):
        from mutagent.sandbox._app_impl import _build_namespace_dict
        from mutagent.sandbox.app import SandboxApp
        app = SandboxApp()
        app.add_namespace(_make_ns("uniq_help_merged",
                                   functions={"a": lambda: 1}))
        app.add_namespace(_make_ns("uniq_help_merged",
                                   functions={"b": lambda: 2}))
        help_fn = _build_namespace_dict(app)["help"]
        text = help_fn()
        # Layer 1 列表里能看到 merged ns
        assert "uniq_help_merged" in text
        # 多 provider badge（2 providers）
        assert "[2 providers]" in text


# ---------------------------------------------------------------------------
# connection_status 纯函数 — R3
# ---------------------------------------------------------------------------

class TestConnectionStatus:
    """验证 ``connection_status`` 与文本/UI 渲染层的共享泛型。"""

    def _make_ns_with_state(self, state: str | None, error: str = "",
                             has_conn: bool = True) -> Namespace:
        ns = Namespace("state_probe")
        ns.connection_state = state
        ns.connection_error = error
        if has_conn:
            # 任意非 None 值即可（connection_status 只检查 is None）
            ns._connection = object()  # type: ignore[assignment]
        return ns

    def test_non_mcp_namespace_returns_none_none(self):
        from mutagent.sandbox._namespace import connection_status
        ns = self._make_ns_with_state(None, has_conn=False)
        assert connection_status(ns) == (None, None)

    def test_connected_returns_state_none_reason(self):
        from mutagent.sandbox._namespace import connection_status
        ns = self._make_ns_with_state("connected")
        assert connection_status(ns) == ("connected", None)

    def test_connecting_returns_state_none_reason(self):
        from mutagent.sandbox._namespace import connection_status
        ns = self._make_ns_with_state("connecting")
        assert connection_status(ns) == ("connecting", None)

    def test_disconnected_returns_state_none_reason(self):
        from mutagent.sandbox._namespace import connection_status
        ns = self._make_ns_with_state("disconnected")
        assert connection_status(ns) == ("disconnected", None)

    def test_failed_without_error_returns_state_none_reason(self):
        from mutagent.sandbox._namespace import connection_status
        ns = self._make_ns_with_state("failed", error="")
        assert connection_status(ns) == ("failed", None)

    def test_failed_short_error_returned_verbatim(self):
        from mutagent.sandbox._namespace import connection_status
        ns = self._make_ns_with_state("failed", error="ECONNREFUSED")
        state, reason = connection_status(ns)
        assert state == "failed"
        assert reason == "ECONNREFUSED"

    def test_failed_takes_first_line_only(self):
        from mutagent.sandbox._namespace import connection_status
        err = "first line\nsecond line should be ignored\nthird"
        ns = self._make_ns_with_state("failed", error=err)
        _, reason = connection_status(ns)
        assert reason == "first line"

    def test_failed_long_error_truncated_to_60(self):
        from mutagent.sandbox._namespace import connection_status
        err = "x" * 200
        ns = self._make_ns_with_state("failed", error=err)
        _, reason = connection_status(ns)
        assert reason is not None
        assert len(reason) == 60
        assert reason.endswith("...")

    def test_format_state_label_accepts_ns(self):
        """新签名兼容：直接传 ns。"""
        from mutagent.sandbox._namespace import _format_state_label
        ns = self._make_ns_with_state("failed", error="boom")
        assert _format_state_label(ns) == "[failed: boom]"

    def test_format_state_label_legacy_signature(self):
        """旧签名 ``(state, error)`` 保持可用——旧消费者不破。"""
        from mutagent.sandbox._namespace import _format_state_label
        assert _format_state_label("failed", "boom") == "[failed: boom]"
        assert _format_state_label("connecting", None) == "[connecting...]"
        assert _format_state_label(None, None) == ""

    def test_text_and_ui_truncation_consistent(self):
        """文本端与 UI 端在同一长 reason 上的 reason 内容等价。"""
        from mutagent.sandbox._namespace import (
            _format_state_label,
            connection_status,
        )
        err = "a" * 100
        ns = self._make_ns_with_state("failed", error=err)
        _, reason_ns = connection_status(ns)
        text_label = _format_state_label(ns)
        # text_label 格式 '[failed: <reason>]'，提出 reason
        assert text_label.startswith("[failed: ")
        assert text_label.endswith("]")
        reason_from_text = text_label[len("[failed: "):-1]
        assert reason_from_text == reason_ns
