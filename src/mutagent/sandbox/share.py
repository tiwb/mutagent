"""Pysandbox namespace sharing — server side.

把一个 ``SandboxApp`` 的 namespace registry 通过 MCP 信道分享给
对端 pysandbox（典型场景：mutbot 把自身能力 share 给 mutagent）。

设计文档：
``mutagent/docs/specifications/feature-pysandbox-namespace-sharing.md``

mutio 不感知 pysandbox 概念，只通过 ``MCPView.extra_capabilities`` /
``MCPView.register_extra_methods`` 钩子接入。本模块提供两个工具：

- :data:`PYSANDBOX_CAPABILITY` — 顶层 capability 字段
- :func:`register_pysandbox_methods` — 在 dispatcher 上注册 3 个扩展方法
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import TYPE_CHECKING, Any

from mutio.mcp.protocol import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    JsonRpcDispatcher,
    JsonRpcError,
)

if TYPE_CHECKING:
    from mutagent.sandbox._namespace import Namespace
    from mutagent.sandbox.app import SandboxApp

logger = logging.getLogger(__name__)


# 协议版本（client 通过 ``capabilities.pysandbox.version`` 检查兼容性）
PYSANDBOX_CAPABILITY: dict[str, Any] = {"pysandbox": {"version": "1"}}


# ---------------------------------------------------------------------------
# 内部 helper
# ---------------------------------------------------------------------------

def _all_namespaces(sandbox: "SandboxApp") -> dict[str, "Namespace"]:
    """收集 sandbox 当前可见的全部 namespace（拍平成单 provider）。

    包含两类来源：
    1. 外部注入（``add_namespace``）—— 走 ``_registry._namespaces``
    2. NamespaceTools Declaration 子类自动发现 —— 走 ``_build_declaration_namespaces``

    合并策略与 ``_build_namespace_dict``（exec_code 路径）严格一致：
    用同一个 temp_registry 按 decl 先 + external 后的顺序装入，
    同名多 provider 时拍平成单 :class:`Namespace`：

    - ``description`` / ``provider_kind`` 取 :func:`primary_of`
    - ``functions`` 集 = view 合并后的 active 集，不丢 external 的非冲突函数

    这保证 export 函数集与本地 ``exec_code`` 可见函数集严格一致。
    """
    # 复用 _app_impl 的 declaration 发现逻辑，避免重复实现
    from mutagent.sandbox._app_impl import _build_declaration_namespaces
    from mutagent.sandbox._namespace import (
        MergedNamespaceView,
        NamespaceRegistry,
        flatten_view,
    )

    decl_namespaces = _build_declaration_namespaces(sandbox)
    registry = getattr(sandbox, "_registry", None)

    temp_registry = NamespaceRegistry()
    # decl 先注册 → 本地 NamespaceTools 优先于外部 peer（与 exec_code 同序）
    for ns in decl_namespaces.values():
        temp_registry.add(ns)
    if registry is not None:
        for providers in registry._namespaces.values():
            for p in providers:
                temp_registry.add(p)

    result: dict[str, "Namespace"] = {}
    for name in temp_registry._namespaces:
        ns = temp_registry.get(name)
        if isinstance(ns, MergedNamespaceView):
            result[name] = flatten_view(ns)
        elif ns is not None:
            result[name] = ns
    return result


def _json_safe(value: Any) -> Any:
    """对返回值做 JSON 兜底序列化。

    Namespace 函数返回任何 Python 对象都可能。优先尝试 JSON 原生序列化；
    失败时退化为 ``repr()``，避免单个调用把整个 RPC 响应弄炸。
    """
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _describe_function(fn: Any) -> dict[str, Any]:
    """对单个 namespace 函数生成 describe 条目。

    - signature: ``inspect.signature(fn).__str__()`` 字符串，help() 直接渲染
    - doc: 完整 docstring（None 时空串）
    - kwargs_schema: v1 阶段统一返回空 dict（type-hint 推断列入 v1.1）
    """
    try:
        sig_str = str(inspect.signature(fn))
    except (TypeError, ValueError):
        sig_str = "(...)"
    doc = inspect.getdoc(fn) or ""
    return {
        "signature": sig_str,
        "doc": doc,
        "kwargs_schema": {},
    }


# ---------------------------------------------------------------------------
# JSON-RPC handler factory
# ---------------------------------------------------------------------------

def register_pysandbox_methods(
    dispatch: JsonRpcDispatcher,
    sandbox: "SandboxApp",
) -> None:
    """在 view 的 dispatcher 上注册 3 个 pysandbox namespace sharing 扩展方法。

    - ``pysandbox/namespaces.list``     — 列出所有 namespace 名 + 描述 + 函数数
    - ``pysandbox/namespaces.describe`` — 返回某个 namespace 全部函数的签名和文档
    - ``pysandbox/namespaces.call``     — 调用某个函数（kwargs only）

    实现路径：直查 sandbox registry，**不过 pysandbox 的 Python 代码解析**。
    比 pysandbox tool 路径更直接，也更快。
    """

    async def _handle_list(params: dict[str, Any]) -> dict[str, Any]:
        namespaces = _all_namespaces(sandbox)
        items = []
        for name, ns in sorted(namespaces.items()):
            items.append({
                "name": name,
                "description": ns._description or "",
                "function_count": len(ns._functions),
            })
        return {"namespaces": items}

    async def _handle_describe(params: dict[str, Any]) -> dict[str, Any]:
        ns_name = params.get("namespace")
        if not isinstance(ns_name, str) or not ns_name:
            raise JsonRpcError(INVALID_PARAMS, "Missing 'namespace'")
        namespaces = _all_namespaces(sandbox)
        ns = namespaces.get(ns_name)
        if ns is None:
            raise JsonRpcError(
                METHOD_NOT_FOUND, f"Namespace not found: {ns_name}")
        functions: dict[str, dict[str, Any]] = {}
        for fn_name, fn in ns._functions.items():
            functions[fn_name] = _describe_function(fn)
        return {
            "name": ns_name,
            "description": ns._description or "",
            "functions": functions,
        }

    async def _handle_call(params: dict[str, Any]) -> Any:
        ns_name = params.get("namespace")
        fn_name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(ns_name, str) or not ns_name:
            raise JsonRpcError(INVALID_PARAMS, "Missing 'namespace'")
        if not isinstance(fn_name, str) or not fn_name:
            raise JsonRpcError(INVALID_PARAMS, "Missing 'name'")
        if not isinstance(arguments, dict):
            raise JsonRpcError(INVALID_PARAMS, "'arguments' must be an object")

        namespaces = _all_namespaces(sandbox)
        ns = namespaces.get(ns_name)
        if ns is None:
            raise JsonRpcError(
                METHOD_NOT_FOUND, f"Namespace not found: {ns_name}")
        fn = ns._functions.get(fn_name)
        if fn is None:
            raise JsonRpcError(
                METHOD_NOT_FOUND,
                f"Function not found: {ns_name}.{fn_name}")

        # 业务异常 → JSON-RPC error（不复用 transport-error 通道,不触发 client 重连）
        try:
            # async NamespaceTools 方法被 _wrap_async 包成 sync wrapper
            # 后，原 coroutine 函数被挂在 ``_async_original`` 上。本 handler
            # 本身已在主 loop 的 async 上下文里，直接 await 原 coroutine
            # 可避免「sync wrapper run_coroutine_threadsafe 同线程等」死锁。
            async_original = getattr(fn, "_async_original", None)
            if async_original is not None:
                result = await async_original(**arguments)
            else:
                result = fn(**arguments)
                # 同步函数返回 awaitable 时 await（罕见，但兼容）
                if inspect.isawaitable(result):
                    result = await result
        except JsonRpcError:
            raise
        except TypeError as exc:
            # kwargs 不匹配 / 参数缺失 → INVALID_PARAMS 更合适
            raise JsonRpcError(INVALID_PARAMS, str(exc)) from exc
        except Exception as exc:
            logger.exception(
                "pysandbox/namespaces.call %s.%s raised", ns_name, fn_name)
            raise JsonRpcError(INTERNAL_ERROR, str(exc)) from exc

        return _json_safe(result)

    dispatch.add_method("pysandbox/namespaces.list", _handle_list)
    dispatch.add_method("pysandbox/namespaces.describe", _handle_describe)
    dispatch.add_method("pysandbox/namespaces.call", _handle_call)
