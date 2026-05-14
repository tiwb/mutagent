"""Pysandbox namespace sharing — client side.

复用 ``MCPConnection`` 的连接 / 重连 / 状态机，调对端的
``pysandbox/namespaces.*`` 扩展方法，把远端 namespaces 平铺融合进本地
sandbox。

设计文档：
``mutagent/docs/specifications/feature-pysandbox-namespace-sharing.md``

外部入口仅一个：:func:`build_peer_namespaces`，由 ``MCPConnection._do_rebuild``
在标准 tools 路径之后调用。其余符号均为内部实现。
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from mutagent.sandbox._namespace import Namespace

if TYPE_CHECKING:
    from mutagent.sandbox._adapter_mcp import HTTPMCPClient, MCPConnection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Peer client — 在 HTTPMCPClient 上加 3 个扩展方法
# ---------------------------------------------------------------------------


class PysandboxPeerClient:
    """薄封装：在已经 connect 完成的 ``HTTPMCPClient`` 上调 pysandbox 扩展方法。

    所有方法直接走 ``MCPClient.request`` 通用 JSON-RPC 入口；
    传输错由 ``HTTPMCPClient`` / ``MCPConnection`` 上层捕获重连。
    """

    def __init__(self, http_client: "HTTPMCPClient") -> None:
        self._http = http_client

    async def list_namespaces(self) -> list[dict[str, Any]]:
        """返回 ``[{name, description, function_count}, ...]``。"""
        result = await self._http._mcp.request(
            "pysandbox/namespaces.list", {})
        return list(result.get("namespaces", []))

    async def describe_namespace(self, name: str) -> dict[str, Any]:
        """返回 ``{name, description, functions: {fn: {signature, doc, kwargs_schema}}}``。"""
        return await self._http._mcp.request(
            "pysandbox/namespaces.describe", {"namespace": name})

    async def call_namespace(
        self, namespace: str, name: str, arguments: dict[str, Any],
    ) -> Any:
        """调用远端 ``namespace.name(**arguments)``，返回原始 result。"""
        return await self._http._mcp.request(
            "pysandbox/namespaces.call",
            {"namespace": namespace, "name": name, "arguments": arguments},
        )


# ---------------------------------------------------------------------------
# Function wrapper — 对齐 _make_tool_func 的结构
# ---------------------------------------------------------------------------


def _make_namespace_func(
    conn: "MCPConnection",
    ns_name: str,
    fn_name: str,
    doc: str,
    params: list[dict[str, Any]] | None = None,
) -> Any:
    """为远端 namespace 函数生成本地 Python 函数。

    结构对齐 ``_adapter_mcp._make_tool_func``：
    - 调用前 ``ensure_connected``
    - 传输错时 ``mark_disconnected`` + ``reconnect`` 后重试一次
    - 通过 ``run_coroutine_threadsafe`` 切回 main_loop（httpx 资源约束）

    与 tool wrapper 的差别仅在最终调用：``call_namespace`` 而非 ``call_tool``。

    签名层：
    - 新 server 返回的 ``params`` 存在时，用 ``_signature.build_signature`` 构造
      真签名挂在 ``__signature__`` 上；wrapper 内用 ``sig.bind`` 规范化参数
    - ``params`` 缺失（老 server）或构造失败，回落为旧版 ``(**kwargs)`` wrapper；
      展示层会自然降级为 ``ns_func(**kwargs)``，参数详细信息依 Annotations 段补充
    - ``__doc__`` 不拼接签名首行（去重复展示 bug）

    注：iter3 后不再写入 ``_pysandbox_signature_str`` 属性，format_callable_signature
    已去除 fallback 分发。
    """
    # 延迟 import 避免循环
    from mutagent.sandbox._adapter_mcp import HTTPMCPClient, _is_transport_error
    from mutagent.sandbox._signature import try_build_signature

    async def call_with_retry(kwargs: dict[str, Any]) -> Any:
        await conn.ensure_connected()
        client = conn.client
        assert isinstance(client, HTTPMCPClient)
        peer = PysandboxPeerClient(client)
        try:
            return await peer.call_namespace(ns_name, fn_name, kwargs)
        except Exception as exc:
            if not _is_transport_error(exc):
                raise
            conn.mark_disconnected(str(exc) or exc.__class__.__name__)
            await conn.reconnect()
            client = conn.client
            assert isinstance(client, HTTPMCPClient)
            peer = PysandboxPeerClient(client)
            return await peer.call_namespace(ns_name, fn_name, kwargs)

    # _async_original: 供 share.py:_handle_call 直接 await，避免
    # sync wrapper 的同线程死锁（与 _wrap_async 模式一致）。
    async def _ns_async(**kwargs: Any) -> Any:
        return await call_with_retry(kwargs)

    # 构真签名：仅在 server 提供结构化 params 时尝试
    # 注意：空列表亦是合法入参（无参函数），用 is not None 而非真值测试
    sig = None
    if params is not None:
        sig = try_build_signature(
            params, context=f"pysandbox {ns_name}.{fn_name}")

    if sig is not None:
        _bind_sig = sig

        def ns_func(*args: Any, **kwargs: Any) -> Any:
            bound = _bind_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            future = asyncio.run_coroutine_threadsafe(
                call_with_retry(dict(bound.arguments)), conn.main_loop)
            return future.result(timeout=120)

        ns_func.__signature__ = sig  # type: ignore[attr-defined]
    else:
        def ns_func(**kwargs: Any) -> Any:  # type: ignore[misc]
            future = asyncio.run_coroutine_threadsafe(
                call_with_retry(kwargs), conn.main_loop)
            return future.result(timeout=120)

    ns_func.__name__ = fn_name
    ns_func.__doc__ = doc  # 停止向 doc 拼接签名首行
    ns_func._async_original = _ns_async  # type: ignore[attr-defined]
    return ns_func


# ---------------------------------------------------------------------------
# 主入口 — build_peer_namespaces
# ---------------------------------------------------------------------------


def has_pysandbox_capability(init_result: dict[str, Any]) -> bool:
    """检测对端 initialize 响应是否声明了 pysandbox capability（D3）。"""
    caps = init_result.get("capabilities") or {}
    return isinstance(caps.get("pysandbox"), dict)


async def build_peer_namespaces(
    conn: "MCPConnection",
    init_result: dict[str, Any],
    client: "HTTPMCPClient",
) -> list[Namespace]:
    """从对端拉取并构建 peer namespaces。

    流程（对齐 D4 Eager + 单条容错）：

    1. ``pysandbox/namespaces.list`` —— 失败往上抛（视为整 source 失败）
    2. 对每个 ns 立刻 ``describe`` —— 单条失败 skip + WARNING
    3. 描述区追加 ``(shared from <source_name>)`` 标记（D6 半透明归属）
    4. 函数装入新建的 :class:`Namespace`，状态字段挂到 ``conn``

    Returns:
        按 server 返回顺序构建的 namespace 列表。重名冲突由调用方
        （``MCPConnection`` / ``SandboxApp``）按 D1 处理。
    """
    peer = PysandboxPeerClient(client)
    items = await peer.list_namespaces()  # 失败往上抛 → MCPConnection 转 failed

    source_label = conn.ns_name
    namespaces: list[Namespace] = []
    for item in items:
        name = item.get("name") or ""
        if not name:
            continue
        try:
            described = await peer.describe_namespace(name)
        except Exception as exc:
            logger.warning(
                "MCP '%s': describe namespace %r failed, skipped: %s",
                conn.ns_name, name, exc)
            continue

        base_desc = (described.get("description") or "").rstrip()
        # D6: 在描述末尾追加来源标记
        if base_desc:
            full_desc = f"{base_desc}\n\n(shared from {source_label})"
        else:
            full_desc = f"(shared from {source_label})"

        ns = Namespace(name, description=full_desc, provider_kind="peer")
        # peer namespace 也要随 conn 一起跑状态，help() 才能正确显示
        ns._connection = conn  # type: ignore[attr-defined]
        ns.connection_state = conn.state  # type: ignore[attr-defined]
        ns.connection_error = conn.last_error  # type: ignore[attr-defined]

        functions = described.get("functions") or {}
        for fn_name, info in functions.items():
            if not isinstance(info, dict):
                continue
            doc = info.get("doc") or ""
            # params 是新 server 的 additive 可选字段，缺失不影响老 server 兼容
            params = info.get("params")
            if not isinstance(params, list):
                params = None
            fn = _make_namespace_func(
                conn, name, fn_name, doc, params)
            # description 取 doc 首段（与 tool 路径一致：register 时存全文）
            ns.register(fn_name, fn, doc)

        namespaces.append(ns)

    return namespaces
