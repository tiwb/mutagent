"""SandboxApp 实现 — cache+rebuild、generation 懒发现。"""

import asyncio
import inspect
import logging
import sys
import time
from typing import Any

import mutobj
import mutagent
from mutagent.sandbox.app import SandboxApp
from mutagent.sandbox._engine import execute
from mutagent.sandbox._namespace import Namespace, NamespaceRegistry
from mutagent.sandbox._adapter_mcp import bridge_mcp_server, AnyMCPClient
from mutagent.sandbox._adapter_cli import build_cli_namespace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 内部状态 helpers
# ---------------------------------------------------------------------------

def _get_mcp_config(self: SandboxApp) -> dict[str, dict[str, Any]]:
    return self.config.get("mcp_sources", default={}) or {}


def _get_cli_config(self: SandboxApp) -> dict[str, dict[str, Any]]:
    return self.config.get("cli_sources", default={}) or {}


def _get_mcp_clients(self: SandboxApp) -> dict[str, AnyMCPClient]:
    clients = getattr(self, '_mcp_clients', None)
    if clients is None:
        clients = {}
        object.__setattr__(self, '_mcp_clients', clients)
    return clients


def _get_registry(self: SandboxApp) -> NamespaceRegistry:
    registry = getattr(self, '_registry', None)
    if registry is None:
        registry = NamespaceRegistry()
        object.__setattr__(self, '_registry', registry)
    return registry


def _get_start_time(self: SandboxApp) -> float:
    t = getattr(self, '_start_time', None)
    if t is None:
        t = time.time()
        object.__setattr__(self, '_start_time', t)
    return t


# ---------------------------------------------------------------------------
# NamespaceTools 发现
# ---------------------------------------------------------------------------

def _get_ns_tools_prefix(cls: type) -> str:
    """从 NamespaceTools 子类名推导 namespace 名（去掉 Tools 后缀，小写）。"""
    explicit = cls.__dict__.get('_namespace')
    if explicit is not None:
        return explicit
    name = cls.__name__
    if name.endswith("Tools") and name != "Tools":
        name = name[:-5]
    return name.lower()


def _build_declaration_namespaces(self: SandboxApp) -> dict[str, Namespace]:
    """从 NamespaceTools 子类构建命名空间。"""
    from mutagent.sandbox.namespace import NamespaceTools

    result: dict[str, Namespace] = {}
    for cls in mutobj.discover_subclasses(NamespaceTools):
        ns_name = _get_ns_tools_prefix(cls)
        try:
            instance = cls()
        except Exception:
            logger.debug("Cannot instantiate %s, skipping", cls.__name__)
            continue

        ns = Namespace(ns_name)
        for method_name in dir(cls):
            if method_name.startswith("_"):
                continue
            attr = getattr(cls, method_name, None)
            if attr is None or not (inspect.isfunction(attr) or inspect.ismethod(attr)):
                continue
            # 跳过 Declaration 基类方法
            if method_name in dir(NamespaceTools):
                continue

            bound = getattr(instance, method_name)
            desc = (bound.__doc__ or '').strip().split('\n')[0]

            # async 方法包装为 sync（sandbox 在工作线程中同步执行）
            if inspect.iscoroutinefunction(bound):
                fn = _wrap_async(bound)
                fn.__name__ = method_name
                fn.__doc__ = bound.__doc__
            else:
                fn = bound

            ns.register(method_name, fn, desc)

        if ns._functions:
            result[ns_name] = ns
            logger.info("Discovered NamespaceTools: %s (%d functions)",
                        ns_name, len(ns._functions))

    return result


def _wrap_async(coro_fn: Any) -> Any:
    """将 async 函数包装为 sync，在工作线程中安全调用 event loop。"""
    def wrapper(**kwargs: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(coro_fn(**kwargs), loop)
            return future.result(timeout=120)
        except RuntimeError:
            return asyncio.run(coro_fn(**kwargs))
    return wrapper


# ---------------------------------------------------------------------------
# Namespace dict 构建（cache + rebuild）
# ---------------------------------------------------------------------------

def _build_namespace_dict(self: SandboxApp) -> dict[str, Any]:
    """构建完整 namespace dict，带缓存。

    缓存失效条件：
    - mutobj 类注册表 generation 变化（NamespaceTools 新增/变更）
    - 显式调用 setup/reload（MCP/CLI 配置变更）
    """
    cached = getattr(self, '_cached_ns', None)
    cached_gen = getattr(self, '_cached_gen', -1)
    current_gen = mutobj.get_registry_generation()

    if cached is not None and cached_gen == current_gen:
        return cached

    registry = _get_registry(self)

    # 从数据源完整重建
    ns_dict: dict[str, Any] = {}

    # 1. MCP + CLI 命名空间（已在 registry 中）
    for name, ns in registry._namespaces.items():
        ns_dict[name] = ns

    # 2. NamespaceTools（Declaration 自动发现）
    decl_namespaces = _build_declaration_namespaces(self)
    ns_dict.update(decl_namespaces)

    # 3. help 函数
    all_namespaces = dict(registry._namespaces)
    all_namespaces.update(decl_namespaces)
    temp_registry = NamespaceRegistry()
    for ns in all_namespaces.values():
        temp_registry.add(ns)
    ns_dict['help'] = temp_registry._make_help()

    # 缓存
    object.__setattr__(self, '_cached_ns', ns_dict)
    object.__setattr__(self, '_cached_gen', current_gen)
    return ns_dict


def _invalidate_cache(self: SandboxApp) -> None:
    object.__setattr__(self, '_cached_ns', None)
    object.__setattr__(self, '_cached_gen', -1)


# ---------------------------------------------------------------------------
# SandboxApp @impl
# ---------------------------------------------------------------------------

@mutagent.impl(SandboxApp.setup)
async def _setup(self: SandboxApp) -> None:
    _get_start_time(self)  # 初始化启动时间
    registry = _get_registry(self)
    clients = _get_mcp_clients(self)

    mcp_config = _get_mcp_config(self)
    cli_config = _get_cli_config(self)

    # MCP 连接
    for ns_name, server_config in mcp_config.items():
        try:
            ns, client = await bridge_mcp_server(ns_name, server_config)
            registry.add(ns)
            clients[ns_name] = client
        except Exception as e:
            logger.warning("Failed to connect MCP '%s': %s", ns_name, e)
            print(f"Warning: Failed to connect MCP '{ns_name}': {e}",
                  file=sys.stderr)

    # CLI 命名空间
    if cli_config:
        cli_ns = build_cli_namespace(cli_config)
        registry.add(cli_ns)

    _invalidate_cache(self)


@mutagent.impl(SandboxApp.exec_code)
def _exec_code(self: SandboxApp, code: str,
               state: dict[str, Any] | None = None) -> dict[str, Any]:
    ns_dict = _build_namespace_dict(self)
    return execute(code, ns_dict, state)


@mutagent.impl(SandboxApp.reload)
async def _reload(self: SandboxApp) -> dict[str, Any]:
    await self.shutdown()

    # 重置 registry
    object.__setattr__(self, '_registry', NamespaceRegistry())

    await self.setup()

    registry = _get_registry(self)
    ns_count = len(registry._namespaces)
    return {"namespaces": ns_count}


@mutagent.impl(SandboxApp.shutdown)
async def _shutdown(self: SandboxApp) -> None:
    clients = _get_mcp_clients(self)
    for client in clients.values():
        try:
            await client.close()
        except Exception:
            pass
    clients.clear()
    _invalidate_cache(self)
