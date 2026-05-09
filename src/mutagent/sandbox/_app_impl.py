"""SandboxApp 实现 — 纯 sync namespace registry + 执行引擎。

设计要点：
- 不读 config，不发起 MCP 连接，不创建 event loop
- ``add_namespace`` / ``remove_namespace`` 自动 invalidate cache
- ``add_namespace(ns, on_remove=...)`` 把清理回调托管给 sandbox，
  ``close()`` 时统一调用，方便外部用一行收尾
- NamespaceTools (Declaration 自动发现) 仍然按需懒加载
"""

import asyncio
import inspect
import logging
import time
from typing import Any, Callable

import mutobj
import mutagent
from mutagent.sandbox.app import SandboxApp, CleanupCallback
from mutagent.sandbox._engine import execute
from mutagent.sandbox._namespace import Namespace, NamespaceRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 内部状态 helpers
# ---------------------------------------------------------------------------

def _get_registry(self: SandboxApp) -> NamespaceRegistry:
    registry = getattr(self, '_registry', None)
    if registry is None:
        registry = NamespaceRegistry()
        object.__setattr__(self, '_registry', registry)
    return registry


def _get_cleanups(self: SandboxApp) -> dict[int, tuple[Namespace, CleanupCallback]]:
    """id(ns) -> (ns, on_remove)。

    multi-provider 下同名 ns 有多个实例，按名存会互盖。
    改为按实例 id 存，remove 时才能唯一定位。
    为了能从 name 反查 cleanup，同时保存 ns 引用。
    """
    cleanups = getattr(self, '_cleanups', None)
    if cleanups is None:
        cleanups = {}
        object.__setattr__(self, '_cleanups', cleanups)
    return cleanups


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

        ns = Namespace(ns_name, description=inspect.getdoc(cls) or "")
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
    - 显式调用 add_namespace / remove_namespace（自动 invalidate）

    multi-provider：同名 2+ providers 走 ``MergedNamespaceView``；
    单 provider 名仍返回 Namespace 实例。详 ``_namespace.NamespaceRegistry``。
    """
    cached = getattr(self, '_cached_ns', None)
    cached_gen = getattr(self, '_cached_gen', -1)
    current_gen = mutobj.get_registry_generation()

    if cached is not None and cached_gen == current_gen:
        return cached

    registry = _get_registry(self)

    # NamespaceTools（Declaration 自动发现）— 按名合入 temp registry
    decl_namespaces = _build_declaration_namespaces(self)

    # 构合完整 view：外部注入的全部 providers + decl namespaces
    # 直接复用主 registry 的 provider list（view 实例也复用，
    # WARN-once 状态才能跨 cache lifetime 保留。。。但 cache 按代位
    # 重建的场景下，view 实例仍是 registry._views[name] 同一个，OK）
    if decl_namespaces:
        # decl ns 不在 main registry，需合入临时视图
        temp_registry = NamespaceRegistry()
        for providers in registry._namespaces.values():
            for p in providers:
                temp_registry.add(p)
        for ns in decl_namespaces.values():
            temp_registry.add(ns)
        ns_dict = temp_registry.build_namespace_dict()
    else:
        # 没 decl namespace 时直接走主 registry，避免 temp view 覆盖主 view、
        # 导致 WARN-once 跨 cache 周期重复触发
        ns_dict = registry.build_namespace_dict()

    # 缓存
    object.__setattr__(self, '_cached_ns', ns_dict)
    object.__setattr__(self, '_cached_gen', current_gen)
    return ns_dict


def _invalidate_cache(self: SandboxApp) -> None:
    object.__setattr__(self, '_cached_ns', None)
    object.__setattr__(self, '_cached_gen', -1)


async def _invoke_cleanup(name: str, cb: CleanupCallback) -> None:
    """调用 on_remove 回调，sync/async 统一处理，异常吞掉只记日志。"""
    try:
        result = cb()
        if inspect.isawaitable(result):
            await result
    except Exception as e:
        logger.warning("on_remove for namespace '%s' failed: %s", name, e)


def _schedule_cleanup_sync(name: str, cb: CleanupCallback) -> None:
    """在 sync 上下文 (remove_namespace) 里调度 cleanup。

    - 已有 running loop：用 asyncio.ensure_future 提交任务
    - 没有 loop：直接同步调；async 回调用 asyncio.run 跑完
    """
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_invoke_cleanup(name, cb))
    except RuntimeError:
        try:
            result = cb()
            if inspect.isawaitable(result):
                asyncio.run(result)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning("on_remove for namespace '%s' failed: %s", name, e)


# ---------------------------------------------------------------------------
# SandboxApp @impl
# ---------------------------------------------------------------------------

@mutagent.impl(SandboxApp.add_namespace)
def _add_namespace(
    self: SandboxApp,
    ns: Namespace,
    on_remove: CleanupCallback | None = None,
) -> None:
    """注入 namespace（multi-provider）。

    同名 ns 不再互相覆盖：作为同名 namespace 的另一个 provider 并存，
    调用时走 :class:`MergedNamespaceView` 按「先注册先赢」解析。
    """
    _get_start_time(self)
    registry = _get_registry(self)
    cleanups = _get_cleanups(self)

    registry.add(ns)
    if on_remove is not None:
        cleanups[id(ns)] = (ns, on_remove)

    _invalidate_cache(self)
    logger.debug("Namespace '%s' added (kind=%s, on_remove=%s)",
                 ns.name, ns.provider_kind, on_remove is not None)


@mutagent.impl(SandboxApp.remove_namespace)
def _remove_namespace(self: SandboxApp, name: str) -> None:
    """按 name 移除该名下的**全部** providers（向后兼容接口）。

    同时调度所有被移除 provider 的 cleanup。
    按实例级别移除请用 :func:`_remove_namespace_provider`。
    """
    registry = _get_registry(self)
    cleanups = _get_cleanups(self)

    providers = list(registry._namespaces.get(name, ()))
    registry.remove(name)
    # 收 cleanup
    for p in providers:
        entry = cleanups.pop(id(p), None)
        if entry is not None:
            _, cb = entry
            _schedule_cleanup_sync(name, cb)

    _invalidate_cache(self)
    logger.debug("Namespace '%s' removed (%d providers)", name, len(providers))


def _remove_namespace_provider(self: SandboxApp, ns: Namespace) -> bool:
    """按实例移除一个 provider。

    不是 Declaration 接口 —— 作为 SandboxApp 上的辅助函数暴露（
    调用者可走 ``sandbox_app.remove_provider(ns)``）。详
    feature-namespace-multi-provider。
    """
    registry = _get_registry(self)
    cleanups = _get_cleanups(self)

    removed = registry.remove_provider(ns)
    if not removed:
        return False

    entry = cleanups.pop(id(ns), None)
    if entry is not None:
        _, cb = entry
        _schedule_cleanup_sync(ns.name, cb)

    _invalidate_cache(self)
    logger.debug("Namespace provider '%s' (%s) removed",
                 ns.name, ns.provider_kind)
    return True


# 辅助方法挂到 SandboxApp 上，方便外部调用 sandbox_app.remove_provider(ns)
SandboxApp.remove_provider = _remove_namespace_provider  # type: ignore[attr-defined]


@mutagent.impl(SandboxApp.exec_code)
def _exec_code(self: SandboxApp, code: str,
               state: dict[str, Any] | None = None) -> dict[str, Any]:
    ns_dict = _build_namespace_dict(self)
    return execute(code, ns_dict, state)


@mutagent.impl(SandboxApp.close)
async def _close(self: SandboxApp) -> None:
    cleanups = _get_cleanups(self)
    registry = _get_registry(self)

    # 拷贝并清空，避免重入
    items = list(cleanups.values())
    cleanups.clear()
    for name in list(registry._namespaces):
        registry.remove(name)

    for ns, cb in items:
        await _invoke_cleanup(ns.name, cb)

    _invalidate_cache(self)


@mutagent.impl(SandboxApp.format_result)
def _format_result(self: SandboxApp, result: dict[str, Any]) -> tuple[str, bool]:
    if "error" in result:
        text = result["error"]
        if result.get("traceback"):
            text += "\n" + result["traceback"]
        return text, True

    parts: list[str] = []
    if result.get("stdout"):
        parts.append(result["stdout"])
    value = result.get("result")
    if value is not None:
        if isinstance(value, str):
            parts.append(value)
        else:
            parts.append(repr(value))
    return ("\n".join(parts) if parts else "(no output)"), False
