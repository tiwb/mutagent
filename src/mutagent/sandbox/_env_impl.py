"""SandboxEnv 实现 — 纯 sync namespace registry + 执行引擎。

设计要点：
- 不读 config，不发起 MCP 连接，不创建 event loop
- ``add_namespace`` / ``remove_namespace`` 自动 invalidate cache
- ``add_namespace(ns, on_remove=...)`` 把清理回调托管给 sandbox，
  ``close()`` 时统一调用，方便外部用一行收尾
- NamespaceTools (Declaration 自动发现) 仍然按需懒加载
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Iterator, cast

import mutobj
from mutagent.sandbox.env import SandboxEnv

if TYPE_CHECKING:
    from mutagent.sandbox.mcp import MCPConnection

# 类型别名：on_remove 回调可以是 sync 或 async（从 app.py 移入）
CleanupCallback = Callable[[], Any]
from mutagent.sandbox._engine import execute
from mutagent.sandbox._namespace_impl import (
    MergedNamespaceView,
    Namespace,
    NamespaceRegistry,
)

logger = logging.getLogger(__name__)


class SandboxEnvRuntime(mutobj.Extension[SandboxEnv]):
    """SandboxEnv internal runtime state."""

    registry: NamespaceRegistry | None = None
    cleanups: dict[int, tuple[Namespace, CleanupCallback]] | None = None
    mcp_conns: dict[str, MCPConnection] | None = None
    start_time: float | None = None
    async_loop: asyncio.AbstractEventLoop | None = None
    async_loop_thread_id: int | None = None
    cached_ns: dict[str, Any] | None = None
    cached_gen: int = -1


def _runtime(sandbox: SandboxEnv) -> SandboxEnvRuntime:
    return SandboxEnvRuntime.get_or_create(sandbox)


# ---------------------------------------------------------------------------
# 内部状态 helpers
# ---------------------------------------------------------------------------

def _get_registry(sandbox: SandboxEnv) -> NamespaceRegistry:
    rt = _runtime(sandbox)
    registry = rt.registry
    if registry is None:
        registry = NamespaceRegistry()
        rt.registry = registry
    return registry


def peek_registry(sandbox: SandboxEnv) -> NamespaceRegistry | None:
    rt = SandboxEnvRuntime.get(sandbox)
    return None if rt is None else rt.registry


def _get_cleanups(sandbox: SandboxEnv) -> dict[int, tuple[Namespace, CleanupCallback]]:
    """id(ns) -> (ns, on_remove)。

    multi-provider 下同名 ns 有多个实例，按名存会互盖。
    改为按实例 id 存，remove 时才能唯一定位。
    为了能从 name 反查 cleanup，同时保存 ns 引用。
    """
    rt = _runtime(sandbox)
    cleanups = rt.cleanups
    if cleanups is None:
        cleanups = {}
        rt.cleanups = cleanups
    return cleanups


def _get_mcp_conns(sandbox: SandboxEnv) -> dict[str, MCPConnection]:
    rt = _runtime(sandbox)
    conns = rt.mcp_conns
    if conns is None:
        conns = {}
        rt.mcp_conns = conns
    return conns


def _get_start_time(sandbox: SandboxEnv) -> float:
    rt = _runtime(sandbox)
    t = rt.start_time
    if t is None:
        t = time.time()
        rt.start_time = t
    return t


def get_async_loop(sandbox: SandboxEnv) -> asyncio.AbstractEventLoop | None:
    rt = SandboxEnvRuntime.get(sandbox)
    return None if rt is None else rt.async_loop


def require_async_loop(sandbox: SandboxEnv) -> asyncio.AbstractEventLoop:
    loop = get_async_loop(sandbox)
    if loop is None:
        raise RuntimeError(
            "SandboxEnv._async_loop not set; "
            "caller must call sandbox.bind_main_loop() before exec_code"
        )
    return loop


def _get_async_loop_thread_id(sandbox: SandboxEnv) -> int | None:
    rt = SandboxEnvRuntime.get(sandbox)
    return None if rt is None else rt.async_loop_thread_id


# ---------------------------------------------------------------------------
# NamespaceTools 发现
# ---------------------------------------------------------------------------

def _get_ns_tools_prefix(cls: type) -> str:
    """从 NamespaceTools 子类名推导 namespace 名（去掉 Tools 后缀，小写）。"""
    explicit = cls.__dict__.get('namespace')
    if explicit is not None:
        return explicit
    name = cls.__name__
    if name.endswith("Tools") and name != "Tools":
        name = name[:-5]
    return name.lower()


def _build_declaration_namespaces(self: SandboxEnv) -> dict[str, Namespace]:
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

            # async 方法包装为 sync：投递到 app 捕获的主 loop 执行
            if inspect.iscoroutinefunction(bound):
                fn = _wrap_async(self, bound)
                fn.__name__ = method_name
                fn.__doc__ = bound.__doc__
            else:
                fn = bound

            ns.register(method_name, fn, desc)

        if ns.functions:
            result[ns_name] = ns
            logger.info("Discovered NamespaceTools: %s (%d functions)",
                        ns_name, len(ns.functions))

    return result


def _wrap_async(sandbox: SandboxEnv, coro_fn: Any) -> Callable[..., Any]:
    """将 async NamespaceTools 方法包装为 sync。

    把 coroutine 投递到 SandboxEnv 上捕获的主 event loop 执行，
    不创建临时 event loop，避免 async tool 在错误线程/loop 执行。

    主 loop 由调用方（PySandboxTools / SandboxToolkit / 其他 entry）在
    ``run_in_executor`` 前通过 ``sandbox.bind_main_loop()`` 注入。

    返回的 wrapper 上挂 ``_async_original`` 属性，指向原始 coroutine
    函数，供已经在主 loop 异步上下文里的调用方（如
    ``_mcp_share.py:_handle_call``）绕过 sync wrapper 直接 ``await``，
    避免「同线程同步等自己排队的 coroutine」死锁。

    超时行为可通过 SandboxEnv 属性配置：
    - ``_wrap_async_timeout``: float | None — 超时秒数，None 使用默认 120s
    - ``_on_wrap_async_timeout``: Callable | None — 超时回调
      签名: ``(fn_name: str, future: concurrent.futures.Future) -> Any``
      返回值作为 wrapper 返回值。未设置时超时抛 TimeoutError。
    """
    fn_name = coro_fn.__name__

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        loop = require_async_loop(sandbox)

        # 同线程死锁保护
        loop_thread_id = _get_async_loop_thread_id(sandbox)
        if loop_thread_id is not None and threading.get_ident() == loop_thread_id:
            raise RuntimeError(
                "Cannot synchronously call async NamespaceTools from "
                "the target event loop thread; use await or call from worker thread"
            )

        # 参数规范化：有真签名时 bind 位置参数 + 填充默认值
        if _sig is not None:
            bound = _sig.bind(*args, **kwargs)
            bound.apply_defaults()
            call_kwargs: dict[str, Any] = dict(bound.arguments)
        else:
            if args:
                raise TypeError(
                    f"{fn_name}() takes 0 positional arguments "
                    f"but {len(args)} {'was' if len(args) == 1 else 'were'} given"
                )
            call_kwargs = kwargs

        timeout = getattr(sandbox, '_wrap_async_timeout', None) or 120
        on_timeout = getattr(sandbox, '_on_wrap_async_timeout', None)

        future = asyncio.run_coroutine_threadsafe(coro_fn(**call_kwargs), loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            if future.done():
                return future.result()
            if on_timeout is not None:
                return on_timeout(fn_name, future)
            raise

    # ---------- 真签名 + 位置调用支持 ----------
    # 从 coro_fn 提取 inspect.Signature，去掉 self 后用于:
    # (a) wrapper.__signature__ → help() 展示真签名
    # (b) sig.bind().apply_defaults() → 位置参数 → kwargs 规范化
    # 签名不可解析时 _sig 为 None，wrapper 回落为仅 (**kwargs)。
    try:
        orig_sig = inspect.signature(coro_fn)
        params = list(orig_sig.parameters.values())
        if params and params[0].name == "self":
            params = params[1:]
        _sig: inspect.Signature | None = inspect.Signature(
            params, return_annotation=orig_sig.return_annotation
        )
    except (ValueError, TypeError):
        _sig = None

    if _sig is not None:
        wrapper.__signature__ = _sig  # type: ignore[attr-defined]

    # 暴露原 coroutine 函数：异步上下文（如 _mcp_share.py 的 RPC handler）
    # 检测到该属性后可直接 await，跳过 sync wrapper 的 future 调度。
    wrapper._async_original = coro_fn  # type: ignore[attr-defined]
    return wrapper


@mutobj.impl(SandboxEnv.bind_main_loop)
def sandbox_env_bind_main_loop(self: SandboxEnv) -> None:
    """把当前 event loop 注入 SandboxEnv，作为 async NamespaceTools 的目标 loop。

    必须在主 loop 线程里调用（典型场景：每个 pysandbox entry 在
    ``run_in_executor`` 之前一次）。重复调用幂等。

    这是「3 处 entry 都要写的注入代码」的统一入口。新增 entry
    时只需 ``self.app.bind_main_loop()`` 一行，避免遗漏。
    """
    loop = asyncio.get_running_loop()
    rt = _runtime(self)
    rt.async_loop = loop
    rt.async_loop_thread_id = threading.get_ident()


# ---------------------------------------------------------------------------
# Namespace 收集 — sandbox 可见集合的单一来源
# ---------------------------------------------------------------------------

def collect_namespaces(
    sandbox: SandboxEnv,
) -> dict[str, Namespace | MergedNamespaceView]:
    """sandbox 可见的全部 namespace（decl 先 + external 后，同名走 merged view）。

    这是 ``_build_namespace_dict``（exec_code 路径）与 ``_mcp_share.py::_all_namespaces``
    （跨进程序列化路径）共享的单一合并入口，保证两条路径看到的 namespace 可见集
    严格一致。

    单 provider 名返回原始 :class:`Namespace`；同名 2+ providers 返回
    :class:`MergedNamespaceView`（按「先注册先赢」解析）。

    WARN-once 稳定性：
    - 无 decl namespace 时走主 registry 的原生 view（跨调用稳定）
    - 有 decl namespace 时走 temp_registry（每次构造新 view，WARN 按需触发）
    """
    decl_namespaces = _build_declaration_namespaces(sandbox)
    registry = _get_registry(sandbox)

    if not decl_namespaces:
        # 主 registry 视图直通 —— 复用原生 MergedNamespaceView，WARN-once 稳定
        result: dict[str, Namespace | MergedNamespaceView] = {}
        for name in registry.namespaces:
            ns = registry.get(name)
            if ns is not None:
                result[name] = ns
        return result

    # decl 先注册 → 本地 NamespaceTools 优先于外部 peer（与 exec_code 同序）
    temp_registry = NamespaceRegistry()
    for ns in decl_namespaces.values():
        temp_registry.add(ns)
    for providers in registry.namespaces.values():
        for p in providers:
            temp_registry.add(p)

    result = {}
    for name in temp_registry.namespaces:
        ns = temp_registry.get(name)
        if ns is not None:
            result[name] = ns
    return result


# ---------------------------------------------------------------------------
# Namespace dict 构建（cache + rebuild）
# ---------------------------------------------------------------------------

def _build_namespace_dict(self: SandboxEnv) -> dict[str, Any]:
    """构建完整 namespace dict，带缓存。

    缓存失效条件：
    - mutobj 类注册表 generation 变化（NamespaceTools 新增/变更）
    - 显式调用 add_namespace / remove_namespace（自动 invalidate）

    multi-provider：同名 2+ providers 走 ``MergedNamespaceView``；
    单 provider 名仍返回 Namespace 实例。详 ``_namespace.NamespaceRegistry``。

    ``help`` 键走 sandbox 视角：闭包捕获 sandbox，通过 ``iter_namespaces`` /
    ``get_namespace`` 访问当前可见集，与 exec_code 路径严格一致。
    """
    rt = _runtime(self)
    cached = rt.cached_ns
    cached_gen = rt.cached_gen
    current_gen = mutobj.get_registry_generation()

    if cached is not None and cached_gen == current_gen:
        return cached

    collected = collect_namespaces(self)
    ns_dict: dict[str, Any] = dict(collected)
    ns_dict['help'] = _make_sandbox_help(self)

    # 缓存
    rt.cached_ns = ns_dict
    rt.cached_gen = current_gen
    return ns_dict


def _make_sandbox_help(sandbox: SandboxEnv) -> Callable[..., Any]:
    """生成 sandbox-bound ``help()``。

    与 ``NamespaceRegistry._make_help`` 的区别：数据源是 sandbox 而非 registry，
    所以 Layer 1 列表包含 decl + external 合并后的完整集合（与 exec_code
    可见集一致），而非仅外部注入的 registry 视角。
    """
    from mutagent.sandbox._namespace_impl import (
        render_function,
        render_namespace,
        render_registry_from_namespaces,
    )

    def help(func_or_name: Any = None) -> str:
        """查看 namespace / 函数的文档。

        - help()                           列出所有 namespace
        - help(namespace)                  namespace 完整说明 + 函数列表
        - help(namespace.function)         函数签名 + 完整文档
        - help("namespace.function")       同上（字符串形式）
        """
        # Layer 1: 列所有 namespace
        if func_or_name is None:
            return render_registry_from_namespaces(
                list(sandbox_env_iter_namespaces(sandbox)))

        # Layer 2: 聚焦某个 namespace（含 view）
        if isinstance(func_or_name, (Namespace, MergedNamespaceView)):
            return render_namespace(func_or_name)

        # Layer 3: 聚焦某个函数
        if callable(func_or_name):
            return render_function(func_or_name)

        if isinstance(func_or_name, str):
            parts = func_or_name.split('.', 1)
            if len(parts) == 2:
                ns = sandbox_env_get_namespace(sandbox, parts[0])
                if ns is not None and parts[1] in ns.functions:
                    return render_function(
                        ns.functions[parts[1]],
                        ns_name=parts[0],
                        fn_name=parts[1])
            return f"(no documentation for '{func_or_name}')"

        return "(no documentation)"

    return help


def _invalidate_cache(self: SandboxEnv) -> None:
    rt = _runtime(self)
    rt.cached_ns = None
    rt.cached_gen = -1


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
# SandboxEnv @impl
# ---------------------------------------------------------------------------

def sandbox_env_add_namespace(
    self: SandboxEnv,
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


def sandbox_env_remove_namespace(self: SandboxEnv, name: str) -> None:
    """按 name 移除该名下的**全部** providers（向后兼容接口）。

    同时调度所有被移除 provider 的 cleanup。
    按实例级别移除请用 :func:`_remove_namespace_provider`。
    """
    registry = _get_registry(self)
    cleanups = _get_cleanups(self)

    providers = list(registry.namespaces.get(name, ()))
    registry.remove(name)
    # 收 cleanup
    for p in providers:
        entry = cleanups.pop(id(p), None)
        if entry is not None:
            _, cb = entry
            _schedule_cleanup_sync(name, cb)

    _invalidate_cache(self)
    logger.debug("Namespace '%s' removed (%d providers)", name, len(providers))


def sandbox_env_remove_provider(self: SandboxEnv, ns: Namespace) -> bool:
    """按实例移除一个 provider。详 feature-namespace-multi-provider。"""
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


@mutobj.impl(SandboxEnv.connect_source)
def sandbox_env_connect_source(self: SandboxEnv, conn: Any) -> None:
    """注入一个 MCP 连接源。幂等。"""
    from mutagent.sandbox._mcp_impl import MCPConnectionImpl

    _get_mcp_conns(self)[conn.name] = conn
    # 避免重复注册（panel Connect 可能调用多次）
    impl = mutobj.implementation_of(conn, MCPConnectionImpl)
    impl.sandbox = self
    registry = _get_registry(self)
    existing = registry.namespaces.get(conn.namespace.name, [])
    if conn.namespace not in existing:
        sandbox_env_add_namespace(self, conn.namespace, on_remove=conn.close)


@mutobj.impl(SandboxEnv.disconnect_source)
def sandbox_env_disconnect_source(self: SandboxEnv, name: str) -> None:
    """移除一个 MCP 连接源。幂等。"""
    conn = _get_mcp_conns(self).pop(name, None)
    if conn is not None:
        sandbox_env_remove_provider(self, cast(Namespace, conn.namespace))


@mutobj.impl(SandboxEnv.list_sources)
def sandbox_env_list_sources(self: SandboxEnv) -> dict[str, Any]:
    return dict(_get_mcp_conns(self))


def sandbox_env_iter_namespaces(
    self: SandboxEnv,
) -> Iterator[Namespace | MergedNamespaceView]:
    """按名排序遍历 sandbox 可见的全部 namespace。

    实现沿用 ``_build_namespace_dict`` 的缓存路径（跳过 ``help`` 键），
    与 ``exec_code`` 内看到的集合严格一致。
    """
    ns_dict = _build_namespace_dict(self)
    for name in sorted(k for k in ns_dict if k != 'help'):
        yield ns_dict[name]


def sandbox_env_get_namespace(
    self: SandboxEnv, name: str,
) -> Namespace | MergedNamespaceView | None:
    """按名获取 namespace；不存在返回 ``None``。与 ``iter_namespaces`` 来自同一可见集。"""
    if name == 'help':
        return None
    ns_dict = _build_namespace_dict(self)
    value = ns_dict.get(name)
    # help 键是 callable，其他都是 Namespace / MergedNamespaceView
    if isinstance(value, (Namespace, MergedNamespaceView)):
        return value
    return None


@mutobj.impl(SandboxEnv.exec_code)
def sandbox_env_exec_code(self: SandboxEnv, code: str,
                          state: dict[str, Any] | None = None) -> dict[str, Any]:
    ns_dict = _build_namespace_dict(self)
    return execute(code, ns_dict, state)


@mutobj.impl(SandboxEnv.close)
async def sandbox_env_close(self: SandboxEnv) -> None:
    cleanups = _get_cleanups(self)
    registry = _get_registry(self)
    mcp_conns = _get_mcp_conns(self)

    # 拷贝并清空，避免重入
    items = list(cleanups.values())
    cleanups.clear()
    for name in list(registry.namespaces):
        registry.remove(name)
    mcp_conns.clear()

    for ns, cb in items:
        await _invoke_cleanup(ns.name, cb)

    _invalidate_cache(self)


@mutobj.impl(SandboxEnv.format_result)
def sandbox_env_format_result(self: SandboxEnv, result: dict[str, Any]) -> tuple[str, bool]:
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
