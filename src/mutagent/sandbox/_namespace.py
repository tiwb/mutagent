"""命名空间机制 — 能力源函数的分组访问和按需查询。

Multi-provider 模型（feature-namespace-multi-provider）：

- ``Namespace`` 仍是一个 **provider**，内部结构不变，只新增 ``provider_kind``
  元数据用于策略判断。
- ``NamespaceRegistry._namespaces`` 由 ``dict[str, Namespace]`` 改为
  ``dict[str, list[Namespace]]``，同名 namespace 可以共存。
- 同名 2+ providers 时 ``get(name)`` 返回 :class:`MergedNamespaceView`，
  view 持有 providers 列表的引用并按「先注册先赢」策略解析函数；
  发生函数冲突时 WARNING 一次（同 providers 签名内不重复）。
- 单 provider 名 ``get(name)`` 仍直接返回 ``Namespace``，
  保持向后兼容（旧测试与下游用法不变）。
"""

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from mutagent.sandbox._adapter_mcp import MCPConnection

logger = logging.getLogger(__name__)


ProviderKind = Literal["builtin", "tool", "peer", "cli"]


def _first_line(text: str) -> str:
    """提取文本首行（非空、strip）。空文本返回空串。"""
    if not text:
        return ""
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s
    return ""


class Namespace:
    """命名空间对象，通过 . 访问其中的函数。

    >>> ns = Namespace("browser", description="浏览器自动化工具")
    >>> ns.register("navigate", some_func, "Navigate to URL")
    >>> ns.navigate(url="https://example.com")
    """

    def __init__(self, name: str, description: str = "",
                 provider_kind: ProviderKind = "builtin"):
        self._name = name
        self._description = description
        # provider_kind 用于 multi-provider 视图渲染时的归属标签 / 策略判断；
        # 单 provider 路径下不影响行为。
        self.provider_kind: ProviderKind = provider_kind
        self._functions: dict[str, Callable] = {}
        self._descriptions: dict[str, str] = {}
        # 仅 MCP namespace 有意义；其他 namespace 留 None，渲染时跳过状态显示
        self._connection: "MCPConnection | None" = None
        self.connection_state: str | None = None
        self.connection_error: str | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def register(self, func_name: str, func: Callable,
                 description: str = "") -> None:
        """注册一个函数到命名空间。

        description 存完整文本，展示时由调用方决定取首行还是全文。
        """
        self._functions[func_name] = func
        self._descriptions[func_name] = description or (func.__doc__ or '')
        # 给函数附加 namespace 归属，help(fn) 展示时能拼前缀。best-effort。
        try:
            func.__namespace__ = self._name  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass

    def __getattr__(self, name: str) -> Any:
        # 私有属性 / dunder 不走懒触发，否则会与 hasattr 检查死循环
        if name.startswith('_'):
            raise AttributeError(name)
        # 已注册（含连过又断的「last seen」函数）直接返回
        functions = self.__dict__.get('_functions', {})
        if name in functions:
            return functions[name]
        # MCP namespace 且未连：阻塞式触发 ensure_connected
        connection = self.__dict__.get('_connection')
        state = self.__dict__.get('connection_state')
        if connection is not None and state != "connected":
            future = asyncio.run_coroutine_threadsafe(
                connection.ensure_connected(), connection.main_loop)
            try:
                # 30s 余量覆盖 stdio 冷启动 / npx 下载场景
                future.result(timeout=30)
            except Exception as exc:
                # 重连失败 — 给用户清晰的错误信息（包含原因）
                raise AttributeError(
                    f"'{self._name}' is not connected: {exc}") from exc
            if name in functions:
                return functions[name]
        raise AttributeError(
            f"'{self._name}' has no function '{name}'")

    def __repr__(self) -> str:
        count = len(self._functions)
        return f"<Namespace '{self._name}' ({count} functions)>"


# ---------------------------------------------------------------------------
# MergedNamespaceView — 多 provider 同名时暴露给沙箱的合并视图
# ---------------------------------------------------------------------------


@dataclass
class ResolvedFn:
    """一个函数名在 multi-provider 解析后的归属信息。"""
    active: Namespace
    fn: Callable
    shadowed: list[Namespace] = field(default_factory=list)


class MergedNamespaceView:
    """同名 namespace 多 provider 共存时，暴露给 sandbox 的合并视图。

    设计要点：
    - **不持有状态**：``providers`` 字段是对 registry 内部 list 的引用，
      registry 端 append/remove 后视图自动看到最新内容。
    - **薄缓存**：``_resolved_cache_key = tuple(id(p) for p in providers)``，
      providers 列表变化时自动失效，重新触发一次冲突 WARNING。
    - **「先注册先赢」**：函数同名时按 providers 顺序首个胜出，其余进 shadowed。
    - **`_functions` / `_description` / `_connection` 等属性兼容 `Namespace` 接口**，
      让 help 渲染层无需对 view 做大量分支判断。
    """

    def __init__(self, name: str, providers: list[Namespace]) -> None:
        self._name = name
        # 直接持有 registry 内部 list 的引用，**不要拷贝**
        self._providers = providers
        # 解析结果缓存（key = providers id 序列）
        self._resolved_cache: dict[str, ResolvedFn] | None = None
        self._resolved_cache_key: tuple[int, ...] | None = None
        # 已 WARN 过的 (fn_name, providers_id_sig)；providers 变化后 sig 变，自动重新 WARN
        self._warned_keys: set[tuple[str, tuple[int, ...]]] = set()

    @property
    def name(self) -> str:
        return self._name

    @property
    def providers(self) -> list[Namespace]:
        return self._providers

    # -- “选主”语义：displayed / primary（唯一权威）---------------------

    @property
    def displayed(self) -> list[Namespace]:
        """参与渲染的 provider 列表：贡献了至少一个 active 函数的 provider。

        两个过滤条件：
        1. ``_functions`` 非空（过滤空壳 tool ns、连接前函数表空）
        2. 至少有一个 active（非 shadowed）函数（过滤全被覆盖的 provider）

        这是“权威 provider”的根本属性，primary / multi-provider 渲染都从这里派生。
        """
        if not self._providers:
            return []
        resolved = self._resolved_functions()
        active_ids = {id(rf.active) for rf in resolved.values()}
        return [p for p in self._providers
                if p._functions and id(p) in active_ids]

    @property
    def primary(self) -> Namespace | None:
        """合并视图的主 provider。``displayed[0]`` 的派生。

        全空壳时退化为 ``_providers[0]``；无 provider 返回 None。
        """
        d = self.displayed
        if d:
            return d[0]
        return self._providers[0] if self._providers else None

    # -- 兼容 Namespace 接口（供 help 渲染 / 同代码路径复用）-----------------

    @property
    def _description(self) -> str:
        # 描述走 primary；全空壳 view 由 primary 退化为 _providers[0] 负责。
        p = self.primary
        return p._description if p is not None else ""

    @property
    def description(self) -> str:
        return self._description

    @property
    def _connection(self) -> "MCPConnection | None":
        # 优先返回 connected provider 的 connection；否则取第一个非 None
        for p in self._providers:
            if p.connection_state == "connected" and p._connection is not None:
                return p._connection
        for p in self._providers:
            if p._connection is not None:
                return p._connection
        return None

    @property
    def connection_state(self) -> str | None:
        # any connected -> connected; any connecting -> connecting; else first non-None
        states = [p.connection_state for p in self._providers]
        if any(s == "connected" for s in states):
            return "connected"
        if any(s == "connecting" for s in states):
            return "connecting"
        for s in states:
            if s is not None:
                return s
        return None

    @property
    def connection_error(self) -> str | None:
        # 取第一个 failed provider 的 error
        for p in self._providers:
            if p.connection_state == "failed":
                return p.connection_error
        return None

    @property
    def _functions(self) -> dict[str, Callable]:
        """合并后的函数表（active 视角）— 给 help / dir() 用。"""
        return {fn_name: rf.fn
                for fn_name, rf in self._resolved_functions().items()}

    @property
    def _descriptions(self) -> dict[str, str]:
        """合并后的函数 description（取 active provider 上的描述）。"""
        result: dict[str, str] = {}
        for fn_name, rf in self._resolved_functions().items():
            result[fn_name] = rf.active._descriptions.get(fn_name, '')
        return result

    # -- 解析 ----------------------------------------------------------

    def _current_sig(self) -> tuple[int, ...]:
        return tuple(id(p) for p in self._providers)

    def _resolved_functions(self) -> dict[str, ResolvedFn]:
        sig = self._current_sig()
        if (self._resolved_cache is not None
                and self._resolved_cache_key == sig):
            return self._resolved_cache

        resolved: dict[str, ResolvedFn] = {}
        # 先注册先赢
        for p in self._providers:
            for fn_name, fn in p._functions.items():
                if fn_name in resolved:
                    resolved[fn_name].shadowed.append(p)
                else:
                    resolved[fn_name] = ResolvedFn(active=p, fn=fn)

        # 触发 WARNING（同一签名 + 同一 fn 名只 WARN 一次）
        for fn_name, rf in resolved.items():
            if not rf.shadowed:
                continue
            key = (fn_name, sig)
            if key in self._warned_keys:
                continue
            self._warned_keys.add(key)
            shadowed_desc = ", ".join(
                f"{p.provider_kind}#{id(p):x}" for p in rf.shadowed)
            logger.warning(
                "namespace %r function %r: active=%s#%x, shadowed=[%s]",
                self._name, fn_name, rf.active.provider_kind, id(rf.active),
                shadowed_desc)

        self._resolved_cache = resolved
        self._resolved_cache_key = sig
        return resolved

    def invalidate(self) -> None:
        """显式失效缓存（registry 在 add/remove provider 后调用）。"""
        self._resolved_cache = None
        self._resolved_cache_key = None

    # -- 沙箱调用入口 ---------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # 私有属性 / dunder 不走懒触发
        if name.startswith('_'):
            raise AttributeError(name)
        providers = self.__dict__.get('_providers', [])
        if not providers:
            raise AttributeError(f"'{self._name}' has no providers")

        # 1) 直接命中
        resolved = self._resolved_functions()
        if name in resolved:
            return resolved[name].fn

        # 2) 触发各未连 provider 的 ensure_connected（懒触发）；
        #    单个 provider 失败不阻塞其他，最大化「至少能拿到一个 active 函数」
        for p in providers:
            conn = p._connection
            state = p.connection_state
            if conn is not None and state != "connected":
                future = asyncio.run_coroutine_threadsafe(
                    conn.ensure_connected(), conn.main_loop)
                try:
                    future.result(timeout=30)
                except Exception:
                    pass

        # 3) 重新解析（providers 列表对象引用不变，但内部函数表可能更新；
        #    invalidate cache key 强制 rebuild）
        self.invalidate()
        resolved = self._resolved_functions()
        if name in resolved:
            return resolved[name].fn
        raise AttributeError(f"'{self._name}' has no function '{name}'")

    def __repr__(self) -> str:
        n = sum(len(p._functions) for p in self._providers)
        return (f"<MergedNamespaceView '{self._name}' "
                f"({len(self._providers)} providers, {n} functions)>")


# ---------------------------------------------------------------------------
# NamespaceRegistry — multi-provider 注册表
# ---------------------------------------------------------------------------


# 给 type hint / isinstance 用的统一类型别名
NamespaceLike = "Namespace | MergedNamespaceView"


# ---------------------------------------------------------------------------
# 模块级 helper：view / Namespace 统一的“选主”访问入口
# ---------------------------------------------------------------------------

def primary_of(ns: "Namespace | MergedNamespaceView") -> "Namespace":
    """统一访问主 provider — ``Namespace`` / ``MergedNamespaceView`` 通吃。

    - ``Namespace``：返回自身
    - ``MergedNamespaceView``：返回 ``view.primary``（无 displayed 时退化首个 provider）

    用于消费者（help 渲染、share export、adapter 描述提取）避免到处写 ``isinstance``。
    """
    if isinstance(ns, MergedNamespaceView):
        return ns.primary or ns._providers[0]
    return ns


def displayed_of(ns: "Namespace | MergedNamespaceView") -> list["Namespace"]:
    """统一访问 displayed providers 列表。

    单 ``Namespace`` 返回空列表（表示“单 provider 路径，无多 provider 渲染”）。
    """
    if isinstance(ns, MergedNamespaceView):
        return ns.displayed
    return []


def flatten_view(view: MergedNamespaceView) -> Namespace:
    """把 multi-provider view 拍平成对端可见的单 ``Namespace``。

    - ``description`` / ``provider_kind`` 走 :func:`primary_of`
    - ``functions`` / ``descriptions`` 走 view 合并后的 active 集
      （与 exec_code 路径函数可见集完全一致）

    拍平后的临时 ``Namespace`` **不挂** ``_connection`` / state：
    对端拿到的是“快照”，不应感知本端的 MCP 连接细节。
    """
    p = primary_of(view)
    flat = Namespace(view.name, description=p._description,
                     provider_kind=p.provider_kind)
    # view._functions / _descriptions 已是 active 视角的合并集
    for fn_name, fn in view._functions.items():
        flat.register(fn_name, fn, view._descriptions.get(fn_name, ""))
    return flat


class NamespaceRegistry:
    """管理所有命名空间，提供 help() 按需查询。

    存储模型：``_namespaces: dict[str, list[Namespace]]``。
    单 provider 名通过 ``get(name)`` 返回 ``Namespace``；
    多 provider 名返回 :class:`MergedNamespaceView`（实例 stable，
    跨调用复用，避免 WARN-once 失效）。
    """

    def __init__(self) -> None:
        self._namespaces: dict[str, list[Namespace]] = {}
        # 同名 2+ providers 时缓存 view 实例，保持 warn 状态稳定
        self._views: dict[str, MergedNamespaceView] = {}

    def add(self, ns: Namespace) -> None:
        """append 一个 provider；不替换已存在的同名 ns。"""
        lst = self._namespaces.setdefault(ns.name, [])
        lst.append(ns)
        v = self._views.get(ns.name)
        if v is not None:
            v.invalidate()

    def remove(self, name: str) -> None:
        """按 name 移除该名下**所有** providers（保留向后兼容签名）。

        新增的 :meth:`remove_provider` 才是按实例移除。
        """
        self._namespaces.pop(name, None)
        self._views.pop(name, None)

    def remove_provider(self, ns: Namespace) -> bool:
        """按实例移除一个 provider。list 空了 pop key + view。

        Returns:
            ``True`` 若确实移除了一项；``False`` 表示未找到。
        """
        lst = self._namespaces.get(ns.name)
        if not lst:
            return False
        try:
            lst.remove(ns)
        except ValueError:
            return False
        if not lst:
            self._namespaces.pop(ns.name, None)
            self._views.pop(ns.name, None)
        else:
            v = self._views.get(ns.name)
            if v is not None:
                v.invalidate()
        return True

    def get(self, name: str) -> NamespaceLike | None:
        """获取一个 namespace。多 provider 时返回 view，单 provider 时返回原 Namespace。"""
        lst = self._namespaces.get(name)
        if not lst:
            return None
        if len(lst) == 1:
            return lst[0]
        v = self._views.get(name)
        if v is None:
            v = MergedNamespaceView(name, lst)
            self._views[name] = v
        return v

    def build_namespace_dict(self) -> dict[str, Any]:
        """构建注入 sandbox 的命名空间字典。

        包含所有命名空间对象（按需 wrap 为 view）+ help。
        """
        ns_dict: dict[str, Any] = {}
        for name in self._namespaces:
            ns_dict[name] = self.get(name)
        ns_dict['help'] = self._make_help()
        return ns_dict

    def _make_help(self) -> Callable:
        registry = self

        def help(func_or_name: Any = None) -> str:
            """查看 namespace / 函数的文档。

            - help()                           列出所有 namespace
            - help(namespace)                  namespace 完整说明 + 函数列表
            - help(namespace.function)         函数签名 + 完整文档
            - help("namespace.function")       同上（字符串形式）
            """
            # Layer 1: 列所有 namespace
            if func_or_name is None:
                return _render_registry(registry)

            # Layer 2: 聚焦某个 namespace（含 view）
            if isinstance(func_or_name, (Namespace, MergedNamespaceView)):
                return _render_namespace(func_or_name)

            # Layer 3: 聚焦某个函数
            if callable(func_or_name):
                return _render_function(func_or_name)

            if isinstance(func_or_name, str):
                parts = func_or_name.split('.', 1)
                if len(parts) == 2:
                    ns = registry.get(parts[0])
                    if ns is not None and parts[1] in ns._functions:
                        return _render_function(ns._functions[parts[1]],
                                                ns_name=parts[0],
                                                fn_name=parts[1])
                return f"(no documentation for '{func_or_name}')"

            return "(no documentation)"

        return help


# ---------------------------------------------------------------------------
# 渲染函数 — 分层显示
# ---------------------------------------------------------------------------

def _format_state_label(state: str | None, error: str | None) -> str:
    """渲染连接状态标签。connected / None 不显示标签。"""
    if state in (None, "connected"):
        return ""
    if state == "connecting":
        return "[connecting...]"
    if state == "disconnected":
        return "[disconnected]"
    if state == "failed":
        reason = (error or "").strip().splitlines()[0] if error else ""
        if len(reason) > 60:
            reason = reason[:57] + "..."
        return f"[failed: {reason}]" if reason else "[failed]"
    return f"[{state}]"


def _format_function_count(ns: "NamespaceLike") -> str:
    """函数数显示：连过的显示真实数；从未连过的（MCP 且无函数）显示 (? functions)。"""
    count = len(ns._functions)
    is_mcp = ns._connection is not None
    state = ns.connection_state
    if is_mcp and count == 0 and state != "connected":
        return "(? functions)"
    return f"({count} functions)"


def _displayed_providers(ns: "NamespaceLike") -> list[Namespace]:
    """[Deprecated alias] 请改用 :func:`displayed_of`。

    原算法已上提为 ``MergedNamespaceView.displayed`` property（带缓存），
    本函数仅为保证外部 import 不断裂而保留。
    """
    return displayed_of(ns)


def _render_registry(registry: "NamespaceRegistry") -> str:
    """Layer 1: 列所有 namespace（首行摘要）。"""
    names = sorted(registry._namespaces.keys())
    if not names:
        return "No namespaces registered."

    max_name = max(len(n) for n in names)

    lines = ["Available namespaces:", ""]
    for name in names:
        ns = registry.get(name)
        if ns is None:
            continue
        desc = _first_line(ns._description)
        count_text = _format_function_count(ns)
        label = _format_state_label(ns.connection_state, ns.connection_error)
        padded = f"{name:<{max_name}}"
        # multi-provider badge — 按 displayed（贡献函数的 provider）数算
        provider_badge = ""
        displayed = displayed_of(ns)
        if len(displayed) > 1:
            provider_badge = f"[{len(displayed)} providers]"
        suffix_parts: list[str] = []
        if provider_badge:
            suffix_parts.append(provider_badge)
        if label:
            suffix_parts.append(label)
        if desc:
            suffix_parts.append(f"— {desc}")
        suffix_parts.append(count_text)
        lines.append(f"  {padded} " + " ".join(suffix_parts))
    lines.append("")
    lines.append("Use help(<namespace>) for details, "
                 "e.g. help(" + names[0] + ").")
    return '\n'.join(lines)


def _provider_label(p: Namespace) -> str:
    """给单个 provider 渲染一行简要标签：'<kind>, <state>'。"""
    state = p.connection_state or "—"
    return f"{p.provider_kind}, {state}"


def _render_namespace(ns: "NamespaceLike") -> str:
    """Layer 2: namespace 完整 description + 函数首行摘要列表。

    支持 ``Namespace`` 和 ``MergedNamespaceView``。
    多 provider 时额外渲染 Providers 段，并在每个函数后标注归属 +
    shadowed 列表（如有）。
    """
    lines = [f"Namespace: {ns._name}", ""]

    desc = ns._description.strip() if ns._description else ""
    if desc:
        lines.append(desc)
        lines.append("")

    # 多 provider 时列出 providers + 状态（仅算 displayed = 贡献函数的 provider）
    displayed = displayed_of(ns)
    is_multi = len(displayed) > 1
    if is_multi:
        lines.append(f"Providers ({len(displayed)}):")
        for i, p in enumerate(displayed, 1):
            label = _format_state_label(p.connection_state, p.connection_error)
            label_str = f" {label}" if label else ""
            lines.append(f"  [{i}] kind={p.provider_kind}, "
                         f"state={p.connection_state or '—'}, "
                         f"functions={len(p._functions)}{label_str}")
        lines.append("")

    # 单 provider Namespace 的失败状态 hint（保持原行为）
    if isinstance(ns, Namespace):
        if ns.connection_state == "failed":
            reason = (ns.connection_error or "").strip() or "(unknown)"
            lines.append(f"⚠ Connection failed: {reason}")
            last_attempt = None
            connection = ns._connection
            if connection is not None and getattr(connection, "last_attempt_at", None):
                last_attempt = time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(connection.last_attempt_at))
            if last_attempt:
                lines.append(f"  Last attempt: {last_attempt}")
            lines.append("  Calling any function will retry the connection.")
            lines.append("")
        elif ns.connection_state in ("connecting", "disconnected") and ns._connection is not None:
            label = ns.connection_state
            lines.append(f"(connection state: {label})")
            lines.append("")

    # 函数列表：多 displayed provider 才标注归属（编号引用 Providers 段）；
    # 单 displayed provider 退化为单 provider 路径，无 [from ...] 标签
    if is_multi:
        idx_map = {id(p): i for i, p in enumerate(displayed, 1)}
        resolved = ns._resolved_functions()  # type: ignore[union-attr]
        count = len(resolved)
        lines.append(f"{count} Functions:")
        lines.append("")
        if resolved:
            fnames = sorted(resolved.keys())
            max_fname = max(len(f) for f in fnames)
            for fname in fnames:
                rf = resolved[fname]
                fdesc = _first_line(rf.active._descriptions.get(fname, ''))
                padded = f"{fname:<{max_fname}}"
                # active 必然在 displayed 中（自带函数）；防御式 fallback 到 ?
                active_idx = idx_map.get(id(rf.active), '?')
                origin = f"[from #{active_idx}]"
                shadow = ""
                if rf.shadowed:
                    shadow_idx = [
                        f"#{idx_map.get(id(p), '?')}" for p in rf.shadowed]
                    shadow = " (shadowed: " + ", ".join(shadow_idx) + ")"
                if fdesc:
                    lines.append(f"  {padded}  {fdesc}  {origin}{shadow}")
                else:
                    lines.append(f"  {padded}  {origin}{shadow}")
    else:
        # 单 provider 路径 — 与改造前一致
        functions = ns._functions
        descriptions = ns._descriptions
        count = len(functions)
        lines.append(f"{count} Functions:")
        lines.append("")
        if functions:
            fnames = sorted(functions.keys())
            max_fname = max(len(f) for f in fnames)
            for fname in fnames:
                fdesc = _first_line(descriptions.get(fname, ''))
                padded = f"{fname:<{max_fname}}"
                if fdesc:
                    lines.append(f"  {padded}  {fdesc}")
                else:
                    lines.append(f"  {padded}")

    lines.append("")
    lines.append(f"Use help({ns._name}.<function>) for function details.")
    return '\n'.join(lines)


def _render_function(func: Callable, ns_name: str = "",
                     fn_name: str = "") -> str:
    """Layer 3: 函数签名 + 完整 docstring。"""
    name = fn_name or getattr(func, '__name__', str(func))
    prefix = ns_name or getattr(func, '__namespace__', '')
    qualified = f"{prefix}.{name}" if prefix else name

    doc = getattr(func, '__doc__', None) or '(no documentation)'
    try:
        sig = inspect.signature(func)
        return f"{qualified}{sig}\n\n{doc}"
    except (ValueError, TypeError):
        return f"{qualified}\n\n{doc}"
