"""命名空间机制 — 能力源函数的分组访问和按需查询。"""

import asyncio
import inspect
import time
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from mutagent.sandbox._adapter_mcp import MCPConnection


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

    def __init__(self, name: str, description: str = ""):
        self._name = name
        self._description = description
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


class NamespaceRegistry:
    """管理所有命名空间，提供 help() 按需查询。"""

    def __init__(self):
        self._namespaces: dict[str, Namespace] = {}

    def add(self, ns: Namespace) -> None:
        self._namespaces[ns.name] = ns

    def remove(self, name: str) -> None:
        self._namespaces.pop(name, None)

    def get(self, name: str) -> Namespace | None:
        return self._namespaces.get(name)

    def build_namespace_dict(self) -> dict[str, Any]:
        """构建注入 sandbox 的命名空间字典。

        包含所有命名空间对象 + help。
        """
        ns_dict: dict[str, Any] = {}

        for name, ns in self._namespaces.items():
            ns_dict[name] = ns

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

            # Layer 2: 聚焦某个 namespace
            if isinstance(func_or_name, Namespace):
                return _render_namespace(func_or_name)

            # Layer 3: 聚焦某个函数
            if callable(func_or_name):
                return _render_function(func_or_name)

            if isinstance(func_or_name, str):
                parts = func_or_name.split('.', 1)
                if len(parts) == 2:
                    ns = registry.get(parts[0])
                    if ns and parts[1] in ns._functions:
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


def _format_function_count(ns: "Namespace") -> str:
    """函数数显示：连过的显示真实数；从未连过的（MCP 且无函数）显示 (? functions)。"""
    count = len(ns._functions)
    is_mcp = ns._connection is not None
    state = ns.connection_state
    if is_mcp and count == 0 and state != "connected":
        return "(? functions)"
    return f"({count} functions)"


def _render_registry(registry: "NamespaceRegistry") -> str:
    """Layer 1: 列所有 namespace（首行摘要）。"""
    names = sorted(registry._namespaces.keys())
    if not names:
        return "No namespaces registered."

    max_name = max(len(n) for n in names)

    lines = ["Available namespaces:", ""]
    for name in names:
        ns = registry._namespaces[name]
        desc = _first_line(ns._description)
        count_text = _format_function_count(ns)
        label = _format_state_label(ns.connection_state, ns.connection_error)
        padded = f"{name:<{max_name}}"
        # 状态标签紧跟 namespace 名后；desc 在标签之后
        suffix_parts: list[str] = []
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


def _render_namespace(ns: Namespace) -> str:
    """Layer 2: namespace 完整 description + 函数首行摘要列表。"""
    lines = [f"Namespace: {ns._name}", ""]

    desc = ns._description.strip() if ns._description else ""
    if desc:
        lines.append(desc)
        lines.append("")

    # MCP 失败状态附 hint
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

    count = len(ns._functions)
    lines.append(f"{count} Functions:")
    lines.append("")

    if ns._functions:
        fnames = sorted(ns._functions.keys())
        max_fname = max(len(f) for f in fnames)
        for fname in fnames:
            fdesc = _first_line(ns._descriptions.get(fname, ''))
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
