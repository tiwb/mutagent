"""命名空间机制 — 能力源函数的分组访问和按需查询。"""

import inspect
from typing import Any, Callable


class Namespace:
    """命名空间对象，通过 . 访问其中的函数。

    >>> ns = Namespace("browser")
    >>> ns.register("navigate", some_func, "Navigate to URL")
    >>> ns.navigate(url="https://example.com")
    """

    def __init__(self, name: str):
        self._name = name
        self._functions: dict[str, Callable] = {}
        self._descriptions: dict[str, str] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, func_name: str, func: Callable,
                 description: str = "") -> None:
        """注册一个函数到命名空间。"""
        self._functions[func_name] = func
        self._descriptions[func_name] = description or (
            func.__doc__.strip().split('\n')[0] if func.__doc__ else '')

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(name)
        if name in self._functions:
            return self._functions[name]
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
            """查看函数的详细文档。

            用法: help(playwright.browser_navigate)
            """
            if func_or_name is None:
                return "Usage: help(function) — e.g. help(playwright.browser_navigate)"

            if callable(func_or_name):
                name = getattr(func_or_name, '__name__', str(func_or_name))
                doc = getattr(func_or_name, '__doc__', None) or '(no documentation)'
                try:
                    sig = inspect.signature(func_or_name)
                    return f"{name}{sig}\n{doc}"
                except (ValueError, TypeError):
                    return f"{name}\n{doc}"

            if isinstance(func_or_name, str):
                # 尝试 "namespace.func" 格式
                parts = func_or_name.split('.', 1)
                if len(parts) == 2:
                    ns = registry.get(parts[0])
                    if ns and parts[1] in ns._functions:
                        fn = ns._functions[parts[1]]
                        doc = fn.__doc__ or '(no documentation)'
                        return f"{func_or_name}\n{doc}"
                return f"(no documentation for '{func_or_name}')"

            if isinstance(func_or_name, Namespace):
                lines = [f"Namespace: {func_or_name.name}", ""]
                for fname in sorted(func_or_name._functions):
                    func = func_or_name._functions[fname]
                    desc = func_or_name._descriptions.get(fname, '')
                    entry = f"  {fname}"
                    if desc:
                        entry += f" — {desc}"
                    lines.append(entry)
                return '\n'.join(lines)

            return "(no documentation)"

        return help
