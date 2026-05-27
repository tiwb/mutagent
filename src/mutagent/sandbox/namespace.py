"""mutagent.sandbox.namespace -- NamespaceTools Declaration。

本进程能力源的自动发现机制。子类的 public 方法自动注册为 sandbox 命名空间函数。
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

import mutobj


class NamespaceProtocol(Protocol):
    """命名空间对象，通过 . 访问其中的函数。
    """

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    def __getattr__(self, name: str) -> Any: ...


class NamespaceTools(mutobj.Declaration):
    """声明一组注入 sandbox 命名空间的函数。

    namespace 名从类名推导（去掉 ``Tools`` 后缀），或用 ``_namespace`` 显式指定。
    子类的 public 方法（不以 ``_`` 开头）自动注册为命名空间函数。

    Attributes:
        _namespace: 显式指定 namespace 名。None = 从类名自动推导。

    Example::

        class WebTools(NamespaceTools):
            def search(self, query: str) -> str:
                '''搜索网页。'''  # → web.search()
                ...

            def fetch(self, url: str) -> str:
                '''获取网页内容。'''  # → web.fetch()
                ...
    """

    _namespace: ClassVar[str | None] = None


from . import _namespace_impl as _namespace_impl  # noqa: E402,F401
