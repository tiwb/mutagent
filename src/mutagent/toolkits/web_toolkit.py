"""mutagent.toolkits.web_toolkit -- WebToolkit 及 SearchImpl/FetchImpl 声明。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mutobj
from mutagent.core.tools import Toolkit

if TYPE_CHECKING:
    from mutagent.app.config import Config
    from mutagent.core.messages import ToolSchema


# ---------------------------------------------------------------------------
# Provider 抽象基类
# ---------------------------------------------------------------------------

class SearchImpl(mutobj.Declaration):
    """搜索实现基类。

    子类通过 mutobj 子类发现机制自动注册。
    每个子类代表一种搜索后端（如 Jina、SearXNG）。

    子类需设置 ``name`` 和 ``description`` 类变量。
    """

    name = ""
    description = ""

    async def search(self, query: str, max_results: int = 5) -> str:
        """执行搜索并返回格式化结果。

        Args:
            query: 搜索关键词。
            max_results: 最大返回结果数。
        """
        ...


class FetchImpl(mutobj.Declaration):
    """网页内容提取实现基类。

    负责将原始 HTML 转换为 clean HTML 或 Markdown。
    ``raw`` 格式由 WebToolkit 内置处理，不经过 FetchImpl。

    子类需设置 ``name`` 和 ``description`` 类变量。
    """

    name = ""
    description = ""

    async def fetch(self, url: str, format: str = "markdown") -> str:
        """获取并提取网页内容。

        Args:
            url: 网页 URL。
            format: 输出格式 — ``"markdown"`` 或 ``"html"``。
        """
        ...


# ---------------------------------------------------------------------------
# WebToolkit
# ---------------------------------------------------------------------------

class WebToolkit(Toolkit):
    """Web 信息检索工具集。

    Attributes:
        config: 配置对象，用于读取 API key 等设置。
    """

    config: Config

    def search(self, query: str, max_results: int = 5, impl: str = "") -> str:
        """搜索 Web 并返回结果摘要。

        Args:
            query: 搜索关键词。
            max_results: 最大返回结果数。
            impl: 使用的搜索实现。
        """
        ...

    def fetch(self, url: str, format: str = "markdown", impl: str = "") -> str:
        """读取网页内容并返回文本。

        Args:
            url: 要读取的网页 URL。
            format: 输出格式 — ``"markdown"``（默认）、``"html"``（提取后的正文）、``"raw"``（原始网页）。
            impl: 使用的获取实现。
        """
        ...

    def _customize_schema(self, method_name: str, schema: ToolSchema) -> ToolSchema:
        """根据已发现的实现动态调整工具 schema。"""
        ...


from . import _web_toolkit_impl as _web_toolkit_impl, _web_toolkit_impl_jina as _web_toolkit_impl_jina  # noqa: F401, E402
