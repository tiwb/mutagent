"""mutagent.toolkits._web_toolkit_impl_local -- Local content extraction implementation (readability + markdownify).

依赖 ``readability-lxml`` 和 ``markdownify``，通过 ``pip install mutagent[web-extract]`` 安装。
如果依赖未安装，本模块 import 会失败，LocalFetchImpl 不会注册到 class registry。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx
from markdownify import markdownify as md
from readability import Document as ReadabilityDoc

import mutobj
from mutio.net.client import HttpClient
from .web_toolkit import FetchImpl

if TYPE_CHECKING:
    from mutagent.app.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_TIMEOUT = 30
_MAX_CONTENT_CHARS = 50000


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------

async def _httpx_get(url: str) -> str:
    """通过 httpx 获取网页原始 HTML。"""
    async with HttpClient.create() as client:
        resp = await client.get(url, timeout=_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
    return resp.text


def _extract(html: str, url: str) -> tuple[str, str]:
    """使用 readability 提取正文。

    Returns:
        (title, clean_html) 元组。
    """
    doc = ReadabilityDoc(html, url=url)
    title = doc.short_title()
    clean_html = doc.summary(html_partial=True)
    return title, clean_html


def _to_markdown(html: str) -> str:
    """使用 markdownify 将 HTML 转换为 Markdown。"""
    return md(html, heading_style="ATX", strip=["img"])


def _format_result(title: str, content: str, url: str) -> str:
    """格式化输出结果，添加标题和截断。"""
    if len(content) > _MAX_CONTENT_CHARS:
        content = content[:_MAX_CONTENT_CHARS] + "\n\n[content truncated]"
    parts: list[str] = []
    if title:
        parts.append(f"# {title}\n")
        parts.append(f"URL: {url}\n")
    parts.append(content)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LocalFetchImpl
# ---------------------------------------------------------------------------

class LocalFetchImpl(FetchImpl):
    """本地内容提取（readability + markdownify）。"""

    name = "local"
    description = "Local extraction"
    config: Config

    async def fetch(self, url: str, format: str = "markdown") -> str:
        """获取并提取网页内容。"""
        ...


@mutobj.impl(LocalFetchImpl.fetch)
async def _local_fetch(self: LocalFetchImpl, url: str, format: str = "markdown") -> str:
    """本地提取实现。"""
    try:
        raw_html = await _httpx_get(url)
    except httpx.TimeoutException:
        return f"Fetch timed out ({_TIMEOUT}s). Please try again later."
    except httpx.HTTPError as exc:
        logger.warning("Web fetch failed for %s: %s", url, exc)
        return f"Fetch failed: {exc}"

    title, clean_html = _extract(raw_html, url)

    if format == "html":
        return _format_result(title, clean_html, url)

    # markdown（默认）
    markdown_content = _to_markdown(clean_html)
    return _format_result(title, markdown_content, url)
