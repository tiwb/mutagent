"""mutagent.builtins.web_jina -- Jina AI 搜索和获取实现。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from urllib.parse import quote

import httpx

import mutagent
from mutio.net.client import HttpClient
from mutagent.toolkits.web_toolkit import FetchImpl, SearchImpl

if TYPE_CHECKING:
    from mutagent.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_SEARCH_API = "https://s.jina.ai/"
_READER_API = "https://r.jina.ai/"
_TIMEOUT = 30
_MAX_CONTENT_CHARS = 50000


# ---------------------------------------------------------------------------
# 请求头
# ---------------------------------------------------------------------------

def _get_headers(config: Config) -> dict[str, str]:
    """构建请求头，包含可选的 API key。"""
    headers: dict[str, str] = {"Accept": "application/json"}
    api_key = config.get("WebToolkit.jina_api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


# ---------------------------------------------------------------------------
# JinaSearchImpl
# ---------------------------------------------------------------------------

class JinaSearchImpl(SearchImpl):
    """Jina AI 搜索。"""

    name = "jina"
    description = "Jina Search API"
    config: Config

    async def search(self, query: str, max_results: int = 5) -> str:
        """通过 Jina Search API 搜索 Web。"""
        ...


@mutagent.impl(JinaSearchImpl.search)
async def _jina_search(self: JinaSearchImpl, query: str, max_results: int = 5) -> str:
    """Jina Search API 实现。"""
    encoded_query = quote(query)
    url = f"{_SEARCH_API}{encoded_query}"
    headers = _get_headers(self.config)

    try:
        async with HttpClient.create() as client:
            resp = await client.get(url, headers=headers, timeout=_TIMEOUT)
            if resp.status_code in (401, 429):
                body = resp.text[:500]
                raise RuntimeError(
                    f"Jina API HTTP {resp.status_code}.\n"
                    f"Response: {body}"
                )
            resp.raise_for_status()
    except httpx.TimeoutException:
        raise RuntimeError(f"Search timed out ({_TIMEOUT}s). Please try again later.")
    except httpx.HTTPError as exc:
        logger.warning("Web search failed: %s", exc)
        raise RuntimeError(f"Search failed: {exc}") from exc

    try:
        data = resp.json()
    except ValueError:
        return "Search returned an unparseable response."

    items = data.get("data", [])
    if not items:
        return f"No results found for \"{query}\"."

    items = items[:max_results]

    parts: list[str] = []
    parts.append(f"Search results for \"{query}\" ({len(items)} items)\n")
    for i, item in enumerate(items, 1):
        title = item.get("title", "(untitled)")
        item_url = item.get("url", "")
        description = item.get("description", "")
        parts.append(f"### {i}. {title}")
        if item_url:
            parts.append(f"URL: {item_url}")
        if description:
            parts.append(f"{description}")
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# JinaFetchImpl
# ---------------------------------------------------------------------------

class JinaFetchImpl(FetchImpl):
    """Jina Reader API 获取（需配置 API key）。"""

    name = "jina"
    description = "Jina Reader API"
    config: Config

    async def fetch(self, url: str, format: str = "markdown") -> str:
        """通过 Jina Reader API 获取网页内容。"""
        ...


@mutagent.impl(JinaFetchImpl.fetch)
async def _jina_fetch(self: JinaFetchImpl, url: str, format: str = "markdown") -> str:
    """Jina Reader API 实现。"""
    if format == "html":
        return "Jina Reader API does not support html format. Use local impl or format=\"markdown\"."

    reader_url = f"{_READER_API}{url}"
    headers = _get_headers(self.config)

    try:
        async with HttpClient.create() as client:
            resp = await client.get(reader_url, headers=headers, timeout=_TIMEOUT)
            if resp.status_code in (401, 429):
                body = resp.text[:500]
                raise RuntimeError(
                    f"Jina API HTTP {resp.status_code}.\n"
                    f"Response: {body}"
                )
            resp.raise_for_status()
    except httpx.TimeoutException:
        raise RuntimeError(f"Fetch timed out ({_TIMEOUT}s). Please try again later.")
    except httpx.HTTPError as exc:
        logger.warning("Web fetch failed for %s: %s", url, exc)
        raise RuntimeError(f"Fetch failed: {exc}") from exc

    try:
        data = resp.json()
    except ValueError:
        content = resp.text
        if len(content) > _MAX_CONTENT_CHARS:
            content = content[:_MAX_CONTENT_CHARS] + "\n\n[content truncated]"
        return content

    page = data.get("data", {})
    title = page.get("title", "")
    content = page.get("content", "")

    if not content:
        return f"Failed to extract content from {url}."

    parts: list[str] = []
    if title:
        parts.append(f"# {title}\n")
        parts.append(f"URL: {url}\n")
    if len(content) > _MAX_CONTENT_CHARS:
        content = content[:_MAX_CONTENT_CHARS] + "\n\n[内容已截断]"
    parts.append(content)

    return "\n".join(parts)
