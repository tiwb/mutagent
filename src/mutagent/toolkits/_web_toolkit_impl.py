"""mutagent.toolkits._web_toolkit_impl -- WebToolkit 实现：raw 内置 + provider 分发 + 动态描述。"""

from __future__ import annotations

import logging

import httpx
import mutobj

from mutio.net.client import HttpClient
from mutagent.core.messages import ToolSchema
from .web_toolkit import FetchImpl, SearchImpl, WebToolkit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_TIMEOUT = 30
_MAX_CONTENT_CHARS = 50000


# ---------------------------------------------------------------------------
# Provider 发现
# ---------------------------------------------------------------------------

def _discover_impls(base_cls: type) -> dict[str, type]:
    """发现所有已注册的实现子类，返回 {name: cls} 映射。"""
    return {cls.name: cls for cls in mutobj.discover_subclasses(base_cls)}


# ---------------------------------------------------------------------------
# 内置 raw 获取
# ---------------------------------------------------------------------------

async def _httpx_get_raw(url: str) -> str:
    """内置 raw 获取，仅依赖 httpx。"""
    try:
        async with HttpClient.create() as client:
            resp = await client.get(url, timeout=_TIMEOUT, follow_redirects=True)
            resp.raise_for_status()
    except httpx.TimeoutException:
        return f"Fetch timed out ({_TIMEOUT}s). Please try again later."
    except httpx.HTTPError as exc:
        logger.warning("Web fetch failed for %s: %s", url, exc)
        return f"Fetch failed: {exc}"

    content = resp.text
    if len(content) > _MAX_CONTENT_CHARS:
        content = content[:_MAX_CONTENT_CHARS] + "\n\n[content truncated]"
    return f"# {url}\n\n{content}"


# ---------------------------------------------------------------------------
# WebToolkit.search 实现
# ---------------------------------------------------------------------------

@mutobj.impl(WebToolkit.search)
async def web_toolkit_search(self: WebToolkit, query: str, max_results: int = 5, impl: str = "") -> str:
    """搜索 Web 并返回结果摘要。"""
    search_impls = _discover_impls(SearchImpl)
    if not search_impls:
        return "No search implementation available. Ensure a SearchImpl is registered."
    name = impl or "jina"
    impl_cls = search_impls.get(name)
    if impl_cls is None:
        available = ", ".join(search_impls.keys())
        return f"Unknown search impl \"{name}\". Available: {available}"
    instance = impl_cls(config=self.config)
    return await instance.search(query, max_results)


# ---------------------------------------------------------------------------
# WebToolkit.fetch 实现
# ---------------------------------------------------------------------------

@mutobj.impl(WebToolkit.fetch)
async def web_toolkit_fetch(self: WebToolkit, url: str, format: str = "markdown", impl: str = "") -> str:
    """读取网页内容并返回文本。"""
    # raw 格式：WebToolkit 内置，不需要 FetchImpl
    if format == "raw":
        return await _httpx_get_raw(url)

    # html/markdown 格式：需要 FetchImpl
    fetch_impls = _discover_impls(FetchImpl)
    if not fetch_impls:
        return (
            f"Format \"{format}\" requires content extraction dependencies.\n"
            "Currently only format=\"raw\" (raw HTML) is supported.\n\n"
            "Install local extraction: pip install mutagent[web-extract]"
        )
    name = impl or "local"
    impl_cls = fetch_impls.get(name)
    if impl_cls is None:
        available = ", ".join(fetch_impls.keys())
        return f"Unknown fetch impl \"{name}\". Available: {available}"
    instance = impl_cls(config=self.config)
    return await instance.fetch(url, format)


# ---------------------------------------------------------------------------
# WebToolkit._customize_schema 实现
# ---------------------------------------------------------------------------

@mutobj.impl(WebToolkit._customize_schema)
def web_toolkit_customize_schema(self: WebToolkit, method_name: str, schema: ToolSchema) -> ToolSchema:
    """根据已发现的实现动态调整工具 schema。"""
    if method_name == "search":
        impls = mutobj.discover_subclasses(SearchImpl)
        if impls:
            impl_list = ", ".join(f"{c.name} ({c.description})" for c in impls)
            desc = f"Search the web and return result summaries. Available: {impl_list}."
        else:
            desc = "Search the web and return result summaries. (no search impl available)"
        return ToolSchema(
            name=schema.name,
            description=desc,
            input_schema=schema.input_schema,
        )

    if method_name == "fetch":
        fetch_impls = mutobj.discover_subclasses(FetchImpl)
        props = dict(schema.input_schema.get("properties", {}))
        if fetch_impls:
            impl_list = ", ".join(f"{c.name} ({c.description})" for c in fetch_impls)
            desc = f"Fetch web page content as text. Available: {impl_list}."
            props["format"] = {
                "type": "string",
                "description": (
                    'Output format — "markdown" (default), '
                    '"html" (extracted body), "raw" (raw HTML).'
                ),
                "default": "markdown",
            }
            props["impl"] = {
                "type": "string",
                "description": f"Extraction impl (default \"{fetch_impls[0].name}\").",
                "default": "",
            }
        else:
            desc = "Fetch raw web page content (HTML)."
            props.pop("format", None)
            props.pop("impl", None)
        new_input = dict(schema.input_schema)
        new_input["properties"] = props
        new_input["required"] = ["url"]
        return ToolSchema(
            name=schema.name,
            description=desc,
            input_schema=new_input,
        )

    return schema
