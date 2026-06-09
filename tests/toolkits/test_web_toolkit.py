"""Tests for WebToolkit declaration, schema, provider discovery, and implementations."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import mutagent
import mutobj
from mutagent.app.config import Config
from mutagent.core.messages import ToolUseBlock, ToolSchema
from mutagent.toolkits._web_toolkit_impl_jina import JinaSearchImpl, JinaFetchImpl
from mutagent.toolkits._web_toolkit_impl_local import LocalFetchImpl
from mutagent.toolkits.web_toolkit import FetchImpl, SearchImpl, WebToolkit
from mutagent.core.tools import ToolSet
from mutobj import get_declaration_func
from mutagent.core._tools_impl import _make_schema

import mutagent.toolkits._web_toolkit_impl_local  # noqa: F401  -- register LocalFetchImpl


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """无 API key 的空配置。"""
    c = Config()
    c.load_from_dict({})
    return c


@pytest.fixture
def config_with_key():
    """包含 Jina API key 的配置。"""
    c = Config()
    c.load_from_dict({"WebToolkit": {"jina_api_key": "test-key-123"}})
    return c


@pytest.fixture
def toolkit(config):
    return WebToolkit(config=config)


@pytest.fixture
def toolkit_with_key(config_with_key):
    return WebToolkit(config=config_with_key)


@pytest.fixture
def tool_set(toolkit):
    ts = ToolSet()
    ts.add(toolkit)
    return ts


# ---------------------------------------------------------------------------
# httpx mock helpers
# ---------------------------------------------------------------------------

def _make_mock_client(response: MagicMock) -> AsyncMock:
    """创建模拟 httpx.AsyncClient 作为异步上下文管理器。"""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=response)
    return mock_client


def _make_mock_client_with_error(error: Exception) -> AsyncMock:
    """创建模拟 httpx.AsyncClient，其 get 方法抛出异常。"""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=error)
    return mock_client


# ---------------------------------------------------------------------------
# Declaration Tests
# ---------------------------------------------------------------------------

class TestWebToolkitDeclaration:

    def test_inherits_from_toolkit(self):
        assert issubclass(WebToolkit, mutagent.Toolkit)

    def test_inherits_from_declaration(self):
        assert issubclass(WebToolkit, mutobj.Declaration)

    def test_uses_declaration_meta(self):
        assert issubclass(WebToolkit, mutobj.Declaration)

    def test_declared_methods(self):
        assert mutobj.get_declaration_func(WebToolkit, "search") is not None
        assert mutobj.get_declaration_func(WebToolkit, "fetch") is not None

    def test_has_config_attribute(self, toolkit):
        assert hasattr(toolkit, "config")


class TestSearchImplDeclaration:

    def test_inherits_from_declaration(self):
        assert issubclass(SearchImpl, mutobj.Declaration)

    def test_has_name_and_description(self):
        assert JinaSearchImpl.name == "jina"
        assert JinaSearchImpl.description == "Jina Search API"


class TestFetchImplDeclaration:

    def test_inherits_from_declaration(self):
        assert issubclass(FetchImpl, mutobj.Declaration)

    def test_local_impl_has_name(self):
        assert LocalFetchImpl.name == "local"
        assert LocalFetchImpl.description == "Local extraction"

    def test_jina_impl_has_name(self):
        assert JinaFetchImpl.name == "jina"
        assert JinaFetchImpl.description == "Jina Reader API"


# ---------------------------------------------------------------------------
# Provider Discovery Tests
# ---------------------------------------------------------------------------

class TestProviderDiscovery:

    def test_discover_search_impls(self):
        impls = mutobj.discover_subclasses(SearchImpl)
        names = {cls.name for cls in impls}
        assert "jina" in names

    def test_discover_fetch_impls(self):
        impls = mutobj.discover_subclasses(FetchImpl)
        names = {cls.name for cls in impls}
        assert "local" in names
        assert "jina" in names


# ---------------------------------------------------------------------------
# Tool Registration Tests
# ---------------------------------------------------------------------------

class TestWebToolkitRegistration:

    def test_tool_names(self, tool_set):
        names = {s.name for s in tool_set.get_tools()}
        assert names == {"Web-search", "Web-fetch"}

    def test_tool_count(self, tool_set):
        assert len(tool_set.get_tools()) == 2

    def test_query_search(self, tool_set):
        schema = tool_set.query("Web-search")
        assert schema is not None
        assert isinstance(schema, ToolSchema)

    def test_query_fetch(self, tool_set):
        schema = tool_set.query("Web-fetch")
        assert schema is not None
        assert isinstance(schema, ToolSchema)

    def test_add_with_methods_filter(self, toolkit):
        ts = ToolSet()
        ts.add(toolkit, methods=["search"])
        names = {s.name for s in ts.get_tools()}
        assert names == {"Web-search"}


# ---------------------------------------------------------------------------
# Schema Tests
# ---------------------------------------------------------------------------

class TestWebToolkitSchema:

    def test_search_schema(self):
        decl = get_declaration_func(WebToolkit, "search") or getattr(WebToolkit, "search")
        schema = _make_schema(decl, "Web-search")
        assert schema.name == "Web-search"
        assert schema.description
        props = schema.input_schema["properties"]
        assert "query" in props
        assert "max_results" in props
        assert "query" in schema.input_schema["required"]
        assert "max_results" not in schema.input_schema.get("required", [])

    def test_fetch_schema(self):
        decl = get_declaration_func(WebToolkit, "fetch") or getattr(WebToolkit, "fetch")
        schema = _make_schema(decl, "Web-fetch")
        assert schema.name == "Web-fetch"
        assert schema.description
        props = schema.input_schema["properties"]
        assert "url" in props
        assert "url" in schema.input_schema["required"]


# ---------------------------------------------------------------------------
# Dynamic Schema Tests (_customize_schema)
# ---------------------------------------------------------------------------

class TestCustomizeSchema:

    def test_fetch_schema_has_format_and_impl(self, tool_set):
        """有 FetchImpl 时 schema 包含 format 和 impl。"""
        schema = tool_set.query("Web-fetch")
        props = schema.input_schema["properties"]
        assert "format" in props
        assert "impl" in props
        assert "local" in schema.description

    def test_search_schema_has_impl_info(self, tool_set):
        """search 描述包含已发现的搜索实现。"""
        schema = tool_set.query("Web-search")
        assert "jina" in schema.description


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

class TestWebToolkitConfig:

    def test_no_api_key(self, toolkit):
        assert toolkit.config.root.get("WebToolkit.jina_api_key") is None

    def test_with_api_key(self, toolkit_with_key):
        assert toolkit_with_key.config.root.get("WebToolkit.jina_api_key") == "test-key-123"


# ---------------------------------------------------------------------------
# Search Implementation Tests (JinaSearchImpl, mocked)
# ---------------------------------------------------------------------------

def _mock_search_response(items):
    """构造 Jina Search API 的模拟 JSON 响应。"""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "code": 200,
        "status": 20000,
        "data": [
            {
                "title": item["title"],
                "url": item["url"],
                "description": item.get("description", ""),
            }
            for item in items
        ],
    }
    resp.raise_for_status = MagicMock()
    return resp


class TestJinaSearchImpl:

    async def test_search_returns_results(self, tool_set):
        mock_resp = _mock_search_response([
            {"title": "Python", "url": "https://python.org", "description": "Official site"},
            {"title": "W3Schools", "url": "https://w3schools.com", "description": "Tutorials"},
        ])
        mock_client = _make_mock_client(mock_resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            block = ToolUseBlock(id="t1", name="Web-search", input={"query": "python"})
            result = await tool_set.dispatch(block)
        assert not result.is_error
        assert "Python" in result.content
        assert "https://python.org" in result.content

    async def test_search_respects_max_results(self, tool_set):
        mock_resp = _mock_search_response([
            {"title": f"Result {i}", "url": f"https://example.com/{i}", "description": ""}
            for i in range(10)
        ])
        mock_client = _make_mock_client(mock_resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            block = ToolUseBlock(id="t1", name="Web-search", input={"query": "test", "max_results": 3})
            result = await tool_set.dispatch(block)
        assert not result.is_error
        assert "Result 0" in result.content
        assert "Result 2" in result.content
        assert "Result 3" not in result.content

    async def test_search_empty_results(self, tool_set):
        mock_resp = _mock_search_response([])
        mock_client = _make_mock_client(mock_resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            block = ToolUseBlock(id="t1", name="Web-search", input={"query": "xyzzy123"})
            result = await tool_set.dispatch(block)
        assert not result.is_error
        assert "No results found" in result.content

    async def test_search_timeout(self, tool_set):
        mock_client = _make_mock_client_with_error(httpx.TimeoutException("timeout"))
        with patch("httpx.AsyncClient", return_value=mock_client):
            block = ToolUseBlock(id="t1", name="Web-search", input={"query": "test"})
            result = await tool_set.dispatch(block)
        assert result.is_error is True
        assert "timed out" in result.content

    async def test_search_request_error(self, tool_set):
        mock_client = _make_mock_client_with_error(
            httpx.ConnectError("connection refused")
        )
        with patch("httpx.AsyncClient", return_value=mock_client):
            block = ToolUseBlock(id="t1", name="Web-search", input={"query": "test"})
            result = await tool_set.dispatch(block)
        assert result.is_error is True
        assert "Search failed" in result.content

    async def test_search_sends_api_key(self, toolkit_with_key):
        mock_resp = _mock_search_response([])
        mock_client = _make_mock_client(mock_resp)
        ts = ToolSet()
        ts.add(toolkit_with_key)
        with patch("httpx.AsyncClient", return_value=mock_client):
            block = ToolUseBlock(id="t1", name="Web-search", input={"query": "test"})
            await ts.dispatch(block)
        call_kwargs = mock_client.get.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert headers.get("Authorization") == "Bearer test-key-123"

    async def test_search_401_friendly_message(self, tool_set):
        resp = MagicMock()
        resp.status_code = 401
        resp.text = "Unauthorized"
        mock_client = _make_mock_client(resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            block = ToolUseBlock(id="t1", name="Web-search", input={"query": "test"})
            result = await tool_set.dispatch(block)
        assert result.is_error is True
        assert "401" in result.content

    async def test_search_429_friendly_message(self, tool_set):
        resp = MagicMock()
        resp.status_code = 429
        resp.text = "Too Many Requests"
        mock_client = _make_mock_client(resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            block = ToolUseBlock(id="t1", name="Web-search", input={"query": "test"})
            result = await tool_set.dispatch(block)
        assert result.is_error is True
        assert "429" in result.content


# ---------------------------------------------------------------------------
# Fetch Raw (built-in) Tests
# ---------------------------------------------------------------------------

class TestFetchRaw:

    async def test_raw_returns_html(self, tool_set):
        resp = MagicMock()
        resp.text = "<html><body>Hello</body></html>"
        resp.raise_for_status = MagicMock()
        mock_client = _make_mock_client(resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            block = ToolUseBlock(id="t1", name="Web-fetch",
                                 input={"url": "https://example.com", "format": "raw"})
            result = await tool_set.dispatch(block)
        assert not result.is_error
        assert "<html>" in result.content
        assert "Hello" in result.content

    async def test_raw_truncates_long_content(self, tool_set):
        resp = MagicMock()
        resp.text = "x" * 60000
        resp.raise_for_status = MagicMock()
        mock_client = _make_mock_client(resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            block = ToolUseBlock(id="t1", name="Web-fetch",
                                 input={"url": "https://example.com", "format": "raw"})
            result = await tool_set.dispatch(block)
        assert "truncated" in result.content

    async def test_raw_timeout(self, tool_set):
        mock_client = _make_mock_client_with_error(httpx.TimeoutException("timeout"))
        with patch("httpx.AsyncClient", return_value=mock_client):
            block = ToolUseBlock(id="t1", name="Web-fetch",
                                 input={"url": "https://example.com", "format": "raw"})
            result = await tool_set.dispatch(block)
        assert "timed out" in result.content

    async def test_raw_http_error(self, tool_set):
        mock_client = _make_mock_client_with_error(
            httpx.ConnectError("connection refused")
        )
        with patch("httpx.AsyncClient", return_value=mock_client):
            block = ToolUseBlock(id="t1", name="Web-fetch",
                                 input={"url": "https://example.com", "format": "raw"})
            result = await tool_set.dispatch(block)
        assert "Fetch failed" in result.content


# ---------------------------------------------------------------------------
# Fetch Local (LocalFetchImpl) Tests
# ---------------------------------------------------------------------------

_SAMPLE_HTML = """
<html><head><title>Test Page</title></head>
<body>
<nav>Navigation</nav>
<article>
<h1>Article Title</h1>
<p>This is the main content of the article with enough text to pass readability thresholds.
It has multiple paragraphs to ensure the content extraction works correctly.</p>
<p>Second paragraph with <strong>bold</strong> and <a href="https://example.com">a link</a>.</p>
<p>Third paragraph with more meaningful content to help the readability algorithm identify
this as the main article body rather than boilerplate navigation text.</p>
</article>
<footer>Footer</footer>
</body></html>
"""


class TestLocalFetchImpl:

    async def test_fetch_markdown_default(self, tool_set):
        resp = MagicMock()
        resp.text = _SAMPLE_HTML
        resp.raise_for_status = MagicMock()
        mock_client = _make_mock_client(resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            block = ToolUseBlock(id="t1", name="Web-fetch",
                                 input={"url": "https://example.com"})
            result = await tool_set.dispatch(block)
        assert not result.is_error
        assert "Article Title" in result.content
        # markdown 格式不包含 HTML 标签
        assert "<article>" not in result.content

    async def test_fetch_html_format(self, tool_set):
        resp = MagicMock()
        resp.text = _SAMPLE_HTML
        resp.raise_for_status = MagicMock()
        mock_client = _make_mock_client(resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            block = ToolUseBlock(id="t1", name="Web-fetch",
                                 input={"url": "https://example.com", "format": "html"})
            result = await tool_set.dispatch(block)
        assert not result.is_error
        # html 格式包含 HTML 标签
        assert "<" in result.content

    async def test_fetch_timeout(self, tool_set):
        mock_client = _make_mock_client_with_error(httpx.TimeoutException("timeout"))
        with patch("httpx.AsyncClient", return_value=mock_client):
            block = ToolUseBlock(id="t1", name="Web-fetch",
                                 input={"url": "https://example.com"})
            result = await tool_set.dispatch(block)
        assert "timed out" in result.content

    async def test_fetch_http_error(self, tool_set):
        mock_client = _make_mock_client_with_error(
            httpx.ConnectError("connection refused")
        )
        with patch("httpx.AsyncClient", return_value=mock_client):
            block = ToolUseBlock(id="t1", name="Web-fetch",
                                 input={"url": "https://example.com"})
            result = await tool_set.dispatch(block)
        assert "Fetch failed" in result.content


# ---------------------------------------------------------------------------
# Fetch Jina (JinaFetchImpl) Tests
# ---------------------------------------------------------------------------

def _mock_jina_fetch_response(title, content):
    """构造 Jina Reader API 的模拟 JSON 响应。"""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "code": 200,
        "status": 20000,
        "data": {
            "title": title,
            "url": "https://example.com",
            "content": content,
        },
    }
    resp.raise_for_status = MagicMock()
    return resp


class TestJinaFetchImpl:

    async def test_jina_fetch_returns_content(self, tool_set):
        mock_resp = _mock_jina_fetch_response("Example", "Hello, world!")
        mock_client = _make_mock_client(mock_resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            block = ToolUseBlock(id="t1", name="Web-fetch",
                                 input={"url": "https://example.com", "impl": "jina"})
            result = await tool_set.dispatch(block)
        assert not result.is_error
        assert "Example" in result.content
        assert "Hello, world!" in result.content

    async def test_jina_fetch_empty_content(self, tool_set):
        mock_resp = _mock_jina_fetch_response("Empty", "")
        mock_client = _make_mock_client(mock_resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            block = ToolUseBlock(id="t1", name="Web-fetch",
                                 input={"url": "https://example.com", "impl": "jina"})
            result = await tool_set.dispatch(block)
        assert "Failed to extract" in result.content

    async def test_jina_fetch_html_not_supported(self, tool_set):
        block = ToolUseBlock(id="t1", name="Web-fetch",
                             input={"url": "https://example.com", "format": "html", "impl": "jina"})
        result = await tool_set.dispatch(block)
        assert "does not support html" in result.content


# ---------------------------------------------------------------------------
# Dispatch & Provider Selection Tests
# ---------------------------------------------------------------------------

class TestProviderDispatch:

    async def test_fetch_default_impl_is_local(self, tool_set):
        resp = MagicMock()
        resp.text = _SAMPLE_HTML
        resp.raise_for_status = MagicMock()
        mock_client = _make_mock_client(resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            block = ToolUseBlock(id="t1", name="Web-fetch",
                                 input={"url": "https://example.com"})
            result = await tool_set.dispatch(block)
        assert not result.is_error
        # local 返回 markdown，不含 HTML 标签
        assert "<html>" not in result.content

    async def test_search_default_impl_is_jina(self, tool_set):
        mock_resp = _mock_search_response([
            {"title": "Test", "url": "https://example.com", "description": "test"},
        ])
        mock_client = _make_mock_client(mock_resp)
        with patch("httpx.AsyncClient", return_value=mock_client):
            block = ToolUseBlock(id="t1", name="Web-search", input={"query": "test"})
            result = await tool_set.dispatch(block)
        assert not result.is_error
        assert "Test" in result.content

    async def test_unknown_search_impl(self, tool_set):
        block = ToolUseBlock(id="t1", name="Web-search",
                             input={"query": "test", "impl": "nonexistent"})
        result = await tool_set.dispatch(block)
        assert "Unknown search impl" in result.content

    async def test_unknown_fetch_impl(self, tool_set):
        block = ToolUseBlock(id="t1", name="Web-fetch",
                             input={"url": "https://example.com", "impl": "nonexistent"})
        result = await tool_set.dispatch(block)
        assert "Unknown fetch impl" in result.content
