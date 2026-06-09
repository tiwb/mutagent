"""Tests for ToolSet and Toolkit."""

import pytest

from mutagent.core._tools_impl import ToolSetRuntime, get_current_tool_call
from mutagent.core.tools import ToolSet, Toolkit
from mutagent.core.messages import ToolUseBlock, ToolSchema
import mutobj
from mutobj import Declaration


# ---------------------------------------------------------------------------
# Fixture: a mock Toolkit for add/remove/query/dispatch tests
# ---------------------------------------------------------------------------

class MockToolkit(Toolkit):
    """Mock toolkit providing inspect / view_source / define / save."""

    def inspect(self, module_path: str) -> str:
        """Inspect a module.

        Args:
            module_path: Path to the module.
        """
        return f"Module info for {module_path}"

    def view_source(self, module_path: str) -> str:
        """View source of a module.

        Args:
            module_path: Path to the module.
        """
        return f"Source of {module_path}"

    def define(self, module_path: str, source: str) -> str:
        """Define a module.

        Args:
            module_path: Path to the module.
            source: Module source code.
        """
        return "OK"

    def save(self, module_path: str) -> str:
        """Save a module.

        Args:
            module_path: Path to the module.
        """
        if module_path == "nonexistent.mod":
            return "Error: module not found"
        return "OK"


# ---------------------------------------------------------------------------
# ToolSet Declaration Tests
# ---------------------------------------------------------------------------

class TestToolSetDeclaration:

    def test_inherits_from_declaration(self):
        assert issubclass(ToolSet, Declaration)

    def test_uses_declaration_meta(self):
        assert issubclass(ToolSet, mutobj.Declaration)

    def test_declared_methods(self):
        assert mutobj.get_declaration_func(ToolSet, "add") is not None
        assert mutobj.get_declaration_func(ToolSet, "remove") is not None
        assert mutobj.get_declaration_func(ToolSet, "query") is not None
        assert mutobj.get_declaration_func(ToolSet, "get_tools") is not None
        assert mutobj.get_declaration_func(ToolSet, "dispatch") is not None


# ---------------------------------------------------------------------------
# ToolSet.add() Tests
# ---------------------------------------------------------------------------

class TestToolSetAddFromObject:

    @pytest.fixture
    def tool_set(self):
        return ToolSet()

    @pytest.fixture
    def mock_tools(self):
        return MockToolkit()

    def test_add_registers_all_public_methods(self, tool_set, mock_tools):
        tool_set.add(mock_tools)
        schemas = tool_set.get_tools()
        names = {s.name for s in schemas}
        assert "Mock-inspect" in names
        assert "Mock-view_source" in names
        assert "Mock-define" in names
        assert "Mock-save" in names

    def test_add_with_methods_filter(self, tool_set, mock_tools):
        tool_set.add(mock_tools, methods=["inspect", "view_source"])
        schemas = tool_set.get_tools()
        names = {s.name for s in schemas}
        assert names == {"Mock-inspect", "Mock-view_source"}

    def test_add_single_method(self, tool_set, mock_tools):
        tool_set.add(mock_tools, methods=["define"])
        schemas = tool_set.get_tools()
        assert len(schemas) == 1
        assert schemas[0].name == "Mock-define"


# ---------------------------------------------------------------------------
# ToolSet.remove() Tests
# ---------------------------------------------------------------------------

class TestToolSetRemove:

    @pytest.fixture
    def populated_set(self):
        ts = ToolSet()
        ts.add(MockToolkit())
        return ts

    def test_remove_existing(self, populated_set):
        assert populated_set.remove("Mock-inspect") is True
        names = {s.name for s in populated_set.get_tools()}
        assert "Mock-inspect" not in names

    def test_remove_nonexistent(self, populated_set):
        assert populated_set.remove("nonexistent_tool") is False

    async def test_remove_then_dispatch_fails(self, populated_set):
        populated_set.remove("Mock-inspect")
        block = ToolUseBlock(id="tc_1", name="Mock-inspect", input={})
        result = await populated_set.dispatch(block)
        assert result.is_error
        assert "Unknown tool" in result.content


# ---------------------------------------------------------------------------
# ToolSet.query() Tests
# ---------------------------------------------------------------------------

class TestToolSetQuery:

    @pytest.fixture
    def populated_set(self):
        ts = ToolSet()
        ts.add(MockToolkit())
        return ts

    def test_query_existing(self, populated_set):
        schema = populated_set.query("Mock-inspect")
        assert schema is not None
        assert schema.name == "Mock-inspect"
        assert isinstance(schema, ToolSchema)

    def test_query_nonexistent(self, populated_set):
        assert populated_set.query("nonexistent") is None


# ---------------------------------------------------------------------------
# ToolSet.dispatch() Tests
# ---------------------------------------------------------------------------

class TestToolSetDispatch:

    @pytest.fixture
    def tool_set(self):
        ts = ToolSet()
        ts.add(MockToolkit())
        return ts

    async def test_dispatch_inspect(self, tool_set):
        block = ToolUseBlock(
            id="tc_1", name="Mock-inspect",
            input={"module_path": "mutagent"},
        )
        result = await tool_set.dispatch(block)
        assert not result.is_error
        assert result.tool_use_id == "tc_1"
        assert "mutagent" in result.content

    async def test_dispatch_define(self, tool_set):
        block = ToolUseBlock(
            id="tc_2", name="Mock-define",
            input={"module_path": "test.mod", "source": "x = 42\n"},
        )
        result = await tool_set.dispatch(block)
        assert not result.is_error
        assert "OK" in result.content

    async def test_dispatch_unknown_tool(self, tool_set):
        block = ToolUseBlock(id="tc_3", name="nonexistent_tool", input={})
        result = await tool_set.dispatch(block)
        assert result.is_error
        assert "Unknown tool" in result.content

    async def test_dispatch_with_error(self, tool_set):
        block = ToolUseBlock(
            id="tc_4", name="Mock-save",
            input={"module_path": "nonexistent.mod"},
        )
        result = await tool_set.dispatch(block)
        assert "Error" in result.content

    async def test_dispatch_async_method(self, tool_set):
        """Async Toolkit methods are properly awaited."""
        class AsyncToolkit(Toolkit):
            async def fetch(self, url: str) -> str:
                """Fetch a URL.

                Args:
                    url: The URL to fetch.
                """
                return f"fetched: {url}"

        tool_set.add(AsyncToolkit())
        block = ToolUseBlock(
            id="tc_async", name="Async-fetch",
            input={"url": "https://example.com"},
        )
        result = await tool_set.dispatch(block)
        assert not result.is_error
        assert "fetched: https://example.com" in result.content


# ---------------------------------------------------------------------------
# ToolSet Empty State Tests
# ---------------------------------------------------------------------------

class TestToolSetEmptyState:

    def test_empty_get_tools(self):
        ts = ToolSet()
        assert ts.get_tools() == []

    async def test_empty_dispatch_fails(self):
        ts = ToolSet()
        block = ToolUseBlock(id="tc_1", name="anything", input={})
        result = await ts.dispatch(block)
        assert result.is_error

    def test_empty_query_returns_none(self):
        ts = ToolSet()
        assert ts.query("anything") is None


# ---------------------------------------------------------------------------
# ToolSet Multiple Sources Tests
# ---------------------------------------------------------------------------

class TestToolSetMultipleSources:

    def test_add_multiple_objects(self):
        """Adding tools from multiple Toolkit instances accumulates them."""
        ts = ToolSet()
        ts.add(MockToolkit(), methods=["inspect"])
        ts.add(MockToolkit(), methods=["define"])

        schemas = ts.get_tools()
        names = {s.name for s in schemas}
        assert "Mock-inspect" in names
        assert "Mock-define" in names

    def test_add_multiple_toolkits(self):
        """Adding two different Toolkit instances."""
        class AlphaToolkit(Toolkit):
            def alpha(self) -> str:
                """Alpha."""
                return "a"

        class BetaToolkit(Toolkit):
            def beta(self) -> str:
                """Beta."""
                return "b"

        ts = ToolSet()
        ts.add(AlphaToolkit())
        ts.add(BetaToolkit())

        schemas = ts.get_tools()
        names = {s.name for s in schemas}
        assert "Alpha-alpha" in names
        assert "Beta-beta" in names


# ---------------------------------------------------------------------------
# Tool Naming Convention Tests
# ---------------------------------------------------------------------------

class TestToolNamingConvention:
    """工具名格式为 '{Prefix}-{method}'，前缀从类名自动推导。"""

    def test_toolkit_suffix_stripped(self):
        """类名以 Toolkit 结尾时，去掉该后缀作为前缀。"""
        class WebToolkit(Toolkit):
            def search(self, query: str) -> str:
                """Search the web."""
                return f"results for {query}"

            def fetch(self, url: str) -> str:
                """Fetch a URL."""
                return f"content of {url}"

        ts = ToolSet()
        ts.add(WebToolkit())
        names = {s.name for s in ts.get_tools()}
        assert names == {"Web-search", "Web-fetch"}

    def test_class_name_without_toolkit_suffix(self):
        """类名不以 Toolkit 结尾时，使用完整类名作为前缀。"""
        class Greeter(Toolkit):
            def say_hello(self) -> str:
                """Say hello."""
                return "hello"

        ts = ToolSet()
        ts.add(Greeter())
        names = {s.name for s in ts.get_tools()}
        assert names == {"Greeter-say_hello"}

    def test_explicit_tool_prefix(self):
        """_tool_prefix 显式指定时覆盖类名推导。"""
        class MyTools(Toolkit):
            tool_prefix = "Util"

            def ping(self) -> str:
                """Ping."""
                return "pong"

        ts = ToolSet()
        ts.add(MyTools())
        names = {s.name for s in ts.get_tools()}
        assert names == {"Util-ping"}

    def test_empty_tool_prefix(self):
        """_tool_prefix 为空字符串时，工具名即方法名。"""
        class FlatTools(Toolkit):
            tool_prefix = ""

            def ping(self) -> str:
                """Ping."""
                return "pong"

        ts = ToolSet()
        ts.add(FlatTools())
        names = {s.name for s in ts.get_tools()}
        assert names == {"ping"}

    async def test_dispatch_uses_prefixed_name(self):
        """dispatch() 必须使用前缀工具名。"""
        class WebToolkit(Toolkit):
            def search(self, query: str) -> str:
                """Search the web."""
                return f"found: {query}"

        ts = ToolSet()
        ts.add(WebToolkit())
        block = ToolUseBlock(
            id="tc_1", name="Web-search",
            input={"query": "python"},
        )
        result = await ts.dispatch(block)
        assert not result.is_error
        assert "found: python" in result.content

    async def test_bare_method_name_dispatch_fails(self):
        """使用不带前缀的方法名 dispatch 会失败。"""
        class WebToolkit(Toolkit):
            def search(self, query: str) -> str:
                """Search the web."""
                return "results"

        ts = ToolSet()
        ts.add(WebToolkit())
        block = ToolUseBlock(
            id="tc_1", name="search",
            input={"query": "test"},
        )
        result = await ts.dispatch(block)
        assert result.is_error
        assert "Unknown tool" in result.content

    def test_query_uses_prefixed_name(self):
        """query() 使用前缀工具名。"""
        class SessionToolkit(Toolkit):
            def create(self, session_type: str) -> str:
                """Create a session."""
                return "created"

        ts = ToolSet()
        ts.add(SessionToolkit())
        schema = ts.query("Session-create")
        assert schema is not None
        assert schema.name == "Session-create"
        assert ts.query("create") is None

    def test_schema_name_is_prefixed(self):
        """ToolSchema.name 使用前缀格式。"""
        class WebToolkit(Toolkit):
            def search(self, query: str) -> str:
                """Search the web.

                Args:
                    query: Search query.
                """
                return "results"

        ts = ToolSet()
        ts.add(WebToolkit())
        schema = ts.query("Web-search")
        assert schema is not None
        assert schema.name == "Web-search"
        assert "Search the web" in schema.description
        assert "query" in schema.input_schema["properties"]

    def test_remove_uses_prefixed_name(self):
        """remove() 使用前缀工具名。"""
        class WebToolkit(Toolkit):
            def search(self, query: str) -> str:
                """Search."""
                return "results"

            def fetch(self, url: str) -> str:
                """Fetch."""
                return "content"

        ts = ToolSet()
        ts.add(WebToolkit())
        assert ts.remove("Web-search") is True
        names = {s.name for s in ts.get_tools()}
        assert names == {"Web-fetch"}

    def test_methods_filter_uses_method_names(self):
        """add(methods=[...]) 过滤参数使用方法名，注册结果使用前缀工具名。"""
        class WebToolkit(Toolkit):
            def search(self, query: str) -> str:
                """Search."""
                return "results"

            def fetch(self, url: str) -> str:
                """Fetch."""
                return "content"

        ts = ToolSet()
        ts.add(WebToolkit(), methods=["search"])
        schemas = ts.get_tools()
        assert len(schemas) == 1
        assert schemas[0].name == "Web-search"


# ---------------------------------------------------------------------------
# Toolkit.owner Binding Tests
# ---------------------------------------------------------------------------

class TestToolkitOwnerBinding:
    """Toolkit.owner 由 ToolSet 在 add() 时设置。"""

    def test_owner_set_on_add(self):
        """add() Toolkit 实例后，owner 指向 ToolSet。"""
        class MyToolkit(Toolkit):
            def do_stuff(self) -> str:
                """Do stuff."""
                return "done"

        toolkit = MyToolkit()
        assert toolkit.owner is None

        ts = ToolSet()
        ts.add(toolkit)
        assert toolkit.owner is ts

    def test_owner_default_none(self):
        """Toolkit 默认 owner 为 None。"""
        class FreshToolkit(Toolkit):
            def noop(self) -> str:
                """No-op."""
                return ""

        assert FreshToolkit().owner is None


# ---------------------------------------------------------------------------
# Dispatch State Tracking Tests
# ---------------------------------------------------------------------------

class TestDispatchStateTracking:
    """dispatch 期间跟踪 _current_tool_call 和清理 _active_ui。"""

    async def test_current_tool_call_during_dispatch(self):
        """dispatch 执行期间 _current_tool_call 指向当前 ToolUseBlock。"""
        captured_tool_call = None

        class SpyToolkit(Toolkit):
            def spy(self) -> str:
                """Capture current tool call."""
                nonlocal captured_tool_call
                captured_tool_call = get_current_tool_call(self.owner)
                return "spied"

        ts = ToolSet()
        toolkit = SpyToolkit()
        ts.add(toolkit)

        block = ToolUseBlock(id="tc_spy", name="Spy-spy", input={})
        await ts.dispatch(block)

        assert captured_tool_call is block
        assert get_current_tool_call(ts) is None

    async def test_current_tool_call_cleared_on_error(self):
        """工具抛异常后 _current_tool_call 仍被清除。"""
        class FailToolkit(Toolkit):
            def fail(self) -> str:
                """Will fail."""
                raise RuntimeError("boom")

        ts = ToolSet()
        ts.add(FailToolkit())

        block = ToolUseBlock(id="tc_fail", name="Fail-fail", input={})
        result = await ts.dispatch(block)

        assert result.is_error
        assert get_current_tool_call(ts) is None

    async def test_active_ui_cleanup_on_dispatch_end(self):
        """dispatch 结束后清理 _active_ui。"""
        class MockUI:
            closed = False

            def close(self):
                self.closed = True

        class UIToolkit(Toolkit):
            def with_ui(self) -> str:
                """Tool that creates UI."""
                ToolSetRuntime.get_or_create(self.owner).active_ui = MockUI()
                return "done"

        ts = ToolSet()
        ts.add(UIToolkit())

        block = ToolUseBlock(id="tc_ui", name="UI-with_ui", input={})
        result = await ts.dispatch(block)

        assert not result.is_error
        assert ToolSetRuntime.get(ts) is None or ToolSetRuntime.get(ts).active_ui is None

    async def test_active_ui_cleanup_on_error(self):
        """工具抛异常后 _active_ui 仍被清理。"""
        class MockUI:
            closed = False

            def close(self):
                self.closed = True

        mock_ui = MockUI()

        class UIToolkit(Toolkit):
            def fail_with_ui(self) -> str:
                """Tool that creates UI then fails."""
                ToolSetRuntime.get_or_create(self.owner).active_ui = mock_ui
                raise RuntimeError("boom")

        ts = ToolSet()
        ts.add(UIToolkit())

        block = ToolUseBlock(id="tc_fail_ui", name="UI-fail_with_ui", input={})
        result = await ts.dispatch(block)

        assert result.is_error
        assert mock_ui.closed
        assert ToolSetRuntime.get(ts) is None or ToolSetRuntime.get(ts).active_ui is None


# ---------------------------------------------------------------------------
# Toolkit.tool_methods Whitelist Tests
# ---------------------------------------------------------------------------

class TestToolMethods:
    """Tests for Toolkit.tool_methods whitelist."""

    def test_tool_methods_whitelist(self):
        """Only methods in tool_methods should be exposed."""
        class SelectiveToolkit(Toolkit):
            tool_methods = ["search"]

            def search(self, query: str) -> str:
                """Search."""
                return query

            def parse(self, html: str) -> str:
                """Parse."""
                return html

        ts = ToolSet()
        ts.add(SelectiveToolkit())
        schemas = ts.get_tools()
        names = {s.name for s in schemas}
        assert "Selective-search" in names
        assert "Selective-parse" not in names

    def test_tool_methods_empty_list(self):
        """Empty tool_methods means no methods exposed."""
        class NoToolsKit(Toolkit):
            tool_methods = []

            def hidden(self) -> str:
                """Hidden."""
                return 'x'

        ts = ToolSet()
        ts.add(NoToolsKit())
        schemas = ts.get_tools()
        assert len(schemas) == 0

    def test_no_tool_methods_backward_compat(self):
        """Without tool_methods, all public methods exposed (backward compat)."""
        class FullToolkit(Toolkit):
            def alpha(self) -> str:
                """Alpha."""
                return 'a'

            def beta(self) -> str:
                """Beta."""
                return 'b'

        ts = ToolSet()
        ts.add(FullToolkit())
        schemas = ts.get_tools()
        names = {s.name for s in schemas}
        assert "Full-alpha" in names
        assert "Full-beta" in names


# ---------------------------------------------------------------------------
# Toolkit._discoverable Tests
# ---------------------------------------------------------------------------

class TestDiscoverable:
    """Tests for Toolkit._discoverable control."""

    def test_discoverable_false_still_works_with_add(self):
        """_discoverable=False toolkit can still be added manually."""
        class ManualKit(Toolkit):
            discoverable = False

            def manual_tool(self) -> str:
                """Manual."""
                return "manual"

        ts = ToolSet()
        ts.add(ManualKit())
        schemas = ts.get_tools()
        names = {s.name for s in schemas}
        assert "ManualKit-manual_tool" in names

    def test_discoverable_default_true(self):
        """_discoverable defaults to True (no explicit _discoverable attribute)."""
        class DefaultKit(Toolkit):
            def default_tool(self) -> str:
                """Default."""
                return "ok"

        assert DefaultKit.discoverable is True  # type: ignore[reportUnknownMemberType]
