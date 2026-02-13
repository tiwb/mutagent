"""Tests for EssentialTools + ToolSelector declarations and selector impl."""

import importlib.util
from pathlib import Path

import pytest

import mutagent
from mutagent.base import MutagentMeta
from mutagent.essential_tools import EssentialTools
from mutagent.messages import ToolCall, ToolResult, ToolSchema
from mutagent.runtime.module_manager import ModuleManager
from mutagent.selector import ToolSelector
from forwardpy.core import _DECLARED_METHODS


# Load selector.impl.py
_selector_impl_path = (
    Path(__file__).resolve().parent.parent
    / "src" / "mutagent" / "builtins" / "selector.impl.py"
)


def _load_impl(path, mod_name):
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Load the selector impl to register @impl
_selector_mod = _load_impl(_selector_impl_path, "mutagent.builtins.selector_impl")


class TestEssentialToolsDeclaration:

    def test_inherits_from_mutagent_object(self):
        assert issubclass(EssentialTools, mutagent.Object)

    def test_uses_mutagent_meta(self):
        assert isinstance(EssentialTools, MutagentMeta)

    def test_declared_methods(self):
        declared = getattr(EssentialTools, _DECLARED_METHODS, set())
        expected = {"inspect_module", "view_source", "patch_module", "save_module", "run_code"}
        assert expected.issubset(declared)

    def test_has_module_manager_attribute(self):
        mgr = ModuleManager()
        tools = EssentialTools(module_manager=mgr)
        assert tools.module_manager is mgr
        mgr.cleanup()


class TestToolSelectorDeclaration:

    def test_inherits_from_mutagent_object(self):
        assert issubclass(ToolSelector, mutagent.Object)

    def test_declared_methods(self):
        declared = getattr(ToolSelector, _DECLARED_METHODS, set())
        assert "get_tools" in declared
        assert "dispatch" in declared


class TestMakeSchemaFromMethod:

    def test_generates_schema(self):
        from mutagent.builtins import selector_impl
        # Use a known module reference
        make_schema = _selector_mod.make_schema_from_method
        mgr = ModuleManager()
        tools = EssentialTools(module_manager=mgr)

        schema = make_schema(tools, "inspect_module")
        assert isinstance(schema, ToolSchema)
        assert schema.name == "inspect_module"
        assert "properties" in schema.input_schema
        assert "module_path" in schema.input_schema["properties"]
        assert "depth" in schema.input_schema["properties"]
        mgr.cleanup()

    def test_required_params_detected(self):
        make_schema = _selector_mod.make_schema_from_method
        mgr = ModuleManager()
        tools = EssentialTools(module_manager=mgr)

        schema = make_schema(tools, "patch_module")
        assert "module_path" in schema.input_schema.get("required", [])
        assert "source" in schema.input_schema.get("required", [])
        mgr.cleanup()

    def test_optional_params_have_defaults(self):
        make_schema = _selector_mod.make_schema_from_method
        mgr = ModuleManager()
        tools = EssentialTools(module_manager=mgr)

        schema = make_schema(tools, "inspect_module")
        depth_prop = schema.input_schema["properties"]["depth"]
        assert depth_prop.get("default") == 2
        mgr.cleanup()


class TestToolSelectorImpl:

    @pytest.fixture
    def selector_with_tools(self):
        mgr = ModuleManager()
        tools = EssentialTools(module_manager=mgr)
        selector = ToolSelector(essential_tools=tools)
        yield selector
        mgr.cleanup()

    @pytest.mark.asyncio
    async def test_get_tools_returns_schemas(self, selector_with_tools):
        schemas = await selector_with_tools.get_tools({})
        assert len(schemas) == 5
        names = {s.name for s in schemas}
        assert names == {"inspect_module", "view_source", "patch_module", "save_module", "run_code"}

    @pytest.mark.asyncio
    async def test_get_tools_schema_structure(self, selector_with_tools):
        schemas = await selector_with_tools.get_tools({})
        for schema in schemas:
            assert isinstance(schema, ToolSchema)
            assert schema.name
            assert schema.description
            assert "type" in schema.input_schema
            assert schema.input_schema["type"] == "object"

    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool(self, selector_with_tools):
        call = ToolCall(id="tc_1", name="nonexistent_tool", arguments={})
        result = await selector_with_tools.dispatch(call)
        assert result.is_error
        assert "Unknown tool" in result.content

    @pytest.mark.asyncio
    async def test_dispatch_tool_error(self, selector_with_tools):
        # run_code with invalid code should return error result
        call = ToolCall(id="tc_2", name="run_code", arguments={"code": "raise ValueError('test')"})
        result = await selector_with_tools.dispatch(call)
        # The tool stub will raise NotImplementedError since we haven't loaded the impl
        assert result.is_error
