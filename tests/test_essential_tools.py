"""Tests for EssentialTools method implementations."""

import importlib.util
import sys
from pathlib import Path

import pytest

from mutagent.essential_tools import EssentialTools
from mutagent.runtime.module_manager import ModuleManager


# Load all .impl.py files to register @impl
_builtins_dir = Path(__file__).resolve().parent.parent / "src" / "mutagent" / "builtins"

def _load_impl(filename, mod_name):
    path = _builtins_dir / filename
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_load_impl("inspect_module.impl.py", "mutagent.builtins.inspect_module_impl")
_load_impl("view_source.impl.py", "mutagent.builtins.view_source_impl")
_load_impl("patch_module.impl.py", "mutagent.builtins.patch_module_impl")
_load_impl("save_module.impl.py", "mutagent.builtins.save_module_impl")
_load_impl("run_code.impl.py", "mutagent.builtins.run_code_impl")


@pytest.fixture
def tools():
    mgr = ModuleManager()
    t = EssentialTools(module_manager=mgr)
    yield t
    mgr.cleanup()


class TestInspectModule:

    def test_inspect_mutagent(self, tools):
        result = tools.inspect_module("mutagent")
        assert "mutagent/" in result

    def test_inspect_default_module(self, tools):
        result = tools.inspect_module()
        assert "mutagent/" in result

    def test_inspect_nonexistent_module(self, tools):
        result = tools.inspect_module("nonexistent.module.xyz")
        assert "not found" in result.lower()

    def test_inspect_shows_classes(self, tools):
        result = tools.inspect_module("mutagent.essential_tools", depth=2)
        assert "EssentialTools" in result

    def test_inspect_depth_limits_output(self, tools):
        result1 = tools.inspect_module("mutagent", depth=1)
        result2 = tools.inspect_module("mutagent", depth=3)
        # Deeper inspection should have more content
        assert len(result2) >= len(result1)


class TestViewSource:

    def test_view_module_source(self, tools):
        result = tools.view_source("mutagent.messages")
        assert "class Message" in result or "class ToolCall" in result

    def test_view_class_source(self, tools):
        result = tools.view_source("mutagent.essential_tools.EssentialTools")
        assert "class EssentialTools" in result

    def test_view_patched_module(self, tools):
        tools.module_manager.patch_module(
            "test_view.patched", "def hello():\n    return 'world'\n"
        )
        result = tools.view_source("test_view.patched.hello")
        assert "return 'world'" in result

    def test_view_nonexistent_target(self, tools):
        result = tools.view_source("nonexistent.module.xyz")
        assert "Error" in result


class TestPatchModule:

    def test_patch_creates_module(self, tools):
        result = tools.patch_module("test_tool_patch.mod1", "x = 42\n")
        assert "OK" in result
        assert "test_tool_patch.mod1" in result
        assert sys.modules["test_tool_patch.mod1"].x == 42

    def test_patch_reports_version(self, tools):
        tools.patch_module("test_tool_patch.ver", "v = 1\n")
        result = tools.patch_module("test_tool_patch.ver", "v = 2\n")
        assert "v2" in result

    def test_patch_syntax_error(self, tools):
        result = tools.patch_module("test_tool_patch.bad", "def f(\n")
        assert "Error" in result


class TestSaveModule:

    def test_save_module(self, tools, tmp_path):
        tools.module_manager.patch_module("test_tool_save.mod", "val = 99\n")
        result = tools.save_module("test_tool_save.mod", str(tmp_path))
        assert "OK" in result

        # Verify file was written
        saved_file = tmp_path / "test_tool_save" / "mod.py"
        assert saved_file.exists()
        assert saved_file.read_text() == "val = 99\n"

    def test_save_unpatched_module(self, tools, tmp_path):
        result = tools.save_module("nonexistent.module", str(tmp_path))
        assert "Error" in result


class TestRunCode:

    def test_run_simple_print(self, tools):
        result = tools.run_code("print('hello')")
        assert "hello" in result

    def test_run_expression(self, tools):
        result = tools.run_code("x = 2 + 3\nprint(x)")
        assert "5" in result

    def test_run_no_output(self, tools):
        result = tools.run_code("x = 42")
        assert "no output" in result.lower()

    def test_run_syntax_error(self, tools):
        result = tools.run_code("def f(\n")
        assert "SyntaxError" in result

    def test_run_runtime_error(self, tools):
        result = tools.run_code("raise ValueError('test error')")
        assert "ValueError" in result
        assert "test error" in result

    def test_run_import(self, tools):
        result = tools.run_code("import sys\nprint(sys.platform)")
        assert result.strip()  # Should have some platform string
