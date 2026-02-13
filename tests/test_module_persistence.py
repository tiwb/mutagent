"""Tests for ModuleManager.save_module (module persistence)."""

import inspect
import linecache
import sys

import pytest
from mutagent.runtime.module_manager import ModuleManager, _replace_code_filename


@pytest.fixture
def mgr():
    manager = ModuleManager()
    yield manager
    manager.cleanup()


class TestSaveModule:

    def test_save_creates_file(self, mgr, tmp_path):
        mgr.patch_module("save_test.mymod", "x = 42\n")
        path = mgr.save_module("save_test.mymod", tmp_path)

        assert path.exists()
        assert path.read_text(encoding="utf-8") == "x = 42\n"

    def test_save_correct_file_path(self, mgr, tmp_path):
        mgr.patch_module("a.b.c", "val = True\n")
        path = mgr.save_module("a.b.c", tmp_path)

        expected = tmp_path / "a" / "b" / "c.py"
        assert path == expected

    def test_save_creates_parent_dirs(self, mgr, tmp_path):
        mgr.patch_module("deep.nested.pkg.mod", "z = 1\n")
        path = mgr.save_module("deep.nested.pkg.mod", tmp_path)

        assert path.exists()
        assert (tmp_path / "deep" / "nested" / "pkg").is_dir()

    def test_save_updates_module_file(self, mgr, tmp_path):
        mgr.patch_module("save_test.filattr", "y = 10\n")
        path = mgr.save_module("save_test.filattr", tmp_path)

        mod = sys.modules["save_test.filattr"]
        assert mod.__file__ == str(path)

    def test_save_updates_co_filename(self, mgr, tmp_path):
        source = "def hello():\n    return 'world'\n"
        mgr.patch_module("save_test.cofn", source)
        path = mgr.save_module("save_test.cofn", tmp_path)

        mod = sys.modules["save_test.cofn"]
        assert mod.hello.__code__.co_filename == str(path)

    def test_save_updates_nested_code_objects(self, mgr, tmp_path):
        source = (
            "def outer():\n"
            "    def inner():\n"
            "        return 1\n"
            "    return inner\n"
        )
        mgr.patch_module("save_test.nested", source)
        path = mgr.save_module("save_test.nested", tmp_path)

        mod = sys.modules["save_test.nested"]
        # outer's co_filename should be updated
        assert mod.outer.__code__.co_filename == str(path)
        # inner's code object is in co_consts of outer
        inner_codes = [
            c for c in mod.outer.__code__.co_consts
            if hasattr(c, "co_filename")
        ]
        assert len(inner_codes) == 1
        assert inner_codes[0].co_filename == str(path)

    def test_save_removes_virtual_linecache(self, mgr, tmp_path):
        mgr.patch_module("save_test.lcache", "a = 1\n")
        virtual_key = "mutagent://save_test.lcache"
        assert virtual_key in linecache.cache

        mgr.save_module("save_test.lcache", tmp_path)
        assert virtual_key not in linecache.cache

    def test_save_inspect_getsource_still_works(self, mgr, tmp_path):
        source = "def greet():\n    return 'hi'\n"
        mgr.patch_module("save_test.insp", source)
        mgr.save_module("save_test.insp", tmp_path)

        mod = sys.modules["save_test.insp"]
        got = inspect.getsource(mod.greet)
        assert "return 'hi'" in got

    def test_save_unpatched_raises(self, mgr, tmp_path):
        with pytest.raises(ValueError, match="never been patched"):
            mgr.save_module("nonexistent.module", tmp_path)

    def test_save_uses_latest_source(self, mgr, tmp_path):
        mgr.patch_module("save_test.ver", "v = 1\n")
        mgr.patch_module("save_test.ver", "v = 2\n")
        path = mgr.save_module("save_test.ver", tmp_path)

        assert path.read_text(encoding="utf-8") == "v = 2\n"


class TestReplaceCodeFilename:

    def test_simple_replacement(self):
        source = "def f():\n    pass\n"
        code = compile(source, "old_file.py", "exec")
        new_code = _replace_code_filename(code, "old_file.py", "new_file.py")
        assert new_code.co_filename == "new_file.py"

    def test_nested_replacement(self):
        source = "def f():\n    def g():\n        pass\n"
        code = compile(source, "old.py", "exec")
        new_code = _replace_code_filename(code, "old.py", "new.py")

        assert new_code.co_filename == "new.py"
        inner_codes = [c for c in new_code.co_consts if hasattr(c, "co_filename")]
        for inner in inner_codes:
            # Recursively check nested code objects
            assert inner.co_filename == "new.py"
