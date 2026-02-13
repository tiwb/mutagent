"""Tests for ImplLoader discovery and loading of .impl.py files."""

import sys
import types

import pytest
import mutagent
from mutagent.runtime.impl_loader import ImplLoader
from mutagent.runtime.module_manager import ModuleManager


@pytest.fixture
def mgr():
    manager = ModuleManager()
    yield manager
    manager.cleanup()


class TestDiscover:

    def test_discover_finds_impl_files(self, tmp_path):
        (tmp_path / "foo.impl.py").write_text("pass\n")
        (tmp_path / "bar.impl.py").write_text("pass\n")
        (tmp_path / "normal.py").write_text("pass\n")

        loader = ImplLoader()
        found = loader.discover(tmp_path)

        names = [p.name for p in found]
        assert "foo.impl.py" in names
        assert "bar.impl.py" in names
        assert "normal.py" not in names

    def test_discover_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.impl.py").write_text("pass\n")
        (tmp_path / "top.impl.py").write_text("pass\n")

        loader = ImplLoader()
        found = loader.discover(tmp_path)
        names = [p.name for p in found]
        assert "deep.impl.py" in names
        assert "top.impl.py" in names

    def test_discover_empty_dir(self, tmp_path):
        loader = ImplLoader()
        found = loader.discover(tmp_path)
        assert found == []

    def test_discover_nonexistent_dir(self, tmp_path):
        loader = ImplLoader()
        found = loader.discover(tmp_path / "nonexistent")
        assert found == []

    def test_discover_returns_sorted(self, tmp_path):
        (tmp_path / "z.impl.py").write_text("pass\n")
        (tmp_path / "a.impl.py").write_text("pass\n")
        (tmp_path / "m.impl.py").write_text("pass\n")

        loader = ImplLoader()
        found = loader.discover(tmp_path)
        names = [p.name for p in found]
        assert names == sorted(names)


class TestComputeModuleName:

    def test_simple_name(self, tmp_path):
        impl_path = tmp_path / "foo.impl.py"
        loader = ImplLoader()
        name = loader._compute_module_name(impl_path, tmp_path, "pkg")
        assert name == "pkg.foo"

    def test_nested_name(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        impl_path = sub / "bar.impl.py"
        loader = ImplLoader()
        name = loader._compute_module_name(impl_path, tmp_path, "pkg")
        assert name == "pkg.sub.bar"

    def test_empty_base_package(self, tmp_path):
        impl_path = tmp_path / "mod.impl.py"
        loader = ImplLoader()
        name = loader._compute_module_name(impl_path, tmp_path, "")
        assert name == "mod"


class TestLoadFile:

    def test_load_file_direct_exec(self, tmp_path):
        impl_file = tmp_path / "calc.impl.py"
        impl_file.write_text("result = 2 + 3\n")

        loader = ImplLoader()
        mod = loader.load_file(impl_file, tmp_path, "test_load")

        assert mod.__name__ == "test_load.calc"
        assert mod.result == 5

        # Cleanup
        sys.modules.pop("test_load.calc", None)

    def test_load_file_with_module_manager(self, tmp_path, mgr):
        impl_file = tmp_path / "helper.impl.py"
        impl_file.write_text("value = 'managed'\n")

        loader = ImplLoader(module_manager=mgr)
        mod = loader.load_file(impl_file, tmp_path, "test_mgr")

        assert mod.__name__ == "test_mgr.helper"
        assert mod.value == "managed"
        assert mgr.get_version("test_mgr.helper") == 1

    def test_load_file_executes_impl_registration(self, tmp_path):
        # First, create a declaration module with a stub
        decl_mgr = ModuleManager()
        decl_source = (
            "import mutagent\n"
            "class Worker(mutagent.Object):\n"
            "    def work(self) -> str: ...\n"
        )
        decl_mgr.patch_module("test_impl_reg.decl", decl_source)

        # Create impl file
        impl_file = tmp_path / "worker.impl.py"
        impl_file.write_text(
            "import mutagent\n"
            "from test_impl_reg.decl import Worker\n"
            "@mutagent.impl(Worker.work)\n"
            "def work(self) -> str:\n"
            "    return 'done'\n"
        )

        loader = ImplLoader(module_manager=decl_mgr)
        loader.load_file(impl_file, tmp_path, "test_impl_reg.impls")

        Worker = sys.modules["test_impl_reg.decl"].Worker
        obj = Worker()
        assert obj.work() == "done"

        decl_mgr.cleanup()


class TestLoadAll:

    def test_load_all_loads_multiple(self, tmp_path, mgr):
        (tmp_path / "a.impl.py").write_text("x = 1\n")
        (tmp_path / "b.impl.py").write_text("y = 2\n")
        (tmp_path / "regular.py").write_text("z = 3\n")

        loader = ImplLoader(module_manager=mgr)
        modules = loader.load_all(tmp_path, "test_all")

        assert len(modules) == 2
        names = {m.__name__ for m in modules}
        assert "test_all.a" in names
        assert "test_all.b" in names

    def test_load_all_empty_dir(self, tmp_path, mgr):
        loader = ImplLoader(module_manager=mgr)
        modules = loader.load_all(tmp_path, "test_empty")
        assert modules == []

    def test_load_all_with_subdirectories(self, tmp_path, mgr):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.impl.py").write_text("a = 1\n")
        (sub / "nested.impl.py").write_text("b = 2\n")

        loader = ImplLoader(module_manager=mgr)
        modules = loader.load_all(tmp_path, "test_sub")

        assert len(modules) == 2
        names = {m.__name__ for m in modules}
        assert "test_sub.sub.nested" in names
        assert "test_sub.top" in names
