"""Tests for ModuleManager runtime module patching."""

import inspect
import sys

import pytest
import mutagent
from mutagent.base import MutagentMeta
from mutagent.runtime.module_manager import ModuleManager


@pytest.fixture
def mgr():
    manager = ModuleManager()
    yield manager
    manager.cleanup()


class TestPatchModule:

    def test_patch_creates_module(self, mgr):
        mod = mgr.patch_module("test_pkg.mymod", "x = 42\n")
        assert "test_pkg.mymod" in sys.modules
        assert mod.x == 42

    def test_patch_sets_module_attributes(self, mgr):
        mod = mgr.patch_module("test_pkg.attrs", "pass\n")
        assert mod.__name__ == "test_pkg.attrs"
        assert mod.__file__ == "mutagent://test_pkg.attrs"
        assert mod.__package__ == "test_pkg"

    def test_patch_executes_source(self, mgr):
        source = "a = 1\nb = 2\nc = a + b\n"
        mod = mgr.patch_module("test_pkg.calc", source)
        assert mod.c == 3

    def test_repatch_clears_old_namespace(self, mgr):
        mgr.patch_module("test_pkg.repatch", "x = 1\n")
        mod = mgr.patch_module("test_pkg.repatch", "y = 2\n")
        assert mod.y == 2
        assert not hasattr(mod, "x")

    def test_repatch_unregisters_old_impls(self, mgr):
        source1 = (
            "import mutagent\n"
            "class Svc(mutagent.Object):\n"
            "    def run(self) -> str: ...\n"
        )
        mgr.patch_module("test_pkg.svc_decl", source1)

        source_impl = (
            "import mutagent\n"
            "from test_pkg.svc_decl import Svc\n"
            "@mutagent.impl(Svc.run)\n"
            "def run(self) -> str:\n"
            "    return 'v1'\n"
        )
        mgr.patch_module("test_pkg.svc_impl", source_impl)

        mod_decl = sys.modules["test_pkg.svc_decl"]
        obj = mod_decl.Svc()
        assert obj.run() == "v1"

        # Repatch impl with different implementation
        source_impl2 = (
            "import mutagent\n"
            "from test_pkg.svc_decl import Svc\n"
            "@mutagent.impl(Svc.run)\n"
            "def run(self) -> str:\n"
            "    return 'v2'\n"
        )
        mgr.patch_module("test_pkg.svc_impl", source_impl2)
        assert obj.run() == "v2"

    def test_parent_packages_created(self, mgr):
        mgr.patch_module("a.b.c.deep", "val = True\n")
        assert "a" in sys.modules
        assert "a.b" in sys.modules
        assert "a.b.c" in sys.modules
        assert hasattr(sys.modules["a"], "__path__")

    def test_attach_to_parent(self, mgr):
        mgr.patch_module("test_pkg.child", "Z = 99\n")
        parent = sys.modules.get("test_pkg")
        assert parent is not None
        assert hasattr(parent, "child")
        assert parent.child.Z == 99


class TestInspectIntegration:

    def test_inspect_getsource_function(self, mgr):
        source = "def hello():\n    return 'world'\n"
        mod = mgr.patch_module("test_pkg.src", source)
        got = inspect.getsource(mod.hello)
        assert "return 'world'" in got

    def test_inspect_getsource_class(self, mgr):
        source = (
            "import mutagent\n"
            "class MyClass(mutagent.Object):\n"
            "    name: str\n"
            "    def greet(self) -> str: ...\n"
        )
        mod = mgr.patch_module("test_pkg.clsrc", source)
        got = inspect.getsource(mod.MyClass)
        assert "class MyClass" in got

    def test_loader_get_source(self, mgr):
        source = "x = 1\n"
        mod = mgr.patch_module("test_pkg.ldr", source)
        assert mod.__loader__.get_source("test_pkg.ldr") == source

    def test_virtual_filename_in_code(self, mgr):
        source = "def f():\n    pass\n"
        mod = mgr.patch_module("test_pkg.vfn", source)
        assert mod.f.__code__.co_filename == "mutagent://test_pkg.vfn"


class TestHistoryAndVersioning:

    def test_version_increments(self, mgr):
        assert mgr.get_version("test_pkg.ver") == 0
        mgr.patch_module("test_pkg.ver", "v = 1\n")
        assert mgr.get_version("test_pkg.ver") == 1
        mgr.patch_module("test_pkg.ver", "v = 2\n")
        assert mgr.get_version("test_pkg.ver") == 2

    def test_get_source_returns_latest(self, mgr):
        mgr.patch_module("test_pkg.gsrc", "a = 1\n")
        mgr.patch_module("test_pkg.gsrc", "b = 2\n")
        assert mgr.get_source("test_pkg.gsrc") == "b = 2\n"

    def test_get_source_unpatched_returns_none(self, mgr):
        assert mgr.get_source("nonexistent.module") is None

    def test_get_history(self, mgr):
        mgr.patch_module("test_pkg.hist", "v1 = True\n")
        mgr.patch_module("test_pkg.hist", "v2 = True\n")
        mgr.patch_module("test_pkg.hist", "v3 = True\n")
        history = mgr.get_history("test_pkg.hist")
        assert len(history) == 3
        assert history[0].version == 1
        assert history[2].version == 3
        assert history[2].source == "v3 = True\n"


class TestMutagentMetaIntegration:

    def test_inplace_class_update_via_repatch(self, mgr):
        source1 = (
            "import mutagent\n"
            "class Agent(mutagent.Object):\n"
            "    name: str\n"
            "    def run(self) -> str: ...\n"
        )
        mod = mgr.patch_module("test_pkg.agent", source1)
        cls1 = mod.Agent
        id1 = id(cls1)
        obj = cls1(name="test")

        source2 = (
            "import mutagent\n"
            "class Agent(mutagent.Object):\n"
            "    name: str\n"
            "    version: int\n"
            "    def run(self) -> str: ...\n"
            "    def stop(self) -> None: ...\n"
        )
        mod = mgr.patch_module("test_pkg.agent", source2)
        cls2 = mod.Agent

        assert cls1 is cls2
        assert id(cls2) == id1
        assert isinstance(obj, cls2)


class TestCleanup:

    def test_cleanup_removes_modules(self, mgr):
        mgr.patch_module("test_pkg.clean1", "x = 1\n")
        mgr.patch_module("test_pkg.clean2", "y = 2\n")
        assert "test_pkg.clean1" in sys.modules
        assert "test_pkg.clean2" in sys.modules

        mgr.cleanup()
        assert "test_pkg.clean1" not in sys.modules
        assert "test_pkg.clean2" not in sys.modules

    def test_cleanup_resets_state(self, mgr):
        mgr.patch_module("test_pkg.rst", "z = 3\n")
        mgr.cleanup()
        assert mgr.get_version("test_pkg.rst") == 0
        assert mgr.get_source("test_pkg.rst") is None
        assert mgr.get_history("test_pkg.rst") == []
