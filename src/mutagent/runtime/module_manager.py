"""mutagent.runtime.module_manager -- Runtime module patching."""

from __future__ import annotations

import linecache
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PatchRecord:
    """Record of a single module patch operation."""

    module_path: str
    source: str
    version: int
    virtual_filename: str


class _VirtualLoader:
    """Minimal loader satisfying inspect.getsource() for virtual modules."""

    def __init__(self, source: str, filename: str) -> None:
        self._source = source
        self._filename = filename

    def get_source(self, fullname: str) -> str:
        return self._source

    def get_filename(self, fullname: str) -> str:
        return self._filename


class ModuleManager:
    """Manages runtime module patching with source tracking and linecache integration."""

    def __init__(self) -> None:
        self._history: dict[str, list[PatchRecord]] = {}
        self._versions: dict[str, int] = {}
        self._virtual_packages: set[str] = set()
        self._patched_modules: set[str] = set()

    def patch_module(self, module_path: str, source: str) -> types.ModuleType:
        """Patch (or create) a module with new source code.

        Semantics: patch = "write file + restart". The old module namespace
        is completely replaced, not incrementally updated.

        Args:
            module_path: Dotted module path, e.g. "myagent.tools.search"
            source: Python source code for the module

        Returns:
            The patched/created module object
        """
        from forwardpy import unregister_module_impls

        # Bump version
        version = self._versions.get(module_path, 0) + 1
        self._versions[module_path] = version

        virtual_filename = f"mutagent://{module_path}"

        # Ensure parent packages exist
        self._ensure_parent_packages(module_path)

        # Get or create the module
        module = sys.modules.get(module_path)
        if module is not None:
            # Unregister old impls from this module
            unregister_module_impls(module_path)
            # Clear the namespace
            self._clear_module_namespace(module)
        else:
            module = types.ModuleType(module_path)
            sys.modules[module_path] = module

        self._patched_modules.add(module_path)

        # Set up module attributes
        loader = _VirtualLoader(source, virtual_filename)
        module.__name__ = module_path
        module.__file__ = virtual_filename
        module.__loader__ = loader
        module.__package__ = module_path.rpartition(".")[0] or module_path

        # Inject into linecache
        self._inject_linecache(virtual_filename, source)

        # Compile and execute
        code = compile(source, virtual_filename, "exec")
        exec(code, module.__dict__)

        # Attach to parent package
        self._attach_to_parent(module_path, module)

        # Record history
        record = PatchRecord(
            module_path=module_path,
            source=source,
            version=version,
            virtual_filename=virtual_filename,
        )
        self._history.setdefault(module_path, []).append(record)

        return module

    def get_source(self, module_path: str) -> str | None:
        """Get the current source code for a patched module."""
        history = self._history.get(module_path)
        if not history:
            return None
        return history[-1].source

    def get_history(self, module_path: str) -> list[PatchRecord]:
        """Get the full patch history for a module."""
        return list(self._history.get(module_path, []))

    def get_version(self, module_path: str) -> int:
        """Get the current version number (0 if never patched)."""
        return self._versions.get(module_path, 0)

    def _clear_module_namespace(self, module: types.ModuleType) -> None:
        """Clear a module's namespace, preserving essential attributes."""
        preserve = {
            "__name__", "__loader__", "__package__", "__spec__",
            "__path__", "__file__", "__builtins__",
        }
        to_delete = [k for k in module.__dict__ if k not in preserve]
        for k in to_delete:
            del module.__dict__[k]

    def _inject_linecache(self, filename: str, source: str) -> None:
        """Inject source into linecache for inspect.getsource() support."""
        lines = [line + "\n" for line in source.splitlines()]
        linecache.cache[filename] = (len(source), None, lines, filename)

    def _ensure_parent_packages(self, module_path: str) -> None:
        """Create virtual parent packages if they don't exist in sys.modules."""
        parts = module_path.split(".")
        for i in range(1, len(parts)):
            parent_path = ".".join(parts[:i])
            if parent_path not in sys.modules:
                pkg = types.ModuleType(parent_path)
                pkg.__path__ = []
                pkg.__package__ = parent_path
                sys.modules[parent_path] = pkg
                self._virtual_packages.add(parent_path)

    def _attach_to_parent(self, module_path: str, module: types.ModuleType) -> None:
        """Attach a module as an attribute of its parent package."""
        if "." in module_path:
            parent_path, _, child_name = module_path.rpartition(".")
            parent = sys.modules.get(parent_path)
            if parent is not None:
                setattr(parent, child_name, module)

    def save_module(self, module_path: str, directory: str | Path) -> Path:
        """Persist a patched module to the filesystem.

        Writes the latest source to a ``.py`` file under *directory*,
        updates ``module.__file__`` to the real path, replaces
        ``co_filename`` in all code objects, and removes the virtual
        linecache entry.

        Args:
            module_path: Dotted module path, e.g. ``"myagent.tools.search"``
            directory: Root directory to write into.

        Returns:
            The path of the written file.

        Raises:
            ValueError: If the module has never been patched.
        """
        source = self.get_source(module_path)
        if source is None:
            raise ValueError(f"Module {module_path!r} has never been patched")

        # Compute file path: a.b.c → directory/a/b/c.py
        parts = module_path.split(".")
        file_path = Path(directory).joinpath(*parts[:-1], parts[-1] + ".py")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(source, encoding="utf-8")

        # Update module.__file__
        module = sys.modules.get(module_path)
        real_path = str(file_path)
        virtual_filename = f"mutagent://{module_path}"

        if module is not None:
            module.__file__ = real_path

            # Replace co_filename in all code objects defined in this module
            for obj in module.__dict__.values():
                code = getattr(obj, "__code__", None)
                if code is not None and code.co_filename == virtual_filename:
                    obj.__code__ = _replace_code_filename(
                        code, virtual_filename, real_path
                    )

        # Remove virtual linecache entry (real file will be found normally)
        linecache.cache.pop(virtual_filename, None)

        return file_path

    def cleanup(self) -> None:
        """Remove all virtual modules and packages from sys.modules.

        Useful for test teardown.
        """
        for module_path in self._patched_modules:
            sys.modules.pop(module_path, None)
            # Remove linecache entries
            linecache.cache.pop(f"mutagent://{module_path}", None)
        for pkg_path in self._virtual_packages:
            sys.modules.pop(pkg_path, None)
        self._history.clear()
        self._versions.clear()
        self._patched_modules.clear()
        self._virtual_packages.clear()


def _replace_code_filename(code: types.CodeType, old: str, new: str) -> types.CodeType:
    """Recursively replace co_filename in a code object and its nested code objects."""
    new_consts = tuple(
        _replace_code_filename(c, old, new) if isinstance(c, types.CodeType) else c
        for c in code.co_consts
    )
    return code.replace(co_filename=new, co_consts=new_consts)
