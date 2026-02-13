"""mutagent.runtime.impl_loader -- Discover and load .impl.py files."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mutagent.runtime.module_manager import ModuleManager


class ImplLoader:
    """Discovers and loads ``.impl.py`` implementation files.

    Implementation files contain ``@impl`` registrations that provide
    concrete implementations for stub methods declared in ``.py`` files.

    Args:
        module_manager: Optional ModuleManager instance. If provided,
            modules are loaded via ``patch_module`` (with source tracking,
            linecache, etc.). Otherwise, modules are loaded via direct exec.
    """

    def __init__(self, module_manager: ModuleManager | None = None) -> None:
        self._module_manager = module_manager

    def discover(self, package_path: str | Path) -> list[Path]:
        """Scan a directory tree for ``.impl.py`` files.

        Args:
            package_path: Root directory to scan.

        Returns:
            Sorted list of paths to ``.impl.py`` files found.
        """
        root = Path(package_path)
        if not root.is_dir():
            return []
        results = sorted(root.rglob("*.impl.py"))
        return results

    def load_file(
        self,
        impl_path: str | Path,
        package_root: str | Path,
        base_package: str,
    ) -> types.ModuleType:
        """Load a single ``.impl.py`` file, executing its ``@impl`` registrations.

        Args:
            impl_path: Path to the ``.impl.py`` file.
            package_root: Root directory of the package (used to compute
                the dotted module name).
            base_package: Base package name prefix (e.g. ``"mutagent.builtins"``).

        Returns:
            The loaded module object.
        """
        impl_path = Path(impl_path)
        package_root = Path(package_root)
        source = impl_path.read_text(encoding="utf-8")
        module_name = self._compute_module_name(impl_path, package_root, base_package)

        if self._module_manager is not None:
            return self._module_manager.patch_module(module_name, source)

        return self._exec_module(module_name, source, str(impl_path))

    def load_all(
        self,
        package_path: str | Path,
        base_package: str,
    ) -> list[types.ModuleType]:
        """Discover and load all ``.impl.py`` files under a package directory.

        Args:
            package_path: Root directory to scan and load from.
            base_package: Base package name prefix.

        Returns:
            List of loaded module objects.
        """
        package_path = Path(package_path)
        impl_files = self.discover(package_path)
        modules = []
        for impl_path in impl_files:
            mod = self.load_file(impl_path, package_path, base_package)
            modules.append(mod)
        return modules

    def _compute_module_name(
        self,
        impl_path: Path,
        package_root: Path,
        base_package: str,
    ) -> str:
        """Convert a file path to a dotted module name.

        Example: ``package_root/sub/foo.impl.py`` with ``base_package="pkg"``
        becomes ``"pkg.sub.foo"``.
        """
        rel = impl_path.relative_to(package_root)
        # Strip .impl.py suffix: stem gives "foo.impl", strip ".impl"
        parts = list(rel.parent.parts)
        stem = rel.stem  # "foo.impl"
        if stem.endswith(".impl"):
            stem = stem[: -len(".impl")]
        parts.append(stem)

        if base_package:
            return base_package + "." + ".".join(parts)
        return ".".join(parts)

    def _exec_module(
        self,
        module_name: str,
        source: str,
        filename: str,
    ) -> types.ModuleType:
        """Load a module via direct exec (no ModuleManager)."""
        module = types.ModuleType(module_name)
        module.__file__ = filename
        module.__name__ = module_name
        module.__package__ = module_name.rpartition(".")[0] or module_name
        sys.modules[module_name] = module

        code = compile(source, filename, "exec")
        exec(code, module.__dict__)
        return module
