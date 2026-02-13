"""mutagent.builtins.view_source -- view_source tool implementation."""

import importlib
import inspect
import sys

import mutagent
from mutagent.essential_tools import EssentialTools


def _resolve_target(target: str):
    """Resolve a dotted path to an object.

    Tries to import as a module first, then walks attributes.
    Returns the resolved object or raises an error.
    """
    # Try direct module import
    if target in sys.modules:
        return sys.modules[target]

    # Try importing
    try:
        __import__(target)
        return sys.modules[target]
    except ImportError:
        pass

    # Walk the path: module.Class.method
    parts = target.split(".")
    for i in range(len(parts), 0, -1):
        module_path = ".".join(parts[:i])
        mod = sys.modules.get(module_path)
        if mod is None:
            try:
                __import__(module_path)
                mod = sys.modules.get(module_path)
            except ImportError:
                continue
        if mod is not None:
            obj = mod
            for attr_name in parts[i:]:
                obj = getattr(obj, attr_name)
            return obj

    raise ValueError(f"Cannot resolve target: {target}")


@mutagent.impl(EssentialTools.view_source)
def view_source(self: EssentialTools, target: str) -> str:
    """View the source code of a module, class, or function."""
    try:
        obj = _resolve_target(target)
    except (ValueError, AttributeError) as e:
        return f"Error: {e}"

    try:
        return inspect.getsource(obj)
    except (OSError, TypeError) as e:
        return f"Error: Cannot get source for {target}: {e}"
