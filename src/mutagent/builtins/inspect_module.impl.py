"""mutagent.builtins.inspect_module -- inspect_module tool implementation."""

import inspect
import sys
import types
from typing import Any

import mutagent
from mutagent.essential_tools import EssentialTools


def _format_member(name: str, obj: Any, indent: str = "") -> str:
    """Format a single member for display."""
    if isinstance(obj, type):
        return f"{indent}class {name}"
    if callable(obj):
        try:
            sig = inspect.signature(obj)
            return f"{indent}def {name}{sig}"
        except (ValueError, TypeError):
            return f"{indent}def {name}(...)"
    return f"{indent}{name}: {type(obj).__name__}"


def _inspect_module_obj(mod: types.ModuleType, depth: int, current_depth: int = 0) -> str:
    """Recursively inspect a module object."""
    indent = "  " * current_depth
    lines = [f"{indent}{mod.__name__}/"]

    if current_depth >= depth:
        return "\n".join(lines)

    child_indent = "  " * (current_depth + 1)

    # List classes and functions defined in this module
    for name, obj in sorted(inspect.getmembers(mod)):
        if name.startswith("_"):
            continue
        if isinstance(obj, types.ModuleType):
            # Sub-module
            if hasattr(obj, "__name__") and obj.__name__.startswith(mod.__name__ + "."):
                sub_result = _inspect_module_obj(obj, depth, current_depth + 1)
                lines.append(sub_result)
        elif isinstance(obj, type):
            if getattr(obj, "__module__", "") == mod.__name__:
                lines.append(f"{child_indent}class {name}")
                if current_depth + 1 < depth:
                    for mname, mobj in sorted(inspect.getmembers(obj)):
                        if mname.startswith("_"):
                            continue
                        if callable(mobj):
                            try:
                                sig = inspect.signature(mobj)
                                lines.append(f"{child_indent}  def {mname}{sig}")
                            except (ValueError, TypeError):
                                lines.append(f"{child_indent}  def {mname}(...)")
        elif callable(obj):
            if getattr(obj, "__module__", "") == mod.__name__:
                lines.append(_format_member(name, obj, child_indent))

    return "\n".join(lines)


@mutagent.impl(EssentialTools.inspect_module)
def inspect_module(self: EssentialTools, module_path: str = "", depth: int = 2) -> str:
    """Inspect the structure of a Python module."""
    if not module_path:
        module_path = "mutagent"

    mod = sys.modules.get(module_path)
    if mod is None:
        try:
            __import__(module_path)
            mod = sys.modules[module_path]
        except ImportError:
            return f"Module not found: {module_path}"

    return _inspect_module_obj(mod, depth)
