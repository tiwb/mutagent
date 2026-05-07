"""mutagent.builtins.schema -- Tool schema generation from signatures and docstrings."""

from __future__ import annotations

import inspect
import re
from typing import Any

from mutagent.messages import ToolSchema


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}


def _annotation_to_json_type(annotation: Any) -> str:
    """Map a Python type annotation (or its string form) to a JSON Schema type."""
    if annotation is inspect.Parameter.empty:
        return "string"
    s = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", str(annotation))
    return _TYPE_MAP.get(s, "string")


# ---------------------------------------------------------------------------
# Docstring parsing (Google style)
# ---------------------------------------------------------------------------

def parse_docstring(docstring: str | None) -> tuple[str, dict[str, str]]:
    """Parse a Google-style docstring.

    Extracts the description (all text before Args/Returns/...) and the Args section.

    Args:
        docstring: The docstring to parse. May be None.

    Returns:
        A tuple of (description, {param_name: param_description}).
        description includes all text before the first section header (Args, Returns,
        Raises, Notes, etc.).
        param descriptions include continuation lines (indented).
    """
    if not docstring:
        return ("", {})

    lines = docstring.strip().splitlines()

    # Description: all lines before the first section header
    SECTION_HEADERS = (
        "Args", "Arguments", "Parameters",
        "Returns", "Return",
        "Raises", "Raise",
        "Yields", "Yield",
        "Note", "Notes",
        "Example", "Examples",
        "Attributes",
    )
    desc_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(%s)\s*:" % "|".join(SECTION_HEADERS), stripped):
            break
        desc_lines.append(line.rstrip())
    description = "\n".join(desc_lines).strip()

    # Find Args section
    params: dict[str, str] = {}
    in_args = False
    current_param: str | None = None
    current_desc_parts: list[str] = []
    # Indentation of the Args header determines the section level
    args_indent = 0

    for line in lines:
        stripped = line.strip()

        # Detect section headers (Args:, Returns:, Raises:, etc.)
        if re.match(r"^(Args|Arguments|Parameters)\s*:", stripped):
            in_args = True
            args_indent = len(line) - len(line.lstrip())
            continue
        elif in_args and re.match(r"^(Returns?|Raises?|Yields?|Note|Notes|Example|Attributes)\s*:", stripped):
            # End of Args section
            if current_param is not None:
                params[current_param] = " ".join(current_desc_parts).strip()
            in_args = False
            continue

        if not in_args:
            continue

        # Inside Args section
        # A new parameter line: "  param_name: description" or "  param_name (type): description"
        param_match = re.match(r"^(\s+)(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)", line)
        if param_match:
            # Save previous param
            if current_param is not None:
                params[current_param] = " ".join(current_desc_parts).strip()
            current_param = param_match.group(2)
            current_desc_parts = [param_match.group(3).strip()] if param_match.group(3).strip() else []
        elif current_param is not None and stripped:
            # Continuation line (indented more than param line)
            current_desc_parts.append(stripped)

    # Save last param
    if current_param is not None:
        params[current_param] = " ".join(current_desc_parts).strip()

    return (description, params)


# ---------------------------------------------------------------------------
# Declaration method retrieval
# ---------------------------------------------------------------------------

def get_declaration_method(cls: type, method_name: str):
    """Get the original declaration method (before @impl replacement).

    Uses mutobj's _impl_chain to find the method registered with
    source_module == "__default__" (i.e., from the declaration file).

    For classes without @impl (e.g., agent-created Toolkit subclasses),
    falls back to getattr(cls, method_name).

    Args:
        cls: The class to look up.
        method_name: The method name.

    Returns:
        The original declaration function object.
    """
    try:
        from mutobj.core import _impl_chain
        # Traverse MRO: the @impl may target a parent class
        for klass in cls.__mro__:
            chain = _impl_chain.get((klass, method_name), [])
            for func, source_module, seq in chain:
                if source_module == "__default__":
                    return func
    except ImportError:
        pass
    return getattr(cls, method_name)


# ---------------------------------------------------------------------------
# Schema generation
# ---------------------------------------------------------------------------

def make_schema(func: Any, name: str | None = None) -> ToolSchema:
    """Generate a ToolSchema from a function's signature and docstring.

    Uses inspect.signature() for parameter names, types, and defaults.
    Uses parse_docstring() for description and parameter descriptions.

    Args:
        func: The function or method to generate a schema for.
        name: Optional override for the tool name. Defaults to func.__name__.

    Returns:
        A ToolSchema instance.
    """
    func_name = name or getattr(func, '__name__', 'unknown')

    # Get signature
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return ToolSchema(
            name=func_name,
            description=func_name,
            input_schema={"type": "object", "properties": {}},
        )

    # Parse docstring
    doc = inspect.getdoc(func) or ""
    description, param_descs = parse_docstring(doc)
    if not description:
        description = func_name

    # Build properties and required list
    properties: dict[str, Any] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname == "self":
            continue

        json_type = _annotation_to_json_type(param.annotation)
        prop: dict[str, Any] = {
            "type": json_type,
            "description": param_descs.get(pname, pname),
        }

        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(pname)

        properties[pname] = prop

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        input_schema["required"] = required

    return ToolSchema(
        name=func_name,
        description=description,
        input_schema=input_schema,
    )
