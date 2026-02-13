"""mutagent.builtins.selector -- ToolSelector MVP implementation."""

import asyncio
import inspect
from typing import Any, get_type_hints

import mutagent
from mutagent.essential_tools import EssentialTools
from mutagent.messages import ToolCall, ToolResult, ToolSchema
from mutagent.selector import ToolSelector


def _python_type_to_json_type(py_type: Any) -> str:
    """Map a Python type annotation to a JSON Schema type string."""
    if py_type is str:
        return "string"
    if py_type is int:
        return "integer"
    if py_type is float:
        return "number"
    if py_type is bool:
        return "boolean"
    return "string"


def make_schema_from_method(obj: Any, method_name: str) -> ToolSchema:
    """Generate a ToolSchema from a method's signature and docstring.

    Inspects the method's type hints and docstring to build a JSON Schema
    description. The ``self`` parameter is excluded.
    """
    method = getattr(obj, method_name)
    sig = inspect.signature(method)

    # Get docstring
    doc = inspect.getdoc(method) or ""
    description = doc.split("\n")[0] if doc else method_name

    # Build input schema from parameters (skip 'self')
    properties: dict[str, Any] = {}
    required: list[str] = []

    try:
        hints = get_type_hints(method)
    except Exception:
        hints = {}

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        prop: dict[str, Any] = {}
        if param_name in hints:
            prop["type"] = _python_type_to_json_type(hints[param_name])
        else:
            prop["type"] = "string"

        # Extract parameter description from docstring Args section
        prop["description"] = param_name

        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        else:
            prop["default"] = param.default

        properties[param_name] = prop

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        input_schema["required"] = required

    return ToolSchema(
        name=method_name,
        description=description,
        input_schema=input_schema,
    )


_TOOL_METHODS = [
    "inspect_module",
    "view_source",
    "patch_module",
    "save_module",
    "run_code",
]


@mutagent.impl(ToolSelector.get_tools)
async def get_tools(self: ToolSelector, context: dict) -> list[ToolSchema]:
    """MVP: generate schemas from EssentialTools method signatures."""
    schemas = []
    for name in _TOOL_METHODS:
        schema = make_schema_from_method(self.essential_tools, name)
        schemas.append(schema)
    return schemas


@mutagent.impl(ToolSelector.dispatch)
async def dispatch(self: ToolSelector, tool_call: ToolCall) -> ToolResult:
    """MVP: directly invoke the corresponding EssentialTools method."""
    method = getattr(self.essential_tools, tool_call.name, None)
    if method is None:
        return ToolResult(
            tool_call_id=tool_call.id,
            content=f"Unknown tool: {tool_call.name}",
            is_error=True,
        )
    try:
        result = method(**tool_call.arguments)
        if asyncio.iscoroutine(result):
            result = await result
        return ToolResult(tool_call_id=tool_call.id, content=str(result))
    except Exception as e:
        return ToolResult(
            tool_call_id=tool_call.id,
            content=f"{type(e).__name__}: {e}",
            is_error=True,
        )
