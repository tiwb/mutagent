"""mutagent.builtins.selector -- ToolSelector MVP implementation."""

import ast
import asyncio
import inspect
import textwrap
from typing import Any

import mutagent
from mutagent.essential_tools import EssentialTools
from mutagent.messages import ToolCall, ToolResult, ToolSchema
from mutagent.selector import ToolSelector


def _python_type_to_json_type(type_str: str) -> str:
    """Map a Python type annotation string to a JSON Schema type string."""
    mapping = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
    }
    return mapping.get(type_str, "string")


def _parse_method_signature(cls: type, method_name: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse a method's signature from the class source code using AST.

    Returns (docstring, params) where params is a list of dicts with
    keys: name, type, default, required.
    """
    source = inspect.getsource(cls)
    source = textwrap.dedent(source)
    tree = ast.parse(source)

    class_def = tree.body[0]
    for node in class_def.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == method_name:
                return _extract_from_funcdef(node)

    return (method_name, [])


def _extract_from_funcdef(node: ast.FunctionDef) -> tuple[str, list[dict[str, Any]]]:
    """Extract docstring and parameters from an AST FunctionDef node."""
    # Docstring
    docstring = ""
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
        docstring = node.body[0].value.value

    # Parameters
    params = []
    args = node.args
    # Compute defaults offset: defaults align to the end of args
    num_defaults = len(args.defaults)
    num_args = len(args.args)
    default_offset = num_args - num_defaults

    for i, arg in enumerate(args.args):
        if arg.arg == "self":
            continue

        param: dict[str, Any] = {"name": arg.arg}

        # Type annotation
        if arg.annotation:
            param["type"] = _annotation_to_str(arg.annotation)
        else:
            param["type"] = "str"

        # Default value
        default_idx = i - default_offset
        if default_idx >= 0 and default_idx < len(args.defaults):
            default_node = args.defaults[default_idx]
            param["default"] = _ast_literal(default_node)
            param["required"] = False
        else:
            param["required"] = True

        params.append(param)

    return (docstring, params)


def _annotation_to_str(node: ast.expr) -> str:
    """Convert an AST annotation node to a type string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.Attribute):
        return ast.unparse(node)
    return ast.unparse(node)


def _ast_literal(node: ast.expr) -> Any:
    """Extract a literal value from an AST node."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return ast.unparse(node)


def make_schema_from_method(obj: Any, method_name: str) -> ToolSchema:
    """Generate a ToolSchema from a method's source definition.

    Parses the class source code to extract the method signature,
    since forwardpy stubs replace the original signature.
    """
    cls = type(obj)
    docstring, params = _parse_method_signature(cls, method_name)

    # Use first line of docstring as description
    description = docstring.split("\n")[0].strip() if docstring else method_name

    # Build input schema
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param in params:
        prop: dict[str, Any] = {
            "type": _python_type_to_json_type(param["type"]),
            "description": param["name"],
        }
        if not param.get("required", True):
            prop["default"] = param.get("default")
        else:
            required.append(param["name"])
        properties[param["name"]] = prop

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
