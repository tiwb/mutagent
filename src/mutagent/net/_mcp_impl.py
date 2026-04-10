"""mcp.py Declaration 实现 — MCPView @impl + MCPToolProvider。"""

from __future__ import annotations

import inspect
import json
import logging
import secrets
from typing import Any, cast

import mutobj

from mutagent.net.server import Request, Response
from mutagent.net.mcp import MCPToolSet, MCPView
from mutagent.net._mcp_proto import (
    JsonRpcDispatcher,
    JsonRpcError,
    INVALID_PARAMS,
    PROTOCOL_VERSION,
    ServerCapabilities,
    ToolResult,
)
from mutagent.net._protocol import format_sse

logger = logging.getLogger("mutagent.net.mcp")


# ---------------------------------------------------------------------------
# MCPToolProvider — generation 检查 + 懒刷新
# ---------------------------------------------------------------------------


class MCPToolProvider:
    """generation 检查 + 懒刷新，桥接 Declaration 发现到 MCP handler。"""

    def __init__(self, target_view: type[MCPView] | None = None) -> None:
        self._gen: int = -1
        self._tools: dict[str, tuple[MCPToolSet, str]] = {}
        self._target_view = target_view
        # 缓存 target_view 的 path（从实例获取，避免 AttributeDescriptor）
        self._target_path: str = ""
        if target_view is not None:
            self._target_path = target_view().path

    def _match_view(self, toolset: MCPToolSet) -> bool:
        """检查 toolset 是否属于当前 view。

        优先使用 view 属性匹配（精确），其次使用 path 属性匹配。
        """
        if self._target_view is None:
            return True

        # 优先检查 view 属性
        toolset_view = toolset.view
        if toolset_view is not None:
            # 使用类名比较，避免 reload 导致的身份不匹配
            target_name = self._target_view.__name__
            if isinstance(toolset_view, tuple):
                return any(v.__name__ == target_name for v in toolset_view)
            return toolset_view.__name__ == target_name

        # 回退到 path 匹配
        toolset_path = toolset.path
        if not toolset_path:
            # 未指定 view 和 path 的 toolset 匹配所有 view
            return True
        if isinstance(toolset_path, tuple):
            return self._target_path in toolset_path
        return toolset_path == self._target_path

    def refresh(self) -> None:
        gen = mutobj.get_registry_generation()
        if gen != self._gen:
            self._gen = gen
            self._tools = {}
            for cls in mutobj.discover_subclasses(MCPToolSet):
                instance = cls()
                # 过滤：只注册匹配当前 MCPView 的 toolset
                if not self._match_view(instance):
                    continue
                prefix = instance.prefix
                for name in dir(cls):
                    if name.startswith("_"):
                        continue
                    if name in ("prefix", "view", "path"):
                        continue
                    attr = getattr(cls, name, None)
                    if attr is not None and (inspect.isfunction(attr) or inspect.ismethod(attr)):
                        if name in dir(MCPToolSet):
                            continue
                        tool_name = f"{prefix}{name}" if prefix else name
                        self._tools[tool_name] = (instance, name)

    def list_tools(self) -> list[dict[str, Any]]:
        """从类型注解 + docstring 自动生成 tool schema。"""
        self.refresh()
        result: list[dict[str, Any]] = []
        for tool_name, (instance, method_name) in self._tools.items():
            method = getattr(instance, method_name)
            schema = _infer_schema(method)
            # 优先使用声明的 docstring，避免被 @impl 覆盖
            doc = _get_declaration_doc(type(instance), method_name)
            if doc is None:
                doc = method.__doc__ or ""
            result.append({
                "name": tool_name,
                "description": doc,
                "inputSchema": schema,
            })
        return result

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        self.refresh()
        if name not in self._tools:
            raise JsonRpcError(INVALID_PARAMS, f"Unknown tool: {name}")
        instance, method_name = self._tools[name]
        method = getattr(instance, method_name)
        result = await method(**args)
        if isinstance(result, str):
            return ToolResult.text(result)
        if isinstance(result, ToolResult):
            return result
        return ToolResult.text(str(result))


def _get_declaration_doc(cls: type, method_name: str) -> str | None:
    """从 _impl_chain 取回声明时的原始 docstring，避免被 @impl 覆盖。

    沿 MRO 遍历，跳过 __doc__ 为 None 的默认条目（子类 wrapper 可能没有 doc）。
    """
    try:
        from mutobj.core import _impl_chain
        for klass in cls.__mro__:
            chain = _impl_chain.get((klass, method_name), [])
            for func, source_module, _seq in chain:
                if source_module == "__default__":
                    doc = getattr(func, "__doc__", None)
                    if doc is not None:
                        return doc
    except ImportError:
        pass
    return None


def _infer_schema(fn: Any) -> dict[str, Any]:
    """从函数签名推断 JSON Schema（简易版）。"""
    import typing

    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    def _get_json_type(annotation: Any) -> dict[str, Any]:
        """将 Python 类型注解转换为 JSON Schema 类型。"""
        # 基本类型
        if annotation in type_map:
            return {"type": type_map[annotation]}

        # 处理 typing 模块的泛型类型
        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)

        # List[T] -> {"type": "array", "items": ...}
        if origin is list:
            if args:
                return {"type": "array", "items": _get_json_type(args[0])}
            return {"type": "array"}

        # Dict[K, V] -> {"type": "object"}
        if origin is dict:
            return {"type": "object"}

        # Optional[T] = Union[T, None]
        if origin is typing.Union:
            # 过滤掉 NoneType
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                return _get_json_type(non_none_args[0])
            # 多个类型的 Union，降级为 object
            return {"type": "object"}

        # 未知类型默认为 string
        return {"type": "string"}

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            properties[name] = {"type": "string"}
        else:
            properties[name] = _get_json_type(annotation)
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


# ---------------------------------------------------------------------------
# MCPView Extension — 承载运行时状态
# ---------------------------------------------------------------------------


class _MCPViewExt(mutobj.Extension[MCPView]):
    """MCPView 的运行时状态。"""
    _tool_provider: MCPToolProvider | None = None
    _sessions: dict[str, _MCPSession] = mutobj.field(default_factory=dict)
    _dispatch: JsonRpcDispatcher | None = None


class _MCPSession:
    """MCP session 状态。"""
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.initialized = False


def _get_ext(view: MCPView) -> _MCPViewExt:
    ext = cast(_MCPViewExt, _MCPViewExt.get_or_create(view))
    if ext._tool_provider is None:
        ext._tool_provider = MCPToolProvider(target_view=type(view))
        ext._dispatch = JsonRpcDispatcher()
        _setup_handlers(ext, view)
    return ext


def _setup_handlers(ext: _MCPViewExt, view: MCPView) -> None:
    """注册 MCP JSON-RPC 方法。"""
    assert ext._dispatch is not None
    assert ext._tool_provider is not None

    tp = ext._tool_provider

    async def _handle_initialize(params: dict[str, Any]) -> dict[str, Any]:
        tools = tp.list_tools()
        capabilities = ServerCapabilities(
            tools={"listChanged": False} if tools else None,
        )
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": capabilities.to_dict(),
            "serverInfo": {"name": view.name, "version": view.version},
            **({"instructions": view.instructions} if view.instructions else {}),
        }

    async def _handle_initialized(params: dict[str, Any]) -> None:
        pass

    async def _handle_ping(params: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def _handle_tools_list(params: dict[str, Any]) -> dict[str, Any]:
        return {"tools": tp.list_tools()}

    async def _handle_tools_call(params: dict[str, Any]) -> dict[str, Any]:
        tool_name = params.get("name")
        if not tool_name:
            raise JsonRpcError(INVALID_PARAMS, "Missing tool name")
        arguments = params.get("arguments", {})
        try:
            result = await tp.call_tool(tool_name, arguments)
        except JsonRpcError:
            raise
        except Exception as e:
            logger.exception("Tool %s raised exception", tool_name)
            result = ToolResult.error(str(e))
        return result.to_dict()

    ext._dispatch.add_method("initialize", _handle_initialize)
    ext._dispatch.add_notification("notifications/initialized", _handle_initialized)
    ext._dispatch.add_method("ping", _handle_ping)
    ext._dispatch.add_method("tools/list", _handle_tools_list)
    ext._dispatch.add_method("tools/call", _handle_tools_call)


# ---------------------------------------------------------------------------
# MCPView @impl
# ---------------------------------------------------------------------------


async def _send_json_response(
    status: int, data: Any,
    extra_headers: dict[str, str] | None = None,
) -> Response:
    headers = {"content-type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    body = json.dumps(data).encode()
    headers["content-length"] = str(len(body))
    return Response(status=status, body=body, headers=headers)


async def _send_empty_response(status: int) -> Response:
    return Response(status=status, headers={"content-length": "0"})


@mutobj.impl(MCPView.post)
async def _mcp_view_post(self: MCPView, request: Request) -> Response:
    ext = _get_ext(self)
    assert ext._dispatch is not None

    raw = await request.body()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return await _send_json_response(400, {
            "jsonrpc": "2.0", "id": None,
            "error": {"code": -32700, "message": "Parse error"},
        })

    messages = parsed if isinstance(parsed, list) else [parsed]
    has_request = any(
        isinstance(m, dict) and "id" in m and "method" in m
        for m in messages
    )

    if not has_request:
        for msg in messages:
            if isinstance(msg, dict):
                await ext._dispatch.handle(msg)
        return await _send_empty_response(202)

    if isinstance(parsed, list):
        responses = []
        for msg in parsed:
            if isinstance(msg, dict):
                resp = await ext._dispatch.handle(msg)
                if resp is not None:
                    responses.append(resp)
        result_data = responses if len(responses) != 1 else responses[0]
    else:
        result_data = await ext._dispatch.handle(parsed)

    if result_data is None:
        return await _send_empty_response(202)

    extra_headers: dict[str, str] = {}
    if isinstance(parsed, dict) and parsed.get("method") == "initialize":
        session_id = secrets.token_hex(16)
        session = _MCPSession(session_id=session_id)
        ext._sessions[session_id] = session
        extra_headers["mcp-session-id"] = session_id

    accept = request.headers.get("accept", "")

    if "text/event-stream" in accept:
        sse_data = format_sse(json.dumps(result_data), event="message")
        headers = {
            "content-type": "text/event-stream",
            "cache-control": "no-cache",
        }
        headers.update(extra_headers)
        headers["content-length"] = str(len(sse_data))
        return Response(status=200, body=sse_data, headers=headers)
    else:
        return await _send_json_response(200, result_data, extra_headers)


@mutobj.impl(MCPView.delete)
async def _mcp_view_delete(self: MCPView, request: Request) -> Response:
    ext = _get_ext(self)
    session_id = request.headers.get("mcp-session-id", "")
    if session_id and session_id in ext._sessions:
        del ext._sessions[session_id]
        return await _send_empty_response(200)
    else:
        return await _send_empty_response(404)
