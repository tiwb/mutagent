"""MCP 桥接 — 连接外部 MCP server，自动生成命名空间函数。

支持 stdio 模式（subprocess + JSON-RPC over stdin/stdout）。
TODO: 支持 Streamable HTTP 模式（通过 mutagent.net.client.MCPClient）。
"""

import asyncio
import json
import subprocess
import sys
from typing import Any

from mutagent.sandbox._namespace import Namespace

# Windows: 抑制子进程弹出控制台窗口
_POPEN_KWARGS: dict[str, Any] = {}
if sys.platform == "win32":
    _POPEN_KWARGS["creationflags"] = subprocess.CREATE_NO_WINDOW


class StdioMCPClient:
    """Stdio MCP client — 通过 subprocess 连接 MCP server。"""

    def __init__(self, command: str, args: list[str] | None = None,
                 shell: bool = False):
        self._command = command
        self._args = args or []
        self._shell = shell
        self._process: subprocess.Popen | None = None
        self._request_id = 0

    async def connect(self) -> dict[str, Any]:
        """启动 MCP server 子进程并完成 initialize 握手。"""
        if self._shell:
            cmd = self._command + ' ' + ' '.join(self._args)
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                shell=True,
                **_POPEN_KWARGS,
            )
        else:
            self._process = subprocess.Popen(
                [self._command] + self._args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                **_POPEN_KWARGS,
            )
        # MCP initialize 握手
        result = await self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "mutagent-sandbox", "version": "0.1.0"},
        })
        # 发送 initialized 通知
        self._send_notification("notifications/initialized", {})
        return result

    async def list_tools(self) -> list[dict[str, Any]]:
        """获取 server 的 tool 列表。"""
        result = await self._request("tools/list", {})
        return result.get("tools", [])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """调用 tool 并返回结果。"""
        result = await self._request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        # 提取文本内容
        content = result.get("content", [])
        if result.get("isError"):
            texts = [c.get("text", "") for c in content if c.get("type") == "text"]
            raise RuntimeError('\n'.join(texts) if texts else "MCP tool call failed")

        # 返回内容：优先文本，其次原始 content
        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
        if len(texts) == 1:
            # 尝试解析 JSON
            try:
                return json.loads(texts[0])
            except (json.JSONDecodeError, ValueError):
                return texts[0]
        elif texts:
            return '\n'.join(texts)
        return content

    async def close(self) -> None:
        """关闭连接。"""
        if self._process:
            try:
                self._process.stdin.close()
            except Exception:
                pass
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    async def _request(self, method: str, params: dict) -> dict:
        """发送 JSON-RPC 请求并等待响应。"""
        self._request_id += 1
        msg = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        line = json.dumps(msg) + '\n'

        loop = asyncio.get_event_loop()
        # 在线程中执行阻塞 IO
        response = await loop.run_in_executor(None, self._send_and_receive, line)
        return response

    def _send_and_receive(self, line: str) -> dict:
        """同步发送请求并读取响应。"""
        if not self._process or not self._process.stdin or not self._process.stdout:
            raise RuntimeError("MCP server process not running")

        self._process.stdin.write(line)
        self._process.stdin.flush()

        # 读取响应行（跳过通知等非响应消息）
        while True:
            resp_line = self._process.stdout.readline()
            if not resp_line:
                raise RuntimeError("MCP server closed unexpectedly")
            try:
                resp = json.loads(resp_line)
            except json.JSONDecodeError:
                continue
            # 跳过通知（没有 id 的消息）
            if "id" in resp:
                if "error" in resp:
                    err = resp["error"]
                    raise RuntimeError(
                        f"MCP error {err.get('code')}: {err.get('message')}")
                return resp.get("result", {})

    def _send_notification(self, method: str, params: dict) -> None:
        """发送 JSON-RPC 通知（不期望响应）。"""
        msg = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        if self._process and self._process.stdin:
            self._process.stdin.write(json.dumps(msg) + '\n')
            self._process.stdin.flush()


async def bridge_mcp_server(ns_name: str, command: str,
                            args: list[str] | None = None,
                            shell: bool = False) -> tuple[Namespace, StdioMCPClient]:
    """桥接一个 MCP server，返回命名空间和 client。

    Args:
        ns_name: 命名空间名（如 "playwright"）
        command: MCP server 启动命令
        args: 命令参数
        shell: 是否使用 shell 模式启动

    Returns:
        (namespace, client) 元组
    """
    client = StdioMCPClient(command, args, shell=shell)
    await client.connect()

    tools = await client.list_tools()
    ns = Namespace(ns_name)

    for tool in tools:
        tool_name = tool["name"]
        tool_desc = tool.get("description", "")
        input_schema = tool.get("inputSchema", {})

        # 生成同步包装函数
        fn = _make_tool_func(client, tool_name, tool_desc, input_schema)
        ns.register(tool_name, fn, tool_desc)

    return ns, client


def _make_tool_func(client: StdioMCPClient, tool_name: str,
                    description: str,
                    input_schema: dict) -> Any:
    """为一个 MCP tool 生成 Python 函数。"""
    # 从 schema 提取参数信息
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    # 构建文档
    doc_lines = [description, ""]
    if properties:
        doc_lines.append("Args:")
        for pname, pinfo in properties.items():
            ptype = pinfo.get("type", "any")
            pdesc = pinfo.get("description", "")
            req_mark = " (required)" if pname in required else ""
            doc_lines.append(f"    {pname}: {ptype}{req_mark} — {pdesc}")

    doc = '\n'.join(doc_lines)

    def tool_func(**kwargs):
        # 守护进程模式：exec_code 在线程池执行，需要 run_coroutine_threadsafe 回到主 loop
        # 独立模式：没有 running loop，用 asyncio.run
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(
                client.call_tool(tool_name, kwargs), loop)
            return future.result(timeout=120)
        except RuntimeError:
            return asyncio.run(client.call_tool(tool_name, kwargs))

    tool_func.__name__ = tool_name
    tool_func.__doc__ = doc
    return tool_func
