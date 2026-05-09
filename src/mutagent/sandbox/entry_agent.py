"""SandboxToolkit -- Agent 的单一工具入口（entry_agent）。

所有能力（外部 MCP server、CLI 工具、内置 NamespaceTools）通过 namespace 注入沙箱，
LLM 在 pysandbox 中编排多步逻辑，减少 tool call 往返。

接口与 mutbot 的 `mutbot.builtins.pysandbox_toolkit.PySandboxToolkit` 完全一致——
mutbot 后续会删除自己的版本，统一用此版。
"""

from __future__ import annotations

import asyncio
from typing import Any

import mutagent
from mutagent.tools import Toolkit
from mutagent.sandbox.app import SandboxApp, PYSANDBOX_DOC


class SandboxToolkit(Toolkit):
    """pysandbox — 安全的 Python 代码执行环境。

    Attributes:
        _tool_prefix: 空字符串，使 tool 名直接为方法名（"pysandbox"）。
        _tool_methods: 限定只暴露 pysandbox 一个方法。
        _app: 沙箱执行引擎实例。
        _state: 跨 tool call 共享的 REPL 变量字典（per-toolkit 实例隔离）。
    """

    _tool_prefix = ""
    _tool_methods = ["pysandbox"]

    _app: SandboxApp
    _state: dict[str, Any]

    async def pysandbox(self, code: str) -> str:
        loop = asyncio.get_running_loop()
        # 注入主 loop 供 _wrap_async 投递 async NamespaceTools 方法
        self._app.bind_main_loop()
        result = await loop.run_in_executor(
            None, self._app.exec_code, code, self._state)
        text, _is_error = self._app.format_result(result)
        return text


# % 取模运算不会让 Python 把表达式当作函数 docstring，需显式赋值
SandboxToolkit.pysandbox.__doc__ = PYSANDBOX_DOC + """
Variables persist across calls in the same agent session (REPL state).
"""