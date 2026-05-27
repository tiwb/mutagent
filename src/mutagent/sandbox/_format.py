"""向后兼容 shim —— 已迁移到 SandboxEnv.format_result。"""

from __future__ import annotations
from typing import Any


def format_exec_result(result: dict[str, Any]) -> tuple[str, bool]:
    """兼容包装：SandboxEnv.format_result() 的独立函数版本。

    mutbot 旧代码仍用此函数，后续删除 mutbot 的 pysandbox_toolkit 后移除。
    """
    if "error" in result:
        text = result["error"]
        if result.get("traceback"):
            text += "\n" + result["traceback"]
        return text, True

    parts: list[str] = []
    if result.get("stdout"):
        parts.append(result["stdout"])
    value = result.get("result")
    if value is not None:
        if isinstance(value, str):
            parts.append(value)
        else:
            parts.append(repr(value))
    return ("\n".join(parts) if parts else "(no output)"), False
