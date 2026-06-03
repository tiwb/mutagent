"""mutagent.core.context -- AgentContext declaration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mutobj
from mutobj import field

if TYPE_CHECKING:
    from .messages import Message


class AgentContext(mutobj.Declaration):
    """Agent 上下文管理。

    管理系统指令（prompts）和对话历史（messages），提供 token 用量追踪。

    Attributes:
        prompts: 系统指令列表。
        messages: 对话历史列表。
    """

    prompts: list[Message] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)

    def prepare_prompts(self) -> list[Message]:
        """发送前整理系统指令：按 priority 降序排列。"""
        ...

    def prepare_messages(self) -> list[Message]:
        """发送前整理对话历史：默认直接返回。"""
        ...


from . import _context_impl as _context_impl  # noqa: F401, E402
