"""mutagent.context -- AgentContext declaration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mutagent
from mutagent import field

if TYPE_CHECKING:
    from mutagent.messages import Message


class AgentContext(mutagent.Declaration):
    """Agent 上下文管理。

    管理系统指令（prompts）和对话历史（messages），提供 token 用量追踪。

    Attributes:
        context_window: 模型上下文窗口大小（tokens），0 = 未知。
        prompts: 系统指令列表。
        messages: 对话历史列表。
    """

    context_window: int = 0
    prompts: list = field(default_factory=list)
    messages: list = field(default_factory=list)

    def prepare_prompts(self) -> list[Message]:
        """发送前整理系统指令：按 priority 降序排列。"""
        return context_impl.prepare_prompts(self)

    def prepare_messages(self) -> list[Message]:
        """发送前整理对话历史：默认直接返回。"""
        return context_impl.prepare_messages(self)

    def update_usage(self, usage: dict[str, int]) -> None:
        """更新 token 用量。"""
        return context_impl.update_usage(self, usage)

    def get_context_used(self) -> int:
        """获取已使用的 token 数。"""
        return context_impl.get_context_used(self)

    def get_context_percent(self) -> float | None:
        """获取 context 使用百分比。context_window=0 时返回 None。"""
        return context_impl.get_context_percent(self)

    def get_cache_read_tokens(self) -> int:
        """获取累计缓存读取 tokens。"""
        return context_impl.get_cache_read_tokens(self)

    def get_cache_write_tokens(self) -> int:
        """获取累计缓存写入 tokens。"""
        return context_impl.get_cache_write_tokens(self)


from .builtins import context_impl
mutagent.register_module_impls(context_impl)
