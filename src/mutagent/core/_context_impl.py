"""mutagent.core._context_impl -- AgentContext default implementation."""

from __future__ import annotations

import mutobj
from .context import AgentContext
from .messages import Message, Usage


class ContextRuntime(mutobj.Extension[AgentContext]):
    """AgentContext 内部运行时状态：token 用量追踪。"""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0


def _runtime(ctx: AgentContext) -> ContextRuntime:
    return ContextRuntime.get_or_create(ctx)


def update_context_usage(ctx: AgentContext, usage: Usage) -> None:
    """累加 token 用量，包括 cache 字段。"""
    rt = _runtime(ctx)
    rt.total_input_tokens += usage.input_tokens
    rt.total_output_tokens += usage.output_tokens
    rt.total_cache_read_tokens += usage.cache_read_input_tokens
    rt.total_cache_write_tokens += usage.cache_creation_input_tokens


@mutobj.impl(AgentContext.prepare_prompts)
def agent_context_prepare_prompts(self: AgentContext) -> list[Message]:
    """按 priority 降序排列 prompts。"""
    return sorted(self.prompts, key=lambda m: m.priority, reverse=True)


@mutobj.impl(AgentContext.prepare_messages)
def agent_context_prepare_messages(self: AgentContext) -> list[Message]:
    """整理对话历史。"""
    return list(self.messages)

