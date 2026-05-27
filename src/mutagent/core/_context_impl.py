"""mutagent.core._context_impl -- AgentContext default implementation."""

from __future__ import annotations

import mutobj
from .context import AgentContext
from .messages import Message


@mutobj.impl(AgentContext.prepare_prompts)
def agent_context_prepare_prompts(self: AgentContext) -> list[Message]:
    """按 priority 降序排列 prompts。"""
    return sorted(self.prompts, key=lambda m: m.priority, reverse=True)


@mutobj.impl(AgentContext.prepare_messages)
def agent_context_prepare_messages(self: AgentContext) -> list[Message]:
    """整理对话历史。"""
    return list(self.messages)


@mutobj.impl(AgentContext.update_usage)
def agent_context_update_usage(self: AgentContext, usage: dict[str, int]) -> None:
    """累加 token 用量，包括 cache 字段。"""
    total = getattr(self, '_total_input_tokens', 0)
    total += usage.get('input_tokens', 0)
    object.__setattr__(self, '_total_input_tokens', total)

    total_out = getattr(self, '_total_output_tokens', 0)
    total_out += usage.get('output_tokens', 0)
    object.__setattr__(self, '_total_output_tokens', total_out)

    # Cache read — Anthropic: cache_read_input_tokens, OpenAI: cached_tokens → cache_read_input_tokens
    cache_read = getattr(self, '_total_cache_read_tokens', 0)
    cache_read += usage.get('cache_read_input_tokens', 0)
    object.__setattr__(self, '_total_cache_read_tokens', cache_read)

    # Cache write — Anthropic only
    cache_write = getattr(self, '_total_cache_write_tokens', 0)
    cache_write += usage.get('cache_creation_input_tokens', 0)
    object.__setattr__(self, '_total_cache_write_tokens', cache_write)


@mutobj.impl(AgentContext.get_cache_read_tokens)
def agent_context_get_cache_read_tokens(self: AgentContext) -> int:
    """返回累计缓存读取 token 数。"""
    return getattr(self, '_total_cache_read_tokens', 0)


@mutobj.impl(AgentContext.get_cache_write_tokens)
def agent_context_get_cache_write_tokens(self: AgentContext) -> int:
    """返回累计缓存写入 token 数。"""
    return getattr(self, '_total_cache_write_tokens', 0)


@mutobj.impl(AgentContext.get_context_used)
def agent_context_get_context_used(self: AgentContext) -> int:
    """返回最近一次 LLM 调用的 input_tokens（近似 context 用量）。"""
    return getattr(self, '_total_input_tokens', 0)


@mutobj.impl(AgentContext.get_context_percent)
def agent_context_get_context_percent(self: AgentContext) -> float | None:
    """返回 context 使用百分比。context_window=0 时返回 None。"""
    if not self.context_window:
        return None
    used = agent_context_get_context_used(self)
    return used / self.context_window
