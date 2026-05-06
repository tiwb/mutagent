"""mutagent.builtins.context_impl -- AgentContext default implementation."""

from __future__ import annotations

import copy
from datetime import datetime, timezone

import mutagent
from mutagent.context import AgentContext
from mutagent.messages import Message, TextBlock


def _format_timestamp(ts: float) -> str:
    """Unix timestamp → 'YYYY-MM-DD HH:MM' (本地时间)。"""
    if not ts:
        return ""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()
    return dt.strftime("%Y-%m-%d %H:%M")


def _inject_metadata(msg: Message) -> Message:
    """为消息添加 [sender · timestamp] 元信息前缀行。

    系统消息不处理。生成 Message 副本，不修改原始消息。
    """
    if msg.role == "system":
        return msg

    # 确定显示名称
    if msg.role == "user":
        name = msg.sender or "User"
    else:
        name = msg.model or "Assistant"

    time_str = _format_timestamp(msg.timestamp)
    if time_str:
        prefix = f"[{name} · {time_str}]"
    else:
        prefix = f"[{name}]"

    # 找到首个 TextBlock 并注入前缀行
    new_blocks = list(msg.blocks)
    injected = False
    for i, block in enumerate(new_blocks):
        if isinstance(block, TextBlock):
            new_blocks[i] = TextBlock(text=f"{prefix}\n{block.text}")
            injected = True
            break

    if not injected:
        # 没有 TextBlock，在开头插入
        new_blocks.insert(0, TextBlock(text=prefix))

    # 浅拷贝 Message，替换 blocks
    new_msg = copy.copy(msg)
    object.__setattr__(new_msg, 'blocks', new_blocks)
    return new_msg


@mutagent.impl(AgentContext.prepare_prompts)
def prepare_prompts(self: AgentContext) -> list[Message]:
    """按 priority 降序排列 prompts。"""
    return sorted(self.prompts, key=lambda m: m.priority, reverse=True)


@mutagent.impl(AgentContext.prepare_messages)
def prepare_messages(self: AgentContext) -> list[Message]:
    """整理对话历史。启用 message_metadata 时注入发送者和时间前缀。"""
    messages = list(self.messages)
    if self.message_metadata:
        messages = [_inject_metadata(m) for m in messages]
    return messages


@mutagent.impl(AgentContext.update_usage)
def update_usage(self: AgentContext, usage: dict[str, int]) -> None:
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


@mutagent.impl(AgentContext.get_cache_read_tokens)
def get_cache_read_tokens(self: AgentContext) -> int:
    """返回累计缓存读取 token 数。"""
    return getattr(self, '_total_cache_read_tokens', 0)


@mutagent.impl(AgentContext.get_cache_write_tokens)
def get_cache_write_tokens(self: AgentContext) -> int:
    """返回累计缓存写入 token 数。"""
    return getattr(self, '_total_cache_write_tokens', 0)


@mutagent.impl(AgentContext.get_context_used)
def get_context_used(self: AgentContext) -> int:
    """返回最近一次 LLM 调用的 input_tokens（近似 context 用量）。"""
    return getattr(self, '_total_input_tokens', 0)


@mutagent.impl(AgentContext.get_context_percent)
def get_context_percent(self: AgentContext) -> float | None:
    """返回 context 使用百分比。context_window=0 时返回 None。"""
    if not self.context_window:
        return None
    used = get_context_used(self)
    return used / self.context_window
