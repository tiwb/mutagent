"""LLMProvider 实现。

from_spec 由各子类覆盖；
"""

from __future__ import annotations

import fnmatch
from typing import AsyncGenerator

import mutobj
from mutio.codec.json import JsonObject
from mutobj import discover_subclasses, get_registry_generation

from .llm import LLMApiClient
from .messages import (
    Message,
    StreamEvent,
    ToolSchema,
)

# 常见模型的 context window 大小（token 数），作为配置未指定时的兜底。
# 支持通配符：精确匹配优先，通配符按长度降序匹配（更长 = 更具体）。
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Anthropic
    "claude-*": 200_000,
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "o1": 200_000,
    "o3-mini": 200_000,
}


def get_default_context_window(model_id: str) -> int | None:
    """Look up context window for *model_id* from ``MODEL_CONTEXT_WINDOWS``.

    Resolution order:
    1. Exact match (key has no wildcard chars ``*`` / ``?``).
    2. Wildcard patterns via :func:`fnmatch.fnmatch`, longest pattern first.

    Returns ``None`` if no entry matches.
    """
    # 1. Exact match
    if model_id in MODEL_CONTEXT_WINDOWS:
        key = model_id
        if "*" not in key and "?" not in key:
            return MODEL_CONTEXT_WINDOWS[key]

    # 2. Wildcard: collect matching patterns, longest first
    for pattern, val in sorted(
        MODEL_CONTEXT_WINDOWS.items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        if ("*" in pattern or "?" in pattern) and fnmatch.fnmatch(model_id, pattern):
            return val

    return None


# ---------------------------------------------------------------------------
# 短名缓存：自动从 LLMApiClient 子类构建，registry 不变则复用
# ---------------------------------------------------------------------------

_provider_aliases: dict[str, type] | None = None
_provider_aliases_gen: int = -1


def _get_provider_aliases() -> dict[str, type]:
    """从已注册的 LLMApiClient 子类构建 api_type → 类映射。
    
    只使用 api_type classvar 作为 key，不再从类名推导。
    通过 get_registry_generation() 做缓存失效。
    """
    global _provider_aliases, _provider_aliases_gen
    gen = get_registry_generation()
    if _provider_aliases is not None and _provider_aliases_gen == gen:
        return _provider_aliases

    aliases: dict[str, type] = {}
    for sub_cls in discover_subclasses(LLMApiClient):
        if sub_cls is LLMApiClient:
            continue
        api_type = sub_cls.api_type
        if api_type:
            aliases[api_type.lower()] = sub_cls
            aliases[f"{api_type.lower()}provider"] = sub_cls
            aliases[f"{api_type.lower()}apiclient"] = sub_cls
        aliases[sub_cls.__name__.lower()] = sub_cls
    _provider_aliases = aliases
    _provider_aliases_gen = gen
    return aliases


def _iter_provider_lookup_keys(name: str) -> list[str]:
    raw = name.strip()
    if not raw:
        return []

    keys = [raw.lower()]
    if "." in raw:
        keys.append(raw.rsplit(".", 1)[-1].lower())

    index = 0
    while index < len(keys):
        key = keys[index]
        if key.endswith("provider"):
            stripped = key[:-8]
            if stripped:
                keys.append(stripped)
        if key.endswith("apiclient"):
            stripped = key[:-9]
            if stripped:
                keys.append(stripped)
        index += 1

    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _resolve_provider(name: str) -> type:
    """将 provider 类型名解析为 LLMApiClient 子类。

    只支持 api_type 短名（如 "Anthropic"、"OpenAI"），大小写不敏感。
    空字符串默认为 AnthropicApiClient。
    """
    aliases = _get_provider_aliases()
    if not name:
        return aliases.get("anthropic", LLMApiClient)
    for key in _iter_provider_lookup_keys(name):
        if key in aliases:
            return aliases[key]
    raise ValueError(f"Unknown provider type '{name}'")


@mutobj.impl(LLMApiClient.from_spec)
def llm_api_client_from_spec(spec: JsonObject) -> LLMApiClient:
    provider_type = str(spec.get("type", ""))
    provider_cls = _resolve_provider(provider_type)
    return provider_cls(spec)

@mutobj.impl(LLMApiClient.send)
def llm_api_client_send(
        self: LLMApiClient,
        messages: list[Message],
        tools: list[ToolSchema],
        prompts: list[Message] | None = None,
        stream: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:

    raise NotImplementedError(
        f"{type(self).__name__} does not implement 'send'."
    )


# 确保内置 provider 已注册
from . import _llm_impl_anthropic as  _llm_impl_anthropic  # noqa: F401
from . import _llm_impl_copilot  as _llm_impl_copilot  # noqa: F401
from . import _llm_impl_openai as _llm_impl_openai  # noqa: F401