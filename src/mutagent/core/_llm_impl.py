"""LLMProvider 实现。

from_spec 由各子类覆盖；
"""

from __future__ import annotations

import fnmatch

import mutobj
from mutobj import discover_subclasses, get_registry_generation

from .llm import LLMApiClient


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
    # 确保内置 provider 已注册
    from mutagent.core import _llm_impl_anthropic  # noqa: F401
    from mutagent.core import _llm_impl_openai  # noqa: F401
    for sub_cls in discover_subclasses(LLMApiClient):
        if sub_cls is LLMApiClient:
            continue
        api_type = sub_cls.api_type
        if api_type:
            aliases[api_type.lower()] = sub_cls
    _provider_aliases = aliases
    _provider_aliases_gen = gen
    return aliases


def _resolve_provider(name: str) -> type:
    """将 provider 类型名解析为 LLMApiClient 子类。

    只支持 api_type 短名（如 "Anthropic"、"OpenAI"），大小写不敏感。
    空字符串默认为 AnthropicApiClient。
    """
    aliases = _get_provider_aliases()
    key = name.lower()
    if key and key in aliases:
        return aliases[key]
    if not name:
        return aliases.get("anthropic", LLMApiClient)
    raise ValueError(f"Unknown provider type '{name}'")


@mutobj.impl(LLMApiClient.from_spec)
def llm_api_client_from_spec(spec: dict) -> LLMApiClient:
    provider_type = spec.get("type", "")
    provider_cls = _resolve_provider(provider_type)
    return provider_cls(spec)
