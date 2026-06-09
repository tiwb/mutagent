from __future__ import annotations

from mutagent.app.config import Config
from mutagent.core._llm_impl_openai import OpenAIApiClient
from mutagent.llmproxy.routes import (
    configure_llm_proxy,
    reset_llm_proxy_runtime,
)


def setup_function():
    reset_llm_proxy_runtime()


def teardown_function():
    reset_llm_proxy_runtime()


def _make_config() -> Config:
    config = Config()
    config.load_from_dict(
        {
            "logging": {"log_dir": ".mutagent/logs"},
            "providers": {
                "openai": {
                    "type": "OpenAI",
                    "auth_token": "sk-test",
                    "models": {
                        "gpt-main": "gpt-4.1",
                    },
                },
                "anthropic": {
                    "type": "Anthropic",
                    "auth_token": "ak-test",
                    "models": ["claude-sonnet-4-20250514"],
                },
            },
        }
    )
    return config


def test_runtime_lists_models():
    runtime = configure_llm_proxy(_make_config())

    models = runtime.list_models()
    assert {model["name"] for model in models} == {"gpt-main", "claude-sonnet-4-20250514"}


def test_runtime_resolves_alias_and_normalized_model():
    runtime = configure_llm_proxy(_make_config())

    resolved_alias = runtime.resolve_model("gpt-main")
    assert resolved_alias is not None
    assert resolved_alias.backend_model == "gpt-4.1"
    assert isinstance(resolved_alias.provider, OpenAIApiClient)

    resolved_normalized = runtime.resolve_model("claude-sonnet-4")
    assert resolved_normalized is not None
    assert resolved_normalized.backend_model == "claude-sonnet-4-20250514"
    assert resolved_normalized.provider_name == "anthropic"


def test_runtime_invalidates_provider_cache_on_config_change():
    config = _make_config()
    runtime = configure_llm_proxy(config)

    first = runtime.resolve_model("gpt-main")
    assert first is not None
    assert isinstance(first.provider, OpenAIApiClient)
    assert first.provider.api_key == "sk-test"

    config.root.set(
        "providers.openai",
        {
            "type": "OpenAI",
            "auth_token": "sk-updated",
            "models": {"gpt-main": "gpt-4.1"},
        },
    )

    second = runtime.resolve_model("gpt-main")
    assert second is not None
    assert second.provider.api_key == "sk-updated"
    assert second.provider is not first.provider
