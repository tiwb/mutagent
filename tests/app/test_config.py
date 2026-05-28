"""Tests for the Config system (Config + resolve_model/list_models)."""

from __future__ import annotations

from pathlib import Path

import mutobj

from mutagent.app.config import Config
from mutagent.app._config_impl import _expand_env, _resolve_paths_inplace


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_config(data=None):
    """Create a Config pre-populated with load_from_dict."""
    c = Config()
    c.load_from_dict(data or {})
    return c


# ---------------------------------------------------------------------------
# Declaration tests
# ---------------------------------------------------------------------------

class TestConfigDeclaration:

    def test_inherits_from_mutagent_declaration(self):
        assert issubclass(Config, mutobj.Declaration)

    def test_uses_declaration_meta(self):
        assert issubclass(Config, mutobj.Declaration)

    def test_declared_methods(self):
        assert mutobj.get_declaration_func(Config, "get") is not None
        assert mutobj.get_declaration_func(Config, "set") is not None
        assert mutobj.get_declaration_func(Config, "on_change") is not None

    def test_stub_get_returns_default(self):
        config = Config()
        assert config.get("anything") is None
        assert config.get("anything", default="fallback") == "fallback"

    def test_stub_set_does_nothing(self):
        config = Config()
        config.set("key", "value")  # should not raise

    def test_stub_on_change_returns_cancel_fn(self):
        config = Config()
        d = config.on_change("pattern", lambda e: None)
        assert callable(d)
        d()  # should not raise


# ---------------------------------------------------------------------------
# Config load_from_dict tests
# ---------------------------------------------------------------------------

class TestConfigGet:

    def test_get_simple_key(self):
        config = _make_config({"foo": "bar"})
        assert config.get("foo") == "bar"

    def test_get_missing_key_returns_default(self):
        config = _make_config({"foo": "bar"})
        assert config.get("missing") is None
        assert config.get("missing", default="fallback") == "fallback"

    def test_get_dotted_path(self):
        config = _make_config({"a": {"b": {"c": 42}}})
        assert config.get("a.b.c") == 42

    def test_get_dotted_path_missing_segment(self):
        config = _make_config({"a": {"b": 1}})
        assert config.get("a.x.y", default="nope") == "nope"

    def test_empty_data(self):
        config = _make_config()
        assert config.get("anything", default="default") == "default"


class TestConfigSet:

    def test_set_creates_path(self):
        config = _make_config()
        config.set("a.b.c", 42)
        assert config.get("a.b.c") == 42

    def test_set_triggers_callback(self):
        events = []
        config = _make_config()
        config.on_change("a.*", lambda e: events.append(e))
        config.set("a.b", 1)
        assert len(events) == 1
        assert events[0].key == "a.b"
        assert events[0].config is config

    def test_set_does_not_trigger_unrelated(self):
        events = []
        config = _make_config()
        config.on_change("a.*", lambda e: events.append(e))
        config.set("b.c", 1)
        assert len(events) == 0


class TestConfigOnChange:

    def test_cancel_removes_listener(self):
        events = []
        config = _make_config()
        cancel = config.on_change("a", lambda e: events.append(e))
        config.set("a", 1)
        assert len(events) == 1
        cancel()
        config.set("a", 2)
        assert len(events) == 1  # 不再触发

    def test_ancestor_triggers(self):
        """set("providers") 应触发 on_change("providers.anthropic.auth_token")"""
        events = []
        config = _make_config()
        config.on_change("providers.anthropic.auth_token", lambda e: events.append(e))
        config.set("providers", {"anthropic": {"auth_token": "new"}})
        assert len(events) == 1

    def test_double_star_wildcard(self):
        events = []
        config = _make_config()
        config.on_change("providers.**", lambda e: events.append(e))
        config.set("providers.anthropic.auth_token", "new")
        assert len(events) == 1


# ---------------------------------------------------------------------------
# Config.affects() tests
# ---------------------------------------------------------------------------

class TestConfigAffects:

    def test_exact_match(self):
        config = Config()
        assert config.affects("a.b.c", "a.b.c") is True

    def test_single_wildcard(self):
        config = Config()
        assert config.affects("providers.*", "providers.anthropic") is True
        assert config.affects("providers.*", "providers.anthropic.auth_token") is False

    def test_double_wildcard(self):
        config = Config()
        assert config.affects("providers.**", "providers.anthropic") is True
        assert config.affects("providers.**", "providers.anthropic.auth_token") is True

    def test_ancestor_match(self):
        config = Config()
        assert config.affects("providers.anthropic.auth_token", "providers") is True
        assert config.affects("providers.**", "providers") is True

    def test_no_match(self):
        config = Config()
        assert config.affects("providers.*", "agents.xxx") is False


# ---------------------------------------------------------------------------
# Config.resolve_model() tests
# ---------------------------------------------------------------------------

class TestResolveModel:

    def test_list_form(self):
        config = _make_config({
            "providers": {
                "anthropic": {
                    "type": "AnthropicProvider",
                    "base_url": "https://api.anthropic.com",
                    "auth_token": "sk-123",
                    "models": ["claude-sonnet-4", "claude-haiku-4.5"],
                }
            },
        })
        model = config.resolve_model("claude-sonnet-4")
        assert model is not None
        assert model["model_id"] == "claude-sonnet-4"
        assert model["type"] == "AnthropicProvider"
        assert model["auth_token"] == "sk-123"
        assert "models" not in model

    def test_dict_form_key_match(self):
        config = _make_config({
            "providers": {
                "copilot": {
                    "type": "CopilotProvider",
                    "github_token": "ghu_xxx",
                    "models": {
                        "copilot-claude": "claude-sonnet-4",
                        "copilot-gpt": "gpt-4.1",
                    },
                }
            },
        })
        model = config.resolve_model("copilot-claude")
        assert model is not None
        assert model["model_id"] == "claude-sonnet-4"
        assert model["type"] == "CopilotProvider"
        assert model["provider_name"] == "copilot"

    def test_dict_form_value_no_match(self):
        config = _make_config({
            "providers": {
                "copilot": {
                    "type": "CopilotProvider",
                    "models": {"copilot-claude": "claude-sonnet-4"},
                }
            },
        })
        result = config.resolve_model("claude-sonnet-4")
        assert result is None  # 不再 raise SystemExit

    def test_provider_order_priority(self):
        config = _make_config({
            "providers": {
                "copilot": {
                    "type": "CopilotProvider",
                    "github_token": "ghu_xxx",
                    "models": ["claude-sonnet-4"],
                },
                "anthropic": {
                    "type": "AnthropicProvider",
                    "auth_token": "sk-123",
                    "models": ["claude-sonnet-4"],
                },
            },
        })
        model = config.resolve_model("claude-sonnet-4")
        assert model is not None
        assert model["type"] == "CopilotProvider"
        assert model["provider_name"] == "copilot"

    def test_default_model_from_config(self):
        config = _make_config({
            "default_model": "claude-haiku-4.5",
            "providers": {
                "anthropic": {
                    "type": "AnthropicProvider",
                    "auth_token": "k",
                    "models": ["claude-sonnet-4", "claude-haiku-4.5"],
                }
            },
        })
        model = config.resolve_model()
        assert model is not None
        assert model["model_id"] == "claude-haiku-4.5"

    def test_auto_default_first_model(self):
        config = _make_config({
            "providers": {
                "openai": {
                    "type": "OpenAIProvider",
                    "auth_token": "k",
                    "models": ["gpt-4.1", "gpt-4.1-mini"],
                }
            },
        })
        model = config.resolve_model()
        assert model is not None
        assert model["model_id"] == "gpt-4.1"

    def test_not_found_returns_none(self):
        config = _make_config({
            "providers": {
                "anthropic": {
                    "type": "AnthropicProvider",
                    "models": ["claude-sonnet-4"],
                }
            },
        })
        assert config.resolve_model("nonexistent") is None

    def test_no_providers_returns_none(self):
        config = _make_config()
        assert config.resolve_model("anything") is None
        assert config.resolve_model() is None


# ---------------------------------------------------------------------------
# Config.list_models() tests
# ---------------------------------------------------------------------------

class TestListModels:

    def test_list_form(self):
        config = _make_config({
            "providers": {
                "anthropic": {
                    "type": "AnthropicProvider",
                    "models": ["claude-sonnet-4", "claude-haiku-4.5"],
                }
            },
        })
        models = config.list_models()
        assert len(models) == 2
        assert models[0] == {
            "name": "claude-sonnet-4",
            "model_id": "claude-sonnet-4",
            "type": "AnthropicProvider",
            "provider_name": "anthropic",
        }

    def test_dict_form(self):
        config = _make_config({
            "providers": {
                "copilot": {
                    "type": "CopilotProvider",
                    "models": {"my-claude": "claude-sonnet-4"},
                }
            },
        })
        models = config.list_models()
        assert len(models) == 1
        assert models[0]["name"] == "my-claude"
        assert models[0]["model_id"] == "claude-sonnet-4"

    def test_multiple_providers(self):
        config = _make_config({
            "providers": {
                "copilot": {
                    "type": "CopilotProvider",
                    "models": ["claude-sonnet-4"],
                },
                "openai": {
                    "type": "OpenAIProvider",
                    "models": {"my-gpt": "gpt-4.1"},
                },
            },
        })
        models = config.list_models()
        assert len(models) == 2
        assert models[0]["provider_name"] == "copilot"
        assert models[1]["provider_name"] == "openai"

    def test_no_providers(self):
        config = _make_config()
        assert config.list_models() == []


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_resolve_paths_inplace_relative(self, tmp_path):
        data = {"path": ["lib", "ext"]}
        _resolve_paths_inplace(data, tmp_path)
        assert data["path"][0] == str((tmp_path / "lib").resolve())
        assert data["path"][1] == str((tmp_path / "ext").resolve())

    def test_resolve_paths_inplace_absolute(self):
        abs_path = str(Path.home() / "absolute" / "path")
        data = {"path": [abs_path]}
        _resolve_paths_inplace(data, Path("/other"))
        assert data["path"][0] == abs_path

    def test_resolve_paths_inplace_no_path_key(self):
        data = {"foo": "bar"}
        _resolve_paths_inplace(data, Path("/x"))
        assert data == {"foo": "bar"}


# ---------------------------------------------------------------------------
# Environment variable expansion tests
# ---------------------------------------------------------------------------

class TestEnvExpansion:

    def test_expand_dollar_var(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "secret123")
        config = _make_config({"auth_token": "$TEST_KEY"})
        assert config.get("auth_token") == "secret123"

    def test_expand_dollar_brace_var(self, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "abc")
        config = _make_config({"token": "${MY_TOKEN}"})
        assert config.get("token") == "abc"

    def test_undefined_var_preserved(self):
        config = _make_config({"key": "$UNDEFINED_VAR_XYZ"})
        assert config.get("key") == "$UNDEFINED_VAR_XYZ"

    def test_expand_nested_dict(self, monkeypatch):
        monkeypatch.setenv("NESTED_VAL", "deep")
        config = _make_config({"outer": {"inner": {"val": "$NESTED_VAL"}}})
        assert config.get("outer.inner.val") == "deep"

    def test_expand_in_list(self, monkeypatch):
        monkeypatch.setenv("LIST_VAL", "item")
        config = _make_config({"items": ["$LIST_VAL", "static"]})
        result = config.get("items")
        assert result == ["item", "static"]

    def test_non_string_values_unchanged(self):
        config = _make_config({"count": 42, "flag": True})
        assert config.get("count") == 42
        assert config.get("flag") is True

    def test_expand_mixed_text(self, monkeypatch):
        monkeypatch.setenv("HOST", "localhost")
        config = _make_config({"url": "http://$HOST:8080/api"})
        assert config.get("url") == "http://localhost:8080/api"

    def test_expand_multiple_vars(self, monkeypatch):
        monkeypatch.setenv("PROTO", "https")
        monkeypatch.setenv("DOMAIN", "example.com")
        config = _make_config({"url": "$PROTO://$DOMAIN"})
        assert config.get("url") == "https://example.com"
