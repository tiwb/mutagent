"""LLM Settings panel — Declaration + full implementation.

Replaces old settings.py (LLMSettingsPanel Declaration) + _settings_impl.py (916-line impl).
LLMSettingsPanel now extends SettingsPanel from webui.settings.
"""

from __future__ import annotations

import inspect
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar

import httpx
import mutagent
import mutobj
from mutagent.provider import LLMProvider
from mutagent.webui.settings import SettingsPanel
from mutgui import Bind, Callback, ViewBlock

_CHAT_MODEL_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")
_VARIANT_SUFFIXES = ("-mini", "-nano", "-turbo", "-latest", "-preview", "-realtime")
_FEATURED_FAMILIES_PER_PREFIX = 2

_ANTHROPIC_PROVIDER = "mutagent.builtins.anthropic_provider.AnthropicProvider"
_OPENAI_PROVIDER = "mutagent.builtins.openai_provider.OpenAIProvider"

_PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    _ANTHROPIC_PROVIDER: {
        "label": "Anthropic",
        "name_seed": "anthropic",
        "base_url": "https://api.anthropic.com",
        "models": ["claude-sonnet-4", "claude-haiku-4.5", "claude-opus-4"],
        "tag_color": "purple",
    },
    _OPENAI_PROVIDER: {
        "label": "OpenAI",
        "name_seed": "openai",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4.1", "gpt-4.1-mini", "o3"],
        "tag_color": "blue",
    },
}


class LLMSettingsPanel(SettingsPanel):
    """LLM provider 配置面板。

    独立面板文件，由 SettingsDrawer 通过 discover_subclasses 自动发现。
    """

    panel_id: ClassVar[str] = "llm"
    panel_title: ClassVar[str] = "LLM API 设置"
    panel_placement: ClassVar[str] = "settings:10/10"

    _app: Any = None
    _agent: Any = None
    _drafts: dict[str, dict[str, Any]] = mutobj.field(default_factory=dict)

    id: str | int = "llm-settings-panel"

    # ── State fields ──────────────────────────────
    current_step: str = "list"
    editing_key: str = ""
    editing_is_new: bool = False
    provider_name: str = ""
    provider_type: str = _ANTHROPIC_PROVIDER
    provider_type_label: str = "Anthropic"
    base_url: str = ""
    auth_token: str = ""
    models: list[str] = mutobj.field(default_factory=list)
    discovered_models: list[str] = mutobj.field(default_factory=list)
    default_model: str = ""
    error: str = ""
    notice: str = ""

    def __init__(self, *, app: Any, agent: Any) -> None:
        super(LLMSettingsPanel, self).__init__()
        self._app = app
        self._agent = agent
        _load_from_config(self)

    def render(self: LLMSettingsPanel) -> ViewBlock:
        _sync_default_model(self)
        if self.current_step == "edit":
            children = _render_edit(self)
        else:
            children = _render_list(self)
        return ViewBlock([{
            "$component": "div",
            "$id": "llm-settings-body",
            "style": {"paddingBottom": 12},
            "$children": children,
        }])


# ═══════════════════════════════════════════════════════════════
#  Utility functions (unchanged from old _settings_impl.py)
# ═══════════════════════════════════════════════════════════════


def _normalize_models(value: Any) -> list[str]:
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in result:
                result.append(text)
        return result
    if isinstance(value, str):
        result = []
        for part in value.split(","):
            text = part.strip()
            if text and text not in result:
                result.append(text)
        return result
    return []


def _provider_label_from_path(provider_path: str) -> str:
    preset = _PROVIDER_PRESETS.get(provider_path)
    if preset is not None:
        return str(preset["label"])
    class_name = provider_path.rsplit(".", 1)[-1].strip() or "Provider"
    return class_name.removesuffix("Provider") or class_name


def _provider_name_seed(provider_path: str) -> str:
    preset = _PROVIDER_PRESETS.get(provider_path)
    if preset is not None:
        return str(preset["name_seed"])
    label = _provider_label_from_path(provider_path)
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return slug or "provider"


def _provider_protocol(provider_path: str) -> str:
    import mutagent.builtins.anthropic_provider  # noqa: F401
    import mutagent.builtins.openai_provider  # noqa: F401
    from mutagent.builtins.anthropic_provider import AnthropicProvider
    from mutagent.builtins.openai_provider import OpenAIProvider

    try:
        provider_cls = mutobj.resolve_class(provider_path, base_cls=LLMProvider)
    except Exception:
        return "generic"
    if issubclass(provider_cls, AnthropicProvider):
        return "anthropic"
    if issubclass(provider_cls, OpenAIProvider):
        return "openai"
    return "generic"


def _provider_tag_color(provider_path: str) -> str:
    preset = _PROVIDER_PRESETS.get(provider_path)
    if preset is not None:
        return str(preset["tag_color"])
    protocol = _provider_protocol(provider_path)
    if protocol == "anthropic":
        return "purple"
    if protocol == "openai":
        return "blue"
    return "default"


def _provider_base_url_default(provider_path: str) -> str:
    preset = _PROVIDER_PRESETS.get(provider_path)
    if preset is None:
        return ""
    return str(preset["base_url"])


def _provider_default_models(provider_path: str) -> list[str]:
    preset = _PROVIDER_PRESETS.get(provider_path)
    if preset is None:
        return []
    return list(preset["models"])


def _provider_add_button_id(provider_path: str) -> str:
    return f"add-{_provider_name_seed(provider_path)}"


def _available_provider_types() -> list[str]:
    import mutagent.builtins.anthropic_provider  # noqa: F401
    import mutagent.builtins.openai_provider  # noqa: F401

    discovered = {
        f"{cls.__module__}.{cls.__name__}"
        for cls in mutobj.discover_subclasses(LLMProvider)
    }
    discovered.update(_PROVIDER_PRESETS)
    preferred = [_ANTHROPIC_PROVIDER, _OPENAI_PROVIDER]
    ordered = [path for path in preferred if path in discovered]
    ordered.extend(sorted(
        (path for path in discovered if path not in preferred),
        key=lambda path: _provider_label_from_path(path).lower(),
    ))
    return ordered


def _make_provider_draft(name: str, provider_path: str) -> dict[str, Any]:
    models = _provider_default_models(provider_path)
    return {
        "name": name,
        "provider": provider_path,
        "base_url": _provider_base_url_default(provider_path),
        "auth_token": "",
        "models": list(models),
        "discovered_models": list(models),
    }


def _draft_from_config(key: str, config: dict[str, Any]) -> dict[str, Any]:
    provider_path = str(config.get("provider", _ANTHROPIC_PROVIDER)).strip() or _ANTHROPIC_PROVIDER
    models = _normalize_models(config.get("models", []))
    discovered = _provider_default_models(provider_path)
    for model in models:
        if model not in discovered:
            discovered.append(model)
    return {
        "name": key,
        "provider": provider_path,
        "base_url": str(config.get("base_url", _provider_base_url_default(provider_path))),
        "auth_token": str(config.get("auth_token", "")),
        "models": models,
        "discovered_models": discovered,
    }


def _all_model_names(self: LLMSettingsPanel) -> list[str]:
    names: list[str] = []
    for draft in self._drafts.values():
        for model in _normalize_models(draft.get("models", [])):
            if model not in names:
                names.append(model)
    return names


def _sync_default_model(self: LLMSettingsPanel) -> None:
    names = _all_model_names(self)
    if self.default_model and self.default_model in names:
        return
    self.default_model = names[0] if names else ""


def _apply_draft(self: LLMSettingsPanel, draft: dict[str, Any]) -> None:
    self.provider_name = str(draft["name"])
    self.provider_type = str(draft["provider"])
    self.provider_type_label = _provider_label_from_path(self.provider_type)
    self.base_url = str(draft["base_url"])
    self.auth_token = str(draft["auth_token"])
    self.models = list(draft["models"])
    self.discovered_models = list(draft["discovered_models"])


def _persist_current_draft(self: LLMSettingsPanel) -> dict[str, Any]:
    return {
        "name": self.provider_name.strip(),
        "provider": self.provider_type.strip() or _ANTHROPIC_PROVIDER,
        "base_url": self.base_url.strip(),
        "auth_token": self.auth_token.strip(),
        "models": _normalize_models(self.models),
        "discovered_models": _normalize_models(self.discovered_models),
    }


def _unique_provider_name(self: LLMSettingsPanel, base: str) -> str:
    candidate = base
    index = 2
    while candidate in self._drafts:
        candidate = f"{base}-{index}"
        index += 1
    return candidate


def _config_path(self: LLMSettingsPanel) -> Path:
    path = getattr(self._app, "config_path", None)
    if isinstance(path, Path):
        return path
    return (Path.cwd() / ".mutagent" / "config.json").resolve()


def _model_family(name: str) -> str:
    for suffix in _VARIANT_SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def _major_prefix(family: str) -> str:
    prefix = []
    for char in family:
        if char.isalpha():
            prefix.append(char)
            continue
        break
    return "".join(prefix) or family


def _prioritize_models(models_with_ts: list[tuple[str, int]]) -> list[str]:
    families: dict[str, list[tuple[str, int]]] = {}
    for model_id, created in models_with_ts:
        family = _model_family(model_id)
        families.setdefault(family, []).append((model_id, created))
    family_recency = {
        family: max(created for _, created in entries)
        for family, entries in families.items()
    }
    featured: set[str] = set()
    prefixes: dict[str, list[str]] = {}
    for family in families:
        prefixes.setdefault(_major_prefix(family), []).append(family)
    for entries in prefixes.values():
        for family in sorted(
            entries,
            key=lambda name: family_recency[name],
            reverse=True,
        )[:_FEATURED_FAMILIES_PER_PREFIX]:
            featured.add(family)
    result: list[str] = []
    rest: list[str] = []
    for family in sorted(families, key=lambda name: family_recency[name], reverse=True):
        values = [model_id for model_id, _ in sorted(families[family], key=lambda item: item[0])]
        if family in featured:
            result.extend(values)
        else:
            rest.extend(values)
    return result + rest


async def _discover_remote_models(
    base_url: str,
    auth_token: str,
    *,
    chat_filter: bool,
) -> list[str]:
    headers = {"Authorization": f"Bearer {auth_token}"}
    normalized_base = base_url.rstrip("/")
    urls = [f"{normalized_base}/models"]
    if not normalized_base.endswith("/v1"):
        urls.append(f"{normalized_base}/v1/models")
    data: dict[str, Any] | None = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=8.0)) as client:
        for url in urls:
            try:
                response = await client.get(url, headers=headers)
            except Exception:
                continue
            if response.status_code == 200:
                data = response.json()
                break
    if data is None:
        return []
    ranked: list[tuple[str, int]] = []
    for item in data.get("data", []):
        model_id = str(item.get("id", "")).strip()
        if chat_filter and not model_id.startswith(_CHAT_MODEL_PREFIXES):
            continue
        if not model_id:
            continue
        ranked.append((model_id, int(item.get("created", 0) or 0)))
    return _prioritize_models(ranked)


def _write_config(self: LLMSettingsPanel, providers: dict[str, dict[str, Any]], default_model: str) -> None:
    config = self._agent.config
    data = getattr(config, "_data", None)
    if not isinstance(data, dict):
        raise RuntimeError("Current Config implementation cannot be saved from WebUI")
    config.set("providers", providers, source="webui")
    config.set("default_model", default_model, source="webui")
    path = _config_path(self)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _set_message(self: LLMSettingsPanel, *, error: str = "", notice: str = "") -> None:
    self.error = error
    self.notice = notice


def _load_from_config(self: LLMSettingsPanel) -> None:
    providers = self._agent.config.get("providers", default={}) or {}
    self._drafts = {
        key: _draft_from_config(key, deepcopy(config))
        for key, config in providers.items()
        if isinstance(config, dict)
    }
    self.default_model = str(self._agent.config.get("default_model", default="") or "")
    self.current_step = "list"
    self.editing_key = ""
    self.editing_is_new = False
    if self._drafts:
        first_key = next(iter(self._drafts))
        _apply_draft(self, self._drafts[first_key])
        self.editing_key = first_key
    else:
        draft = _make_provider_draft("anthropic", _ANTHROPIC_PROVIDER)
        _apply_draft(self, draft)
    _sync_default_model(self)
    _set_message(self)


def _provider_summary(draft: dict[str, Any]) -> str:
    models = _normalize_models(draft.get("models", []))
    if not models:
        return "No models selected"
    return ", ".join(models)


def _base_url_hint(provider_path: str) -> str:
    protocol = _provider_protocol(provider_path)
    if protocol == "anthropic":
        return (
            "Anthropic 官方地址会直接使用内置模型列表；"
            "Anthropic-compatible 端点会依次尝试 /models 与 /v1/models。"
        )
    if protocol == "openai":
        return "OpenAI-compatible 端点会依次尝试 /models 与 /v1/models。"
    return "会依次尝试 /models 与 /v1/models；如你的 Provider 需要其他发现方式，请手动填写模型列表。"


# ═══════════════════════════════════════════════════════════════
#  User action handlers
# ═══════════════════════════════════════════════════════════════


def _edit_provider(key: str, *, view: LLMSettingsPanel) -> None:
    draft = view._drafts.get(key)
    if draft is None:
        return
    view.current_step = "edit"
    view.editing_key = key
    view.editing_is_new = False
    _apply_draft(view, draft)
    _set_message(view)
    view.invalidate()


def _start_add_provider(provider_path: str, *, view: LLMSettingsPanel) -> None:
    if not provider_path.strip():
        return
    name = _unique_provider_name(view, _provider_name_seed(provider_path))
    draft = _make_provider_draft(name, provider_path)
    view.current_step = "edit"
    view.editing_key = name
    view.editing_is_new = True
    _apply_draft(view, draft)
    _set_message(view)
    view.invalidate()


def _back_to_list(*, view: LLMSettingsPanel) -> None:
    view.current_step = "list"
    if not view.editing_is_new and view.editing_key in view._drafts:
        _apply_draft(view, view._drafts[view.editing_key])
    _sync_default_model(view)
    _set_message(view)
    view.invalidate()


def _save_provider_edits(*, view: LLMSettingsPanel) -> None:
    draft = _persist_current_draft(view)
    provider_name = draft["name"]
    if not provider_name:
        _set_message(view, error="Provider name cannot be empty.")
        view.invalidate()
        return
    if provider_name in view._drafts and (view.editing_is_new or provider_name != view.editing_key):
        _set_message(view, error=f"Provider '{provider_name}' already exists.")
        view.invalidate()
        return
    if not draft["auth_token"]:
        _set_message(view, error="API token cannot be empty.")
        view.invalidate()
        return
    if not draft["base_url"]:
        _set_message(view, error="Base URL cannot be empty.")
        view.invalidate()
        return
    if not draft["models"]:
        _set_message(view, error="Please select at least one model.")
        view.invalidate()
        return
    if not view.editing_is_new and view.editing_key != provider_name:
        view._drafts.pop(view.editing_key, None)
    view._drafts[provider_name] = draft
    view.editing_key = provider_name
    view.editing_is_new = False
    view.current_step = "list"
    _sync_default_model(view)
    _set_message(view, notice=f"Saved provider '{provider_name}'.")
    view.invalidate()


def _delete_provider(*, view: LLMSettingsPanel) -> None:
    if not view.editing_is_new and view.editing_key in view._drafts:
        view._drafts.pop(view.editing_key, None)
        _set_message(view, notice=f"Removed provider '{view.editing_key}'.")
    else:
        _set_message(view)
    view.current_step = "list"
    view.editing_is_new = False
    view.editing_key = next(iter(view._drafts), "")
    if view.editing_key:
        _apply_draft(view, view._drafts[view.editing_key])
    _sync_default_model(view)
    view.invalidate()


async def _discover_models(*, view: LLMSettingsPanel) -> None:
    _set_message(view)
    base_url = view.base_url.strip()
    auth_token = view.auth_token.strip()
    if not auth_token:
        _set_message(view, error="Please enter API token first.")
        view.invalidate()
        return
    try:
        protocol = _provider_protocol(view.provider_type)
        defaults = _provider_default_models(view.provider_type)
        if protocol == "anthropic" and base_url == _provider_base_url_default(view.provider_type) and defaults:
            discovered = list(defaults)
        else:
            discovered = await _discover_remote_models(
                base_url,
                auth_token,
                chat_filter=(protocol == "openai"),
            )
            if not discovered:
                discovered = list(defaults)
        view.discovered_models = discovered
        if not view.models:
            view.models = discovered[: min(len(discovered), 3)]
        _set_message(view, notice=f"Discovered {len(discovered)} models.")
    except Exception as exc:
        _set_message(view, error=str(exc))
    view.invalidate()


async def _save_all_settings(*, view: LLMSettingsPanel) -> None:
    providers: dict[str, dict[str, Any]] = {}
    for key, draft in view._drafts.items():
        provider_name = str(key).strip()
        models = _normalize_models(draft.get("models", []))
        base_url = str(draft.get("base_url", "")).strip()
        auth_token = str(draft.get("auth_token", "")).strip()
        provider_path = str(draft.get("provider", _ANTHROPIC_PROVIDER)).strip() or _ANTHROPIC_PROVIDER
        if not provider_name:
            _set_message(view, error="Provider name cannot be empty.")
            view.invalidate()
            return
        if not auth_token:
            _set_message(view, error=f"Provider '{provider_name}' is missing API token.")
            view.invalidate()
            return
        if not base_url:
            _set_message(view, error=f"Provider '{provider_name}' is missing base URL.")
            view.invalidate()
            return
        if not models:
            _set_message(view, error=f"Provider '{provider_name}' must include at least one model.")
            view.invalidate()
            return
        providers[provider_name] = {
            "provider": provider_path,
            "auth_token": auth_token,
            "base_url": base_url,
            "models": models,
        }
    if not providers:
        _set_message(view, error="Please add at least one provider.")
        view.invalidate()
        return
    all_models = [model for config in providers.values() for model in config["models"]]
    default_model = view.default_model if view.default_model in all_models else all_models[0]
    try:
        _write_config(view, providers, default_model)
        await view.page.notify_models_changed(default_model)
        await view.page.close()
    except Exception as exc:
        _set_message(view, error=str(exc))
        view.invalidate()


# ═══════════════════════════════════════════════════════════════
#  Rendering
# ═══════════════════════════════════════════════════════════════


def _render_message(self: LLMSettingsPanel, *, margin_bottom: int = 12) -> list[dict[str, Any]]:
    if self.error:
        return [{
            "$component": "antd.Alert",
            "$id": "settings-error",
            "type": "error",
            "showIcon": True,
            "message": self.error,
            "style": {"marginBottom": margin_bottom},
        }]
    if self.notice:
        return [{
            "$component": "antd.Alert",
            "$id": "settings-notice",
            "type": "success",
            "showIcon": True,
            "message": self.notice,
            "style": {"marginBottom": margin_bottom},
        }]
    return []


def _render_list(self: LLMSettingsPanel) -> list[dict[str, Any]]:
    items = _render_message(self)
    provider_buttons: list[dict[str, Any]] = []
    for key, draft in self._drafts.items():
        provider_path = str(draft.get("provider", _ANTHROPIC_PROVIDER))
        provider_buttons.append({
            "$component": "antd.Button",
            "$id": f"provider-{key}",
            "block": True,
            "style": {
                "textAlign": "left",
                "height": "auto",
                "padding": "10px 12px",
                "justifyContent": "flex-start",
                "alignItems": "stretch",
            },
            "onClick": Callback(_edit_provider, key, view=self),
            "$children": [{
                "$component": "div",
                "$id": f"provider-text-{key}",
                "style": {
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "8px",
                    "width": "100%",
                },
                "$children": [
                    {
                        "$component": "div",
                        "$id": f"provider-header-{key}",
                        "style": {
                            "display": "flex",
                            "alignItems": "flex-start",
                            "justifyContent": "space-between",
                            "gap": "12px",
                        },
                        "$children": [
                            {
                                "$component": "div",
                                "$id": f"provider-title-{key}",
                                "style": {"fontWeight": 600},
                                "children": key,
                            },
                            {
                                "$component": "antd.Tag",
                                "$id": f"provider-type-{key}",
                                "color": _provider_tag_color(provider_path),
                                "children": _provider_label_from_path(provider_path),
                            },
                        ],
                    },
                    {
                        "$component": "div",
                        "$id": f"provider-summary-{key}",
                        "style": {
                            "fontSize": "12px",
                            "color": "var(--mutgui-text-dim)",
                            "whiteSpace": "normal",
                            "wordBreak": "break-word",
                        },
                        "children": _provider_summary(draft),
                    },
                ],
            }],
        })
    add_buttons = []
    for provider_path in _available_provider_types():
        add_buttons.append({
            "$component": "antd.Button",
            "$id": _provider_add_button_id(provider_path),
            "children": f"Add {_provider_label_from_path(provider_path)}",
            "onClick": Callback(_start_add_provider, provider_path, view=self),
        })
    items.extend([
        {
            "$component": "antd.Typography.Paragraph",
            "$id": "settings-intro",
            "type": "secondary",
            "children": "参考 mutbot Setup-llm：先看 provider 列表，再进入单个 provider 编辑。",
        },
        {
            "$component": "antd.Space",
            "$id": "provider-add-actions",
            "style": {"marginBottom": 16},
            "$children": add_buttons,
        },
        {
            "$component": "div",
            "$id": "provider-list",
            "style": {"display": "flex", "flexDirection": "column", "gap": "8px", "marginBottom": 16},
            "$children": provider_buttons or [{
                "$component": "antd.Empty",
                "$id": "providers-empty",
                "description": "No providers configured yet",
            }],
        },
        {
            "$component": "antd.Form",
            "$id": "settings-list-form",
            "layout": "vertical",
            "$children": [{
                "$component": "antd.Form.Item",
                "$id": "default-model-item",
                "label": "Default Model",
                "$children": [{
                    "$component": "antd.Select",
                    "$id": "default-model",
                    "value": self.default_model or None,
                    "options": [
                        {"label": model, "value": model}
                        for model in _all_model_names(self)
                    ],
                    "placeholder": "Add provider first",
                    "disabled": not bool(self._drafts),
                    "onChange": Bind(self, "default_model", "$0"),
                }],
            }],
        },
        {
            "$component": "div",
            "$id": "config-path",
            "style": {
                "marginTop": 4,
                "marginBottom": 16,
                "fontSize": "12px",
                "color": "var(--mutgui-text-dim)",
            },
            "children": f"Config file: {_config_path(self)}",
        },
        {
            "$component": "antd.Button",
            "$id": "save-all",
            "type": "primary",
            "children": "Save Settings",
            "onClick": Callback(_save_all_settings, view=self),
        },
    ])
    return items


def _render_edit(self: LLMSettingsPanel) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    items.extend([
        {
            "$component": "antd.Space",
            "$id": "edit-header",
            "style": {"marginBottom": 16},
            "$children": [
                {
                    "$component": "antd.Button",
                    "$id": "back-to-list",
                    "children": "← Back",
                    "onClick": Callback(_back_to_list, view=self),
                },
                {
                    "$component": "antd.Tag",
                    "$id": "provider-kind-tag",
                    "color": _provider_tag_color(self.provider_type),
                    "children": self.provider_type_label,
                },
            ],
        },
        {
            "$component": "antd.Form",
            "$id": "settings-edit-form",
            "layout": "vertical",
            "$children": [
                {
                    "$component": "antd.Form.Item",
                    "$id": "key-item",
                    "label": "Provider Name",
                    "$children": [{
                        "$component": "antd.Input",
                        "$id": "provider-name",
                        "value": self.provider_name,
                        "onChange": Bind(self, "provider_name", "$0.target.value"),
                    }],
                },
                {
                    "$component": "antd.Form.Item",
                    "$id": "provider-type-item",
                    "label": "Provider Type",
                    "$children": [{
                        "$component": "antd.Input",
                        "$id": "provider-type",
                        "value": self.provider_type,
                        "disabled": True,
                    }],
                },
                {
                    "$component": "antd.Form.Item",
                    "$id": "base-url-item",
                    "label": "Base URL",
                    "extra": _base_url_hint(self.provider_type),
                    "$children": [{
                        "$component": "antd.Input",
                        "$id": "base-url",
                        "value": self.base_url,
                        "onChange": Bind(self, "base_url", "$0.target.value"),
                    }],
                },
                {
                    "$component": "antd.Form.Item",
                    "$id": "token-item",
                    "label": "API Token",
                    "$children": [{
                        "$component": "antd.Input.Password",
                        "$id": "auth-token",
                        "value": self.auth_token,
                        "onChange": Bind(self, "auth_token", "$0.target.value"),
                    }],
                },
                {
                    "$component": "antd.Form.Item",
                    "$id": "models-item",
                    "$children": [
                        {
                            "$component": "div",
                            "$id": "models-header",
                            "style": {
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "space-between",
                                "gap": "12px",
                                "marginBottom": 8,
                            },
                            "$children": [
                                {
                                    "$component": "div",
                                    "$id": "models-label-wrap",
                                    "$children": [
                                        {
                                            "$component": "div",
                                            "$id": "models-label",
                                            "style": {"fontWeight": 500},
                                            "children": "Models",
                                        },
                                        {
                                            "$component": "div",
                                            "$id": "discover-hint",
                                            "style": {"fontSize": "12px", "color": "var(--mutgui-text-dim)"},
                                            "children": "模型发现会按 Provider Type 对应的协议规则处理",
                                        },
                                    ],
                                },
                                {
                                    "$component": "antd.Button",
                                    "$id": "discover",
                                    "children": "Discover Models",
                                    "onClick": Callback(_discover_models, view=self),
                                },
                            ],
                        },
                        {
                            "$component": "antd.Select",
                            "$id": "models",
                            "mode": "tags",
                            "value": self.models,
                            "options": [
                                {"label": model, "value": model}
                                for model in self.discovered_models
                            ],
                            "tokenSeparators": [","],
                            "onChange": Bind(self, "models", "$0"),
                        },
                    ],
                },
            ],
        },
        *_render_message(self, margin_bottom=10),
        {
            "$component": "antd.Space",
            "$id": "edit-actions",
            "$children": [
                {
                    "$component": "antd.Button",
                    "$id": "delete-provider",
                    "danger": True,
                    "disabled": self.editing_is_new,
                    "children": "Remove Provider",
                    "onClick": Callback(_delete_provider, view=self),
                },
                {
                    "$component": "antd.Button",
                    "$id": "save-provider",
                    "type": "primary",
                    "children": "Save Provider",
                    "onClick": Callback(_save_provider_edits, view=self),
                },
            ],
        },
    ])
    return items
