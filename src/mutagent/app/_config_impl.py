"""Config 默认实现。

提供 ConfigSection.get / get_field / set / on_change / affacts/ section 和
Config.root / resolve_model / list_models / load / load_from_dict / save
的默认实现。
使用 ConfigExt 存储运行时状态（data + listeners）。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, TypeVar, cast
from typing_extensions import TypeForm

T = TypeVar('T')

import mutobj
from mutobj.core._fields import field
from mutio.codec.json import JsonObject, JsonValue
from mutagent.app.config import (
    ChangeCallback, CancelFn, Config, ConfigChangeEvent, ConfigSection,
)


# ── Extension ──────────────────────────────────────────────

class ConfigExt(mutobj.Extension[Config]):
    """Config 的运行时状态。"""
    data: JsonObject = field(default_factory=dict)
    listeners: list[tuple[str, ChangeCallback]] = field(default_factory=list)


def _cext(self: Config) -> ConfigExt:
    return ConfigExt.get_or_create(self)


# ── Helpers ────────────────────────────────────────────────

def _expand_env(value: JsonValue) -> JsonValue:
    """递归展开配置值中的环境变量引用。"""
    if isinstance(value, str):
        return re.sub(
            r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)',
            lambda m: os.environ.get(m.group(1) or m.group(2), m.group(0)),
            value,
        )
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _glob_match(pattern_parts: list[str], key_parts: list[str]) -> bool:
    """glob 风格匹配。* 匹配单段，** 匹配任意段。"""
    return _do_match(pattern_parts, 0, key_parts, 0)


def _do_match(pp: list[str], pi: int, kp: list[str], ki: int) -> bool:
    while pi < len(pp) and ki < len(kp):
        if pp[pi] == "**":
            for skip in range(ki, len(kp) + 1):
                if _do_match(pp, pi + 1, kp, skip):
                    return True
            return False
        if pp[pi] == "*" or pp[pi] == kp[ki]:
            pi += 1
            ki += 1
        else:
            return False
    while pi < len(pp) and pp[pi] == "**":
        pi += 1
    return pi == len(pp) and ki == len(kp)


def _resolve_paths_inplace(data: JsonObject, config_dir: Path) -> None:
    """将 data 中的相对 path 条目解析为绝对路径。"""
    raw_paths = data.get("path")
    if not isinstance(raw_paths, list):
        return
    resolved: list[str] = []
    for p in raw_paths:
        pp = Path(str(p))
        if not pp.is_absolute():
            pp = (config_dir / pp).resolve()
        resolved.append(str(pp))
    data["path"] = resolved  # type: ignore[reportArgumentType]


def _resolve_default_model(config: Config) -> str | None:
    """解析默认模型名。找不到时返回 None。"""
    default = config.root.get_field("default_model", str, default="")
    if default:
        return default
    providers = config.root.section("providers")
    for _prov_name, prov_sec in providers.sections():
        models = prov_sec.get("models", default=[])
        if isinstance(models, list) and models:
            return str(models[0])
        elif isinstance(models, dict) and models:
            return next(iter(models))
    return None


# ── Config @impl ───────────────────────────────────────────
# 基础操作

from mutio.codec.json import check_type


def _full_path(prefix: str, name: str) -> str:
    """拼接 section 前缀与相对路径。"""
    if not prefix:
        return name
    if not name:
        return prefix
    return f"{prefix}.{name}"


@mutobj.impl(ConfigSection.get)
def config_section_get(self: ConfigSection, name: str, *, default: T = None) -> JsonValue | T:
    """读取配置值。name 为相对路径，递归展开环境变量。"""
    ext = _cext(self.config)
    full_name = _full_path(self.prefix, name)
    node = ext.data
    for key in full_name.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return _expand_env(node)


@mutobj.impl(ConfigSection.get_field)
def config_section_get_field(self: ConfigSection, name: str, type: TypeForm[T], /, *, default: T = None) -> T:
    """类型化读取。值不匹配 type 时抛 TypeError，泛型只检测外容器类型。"""
    ext = _cext(self.config)
    full_name = _full_path(self.prefix, name)
    node = ext.data
    for key in full_name.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    value = _expand_env(node)
    if not check_type(value, type):
        raise TypeError(
            f"Config key '{name}': expected {type}, "
            f"got {value.__class__.__name__}"
        )
    return cast(T, value)


@mutobj.impl(ConfigSection.set)
def config_section_set(self: ConfigSection, name: str, value: JsonValue, *, source: str = "") -> None:
    """按点分路径写入 data，触发匹配的 on_change 回调。"""
    full_name = _full_path(self.prefix, name)
    ext = _cext(self.config)
    node: Any = ext.data
    keys = full_name.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value
    event = ConfigChangeEvent(key=full_name, source=source, config=self.config)
    for pattern, cb in ext.listeners:
        if self.affects(pattern, full_name):
            cb(event)


@mutobj.impl(ConfigSection.on_change)
def config_section_on_change(self: ConfigSection, pattern: str, callback: ChangeCallback) -> CancelFn:
    """注册监听。返回 Disposable 用于取消。"""
    ext = _cext(self.config)
    full_pattern = _full_path(self.prefix, pattern)
    entry = (full_pattern, callback)
    ext.listeners.append(entry)
    called = False
    def dispose() -> None:
        nonlocal called
        if called:
            return
        called = True
        ext.listeners.remove(entry)
    return dispose


@mutobj.impl(ConfigSection.affects)
def config_section_affects(self: ConfigSection, pattern: str, key: str) -> bool:
    pattern_parts = pattern.split(".")
    key_parts = key.split(".")

    # 规则 1: key 匹配 pattern
    if _glob_match(pattern_parts, key_parts):
        return True

    # 规则 2: key 是 pattern 的祖先
    if len(key_parts) < len(pattern_parts):
        prefix_match = True
        for i, kp in enumerate(key_parts):
            pp = pattern_parts[i]
            if pp == "**":
                break
            if pp != "*" and pp != kp:
                prefix_match = False
                break
        if prefix_match:
            return True

    return False


@mutobj.impl(Config.root.getter)
def config_root(self: Config) -> ConfigSection:
    """返回根 Section（前缀空）。"""
    return ConfigSection(config=self, prefix="")


@mutobj.impl(ConfigSection.section)
def config_section_section(self: ConfigSection, name: str) -> ConfigSection:
    """返回更深层的子路径视图。"""
    return ConfigSection(config=self.config, prefix=_full_path(self.prefix, name))


@mutobj.impl(ConfigSection.sections)
def config_section_sections(self: ConfigSection) -> Iterable[tuple[str, ConfigSection]]:
    """遍历直接子 section。仅当底层值为 dict 的键才返回。"""
    ext = _cext(self.config)
    node = ext.data
    if self.prefix:
        for key in self.prefix.split("."):
            if not isinstance(node, dict) or key not in node:
                return
            node = node[key]
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        if isinstance(value, dict):
            yield key, ConfigSection(config=self.config, prefix=_full_path(self.prefix, key))


# ── 模型解析 ───────────────────────────────────────────

@mutobj.impl(Config.resolve_model)
def config_resolve_model(self: Config, name: str | None = None) -> JsonObject | None:
    if name is None:
        name = _resolve_default_model(self)
        if name is None:
            return None
    providers = self.root.get_field("providers", JsonObject, default={})
    if not providers:
        return None
    for prov_name, prov_conf in providers.items():
        if not isinstance(prov_conf, dict):
            continue
        models = prov_conf.get("models", [])
        if isinstance(models, list):
            if name in models:
                result: JsonObject = {k: v for k, v in prov_conf.items() if k != "models"}
                result["model_id"] = name
                result["provider_name"] = prov_name
                return result
        elif isinstance(models, dict):
            if name in models:
                model_val = models[name]
                result: JsonObject = {k: v for k, v in prov_conf.items() if k != "models"}
                result["provider_name"] = prov_name
                if isinstance(model_val, str):
                    result["model_id"] = model_val
                elif isinstance(model_val, dict):
                    result["model_id"] = model_val.get("model_id", name)
                    result.update({
                        k: v for k, v in model_val.items() if k != "model_id"
                    })
                else:
                    result["model_id"] = name
                return result
    return None


@mutobj.impl(Config.list_models)
def config_list_models(self: Config) -> list[JsonObject]:
    providers = self.root.get_field("providers", JsonObject, default={})
    if not providers:
        return []
    result: list[JsonObject] = []
    for prov_name, prov_conf in providers.items():
        if not isinstance(prov_conf, dict):
            continue
        provider_cls_path = prov_conf.get("type", "Anthropic")
        models = prov_conf.get("models", [])
        if isinstance(models, list):
            for model_id in models:
                result.append({
                    "name": model_id,
                    "model_id": model_id,
                    "type": provider_cls_path,
                    "provider_name": prov_name,
                })
        elif isinstance(models, dict):
            for alias, model_id in models.items():
                result.append({
                    "name": alias,
                    "model_id": model_id,
                    "type": provider_cls_path,
                    "provider_name": prov_name,
                })
    return result


# ── 持久化 ───────────────────────────────────────────

@mutobj.impl(Config.load)
def config_load(self: Config, path: str) -> None:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    # 项目级配置不存在时 fallback 到用户级 ~/.mutagent/config.json
    if not p.exists():
        user_p = Path.home() / ".mutagent" / "config.json"
        if user_p.exists():
            p = user_p
    self.path = p
    ext = _cext(self)
    if p.exists():
        try:
            ext.data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            ext.data = {}
        _resolve_paths_inplace(ext.data, p.parent)
    else:
        ext.data = {}
    ext.listeners = []


@mutobj.impl(Config.save)
def config_save(self: Config) -> None:
    ext = _cext(self)
    path = self.path
    if path is None:
        raise RuntimeError("Cannot save config without a path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(ext.data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


@mutobj.impl(Config.load_from_dict)
def config_load_from_dict(self: Config, data: JsonObject) -> None:
    self.path = None
    ext = _cext(self)
    ext.data = dict(data)
    ext.listeners = []
