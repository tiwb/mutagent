"""Config 默认实现。

提供 Config.get / set / on_change / affects / resolve_model / list_models /
load / save 的默认实现。
使用 ConfigExt 存储运行时状态（data + listeners）。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import mutobj
from mutagent.app.config import ChangeCallback, CancelFn, Config, ConfigChangeEvent


# ── Extension ──────────────────────────────────────────────

class ConfigExt(mutobj.Extension[Config]):
    """Config 的运行时状态。"""
    data: dict = {}
    listeners: list = []  # list[tuple[str, ChangeCallback]]


def _cext(self: Config) -> ConfigExt:
    return ConfigExt.get_or_create(self)


# ── Helpers ────────────────────────────────────────────────

def _expand_env(value: Any) -> Any:
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


def _resolve_paths_inplace(data: dict, config_dir: Path) -> None:
    """将 data 中的相对 path 条目解析为绝对路径。"""
    raw_paths = data.get("path")
    if not isinstance(raw_paths, list):
        return
    resolved: list[str] = []
    for p in raw_paths:
        pp = Path(p)
        if not pp.is_absolute():
            pp = (config_dir / pp).resolve()
        resolved.append(str(pp))
    data["path"] = resolved


def _resolve_default_model(config: Config) -> str | None:
    """解析默认模型名。找不到时返回 None。"""
    default = config.get("default_model", default="")
    if default:
        return default
    providers = config.get("providers", default={})
    if not providers:
        return None
    for _prov_name, prov_conf in providers.items():
        models = prov_conf.get("models", [])
        if isinstance(models, list) and models:
            return models[0]
        elif isinstance(models, dict) and models:
            return next(iter(models))
    return None


# ── Config @impl ───────────────────────────────────────────
# 基础操作

@mutobj.impl(Config.get)
def config_get(self: Config, name: str, *, default: Any = None) -> Any:
    """读取配置值。name 为点分路径，递归展开环境变量。"""
    ext = _cext(self)
    node = ext.data
    for key in name.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return _expand_env(node)


@mutobj.impl(Config.set)
def config_set(self: Config, name: str, value: Any, *, source: str = "") -> None:
    """按点分路径写入 data，触发匹配的 on_change 回调。"""
    ext = _cext(self)
    node = ext.data
    keys = name.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value
    event = ConfigChangeEvent(key=name, source=source, config=self)
    for pattern, cb in ext.listeners:
        if self.affects(pattern, name):
            cb(event)


@mutobj.impl(Config.on_change)
def config_on_change(self: Config, pattern: str, callback: ChangeCallback) -> CancelFn:
    """注册监听。返回 Disposable 用于取消。"""
    ext = _cext(self)
    entry = (pattern, callback)
    ext.listeners.append(entry)
    called = False
    def dispose() -> None:
        nonlocal called
        if called:
            return
        called = True
        ext.listeners.remove(entry)
    return dispose


@mutobj.impl(Config.affects)
def config_affects(self: Config, pattern: str, key: str) -> bool:
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


# ── 模型解析 ───────────────────────────────────────────

@mutobj.impl(Config.resolve_model)
def config_resolve_model(self: Config, name: str | None = None) -> dict | None:
    if name is None:
        name = _resolve_default_model(self)
        if name is None:
            return None
    providers = self.get("providers", default={})
    if not providers:
        return None
    for prov_name, prov_conf in providers.items():
        models = prov_conf.get("models", [])
        if isinstance(models, list):
            if name in models:
                result = {k: v for k, v in prov_conf.items() if k != "models"}
                result["model_id"] = name
                result["provider_name"] = prov_name
                return result
        elif isinstance(models, dict):
            if name in models:
                model_val = models[name]
                result = {k: v for k, v in prov_conf.items() if k != "models"}
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
def config_list_models(self: Config) -> list[dict]:
    providers = self.get("providers", default={})
    result: list[dict] = []
    for prov_name, prov_conf in providers.items():
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
def config_load(self: Config, config_path: str) -> None:
    p = Path(config_path).expanduser()
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
def config_load_from_dict(self: Config, data: dict) -> None:
    self.path = None
    ext = _cext(self)
    ext.data = dict(data)
    ext.listeners = []
