"""mutagent.app.config -- 可观察的配置容器。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mutobj


@dataclass
class ConfigChangeEvent:
    """配置变更事件。"""
    key: str                    # 被设置的完整路径（如 "providers.anthropic"）
    source: str                 # 变更来源标识（如 "user", "workspace"）
    config: Config = field(repr=False)  # 触发变更的 Config 实例


ChangeCallback = Callable[[ConfigChangeEvent], None]
CancelFn = Callable[[], None]


class Config(mutobj.Declaration):
    """可观察的配置容器。"""

    path: Path | None

    def get(self, name: str, *, default: Any = None) -> Any:
        """读取配置值。name 为点分路径。

        示例：
            config.get("providers.anthropic.auth_token")
            config.get("providers")  # 返回整个 providers dict
            config.get("agents.sub_agent.model", default="claude-sonnet")
        """
        ...

    def set(self, name: str, value: Any, *, source: str = "") -> None:
        """设置配置值并触发变更通知。

        name: 点分路径（如 "providers.anthropic"）
        value: 新值（任意类型）
        source: 变更来源标识（如 "user", "workspace", "runtime"）

        设置一个节点会隐式影响所有子路径。例如：
        set("providers.anthropic", new_dict) 会触发所有监听
        providers.anthropic 及其子路径的回调。
        """
        ...

    def on_change(self, pattern: str, callback: ChangeCallback) -> CancelFn:
        """监听配置变更。

        pattern 支持 glob 风格通配符：
        - 精确路径："providers.anthropic.auth_token"
        - 单级通配 *："providers.*" — 匹配 providers 的任意直接子项
        - 递归通配 **："providers.**" — 匹配 providers 下任意深度
        - 混合："providers.*.models" — 任意 provider 的 models

        触发规则（pattern 与 set 的 key 双向匹配）：
        1. key 匹配 pattern → 触发（监听范围内的 key 被设置）
           on_change("providers.*", cb) + set("providers.anthropic") → ✓
        2. key 是 pattern 的祖先 → 触发（父节点被替换，子路径隐式变更）
           on_change("providers.anthropic.auth_token", cb) + set("providers.anthropic") → ✓
           on_change("providers.**", cb) + set("providers") → ✓
        3. 不相关 → 不触发
           on_change("providers.*", cb) + set("agents.xxx") → ✗
           on_change("providers.*", cb) + set("providers.anthropic.auth_token") → ✗
           （* 只匹配一级，auth_token 是两级深）
        """
        ...

    def affects(self, pattern: str, key: str) -> bool:
        """判断 key 的变更是否影响 pattern 指定的路径。

        双向匹配：
        1. key 匹配 pattern → True（标准 glob）
        2. key 是 pattern 的祖先 → True（父节点被替换，子路径隐式变更）
        3. 不相关 → False

        子类可覆盖以定制匹配策略。
        """
        ...

    def resolve_model(self, name: str | None = None) -> dict | None:
        """从 providers 中查找并组装指定模型的 spec。

        name 为 None 时使用 default_model。找不到时返回 None。
        """
        ...

    def list_models(self) -> list[dict]:
        """列出所有已配置的模型 spec。"""
        ...

    def load(self, path: str) -> None:
        """从 JSON 配置文件加载，设置 self.path。

        路径不存在时初始化为空配置。
        """
        ...

    def load_from_dict(self, data: dict) -> None:
        """从 dict 填充配置（供测试和编程构建用）。

        不设置 self.path，不会写盘。listeners 重置为空。
        """
        ...

    def save(self) -> None:
        """将当前配置序列化写回 self.path。"""
        ...


from . import _config_impl as _config_impl  # noqa: F401, E402
