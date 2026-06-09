"""mutagent.app.config -- 可观察的配置容器。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import UnionType
from typing import Callable, Iterable, TypeVar
from typing_extensions import TypeForm

import mutobj
from mutio.codec.json import JsonObject, JsonValue

T = TypeVar('T')


@dataclass
class ConfigChangeEvent:
    """配置变更事件。"""
    key: str                    # 被设置的完整路径（如 "providers.anthropic"）
    source: str                 # 变更来源标识（如 "user", "workspace"）
    config: Config = field(repr=False)  # 触发变更的 Config 实例


ChangeCallback = Callable[[ConfigChangeEvent], None]
CancelFn = Callable[[], None]


class ConfigSection(mutobj.Declaration):
    """Config 子路径的类型安全视图。

    提供 get / get_field / set / section / on_change，所有路径操作自动加上前缀，
    避免调用方手动拼接和裸 JsonValue 窄化。ConfigSection 实例之间接口完全一致：
    root Section（Config.root）和子 Section（section(name)）可链式调用。
    """

    config: Config      # 父 Config，反向引用
    prefix: str          # 路径前缀（如 "providers.openai"）

    def get(self, name: str, *, default: T = None) -> JsonValue | T:
        """读取配置值。name 为点分路径（相对于本 section 前缀），递归展开环境变量。

        default 仅作为 key 不存在时的回退值。
        """
        ...

    def get_field(self, name: str, type: TypeForm[T] | UnionType, /, *, default: T = None) -> T:
        """类型化读取配置值。

        name 和 type 位置传递，default 必须关键词传递。
        运行时做递归类型检查，不匹配抛 TypeError。
        泛型递归检测元素类型（如 list[str] 会检查每个元素是否为 str）。

        ``type`` 参数签名为 ``type[T] | UnionType``：
        单类型（str/int/float/bool）精确推导 T；Union（str | float）不报错，
        但 T 从 default 推导。Union 字段推荐用 isinstance 逐分支窄化。

        未来 PEP 747 ``TypeForm[T]`` 落地后可替换 ``UnionType``，
        届时 union 的 T 也可从 type 精确推导。
        """
        ...

    def set(self, name: str, value: JsonValue, *, source: str = "") -> None:
        """设置配置值并触发变更通知。

        name: 点分路径（相对于本 section 前缀，如 "openai.api_key"）
        value: 新值（任意类型）
        source: 变更来源标识（如 "user", "workspace", "runtime"）

        设置一个节点会隐式影响所有子路径。例如：
        set("openai", new_dict) 会触发所有监听 openai 及其子路径的回调。
        """
        ...

    def section(self, name: str) -> ConfigSection:
        """获取更深层的子路径视图。链式调用。

            openai = config.root.section("providers").section("openai")
            openai.get_field("api_key", type=str)
        """
        ...

    def sections(self) -> Iterable[tuple[str, ConfigSection]]:
        """遍历直接子 section（底层值为 dict 的子项）。

        底层值为非 dict 类型的键不会出现在结果中。

            for name, sec in config.root.section("providers").sections():
                print(name, sec.get_field("type", type=str))
        """
        ...

    def on_change(self, pattern: str, callback: ChangeCallback) -> CancelFn:
        """监听本 section 内的配置变更。

        pattern 相对于本 section 前缀，支持 glob 风格通配符：
        - 精确路径："openai.auth_token"
        - 单级通配 *："*" — 匹配直接子项
        - 递归通配 **："**" — 匹配任意深度
        - 混合："*.models" — 任意子项的 models

        触发规则（pattern 与 set 的 key 双向匹配）：
        1. key 匹配 pattern → 触发（监听范围内的 key 被设置）
        2. key 是 pattern 的祖先 → 触发（父节点被替换，子路径隐式变更）
        3. 不相关 → 不触发
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


class Config(mutobj.Declaration):
    """可观察的配置容器。"""

    path: Path | None

    @property
    def root(self) -> ConfigSection:
        """根 Section，提供 get/get_field/set/section/on_change 统一入口。

            config.root.get_field("debug", type=bool)
            config.root.section("providers").get_field("openai.type", type=str)
        """
        ...

    def resolve_model(self, name: str | None = None) -> JsonObject | None:
        """从 providers 中查找并组装指定模型的 spec。

        name 为 None 时使用 default_model。找不到时返回 None。
        """
        ...

    def list_models(self) -> list[JsonObject]:
        """列出所有已配置的模型 spec。"""
        ...

    def load(self, path: str) -> None:
        """从 JSON 配置文件加载，设置 self.path。

        路径不存在时初始化为空配置。
        """
        ...

    def load_from_dict(self, data: JsonObject) -> None:
        """从 dict 填充配置（供测试和编程构建用）。

        不设置 self.path，不会写盘。listeners 重置为空。
        """
        ...

    def save(self) -> None:
        """将当前配置序列化写回 self.path。"""
        ...


from . import _config_impl as _config_impl  # noqa: F401, E402
