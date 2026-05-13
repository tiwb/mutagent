"""Wrapper 真签名构造工具。

为 MCP tool wrapper 和 pysandbox namespace wrapper 共享的「`__signature__`
伪装」机制提供底层构造函数。核心设计见
``docs/specifications/refactor-wrapper-faithful-signature.md``。

两条客户端路径的输入形态不同（MCP 是 JSON Schema，pysandbox 是结构化
``params``），但都收敛到统一的 ``ParamSpec`` 列表再调 :func:`build_signature`，
因此参数分组策略（保序 + 智能降级）、annotation 字符串化、默认值 sentinel
处理只有一份实现。

**ParamSpec 约定**（dict）：

- ``name``: 参数名（必填）
- ``kind``: ``"POSITIONAL_OR_KEYWORD"`` / ``"KEYWORD_ONLY"`` /
  ``"POSITIONAL_ONLY"``（可选；缺失时由降级算法推断）
- ``default``: 默认值（可选；**字段存在即表示有默认值，包括 ``None``**）
- ``annotation``: 类型标注字符串（可选；缺失 = 无 annotation）
- ``required``: 是否必填（可选；仅在 ``default`` 缺失时用于辅助判断，
  MCP schema 路径会显式传，pysandbox 路径隐含由 ``default`` 字段推断）
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable, Mapping
from typing import Any

logger = logging.getLogger(__name__)


# JSON Schema type → Python 展示字符串
_JSON_TYPE_MAP = {
    "integer": "int",
    "number": "float",
    "string": "str",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
    "null": "None",
}


def json_type_to_annotation(ptype: Any) -> str:
    """JSON Schema ``type`` → Python 类型展示字符串。

    - 基础类型走映射表
    - 多类型（``["string", "null"]`` 这种）合并为 ``"str | None"`` 之类
    - 不认识的 / 缺失 / 复合结构 → ``"Any"``
    """
    if isinstance(ptype, str):
        return _JSON_TYPE_MAP.get(ptype, "Any")
    if isinstance(ptype, list):
        parts = [_JSON_TYPE_MAP.get(t, "Any") if isinstance(t, str) else "Any"
                 for t in ptype]
        # 去重保序
        seen: list[str] = []
        for p in parts:
            if p not in seen:
                seen.append(p)
        return " | ".join(seen) if seen else "Any"
    return "Any"


def mcp_schema_to_specs(input_schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    """把 MCP tool 的 ``input_schema``（JSON Schema）转为 ``ParamSpec`` 列表。

    - ``properties`` 的顺序被保留（Python 3.7+ dict 保序）
    - ``required`` 成员 → ``required=True``
    - ``default`` 原样透传（JSON 原生值，可安全回传）
    - ``type`` → ``annotation`` 字符串
    """
    specs: list[dict[str, Any]] = []
    properties = input_schema.get("properties") or {}
    required = set(input_schema.get("required") or [])
    if not isinstance(properties, Mapping):
        return specs
    for pname, pinfo in properties.items():
        if not isinstance(pname, str):
            continue
        info = pinfo if isinstance(pinfo, Mapping) else {}
        spec: dict[str, Any] = {"name": pname}
        if pname in required:
            spec["required"] = True
        if "default" in info:
            spec["default"] = info["default"]
        ptype = info.get("type")
        if ptype is not None:
            spec["annotation"] = json_type_to_annotation(ptype)
        specs.append(spec)
    return specs


def build_signature(specs: Iterable[Mapping[str, Any]]) -> inspect.Signature:
    """从 ``ParamSpec`` 列表构造 :class:`inspect.Signature`。

    **参数分组策略：保序 + 智能降级**

    按输入顺序遍历。遇到第一个「非 required 或自带 default」的参数之后，
    把后续所有未显式标记 ``POSITIONAL_ONLY`` 的参数强制降级为
    ``KEYWORD_ONLY``。这样既保留原始顺序（调用者看到的 schema 顺序不被
    重排），又在 required/optional 夹杂的畸形 schema 下不至于构造失败。

    畸形 spec（重名、不合法 kind 等）抛 ``ValueError``；调用方可降级为
    ``(**kwargs)`` wrapper。
    """
    params: list[inspect.Parameter] = []
    seen_names: set[str] = set()
    # 见过可选参数：本身不触发降级，但在这之后出现的 required 必须降为
    # KEYWORD_ONLY（Python 语法约束：POS_OR_KW 无 default 参数不能排在有
    # default 的后面）。
    saw_optional = False
    # 一旦出现 KEYWORD_ONLY，后续不能再回到 POSITIONAL_OR_KEYWORD。
    kw_only_from_now = False

    for spec in specs:
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"invalid param spec: missing name ({spec!r})")
        if name in seen_names:
            raise ValueError(f"duplicate param name: {name}")
        seen_names.add(name)

        has_default = "default" in spec
        # required 字段只在 default 缺失时才参与判断；显式 required=False 也算可选
        required = spec.get("required")
        if required is None:
            required = not has_default
        is_optional = has_default or not required

        kind_hint = spec.get("kind")
        kind: inspect._ParameterKind
        if kind_hint == "POSITIONAL_ONLY":
            kind = inspect.Parameter.POSITIONAL_ONLY
            if is_optional:
                saw_optional = True
        elif kind_hint == "KEYWORD_ONLY":
            kind = inspect.Parameter.KEYWORD_ONLY
            kw_only_from_now = True
        elif kind_hint in (None, "POSITIONAL_OR_KEYWORD"):
            if kw_only_from_now:
                kind = inspect.Parameter.KEYWORD_ONLY
            elif saw_optional and not is_optional:
                # 见过 optional 之后又来 required → 触发降级
                kind = inspect.Parameter.KEYWORD_ONLY
                kw_only_from_now = True
            else:
                kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
                if is_optional:
                    saw_optional = True
        else:
            raise ValueError(
                f"unsupported kind in spec {name!r}: {kind_hint!r}")

        # annotation：字符串形式，缺失 → Parameter.empty
        annotation_raw = spec.get("annotation")
        if annotation_raw is None or annotation_raw == "":
            annotation: Any = inspect.Parameter.empty
        else:
            annotation = str(annotation_raw)

        # default：字段存在（即使值为 None）才算有默认值
        default: Any = spec["default"] if has_default else inspect.Parameter.empty

        params.append(inspect.Parameter(
            name, kind, default=default, annotation=annotation))

    return inspect.Signature(params)


def try_build_signature(
    specs: Iterable[Mapping[str, Any]],
    *,
    context: str = "",
) -> inspect.Signature | None:
    """安全包装 :func:`build_signature`，构造失败时返回 ``None`` 并记日志。

    调用方用返回 ``None`` 作为「降级为 ``(**kwargs)`` wrapper」的信号。
    """
    try:
        return build_signature(specs)
    except Exception as exc:
        logger.warning(
            "build_signature failed%s: %s; falling back to (**kwargs)",
            f" for {context}" if context else "", exc)
        return None
