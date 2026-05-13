"""Wrapper 真签名构造工具。

为 MCP tool wrapper 和 pysandbox namespace wrapper 共享的「`__signature__`
伪装」机制提供底层构造函数。核心设计见
``docs/specifications/refactor-wrapper-faithful-signature.md``。

两条客户端路径的输入形态不同（MCP 是 JSON Schema，pysandbox 是结构化
``params``），但都收敛到统一的 ``ParamSpec`` 列表再调 :func:`build_signature`，
因此参数分组策略（保序 + 智能降级）、annotation 字符串化、默认值 sentinel
处理只有一份实现。

MCP schema 还有一类特殊参数：**协议层可省略，但 schema 没写 default**。这在
Python `inspect.Signature` 里不能直接表达，因为 `Parameter.empty` 的语义是
“必填”。这里用 `_MISSING` sentinel 占位，让 wrapper 既能通过 `sig.bind()`，
又能在 RPC 发送前删掉该 key，把“省略参数”的原始语义还给服务端。

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


class _MissingSentinel:
    """MCP optional-no-default 参数占位符。"""

    def __repr__(self) -> str:
        return "..."

    def __bool__(self) -> bool:
        return False


_MISSING = _MissingSentinel()


class _RawAnnotation:
    """仅用于展示层：让字符串 annotation 经 repr 输出时不带引号。"""

    __slots__ = ("_text",)

    def __init__(self, text: str) -> None:
        self._text = text

    def __repr__(self) -> str:
        return self._text


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
    - ``required`` 成员 → ``required=True``；其余参数显式写 ``required=False``
    - ``default`` 原样透传（JSON 原生值，可安全回传）
    - optional-no-default 参数注入 ``default=_MISSING``，避免被错误构造成必填
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
        else:
            spec["required"] = False
        if "default" in info:
            spec["default"] = info["default"]
        elif pname not in required:
            spec["default"] = _MISSING
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
    # 见过默认值：本身不触发降级，但在这之后出现的无 default 参数必须降为
    # KEYWORD_ONLY（Python 语法约束：POS_OR_KW 无 default 参数不能排在有
    # default 的后面）。
    saw_default = False
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
        kind_hint = spec.get("kind")
        kind: inspect._ParameterKind
        if kind_hint == "POSITIONAL_ONLY":
            kind = inspect.Parameter.POSITIONAL_ONLY
            if has_default:
                saw_default = True
        elif kind_hint == "KEYWORD_ONLY":
            kind = inspect.Parameter.KEYWORD_ONLY
            kw_only_from_now = True
        elif kind_hint in (None, "POSITIONAL_OR_KEYWORD"):
            if kw_only_from_now:
                kind = inspect.Parameter.KEYWORD_ONLY
            elif saw_default and not has_default:
                # 见过默认值之后又来无 default 参数 → 触发降级
                kind = inspect.Parameter.KEYWORD_ONLY
                kw_only_from_now = True
            else:
                kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
                if has_default:
                    saw_default = True
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


def format_signature(sig: inspect.Signature) -> str:
    """格式化签名字符串，仅去掉字符串 annotation 的引号。"""
    params = [
        p.replace(annotation=_RawAnnotation(p.annotation))
        if isinstance(p.annotation, str) else p
        for p in sig.parameters.values()
    ]
    ret = sig.return_annotation
    if isinstance(ret, str):
        ret = _RawAnnotation(ret)
    return str(sig.replace(parameters=params, return_annotation=ret))


def _is_kwargs_only_fallback(sig: inspect.Signature) -> bool:
    """识别 wrapper 构造失败后的 ``(**kwargs)`` 降级签名。"""
    params = list(sig.parameters.values())
    return len(params) == 1 and params[0].kind is inspect.Parameter.VAR_KEYWORD


def _format_schema_signature(input_schema: Mapping[str, Any]) -> str:
    """按现有 MCP settings fallback 规则合成签名字符串。"""
    props = input_schema.get("properties") or {}
    if not isinstance(props, Mapping):
        return "()"
    required = set(input_schema.get("required") or [])
    params: list[str] = []
    for pname, pinfo in props.items():
        ptype = pinfo.get("type", "Any") if isinstance(pinfo, Mapping) else "Any"
        if pname in required:
            params.append(f"{pname}: {ptype}")
        else:
            params.append(f"{pname}: {ptype} = ...")
    return f"({', '.join(params)})"


def format_callable_signature(func: Any) -> str | None:
    """统一格式化 callable 的展示签名，必要时走各类 fallback。"""
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        sig = None
    if sig is not None and not _is_kwargs_only_fallback(sig):
        return format_signature(sig)

    fallback = getattr(func, "_pysandbox_signature_str", None)
    if isinstance(fallback, str) and fallback:
        return fallback

    schema = getattr(func, "_mcp_input_schema", None)
    if isinstance(schema, Mapping):
        return _format_schema_signature(schema)

    if sig is not None:
        return format_signature(sig)
    return None


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
