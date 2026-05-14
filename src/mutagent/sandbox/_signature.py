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
import json
import logging
from collections.abc import Iterable, Mapping
from typing import Any

logger = logging.getLogger(__name__)


class _MissingSentinel:
    """MCP optional-no-default 参数占位符。

    repr 使用 ``<omit>`` 而非 ``...``，避免与 Python ``Ellipsis`` /
    Pydantic ``Field(...)``（含义为「必填」）的反向语义碰撞。尖括号占位符
    在 CLI / template / OpenAPI 圈约定俗成，自解释、零歧义。
    """

    def __repr__(self) -> str:
        return "<omit>"

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


# ---------------------------------------------------------------------------
# MCP schema → docstring 渲染（feature-mcp-schema-help-display.iter2.md）
# ---------------------------------------------------------------------------

# 渲染折行宽度阈值。signature 多行折行 / Annotations JSON 折行共用。
# 80 列 = PEP 8 / Black 默认；agent 训练数据中高频出现的「Python 标准行宽」。
_RENDER_LINE_WIDTH = 80

# 按「三处投影职责唯一」原则，已被 signature / Args 段表达的 schema
# 字段不重复出现在 Annotations 段：
#   - type        → signature annotation
#   - default     → signature default
#   - enum        → signature Literal[...]
#   - description → Args 段散文
# 其余字段（含未知扩展）原词下放 Annotations 段。
_SCHEMA_KEYS_TAKEN_BY_SIGNATURE = frozenset({
    "type",
    "default",
    "enum",
    "description",
})


def _format_json_literal(value: Any) -> str:
    """返回 JSON 风格字面量。"""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _literal_part(value: Any) -> str:
    """为 typing.Literal[...] 输出单个字面量项。

    - str → 双引号（与 spec 示例一致，不走 Python repr 的单引号）
    - True / False / None → Python 关键字拼写（typing.Literal 只认 Python 值）
    - 数字 → repr；其他 → fallback repr
    """
    if value is True:
        return "True"
    if value is False:
        return "False"
    if value is None:
        return "None"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return repr(value)


def _format_literal_annotation(enum_values: Iterable[Any]) -> str | None:
    """把 enum 列表渲染为 ``Literal[...]`` annotation 字符串。

    空 / 非可迭代 返回 ``None``，调用方回落到 ``json_type_to_annotation``。
    同类型 / 混合类型字面量一律走 typing.Literal，Python typing 原生支持。
    """
    try:
        items = list(enum_values)
    except TypeError:
        return None
    if not items:
        return None
    return f"Literal[{', '.join(_literal_part(v) for v in items)}]"


def _format_json_compact(
    obj: Any,
    *,
    max_width: int,
    current_indent: int = 0,
    indent_step: int = 2,
) -> str:
    """Black 风格紧凑 JSON 渲染：能单行就单行，超宽逐层展开。

    Args:
        obj: 要渲染的 JSON 兼容对象（``Mapping`` / ``list`` / 标量）。
            非 JSON 兼容值（dataclass、自定义对象）行为未定义，调用方负责保证。
        max_width: 整行最大列数（含外层缩进与 prefix）。
        current_indent: 该值如展开后，子节点的逻辑左缩进列数。
            紧凑判断公式：``current_indent + len(compact) <= max_width``。
            注：作为简化设计，此处把「行起始列」当作 indent 估算紧凑宽度，
            外层有 ``key: `` 前缀时实际行宽可能略超 ``max_width``
            ——help 文本场景下可接受。
        indent_step: 每层缩进步长，默认 2（Black 风格）。

    分隔符策略：
        - 紧凑模式 → ``separators=(",",":")``（无空格）
        - 展开模式 → 元素间 ``,\n``；对象 key/value 间 ``": "``（含空格）

    复杂度：每节点先 try compact 再决定展开，深嵌套有 O(深度 × 节点数)
    重复序列化成本。MCP schema 实际深度 ≤ 4，可接受。

    未来若有第二个消费方（如 logs dump、debug pretty-print）可上提到 mutio。
    """
    # 标量 → 始终紧凑
    if not isinstance(obj, (Mapping, list)):
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    # 容器 → 先尝试紧凑
    compact = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    if current_indent + len(compact) <= max_width:
        return compact

    # 超宽 → 展开
    inner_indent_n = current_indent + indent_step
    inner_indent = " " * inner_indent_n
    close_indent = " " * current_indent

    if isinstance(obj, list):
        items = [
            _format_json_compact(
                v, max_width=max_width,
                current_indent=inner_indent_n,
                indent_step=indent_step,
            )
            for v in obj
        ]
        body = ",\n".join(f"{inner_indent}{it}" for it in items)
        return f"[\n{body}\n{close_indent}]"

    # Mapping
    parts: list[str] = []
    for k, v in obj.items():
        key_str = json.dumps(k, ensure_ascii=False)
        val_str = _format_json_compact(
            v, max_width=max_width,
            current_indent=inner_indent_n,
            indent_step=indent_step,
        )
        parts.append(f"{inner_indent}{key_str}: {val_str}")
    body = ",\n".join(parts)
    return f"{{\n{body}\n{close_indent}}}"


def format_annotations_section(
    properties: Mapping[str, Any],
    *,
    line_width: int = _RENDER_LINE_WIDTH,
) -> str:
    """根据 MCP schema properties 渲染 ``Annotations:`` 段。

    - 剩余字段 = ``pinfo`` 去掉 ``_SCHEMA_KEYS_TAKEN_BY_SIGNATURE`` 的部分
    - 按 properties 原顺序迭代；property 无剩余字段 → 不输出该行
    - 所有 property 都没剩余 → 返回空串（上层整段省略）
    - 单行 ≤ ``line_width`` → 单行 JSON；超阈值 → ``indent=4`` 多行
    - 全程 ``ensure_ascii=False``，中文 / Unicode 直出原字符

    返回以 ``"Annotations:"`` 顶格头起始、后跟多个 ``    name: <json>``
    行的多行字符串；无字段时返空串。
    """
    if not isinstance(properties, Mapping):
        return ""
    entries: list[str] = []
    for pname, pinfo in properties.items():
        if not isinstance(pname, str) or not isinstance(pinfo, Mapping):
            continue
        # 保顺剔除已被 signature / Args 段表达的字段
        remaining: dict[str, Any] = {
            k: v for k, v in pinfo.items()
            if k not in _SCHEMA_KEYS_TAKEN_BY_SIGNATURE
        }
        if not remaining:
            continue
        prefix = f"    {pname}: "
        # current_indent=4：value 如展开后子节点逻辑缩进起算 4+indent_step=6。
        # prefix 实际占 len(prefix) 列，紧凑宽度判断略保守，help 文本场景可接受。
        formatted = _format_json_compact(
            remaining, max_width=line_width, current_indent=4)
        entries.append(f"{prefix}{formatted}")
    if not entries:
        return ""
    return "Annotations:\n" + "\n".join(entries)


def mcp_schema_to_specs(input_schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    """把 MCP tool 的 ``input_schema``（JSON Schema）转为 ``ParamSpec`` 列表。

    - ``properties`` 的顺序被保留（Python 3.7+ dict 保序）
    - ``required`` 成员 → ``required=True``；其余参数显式写 ``required=False``
    - ``default`` 原样透传（JSON 原生值，可安全回传）
    - optional-no-default 参数注入 ``default=_MISSING``，避免被错误构造成必填
    - ``enum`` → ``annotation`` 升级为 ``Literal[...]``（python first + IDE 补全）
    - ``type`` → ``annotation`` 字符串（仅在无 enum 时使用）
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
        # enum 优先升级 annotation 为 Literal[...]；堆退到 type 映射。
        # 字符串透传，与 _RawAnnotation 展示机制兼容。
        enum_values = info.get("enum")
        literal = (
            _format_literal_annotation(enum_values)
            if isinstance(enum_values, list) else None
        )
        if literal is not None:
            spec["annotation"] = literal
        else:
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

        has_default = "default" in spec or bool(spec.get("default_missing"))
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
        if spec.get("default_missing"):
            default = _MISSING
        else:
            default = spec["default"] if "default" in spec else inspect.Parameter.empty

        params.append(inspect.Parameter(
            name, kind, default=default, annotation=annotation))

    return inspect.Signature(params)


def format_signature(
    sig: inspect.Signature,
    *,
    max_width: int | None = None,
) -> str:
    """格式化签名字符串，仅去掉字符串 annotation 的引号。

    Args:
        sig: ``inspect.Signature`` 实例。
        max_width: 可选。``None``（默认）→ 永远单行（向后兼容
            ``__repr__`` 等老路径）。具体数值 → 单行能装下保持单行，超宽
            切换为 Black 风格多行（每参数一行 + trailing comma，分隔符
            ``*`` / ``/`` 独占一行）。
    """
    params = [
        p.replace(annotation=_RawAnnotation(p.annotation))
        if isinstance(p.annotation, str) else p
        for p in sig.parameters.values()
    ]
    ret = sig.return_annotation
    if isinstance(ret, str):
        ret = _RawAnnotation(ret)
    sig2 = sig.replace(parameters=params, return_annotation=ret)
    single_line = str(sig2)
    if max_width is None or len(single_line) <= max_width:
        return single_line
    # 空参数总是单行（无折行需求）
    if not params:
        return single_line
    return _format_signature_multiline(params, ret)


def _format_signature_multiline(
    params: list[inspect.Parameter],
    return_annotation: Any,
) -> str:
    """Black 风格多行签名渲染。

    - 每参数一行，末尾 trailing comma
    - ``*`` / ``/`` 分隔符独占一行（语法需要的 ``,`` 跟随，不额外补）
    - return annotation 跟在 ``)`` 同一行：``) -> RetType``
    - 闭合 ``)`` 回到第一列，与函数名对齐
    """
    indent = "    "
    out: list[str] = ["("]

    has_var_positional = any(
        p.kind is inspect.Parameter.VAR_POSITIONAL for p in params)
    # 最后一个 POSITIONAL_ONLY 的下标（之后插 ``/`` 分隔符）
    last_pos_only = -1
    for i, p in enumerate(params):
        if p.kind is inspect.Parameter.POSITIONAL_ONLY:
            last_pos_only = i

    star_inserted = False
    for i, p in enumerate(params):
        # 首个 KEYWORD_ONLY 之前插 ``*,``（仅在无 VAR_POSITIONAL 时）
        if (p.kind is inspect.Parameter.KEYWORD_ONLY
                and not has_var_positional
                and not star_inserted):
            out.append(f"{indent}*,")
            star_inserted = True
        out.append(f"{indent}{p},")
        if i == last_pos_only:
            out.append(f"{indent}/,")

    if return_annotation is inspect.Signature.empty:
        out.append(")")
    else:
        out.append(f") -> {inspect.formatannotation(return_annotation)}")

    return "\n".join(out)


def format_callable_signature(
    func: Any,
    *,
    max_width: int | None = None,
) -> str | None:
    """统一格式化 callable 的展示签名。

    所有入口路径（MCP 桥接 / pysandbox peer / 同进程 Namespace / CLI adapter）
    在正常情况下 wrapper 都挂了 ``__signature__``，收敛到 :func:`format_signature`
    主路径。wrapper 构造失败时签名自然降级为 ``(**kwargs)``（Python 原生形态），
    参数详细信息依靠 ``Annotations:`` 段补充。
    """
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return None
    return format_signature(sig, max_width=max_width)


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
