"""`_signature.build_signature` 单元测试。

覆盖设计方案中明确列出的分支：required-first 正常、required 夹在 optional
中自动降级、全 optional、空参、annotation 缺失、default 缺失、JSON Schema
映射、非法 spec 抛错等。
"""

from __future__ import annotations

import inspect

import pytest

from mutagent.sandbox._signature import (
    _MISSING,
    _format_literal_annotation,
    build_signature,
    format_annotations_section,
    format_callable_signature,
    format_signature,
    json_type_to_annotation,
    mcp_schema_to_specs,
    try_build_signature,
)


# ---------------------------------------------------------------------------
# json_type_to_annotation
# ---------------------------------------------------------------------------


class TestJsonTypeToAnnotation:

    def test_basic_types(self) -> None:
        assert json_type_to_annotation("integer") == "int"
        assert json_type_to_annotation("number") == "float"
        assert json_type_to_annotation("string") == "str"
        assert json_type_to_annotation("boolean") == "bool"
        assert json_type_to_annotation("array") == "list"
        assert json_type_to_annotation("object") == "dict"
        assert json_type_to_annotation("null") == "None"

    def test_unknown_type_falls_back_to_any(self) -> None:
        assert json_type_to_annotation("weird") == "Any"
        assert json_type_to_annotation(None) == "Any"
        assert json_type_to_annotation({"oneOf": []}) == "Any"

    def test_union_types(self) -> None:
        assert json_type_to_annotation(["string", "null"]) == "str | None"
        # 去重
        assert json_type_to_annotation(["string", "string"]) == "str"
        # 含未知
        assert json_type_to_annotation(["string", "weird"]) == "str | Any"


# ---------------------------------------------------------------------------
# mcp_schema_to_specs
# ---------------------------------------------------------------------------


class TestMcpSchemaToSpecs:

    def test_required_and_optional(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer", "default": 100},
            },
            "required": ["path"],
        }
        specs = mcp_schema_to_specs(schema)
        assert specs == [
            {"name": "path", "required": True, "annotation": "str"},
            {"name": "limit", "required": False, "default": 100, "annotation": "int"},
        ]

    def test_empty_schema(self) -> None:
        assert mcp_schema_to_specs({}) == []
        assert mcp_schema_to_specs({"type": "object"}) == []

    def test_missing_type(self) -> None:
        schema = {"properties": {"x": {}}, "required": []}
        specs = mcp_schema_to_specs(schema)
        assert specs == [{"name": "x", "required": False, "default": _MISSING}]

    def test_order_preserved(self) -> None:
        schema = {
            "properties": {
                "b": {"type": "string"},
                "a": {"type": "integer"},
                "c": {"type": "boolean"},
            },
            "required": ["b", "c"],
        }
        names = [s["name"] for s in mcp_schema_to_specs(schema)]
        assert names == ["b", "a", "c"]

    def test_default_none_preserved(self) -> None:
        # default: None 必须保留（不是"没 default"）
        schema = {
            "properties": {"x": {"type": "string", "default": None}},
            "required": [],
        }
        specs = mcp_schema_to_specs(schema)
        assert specs[0]["default"] is None
        assert "default" in specs[0]
        assert specs[0]["required"] is False

    def test_optional_without_default_gets_missing_sentinel(self) -> None:
        schema = {
            "properties": {"all": {"type": "boolean"}},
            "required": [],
        }
        specs = mcp_schema_to_specs(schema)
        assert specs == [
            {"name": "all", "required": False, "default": _MISSING, "annotation": "bool"},
        ]


class TestFormatLiteralAnnotation:
    """feature-mcp-schema-help-display.iter2.md 【enum → Literal 升级】。"""

    def test_string_enum(self) -> None:
        assert (
            _format_literal_annotation(["DEBUG", "INFO"])
            == 'Literal["DEBUG", "INFO"]'
        )

    def test_integer_enum(self) -> None:
        assert _format_literal_annotation([1, 2, 3]) == "Literal[1, 2, 3]"

    def test_mixed_type_enum(self) -> None:
        # 混合类型也走 Literal，Python typing 原生支持
        assert (
            _format_literal_annotation(["x", 1, None, True])
            == 'Literal["x", 1, None, True]'
        )

    def test_empty_returns_none(self) -> None:
        assert _format_literal_annotation([]) is None

    def test_non_iterable_returns_none(self) -> None:
        assert _format_literal_annotation(123) is None  # type: ignore[arg-type]


class TestFormatAnnotationsSection:
    """feature-mcp-schema-help-display.iter2.md 【Annotations 段】渲染。"""

    def test_no_remaining_fields_returns_empty(self) -> None:
        # 只有 type/default/enum/description → 全被 signature/Args 表达 → 返空
        properties = {
            "a": {"type": "string", "description": "x"},
            "b": {"type": "integer", "default": 0},
            "c": {"type": "string", "enum": ["x", "y"]},
        }
        assert format_annotations_section(properties) == ""

    def test_basic_constraints_pass_through_as_json(self) -> None:
        properties = {
            "count": {"type": "integer", "minimum": 0, "maximum": 100},
            "name": {"type": "string", "pattern": "^[a-z]+$"},
        }
        result = format_annotations_section(properties)
        assert result == (
            "Annotations:\n"
            '    count: {"minimum": 0, "maximum": 100}\n'
            '    name: {"pattern": "^[a-z]+$"}'
        )

    def test_chinese_value_keeps_original_chars(self) -> None:
        # ensure_ascii=False 验证：Annotations value 包含中文时不转义
        properties = {"name": {"type": "string", "pattern": "^中文$"}}
        result = format_annotations_section(properties)
        assert '"pattern": "^中文$"' in result
        assert "\\u" not in result

    def test_long_value_switches_to_indented_multiline(self) -> None:
        # 超过 100 列阈值 → 多行 JSON
        long_props = {f"key_{i}": {"type": "string"} for i in range(10)}
        properties = {
            "items_field": {
                "type": "object",
                "properties": long_props,
            },
        }
        result = format_annotations_section(properties)
        assert result.startswith("Annotations:\n")
        assert "    items_field: {\n" in result
        # 闭合 } 回到 4 空格缩进
        assert result.rstrip().endswith("    }")
        # 内部 key 8 空格（json indent=4 + 本函数补 4）
        assert '\n        "properties": ' in result

    def test_unknown_extension_field_passes_through(self) -> None:
        properties = {
            "x": {"type": "string", "x-vendor-flag": True, "format": "uuid"},
        }
        result = format_annotations_section(properties)
        assert '"x-vendor-flag": true' in result
        assert '"format": "uuid"' in result

    def test_property_with_no_remaining_skips_line(self) -> None:
        properties = {
            "plain": {"type": "string", "description": "d"},
            "x": {"type": "integer", "minimum": 1},
        }
        result = format_annotations_section(properties)
        assert "plain" not in result
        assert '    x: {"minimum": 1}' in result

    def test_preserves_property_order(self) -> None:
        properties = {
            "b": {"type": "string", "pattern": "b"},
            "a": {"type": "string", "pattern": "a"},
        }
        result = format_annotations_section(properties)
        b_idx = result.index("    b:")
        a_idx = result.index("    a:")
        assert b_idx < a_idx


# ---------------------------------------------------------------------------
# build_signature — 核心分组策略
# ---------------------------------------------------------------------------


class TestBuildSignature:

    def test_empty(self) -> None:
        sig = build_signature([])
        assert list(sig.parameters) == []

    def test_all_required_positional_or_keyword(self) -> None:
        sig = build_signature([
            {"name": "a", "required": True, "annotation": "str"},
            {"name": "b", "required": True, "annotation": "int"},
        ])
        params = list(sig.parameters.values())
        assert [p.name for p in params] == ["a", "b"]
        assert all(p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                   for p in params)
        assert params[0].annotation == "str"
        assert params[1].annotation == "int"
        assert all(p.default is inspect.Parameter.empty for p in params)

    def test_required_first_then_optional_stays_positional(self) -> None:
        # 规范顺序：必选在前、可选在后 → 全部 POSITIONAL_OR_KEYWORD
        sig = build_signature([
            {"name": "a", "required": True},
            {"name": "b", "default": 10},
            {"name": "c", "default": "x"},
        ])
        params = list(sig.parameters.values())
        assert [p.kind for p in params] == [
            inspect.Parameter.POSITIONAL_OR_KEYWORD] * 3
        assert params[1].default == 10
        assert params[2].default == "x"
        # 可以 bind 位置参数
        bound = sig.bind(1, 2)
        bound.apply_defaults()
        assert bound.arguments == {"a": 1, "b": 2, "c": "x"}

    def test_optional_then_required_downgrades_to_keyword_only(self) -> None:
        # 畸形顺序：required 夹在 optional 后 → 自动降级为 kw-only
        sig = build_signature([
            {"name": "a", "default": 1},      # optional first
            {"name": "b", "required": True},  # required 被挤到后面
        ])
        params = list(sig.parameters.values())
        assert params[0].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert params[1].kind == inspect.Parameter.KEYWORD_ONLY
        # 位置调用只能给 a；b 必须 kw
        bound = sig.bind(1, b=5)
        assert bound.arguments == {"a": 1, "b": 5}
        with pytest.raises(TypeError):
            sig.bind(1, 2)  # b 是 kw-only

    def test_all_optional(self) -> None:
        sig = build_signature([
            {"name": "a", "default": 1},
            {"name": "b", "default": 2},
        ])
        params = list(sig.parameters.values())
        assert all(p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                   for p in params)
        bound = sig.bind()
        bound.apply_defaults()
        assert bound.arguments == {"a": 1, "b": 2}

    def test_annotation_missing(self) -> None:
        sig = build_signature([{"name": "x", "required": True}])
        p = sig.parameters["x"]
        assert p.annotation is inspect.Parameter.empty

    def test_annotation_is_string(self) -> None:
        sig = build_signature([
            {"name": "x", "required": True, "annotation": "int | None"},
        ])
        assert sig.parameters["x"].annotation == "int | None"
        # 字符串 annotation 在 str(sig) 里能正常展示
        assert "int | None" in str(sig)

    def test_default_none_is_a_default(self) -> None:
        # default: None 必须当作有默认值处理（下一参数触发降级）
        sig = build_signature([
            {"name": "a", "default": None},
            {"name": "b", "required": True},
        ])
        assert sig.parameters["a"].default is None
        assert sig.parameters["b"].kind == inspect.Parameter.KEYWORD_ONLY

    def test_default_missing_flag_restores_missing_sentinel(self) -> None:
        sig = build_signature([
            {"name": "paths", "annotation": "list", "default_missing": True},
        ])
        assert sig.parameters["paths"].default is _MISSING
        assert str(sig) == "(paths: 'list' = ...)"

    def test_explicit_keyword_only_forces_all_following_kw(self) -> None:
        sig = build_signature([
            {"name": "a", "kind": "KEYWORD_ONLY", "required": True},
            {"name": "b", "required": True},
        ])
        params = list(sig.parameters.values())
        assert params[0].kind == inspect.Parameter.KEYWORD_ONLY
        assert params[1].kind == inspect.Parameter.KEYWORD_ONLY

    def test_explicit_positional_only(self) -> None:
        sig = build_signature([
            {"name": "a", "kind": "POSITIONAL_ONLY", "required": True},
            {"name": "b", "required": True},
        ])
        assert (sig.parameters["a"].kind
                == inspect.Parameter.POSITIONAL_ONLY)
        # 位置调用 a 不能用 kw
        with pytest.raises(TypeError):
            sig.bind(a=1, b=2)
        bound = sig.bind(1, 2)
        assert bound.arguments == {"a": 1, "b": 2}

    def test_bind_unknown_kwarg_raises(self) -> None:
        sig = build_signature([{"name": "a", "required": True}])
        with pytest.raises(TypeError):
            sig.bind(bogus=1)

    def test_duplicate_name_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            build_signature([
                {"name": "a", "required": True},
                {"name": "a", "default": 1},
            ])

    def test_missing_name_raises(self) -> None:
        with pytest.raises(ValueError, match="missing name"):
            build_signature([{"default": 1}])

    def test_unsupported_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported kind"):
            build_signature([
                {"name": "a", "kind": "VAR_POSITIONAL", "required": True},
            ])


class TestTryBuildSignature:

    def test_returns_signature_on_success(self) -> None:
        sig = try_build_signature([{"name": "a", "required": True}])
        assert sig is not None
        assert "a" in sig.parameters

    def test_returns_none_on_failure(self, caplog) -> None:
        sig = try_build_signature([{"default": 1}], context="test_fn")
        assert sig is None
        assert "build_signature failed" in caplog.text


class TestDisplayFormatting:

    def test_format_signature_unquotes_string_annotations(self) -> None:
        sig = inspect.Signature(
            [
                inspect.Parameter(
                    "level",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default="INFO",
                    annotation="str",
                ),
                inspect.Parameter(
                    "args",
                    inspect.Parameter.VAR_POSITIONAL,
                    annotation="list[str]",
                ),
                inspect.Parameter(
                    "verbose",
                    inspect.Parameter.KEYWORD_ONLY,
                    default=False,
                    annotation=bool,
                ),
                inspect.Parameter(
                    "kwargs",
                    inspect.Parameter.VAR_KEYWORD,
                    annotation="Any",
                ),
            ],
            return_annotation="int | None",
        )

        assert format_signature(sig) == (
            "(level: str = 'INFO', *args: list[str], "
            "verbose: bool = False, **kwargs: Any) -> int | None"
        )

    def test_format_signature_keeps_non_string_annotations(self) -> None:
        sig = inspect.Signature(
            [
                inspect.Parameter(
                    "value",
                    inspect.Parameter.POSITIONAL_ONLY,
                    annotation=int,
                ),
            ],
            return_annotation=inspect.Signature.empty,
        )

        assert format_signature(sig) == "(value: int, /)"

    def test_format_callable_signature_uses_pysandbox_fallback(self) -> None:
        def fn(**kwargs):
            return kwargs

        fn._pysandbox_signature_str = "(level: str = 'INFO')"  # type: ignore[attr-defined]
        assert format_callable_signature(fn) == "(level: str = 'INFO')"

    def test_format_callable_signature_uses_mcp_schema_fallback(self) -> None:
        def fn(**kwargs):
            return kwargs

        fn._mcp_input_schema = {  # type: ignore[attr-defined]
            "properties": {
                "level": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["level"],
        }

        assert format_callable_signature(fn) == "(level: string, limit: integer = ...)"


# ---------------------------------------------------------------------------
# End-to-end: MCP schema → signature → bind
# ---------------------------------------------------------------------------


class TestMcpSchemaIntegration:

    def test_typical_mcp_tool_schema(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "offset": {"type": "integer", "default": 0},
                "limit": {"type": "integer", "default": 100},
            },
            "required": ["path"],
        }
        specs = mcp_schema_to_specs(schema)
        sig = build_signature(specs)
        # 位置 + kw 混合调用
        bound = sig.bind("file.txt", limit=50)
        bound.apply_defaults()
        assert bound.arguments == {
            "path": "file.txt", "offset": 0, "limit": 50}

    def test_malformed_schema_optional_before_required_downgrades(self) -> None:
        schema = {
            "properties": {
                "a": {"type": "integer", "default": 0},
                "b": {"type": "string"},
            },
            "required": ["b"],
        }
        specs = mcp_schema_to_specs(schema)
        sig = build_signature(specs)
        assert sig.parameters["b"].kind == inspect.Parameter.KEYWORD_ONLY

    def test_optional_without_default_renders_omitted_default(self) -> None:
        schema = {
            "properties": {
                "level": {"type": "string", "default": "info"},
                "all": {"type": "boolean"},
                "filename": {"type": "string"},
            },
            "required": [],
        }
        specs = mcp_schema_to_specs(schema)
        sig = build_signature(specs)
        bound = sig.bind()
        bound.apply_defaults()
        assert bound.arguments["level"] == "info"
        assert bound.arguments["all"] is _MISSING
        assert bound.arguments["filename"] is _MISSING
        assert str(sig) == (
            "(level: 'str' = 'info', all: 'bool' = ..., "
            "filename: 'str' = ...)"
        )
