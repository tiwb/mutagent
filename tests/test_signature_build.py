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
    build_signature,
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
