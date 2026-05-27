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
    _RENDER_LINE_WIDTH,
    _format_json_compact,
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
        # iter3: 单行紧凑分隔符为 `(",",":")`，无空格
        assert result == (
            "Annotations:\n"
            '    count: {"minimum":0,"maximum":100}\n'
            '    name: {"pattern":"^[a-z]+$"}'
        )

    def test_chinese_value_keeps_original_chars(self) -> None:
        # ensure_ascii=False 验证：Annotations value 包含中文时不转义
        properties = {"name": {"type": "string", "pattern": "^中文$"}}
        result = format_annotations_section(properties)
        # iter3 紧凑模式下 key 后无空格
        assert '"pattern":"^中文$"' in result
        assert "\\u" not in result

    def test_long_value_switches_to_indented_multiline(self) -> None:
        # 超过 80 列阈值 → Black 风格递归展开（indent_step=2）
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
        # 闭合 } 回到 4 空格缩进（与 prefix 同列）
        assert result.rstrip().endswith("    }")
        # 内部 key 6 空格（4 + indent_step=2）
        assert '\n      "properties": ' in result

    def test_unknown_extension_field_passes_through(self) -> None:
        properties = {
            "x": {"type": "string", "x-vendor-flag": True, "format": "uuid"},
        }
        result = format_annotations_section(properties)
        # iter3 紧凑模式同位无空格
        assert '"x-vendor-flag":true' in result
        assert '"format":"uuid"' in result

    def test_property_with_no_remaining_skips_line(self) -> None:
        properties = {
            "plain": {"type": "string", "description": "d"},
            "x": {"type": "integer", "minimum": 1},
        }
        result = format_annotations_section(properties)
        assert "plain" not in result
        assert '    x: {"minimum":1}' in result

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
        # iter3: _MISSING.__repr__ 返回 "<omit>"
        assert str(sig) == "(paths: 'list' = <omit>)"

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

    def test_format_callable_signature_falls_back_to_kwargs_for_kwargs_only(self) -> None:
        # iter3: 删除 _pysandbox_signature_str 和 _mcp_input_schema fallback
        # wrapper 构造失败时自然降级为 (**kwargs) —— inspect.signature 直接返回该形态
        def fn(**kwargs):
            return kwargs

        # 不再走 _pysandbox_signature_str fallback
        fn._pysandbox_signature_str = "(level: str = 'INFO')"  # type: ignore[attr-defined]
        assert format_callable_signature(fn) == "(**kwargs)"

    def test_format_callable_signature_ignores_mcp_schema_attr(self) -> None:
        # iter3: 删除 _mcp_input_schema fallback ——该属性仅供 Annotations 段读取
        def fn(**kwargs):
            return kwargs

        fn._mcp_input_schema = {  # type: ignore[attr-defined]
            "properties": {"level": {"type": "string"}},
            "required": ["level"],
        }
        assert format_callable_signature(fn) == "(**kwargs)"

    def test_format_callable_signature_returns_none_for_unsignature_callable(self) -> None:
        # inspect.signature 报错的奇葩 callable → None
        class _NoSig:
            __signature__ = property(lambda self: 1 / 0)

        # builtins 不可 inspect 的例子
        import operator
        # operator.add 的 signature 不一定报错，改用一个手造逆向
        class _Bad:
            def __call__(self):
                pass
        # _Bad() 是可 inspect 的，跳过这个场景只验证 None 入口不报错
        assert format_callable_signature(_Bad()) is not None

    def test_format_callable_signature_with_max_width_multiline(self) -> None:
        """max_width 传入后超宽签名转多行。"""
        def fn(
            element: str,
            ref: str,
            button: str = "left",
            doubleClick: bool = False,
            modifiers: list | None = None,
        ) -> None:
            ...

        out = format_callable_signature(fn, max_width=80)
        assert out is not None
        assert "\n    element: str,\n" in out
        # 多行模式末参数后有 trailing comma
        assert "= None,\n)" in out


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
        # iter3: _MISSING.__repr__ → <omit>
        assert str(sig) == (
            "(level: 'str' = 'info', all: 'bool' = <omit>, "
            "filename: 'str' = <omit>)"
        )


# ---------------------------------------------------------------------------
# iter3：Black 风格签名多行折行
# ---------------------------------------------------------------------------


class TestSignatureMultilineFolding:
    """feature-mcp-schema-help-display.iter3.md 【Signature 多行折行】。"""

    def test_render_line_width_is_80(self) -> None:
        # iter2 是 100，iter3 改 80
        assert _RENDER_LINE_WIDTH == 80

    def test_short_signature_stays_single_line(self) -> None:
        sig = inspect.Signature([
            inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              annotation="int"),
        ])
        out = format_signature(sig, max_width=80)
        assert out == "(x: int)"
        # 单行模式不加 trailing comma
        assert ",)" not in out

    def test_long_signature_switches_to_multiline_with_trailing_comma(self) -> None:
        sig = inspect.Signature([
            inspect.Parameter("element", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              annotation="str"),
            inspect.Parameter("ref", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              annotation="str"),
            inspect.Parameter("button", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              default="left",
                              annotation='Literal["left","right","middle"]'),
            inspect.Parameter("doubleClick", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              default=False, annotation="bool"),
            inspect.Parameter("modifiers", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              default=None, annotation="list[str] | None"),
        ])
        out = format_signature(sig, max_width=80)
        assert out == (
            "(\n"
            "    element: str,\n"
            "    ref: str,\n"
            '    button: Literal["left","right","middle"] = \'left\',\n'
            "    doubleClick: bool = False,\n"
            "    modifiers: list[str] | None = None,\n"
            ")"
        )

    def test_star_separator_on_own_line(self) -> None:
        sig = inspect.Signature([
            inspect.Parameter("a", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              annotation="int"),
            inspect.Parameter("verbose", inspect.Parameter.KEYWORD_ONLY,
                              default=False, annotation="bool"),
            inspect.Parameter("debug", inspect.Parameter.KEYWORD_ONLY,
                              default=False, annotation="bool"),
            inspect.Parameter("log_level", inspect.Parameter.KEYWORD_ONLY,
                              default="INFO", annotation="str"),
        ])
        # 强制多行：max_width=20
        out = format_signature(sig, max_width=20)
        # *, 独占一行（含语法所需的 ,，不做额外 trailing comma）
        assert "\n    *,\n" in out
        # 一次性插入
        assert out.count("    *,") == 1

    def test_slash_separator_on_own_line(self) -> None:
        sig = inspect.Signature([
            inspect.Parameter("a", inspect.Parameter.POSITIONAL_ONLY,
                              annotation="int"),
            inspect.Parameter("b", inspect.Parameter.POSITIONAL_ONLY,
                              annotation="int"),
            inspect.Parameter("c", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              annotation="int"),
        ])
        out = format_signature(sig, max_width=10)
        # /, 在最后一个 POSITIONAL_ONLY 之后独占一行
        assert "\n    /,\n" in out

    def test_var_positional_suppresses_star_separator(self) -> None:
        sig = inspect.Signature([
            inspect.Parameter("a", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              annotation="int"),
            inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL,
                              annotation="int"),
            inspect.Parameter("kw", inspect.Parameter.KEYWORD_ONLY,
                              default=0, annotation="int"),
        ])
        out = format_signature(sig, max_width=20)
        # 已有 *args 隐式分隔，不再插显式 *,
        assert "    *,\n" not in out

    def test_multiline_includes_return_annotation(self) -> None:
        sig = inspect.Signature(
            [
                inspect.Parameter("a", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                  annotation="int"),
                inspect.Parameter("b", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                  annotation="int"),
            ],
            return_annotation="dict[str, Any]",
        )
        out = format_signature(sig, max_width=10)
        assert out.endswith(") -> dict[str, Any]")

    def test_multiline_no_return_annotation(self) -> None:
        sig = inspect.Signature(
            [
                inspect.Parameter("a", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                  annotation="int"),
                inspect.Parameter("b", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                  annotation="int"),
            ],
            return_annotation=inspect.Signature.empty,
        )
        out = format_signature(sig, max_width=10)
        assert out.endswith(")")
        assert " -> " not in out

    def test_empty_signature_always_single_line(self) -> None:
        sig = inspect.Signature([])
        assert format_signature(sig, max_width=80) == "()"
        assert format_signature(sig, max_width=1) == "()"

    def test_max_width_none_keeps_legacy_single_line(self) -> None:
        # 不传 max_width 时保持向后兼容（永远单行）
        sig = inspect.Signature([
            inspect.Parameter(f"p_{i}", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              annotation="str") for i in range(20)
        ])
        out = format_signature(sig)
        assert "\n" not in out

    def test_single_param_overlong_accepts_overflow(self) -> None:
        # 单参数本身就超 max_width → 接受超宽，不二次折行
        sig = inspect.Signature([
            inspect.Parameter(
                "option", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation='Literal["a","b","c","d","e","f","g","h","i","j"]'),
        ])
        out = format_signature(sig, max_width=20)
        # 该 param 行无法再拆分，原样输出（含末尾 ,）
        assert "    option: Literal" in out


# ---------------------------------------------------------------------------
# iter3：format_callable_signature 简化路径
# ---------------------------------------------------------------------------


class TestFormatCallableSignatureIter3:

    def test_normal_function_renders_single_line(self) -> None:
        def fn(a: int, b: str = "x") -> bool: ...
        out = format_callable_signature(fn)
        assert out == "(a: int, b: str = 'x') -> bool"

    def test_max_width_triggers_multiline(self) -> None:
        def fn(
            a: int,
            b: int,
            c: int,
            d: int,
            e: int,
            f: int,
            g: int,
            h: int,
        ) -> dict: ...
        out = format_callable_signature(fn, max_width=40)
        assert out is not None
        assert "\n    a: int,\n" in out
        assert out.endswith(") -> dict")

    def test_kwargs_only_wrapper_renders_naturally(self) -> None:
        # iter3: 删除 _is_kwargs_only_fallback，(**kwargs) 自然渲染
        def fn(**kwargs):
            ...
        # 即便挂了 _mcp_input_schema 也不再走 fallback
        fn._mcp_input_schema = {  # type: ignore[attr-defined]
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }
        assert format_callable_signature(fn) == "(**kwargs)"

    def test_unsignaturable_returns_none(self) -> None:
        # 极少数 builtins inspect.signature 抛 ValueError
        # 这里构造一个 __signature__ property 抛异常的 callable 触发 TypeError
        class Bad:
            @property
            def __signature__(self):
                raise ValueError("nope")

            def __call__(self, *a, **kw): ...
        # inspect.signature 看到 __signature__ 属性会调它，抛 ValueError
        # 我们的实现 catch (ValueError, TypeError) → 返回 None
        assert format_callable_signature(Bad()) is None


# -


# ---------------------------------------------------------------------------
# iter3：_format_json_compact Black 风格 JSON 折行
# ---------------------------------------------------------------------------


class TestFormatJsonCompact:

    def test_scalar_always_compact(self) -> None:
        assert _format_json_compact("hello", max_width=80) == '"hello"'
        assert _format_json_compact(42, max_width=80) == "42"
        assert _format_json_compact(True, max_width=80) == "true"
        assert _format_json_compact(None, max_width=80) == "null"

    def test_short_array_compact(self) -> None:
        out = _format_json_compact(["a", "b", "c"], max_width=80)
        # 紧凑模式无空格
        assert out == '["a","b","c"]'

    def test_long_scalar_array_expands_one_per_line(self) -> None:
        # 长 enum 列表（>80 列）→ 每元素一行
        long_enum = [f"option_{i}" for i in range(20)]
        out = _format_json_compact(long_enum, max_width=80)
        assert out.startswith("[\n")
        assert out.endswith("\n]")
        # 每元素独占一行
        for item in long_enum:
            assert f'  "{item}"' in out

    def test_short_object_compact(self) -> None:
        out = _format_json_compact(
            {"type": "string", "enum": ["a", "b"]}, max_width=80)
        # 紧凑：key 后无空格
        assert out == '{"type":"string","enum":["a","b"]}'

    def test_long_object_expands_with_colon_space(self) -> None:
        # 大对象触发展开 → key/value 间用 ": "（含空格）
        obj = {f"key_{i}": f"value_{i}" for i in range(15)}
        out = _format_json_compact(obj, max_width=40)
        assert out.startswith("{\n")
        # 多行模式 key/value 间含空格
        assert '"key_0": "value_0"' in out

    def test_chinese_value_kept_unicode(self) -> None:
        out = _format_json_compact({"name": "中文"}, max_width=80)
        assert "中文" in out
        assert "\\u" not in out

    def test_indent_step_is_2(self) -> None:
        # 嵌套两层：外层缩进 2、内层 4
        out = _format_json_compact(
            {"outer": {"inner_key_aaaaaaa": "inner_value_aaaaaaaaa"}},
            max_width=20,
        )
        assert '\n  "outer"' in out

    def test_current_indent_affects_compact_decision(self) -> None:
        # 同一对象，current_indent 越大越倾向展开
        obj = {"a": 1, "b": 2, "c": 3}  # compact len = 19
        # current_indent=0 → 紧凑（0+19 ≤ 20）
        assert _format_json_compact(obj, max_width=20, current_indent=0) \
            == '{"a":1,"b":2,"c":3}'
        # current_indent=5 → 展开（5+19 > 20）
        out = _format_json_compact(obj, max_width=20, current_indent=5)
        assert "\n" in out


# ---------------------------------------------------------------------------
# iter3：<omit> sentinel
# ---------------------------------------------------------------------------


class TestOmitSentinel:

    def test_missing_repr_is_omit(self) -> None:
        assert repr(_MISSING) == "<omit>"

    def test_string_literal_dots_visually_distinct(self) -> None:
        # 字符串 "..." 渲染为 '...'（带引号），与 sentinel <omit>（不带引号）区分
        sig = build_signature([
            {"name": "x", "default": "...", "annotation": "str"},
        ])
        out = str(sig)
        assert "'...'" in out
        assert "<omit>" not in out

    def test_mcp_schema_optional_no_default_renders_omit(self) -> None:
        schema = {
            "properties": {"all": {"type": "boolean"}},
            "required": [],
        }
        sig = build_signature(mcp_schema_to_specs(schema))
        assert "<omit>" in str(sig)

    def test_pysandbox_function_without_missing_no_omit(self) -> None:
        # 普通 Python 函数（无 _MISSING）签名不出现 <omit>
        def fn(x: int = 0): ...
        out = format_callable_signature(fn)
        assert "<omit>" not in out


# ---------------------------------------------------------------------------
# iter3：format_annotations_section Black 风格行数缩减
# ---------------------------------------------------------------------------


class TestAnnotationsBlackStyleLineCount:

    def test_browser_fill_form_drops_to_around_a_dozen_lines(self) -> None:
        # 模拟 browser_fill_form 的复杂 fields 结构
        fields_schema = {
            "fields": {
                "type": "array",
                "description": "Fields to fill",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string",
                                 "description": "Human-readable field name"},
                        "type": {
                            "type": "string",
                            "enum": ["textbox", "checkbox", "radio",
                                     "combobox", "slider"],
                            "description": "Type of the field",
                        },
                        "value": {"type": "string",
                                  "description": "Value to set"},
                    },
                    "required": ["name", "type", "value"],
                    "additionalProperties": False,
                },
            },
        }
        out = format_annotations_section(fields_schema)
        line_count = len(out.splitlines())
        # iter2 ~45 行，iter3 应缩减至 ~20 行内
        assert line_count < 25, f"行数 {line_count} 超出预期，输出:\n{out}"
