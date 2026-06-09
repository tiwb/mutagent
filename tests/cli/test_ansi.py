"""Tests for ANSI color utilities and tool formatting functions."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mutagent.cli import ansi
from mutagent.cli.ansi import (
    _BOLD, _CYAN, _DIM, _GREEN, _RED, _RESET, _YELLOW,
    bold, bold_cyan, bold_red, cyan, dim, green,
    highlight_markdown_line, red, yellow,
    format_tool_call, format_tool_result, _format_value,
    _MAX_SINGLE_LINE, _MAX_VALUE_LEN,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _with_color(enabled: bool):
    """Patch _color_supported to return a fixed value."""
    return patch.object(ansi, '_color_supported', return_value=enabled)


# ---------------------------------------------------------------------------
# Terminal detection tests
# ---------------------------------------------------------------------------

class TestColorSupported:

    def test_no_color_env(self, monkeypatch):
        ansi._color_supported.cache_clear()
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.delenv("FORCE_COLOR", raising=False)
        assert ansi._color_supported() is False
        ansi._color_supported.cache_clear()

    def test_force_color_env(self, monkeypatch):
        ansi._color_supported.cache_clear()
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("FORCE_COLOR", "1")
        assert ansi._color_supported() is True
        ansi._color_supported.cache_clear()

    def test_no_color_takes_priority_over_force(self, monkeypatch):
        ansi._color_supported.cache_clear()
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("FORCE_COLOR", "1")
        assert ansi._color_supported() is False
        ansi._color_supported.cache_clear()

    def test_not_tty(self, monkeypatch):
        ansi._color_supported.cache_clear()
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("FORCE_COLOR", raising=False)
        # In pytest, stdout is not a tty
        assert ansi._color_supported() is False
        ansi._color_supported.cache_clear()


# ---------------------------------------------------------------------------
# Color wrapper function tests
# ---------------------------------------------------------------------------

class TestColorFunctions:

    def test_dim_enabled(self):
        with _with_color(True):
            assert dim("text") == f"{_DIM}text{_RESET}"

    def test_dim_disabled(self):
        with _with_color(False):
            assert dim("text") == "text"

    def test_bold_enabled(self):
        with _with_color(True):
            assert bold("text") == f"{_BOLD}text{_RESET}"

    def test_bold_disabled(self):
        with _with_color(False):
            assert bold("text") == "text"

    def test_green_enabled(self):
        with _with_color(True):
            assert green("text") == f"{_GREEN}text{_RESET}"

    def test_green_disabled(self):
        with _with_color(False):
            assert green("text") == "text"

    def test_red_enabled(self):
        with _with_color(True):
            assert red("text") == f"{_RED}text{_RESET}"

    def test_red_disabled(self):
        with _with_color(False):
            assert red("text") == "text"

    def test_bold_red_enabled(self):
        with _with_color(True):
            assert bold_red("text") == f"{_BOLD}{_RED}text{_RESET}"

    def test_bold_red_disabled(self):
        with _with_color(False):
            assert bold_red("text") == "text"

    def test_yellow_enabled(self):
        with _with_color(True):
            assert yellow("text") == f"{_YELLOW}text{_RESET}"

    def test_yellow_disabled(self):
        with _with_color(False):
            assert yellow("text") == "text"

    def test_cyan_enabled(self):
        with _with_color(True):
            assert cyan("text") == f"{_CYAN}text{_RESET}"

    def test_cyan_disabled(self):
        with _with_color(False):
            assert cyan("text") == "text"

    def test_bold_cyan_enabled(self):
        with _with_color(True):
            assert bold_cyan("text") == f"{_BOLD}{_CYAN}text{_RESET}"

    def test_bold_cyan_disabled(self):
        with _with_color(False):
            assert bold_cyan("text") == "text"

    def test_empty_string(self):
        with _with_color(True):
            assert dim("") == f"{_DIM}{_RESET}"
        with _with_color(False):
            assert dim("") == ""


# ---------------------------------------------------------------------------
# Markdown highlighting tests
# ---------------------------------------------------------------------------

class TestHighlightMarkdownLine:

    def test_heading(self):
        with _with_color(True):
            result = highlight_markdown_line("## Design")
            # Entire line should be cyan
            assert result == f"{_CYAN}## Design{_RESET}"

    def test_heading_levels(self):
        with _with_color(True):
            for n in range(1, 7):
                prefix = "#" * n + " "
                result = highlight_markdown_line(f"{prefix}Title")
                assert result == f"{_CYAN}{prefix}Title{_RESET}"

    def test_unordered_list_dash(self):
        with _with_color(True):
            result = highlight_markdown_line("- item")
            assert _CYAN in result
            assert "item" in result

    def test_unordered_list_asterisk(self):
        with _with_color(True):
            result = highlight_markdown_line("* item")
            assert _CYAN in result

    def test_unordered_list_plus(self):
        with _with_color(True):
            result = highlight_markdown_line("+ item")
            assert _CYAN in result

    def test_ordered_list(self):
        with _with_color(True):
            result = highlight_markdown_line("1. first")
            assert _CYAN in result
            assert "first" in result

    def test_blockquote(self):
        with _with_color(True):
            result = highlight_markdown_line("> quote")
            # Entire line should be cyan
            assert result == f"{_CYAN}> quote{_RESET}"

    def test_bold_markers(self):
        with _with_color(True):
            result = highlight_markdown_line("this is **bold** text")
            # Entire **bold** span should be cyan
            assert f"{_CYAN}**bold**{_RESET}" in result
            assert "this is " in result
            assert " text" in result

    def test_bold_underscore_markers(self):
        with _with_color(True):
            result = highlight_markdown_line("this is __bold__ text")
            assert f"{_CYAN}__bold__{_RESET}" in result

    def test_inline_code(self):
        with _with_color(True):
            result = highlight_markdown_line("use `code` here")
            assert _YELLOW in result
            assert "use " in result
            assert "here" in result

    def test_inline_code_whole_span(self):
        with _with_color(True):
            result = highlight_markdown_line("run `pip install`")
            # The whole `pip install` span should be yellow
            assert f"{_YELLOW}`pip install`{_RESET}" in result

    def test_no_color_returns_unchanged(self):
        with _with_color(False):
            line = "## Heading with `code` and **bold**"
            assert highlight_markdown_line(line) == line

    def test_plain_text_unchanged(self):
        with _with_color(True):
            result = highlight_markdown_line("just plain text")
            assert result == "just plain text"

    def test_heading_with_inline_code(self):
        with _with_color(True):
            result = highlight_markdown_line("## Use `config.json`")
            # Heading → entire line is cyan (no nested inline highlighting)
            assert result == f"{_CYAN}## Use `config.json`{_RESET}"

    def test_list_with_bold(self):
        with _with_color(True):
            result = highlight_markdown_line("- **Toolkit** base class")
            # List marker only highlighted, bold span in rest is highlighted
            assert f"{_CYAN}- {_RESET}" in result
            assert f"{_CYAN}**Toolkit**{_RESET}" in result
            assert "base class" in result

    def test_empty_string(self):
        with _with_color(True):
            assert highlight_markdown_line("") == ""

    def test_indented_list(self):
        with _with_color(True):
            result = highlight_markdown_line("  - nested item")
            assert _CYAN in result


# ---------------------------------------------------------------------------
# _format_value tests
# ---------------------------------------------------------------------------

class TestFormatValue:

    def test_string_value(self):
        assert _format_value("hello") == '"hello"'

    def test_int_value(self):
        assert _format_value(5) == "5"

    def test_float_value(self):
        assert _format_value(3.14) == "3.14"

    def test_bool_value(self):
        assert _format_value(True) == "True"

    def test_none_value(self):
        assert _format_value(None) == "None"

    def test_long_string_truncated(self):
        long_str = "a" * 100
        result = _format_value(long_str)
        assert result.startswith('"')
        assert result.endswith('..."')
        assert len(result) <= _MAX_VALUE_LEN + 2  # quotes

    def test_list_value(self):
        assert _format_value([1, 2, 3]) == "[1, 2, 3]"

    def test_long_repr_truncated(self):
        long_list = list(range(100))
        result = _format_value(long_list)
        assert result.endswith("...")


# ---------------------------------------------------------------------------
# _format_tool_call tests
# ---------------------------------------------------------------------------

class TestFormatToolCall:

    def test_no_args(self):
        result = format_tool_call("Module-inspect", {})
        assert result == "  Module-inspect()"

    def test_single_string_arg(self):
        result = format_tool_call("Module-inspect", {"path": "mutagent.tools"})
        assert 'path="mutagent.tools"' in result
        assert "Module-inspect(" in result

    def test_single_line_format(self):
        result = format_tool_call("func", {"a": "x"})
        assert "\n" not in result
        assert result == '  func(a="x")'

    def test_multi_line_format(self):
        result = format_tool_call("Module-define", {
            "path": "mutagent.my_tool",
            "source": "a" * 70,
        })
        assert "\n" in result
        lines = result.split("\n")
        assert lines[0].strip().endswith("(")
        assert lines[-1].strip() == ")"
        # Each param line should end with comma
        for param_line in lines[1:-1]:
            assert param_line.rstrip().endswith(",")

    def test_mixed_types(self):
        result = format_tool_call("func", {"name": "test", "count": 5, "verbose": True})
        assert 'name="test"' in result
        assert "count=5" in result
        assert "verbose=True" in result

    def test_value_truncation(self):
        result = format_tool_call("func", {"data": "x" * 100})
        assert "..." in result

    def test_long_single_arg_wraps(self):
        # Single arg that makes the line > 80 chars should still be single line
        # if total <= _MAX_SINGLE_LINE, else multi-line
        result = format_tool_call("very_long_function_name", {
            "parameter": "a" * 70,
        })
        # This should be multi-line due to total length
        assert "\n" in result


# ---------------------------------------------------------------------------
# _format_tool_result tests
# ---------------------------------------------------------------------------

class TestFormatToolResult:

    def test_short_result(self):
        result = format_tool_result("Toolkit (class)", is_error=False)
        assert "\u2192" in result
        assert "Toolkit (class)" in result

    def test_multiline_within_preview(self):
        content = "Line 1\nLine 2\nLine 3"
        result = format_tool_result(content, is_error=False)
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result
        assert "..." not in result

    def test_multiline_exceeds_preview(self):
        content = "\n".join(f"line {i}" for i in range(10))
        result = format_tool_result(content, is_error=False)
        assert "line 0" in result
        assert "line 1" in result
        assert "line 2" in result
        assert "line 3" in result
        assert "+6 lines" in result

    def test_error_result(self):
        # In non-tty (test env), no ANSI codes, but function is called
        result = format_tool_result("Module not found", is_error=True)
        assert "\u2192" in result
        assert "Module not found" in result

    def test_empty_content(self):
        result = format_tool_result("", is_error=False)
        assert "\u2192" in result

    def test_single_line_result(self):
        result = format_tool_result("OK", is_error=False)
        assert "\u2192" in result
        assert "OK" in result
        assert "..." not in result

    def test_exactly_preview_lines(self):
        content = "\n".join(f"line {i}" for i in range(4))
        result = format_tool_result(content, is_error=False)
        # Exactly 4 lines should all be shown, no overflow
        assert "line 3" in result
        assert "..." not in result

