"""Tests for TerminalRenderer (CLI event rendering)."""

from io import StringIO
from unittest.mock import patch

import pytest

from mutagent.cli.terminal import TerminalRenderer
from mutagent.core.messages import StreamEvent, ToolUseBlock


# ---------------------------------------------------------------------------
# TerminalRenderer — render_event tests
# ---------------------------------------------------------------------------

class TestTerminalRendererRenderEvent:

    @pytest.fixture
    def r(self):
        return TerminalRenderer()

    def test_text_delta(self, r, capsys):
        r.render_event(StreamEvent(type="text_delta", text="Hello world"))
        captured = capsys.readouterr()
        assert "Hello world" in captured.out

    def test_tool_exec_start_with_args(self, r, capsys):
        tc = ToolUseBlock(id="tc_1", name="Module-inspect",
                          input={"module_path": "mutagent"})
        r.render_event(StreamEvent(type="tool_exec_start", tool_call=tc))
        captured = capsys.readouterr()
        assert "Module-inspect" in captured.out
        assert 'module_path="mutagent"' in captured.out

    def test_tool_exec_start_no_args(self, r, capsys):
        tc = ToolUseBlock(id="tc_1", name="Module-inspect", input={})
        r.render_event(StreamEvent(type="tool_exec_start", tool_call=tc))
        captured = capsys.readouterr()
        assert "Module-inspect()" in captured.out

    def test_tool_exec_end(self, r, capsys):
        tc = ToolUseBlock(id="tc_1", name="Module-inspect", input={},
                          status="done", result="Success result")
        r.render_event(StreamEvent(type="tool_exec_end", tool_call=tc))
        captured = capsys.readouterr()
        assert "\u2192" in captured.out
        assert "Success result" in captured.out

    def test_tool_exec_end_error(self, r, capsys):
        tc = ToolUseBlock(id="tc_1", name="Module-inspect", input={},
                          status="done", result="Failed", is_error=True)
        r.render_event(StreamEvent(type="tool_exec_end", tool_call=tc))
        captured = capsys.readouterr()
        assert "\u2192" in captured.out
        assert "Failed" in captured.out

    def test_tool_exec_end_long_content_truncated(self, r, capsys):
        tc = ToolUseBlock(id="tc_1", name="Module-inspect", input={},
                          status="done",
                          result="\n".join(f"line {i}" for i in range(20)))
        r.render_event(StreamEvent(type="tool_exec_end", tool_call=tc))
        captured = capsys.readouterr()
        assert "..." in captured.out
        assert "+16 lines" in captured.out

    def test_error_event(self, r, capsys):
        r.render_event(StreamEvent(type="error", error="API failed"))
        captured = capsys.readouterr()
        assert "API failed" in captured.err

    def test_turn_done(self, r, capsys):
        r.render_event(StreamEvent(type="turn_done"))
        captured = capsys.readouterr()
        assert captured.out == "\n"

    def test_unknown_event_type_noop(self, r, capsys):
        r.render_event(StreamEvent(type="response_done"))
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# TerminalRenderer — read_input tests
# ---------------------------------------------------------------------------

class TestTerminalRendererReadInput:

    def test_read_input(self):
        r = TerminalRenderer()
        with patch("builtins.input", return_value="  hello  "):
            result = r.read_input()
        assert result == "hello"

    def test_read_input_ctrl_c_bubbles(self):
        """Ctrl+C 不应被吞掉，应冲到 App.run() 走 confirm_exit 分支。"""
        r = TerminalRenderer()
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                r.read_input()

    def test_read_input_ctrl_d_bubbles(self):
        """Ctrl+D / Ctrl+Z 不应被吞掉，应冲到 App.run() 的 EOFError 分支。"""
        r = TerminalRenderer()
        with patch("builtins.input", side_effect=EOFError):
            with pytest.raises(EOFError):
                r.read_input()


# ---------------------------------------------------------------------------
# TerminalRenderer — confirm_exit tests
# ---------------------------------------------------------------------------

class TestTerminalRendererConfirmExit:

    def test_confirm_yes(self):
        r = TerminalRenderer()
        with patch("builtins.input", return_value="y"):
            assert r.confirm_exit() is True

    def test_confirm_empty_is_yes(self):
        r = TerminalRenderer()
        with patch("builtins.input", return_value=""):
            assert r.confirm_exit() is True

    def test_confirm_no(self):
        r = TerminalRenderer()
        with patch("builtins.input", return_value="n"):
            assert r.confirm_exit() is False

    def test_confirm_exhaustion_returns_true(self):
        r = TerminalRenderer()
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            assert r.confirm_exit() is True
