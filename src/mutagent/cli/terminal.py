"""mutagent.cli.terminal -- TerminalRenderer for CLI event display.

A plain class (not Declaration/@impl) that renders Agent StreamEvent
objects to the terminal.  Used by CLI App.run().
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from mutagent.cli.ansi import (
    bold_cyan, bold_red, dim, _format_tool_call, _format_tool_result,
    highlight_markdown_line,
)

if TYPE_CHECKING:
    from mutagent.messages import StreamEvent


class TerminalRenderer:
    """Render Agent StreamEvents to a terminal."""

    def render_event(self, event: StreamEvent) -> None:
        """Render a single StreamEvent."""
        if event.type == "text_delta":
            print(highlight_markdown_line(event.text), end="", flush=True)
        elif event.type == "tool_exec_start":
            name = event.tool_call.name if event.tool_call else "?"
            args = event.tool_call.input if event.tool_call else {}
            call_str = _format_tool_call(name, args)
            print(f"\n{dim(call_str)}", flush=True)
        elif event.type == "tool_exec_end":
            if event.tool_call:
                is_error = event.tool_call.is_error
                result_str = _format_tool_result(
                    event.tool_call.result, is_error,
                )
                print(result_str, flush=True)
        elif event.type == "error":
            print(f"\n{bold_red('[Error: ' + event.error + ']')}",
                  file=sys.stderr, flush=True)
        elif event.type == "turn_done":
            print()

    def read_input(self) -> str:
        """Read a line of user input with a prompt.

        KeyboardInterrupt / EOFError 不在此捕获，由 ``App.run()`` 的顶层
        except 分支处理（分别走 confirm_exit 与正常退出）。
        """
        return input(bold_cyan("> ")).strip()

    def confirm_exit(self) -> bool:
        """Ask the user to confirm exit (y/n)."""
        for _ in range(3):
            try:
                choice = input("\nDo you want to exit? (Y/n) ").strip().lower()
            except KeyboardInterrupt:
                continue
            if choice in ("y", "yes", ""):
                return True
            elif choice in ("n", "no"):
                return False
        print("")
        return True
