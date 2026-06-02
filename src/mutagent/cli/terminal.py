"""mutagent.cli.terminal -- TerminalRenderer for CLI event display.

A plain class (not Declaration/@impl) that renders Agent StreamEvent
objects to the terminal.  Used by CLI App.run().
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import queue

import sys
import threading
from typing import Any

from mutagent.app.app import App
from mutagent.cli.ansi import (
    bold_cyan, bold_red, dim, _format_tool_call, _format_tool_result,
    highlight_markdown_line,
)
from mutagent.core.messages import Message, StreamEvent, TextBlock, ToolResultBlock, ToolUseBlock
from mutagent.core.session import AgentSession


logger = logging.getLogger(__name__)


def _default_session_dir() -> Path:
    return Path.home() / ".mutagent" / "sessions"


def _build_agent_session(app: App, resume: str | None = None) -> AgentSession:
    session = AgentSession()
    session.start_new(
        session_dir=_default_session_dir(),
        cwd=str(Path.cwd()),
        model=app.agent.model,
    )
    if resume is not None:
        session.resume(resume, app.agent.context)
    object.__setattr__(app.agent, "session", session)
    return session


def _make_session_persist_callback(app: App, session: AgentSession):
    def _persist(_event: StreamEvent) -> None:
        try:
            session.sync(app.agent.context)
        except Exception:
            logger.exception("Failed to persist session increment")

    return _persist


class TerminalRenderer:
    """Render Agent StreamEvents to a terminal."""

    def render_event(self, event: StreamEvent) -> None:
        """Render a single StreamEvent."""
        if event.type == "text_delta":
            print(highlight_markdown_line(event.text), end="", flush=True)
        elif event.type == "tool_exec_start" and isinstance(event.tool_call, ToolUseBlock):
            name = event.tool_call.name
            args = event.tool_call.input
            call_str = _format_tool_call(name, args)
            print(f"\n{dim(call_str)}", flush=True)
        elif event.type == "tool_exec_end" and isinstance(event.tool_call, ToolResultBlock):
            is_error = event.tool_call.is_error
            result_str = _format_tool_result(
                event.tool_call.content, is_error,
            )
            print(result_str, flush=True)
        elif event.type == "error":
            print(f"\n{bold_red('[Error: ' + event.error + ']')}",
                  file=sys.stderr, flush=True)
        elif event.type == "turn_done":
            print()

    def render_history(self, messages: list[Message]) -> None:
        """Replay persisted messages with the same formatting as live events."""
        turn_active = False
        for message in messages:
            if self._starts_new_turn(message):
                if turn_active:
                    self.render_event(StreamEvent(type="turn_done"))
                turn_active = True
            self._render_message(message)
        if turn_active:
            self.render_event(StreamEvent(type="turn_done"))

    def _render_message(self, message: Message) -> None:
        for block in message.blocks:
            if isinstance(block, TextBlock):
                if message.role == "user":
                    print(f"{bold_cyan('> ')}{block.text}", flush=True)
                elif message.role == "assistant":
                    self.render_event(StreamEvent(type="text_delta", text=block.text))
            elif isinstance(block, ToolUseBlock):
                self.render_event(StreamEvent(type="tool_exec_start", tool_call=block))
            elif isinstance(block, ToolResultBlock):
                self.render_event(StreamEvent(type="tool_exec_end", tool_call=block))

    def _starts_new_turn(self, message: Message) -> bool:
        return message.role == "user" and any(
            isinstance(block, TextBlock) for block in message.blocks
        )

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


def add_terminal_subcommand(subparsers: Any) -> argparse.ArgumentParser:
    """Register the terminal subparser."""
    parser = subparsers.add_parser(
        "terminal",
        help="Start an interactive chat session (default)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force headless mode (ignored, no TUI)",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="",
        help="Resume a session by path or session id. Without value, resumes the latest session.",
    )
    return parser


def dispatch_terminal(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """在终端 REPL 中运行 agent 会话。"""
    app = App()
    app.load_config(args.config)
    app.setup_agent()
    session = _build_agent_session(app, getattr(args, "resume", None))
    spec = app.config.resolve_model()
    print(f"mutagent ready  (model: {spec.get('model_id', '?') if spec else '?'})")
    print("Type your message. 'exit' or Ctrl+C to quit.\n")

    renderer = TerminalRenderer()
    renderer.render_history(app.agent.context.messages)

    # 启动 asyncio event loop 线程
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    # 在 loop 上连接 MCP/CLI sources
    try:
        asyncio.run_coroutine_threadsafe(
            app.connect_sources(), loop
        ).result(timeout=120)
    except Exception as e:
        logger.warning("connect_sources failed: %s", e)
        print(f"Warning: failed to connect sources: {e}", file=sys.stderr)

    # 订阅 agent 事件（线程安全的 queue.put）
    event_q: queue.Queue[StreamEvent] = queue.Queue()
    app.agent.subscribe(_make_session_persist_callback(app, session))
    app.agent.subscribe(event_q.put)

    waiting_for_input = True  # True when blocked on read_input()

    while True:
        try:
            waiting_for_input = True
            user_input = renderer.read_input()
            waiting_for_input = False

            if not user_input:
                continue

            # exit / /exit 直接退出
            if user_input in ("exit", "/exit"):
                break

            # 提交 agent 任务到 asyncio 线程
            asyncio.run_coroutine_threadsafe(
                app.agent.submit(user_input), loop
            )

            # 主线程同步消费事件
            while True:
                try:
                    evt = event_q.get(timeout=0.2)
                except queue.Empty:
                    if app.agent.is_busy():
                        continue
                    break
                renderer.render_event(evt)
                if evt.type == "turn_done":
                    break

        except KeyboardInterrupt:
            if waiting_for_input:
                # 空输入时 Ctrl+C：询问是否退出
                if renderer.confirm_exit():
                    break
            else:
                # agent 运行中 Ctrl+C：取消当前任务
                app.agent.cancel()
                # 排空 event_q 中残留事件
                try:
                    while True:
                        event_q.get_nowait()
                except queue.Empty:
                    pass
                print("\n[User interrupted]")
        except EOFError:
            # Ctrl+D (Unix) / Ctrl+Z (Windows)
            break
        except Exception as e:
            print(f"\n[Error: {e}]", file=sys.stderr, flush=True)

    # 清理 sandbox + event loop
    sandbox = getattr(app, "sandbox", None)
    try:
        session.sync(app.agent.context)
    except Exception:
        logger.exception("Failed to persist final session state")
    if sandbox is not None:
        try:
            asyncio.run_coroutine_threadsafe(sandbox.close(), loop).result(timeout=5)
        except Exception:
            pass
    try:
        asyncio.run_coroutine_threadsafe(loop.shutdown_asyncgens(), loop).result(timeout=2)
    except Exception:
        pass
    loop.call_soon_threadsafe(loop.stop)
    loop_thread.join(timeout=2)
