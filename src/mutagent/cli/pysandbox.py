"""`mutagent pysandbox` —— 独立执行沙箱代码。

不依赖任何 server 进程，从 config 构造 SandboxApp，连接 mcp_sources / cli_sources，
执行用户代码后关闭并退出。

对齐 python CLI 约定:
  mutagent pysandbox -c "code"      # 单条代码
  mutagent pysandbox script.py      # 脚本文件
  mutagent pysandbox -                # 从 stdin 读
  echo "code" | mutagent pysandbox    # 管道
  mutagent pysandbox                  # 进入交互式 REPL（类似 python）

MSYS2 兼容：含 `/` 的参数（URL/正则/路径）通过 stdin 或 ``--config`` 传入，
避免 Git Bash 自动转换为 Windows 路径。
"""

from __future__ import annotations

import argparse
import asyncio
import code
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from mutagent.runtime.log_store import LogStore, LogStoreHandler, SingleLineFormatter

logger = logging.getLogger(__name__)


def add_pysandbox_subcommand(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "pysandbox",
        help="在沙箱中执行 Python 代码（独立进程，不依赖 server）",
    )
    parser.add_argument(
        "-c",
        dest="code",
        metavar="CODE",
        help="代码字符串（类似 python -c）",
    )
    parser.add_argument(
        "script",
        nargs="?",
        help="脚本文件路径，或 - 从 stdin 读",
    )
    return parser


def _read_code(args: argparse.Namespace) -> str:
    if args.code is not None:
        return args.code
    if args.script == "-":
        return sys.stdin.read()
    if args.script is not None:
        with open(args.script, "r", encoding="utf-8") as f:
            return f.read()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    print(
        "Error: no code provided. Use -c CODE, a script file, or pipe via stdin.",
        file=sys.stderr,
    )
    print("Examples:", file=sys.stderr)
    print('  mutagent pysandbox -c "help()"', file=sys.stderr)
    print("  mutagent pysandbox script.py", file=sys.stderr)
    print('  echo "help()" | mutagent pysandbox', file=sys.stderr)
    sys.exit(2)


def _setup_pysandbox_logging(config: Any) -> None:
    """配置 pysandbox 的基础 logging管线（LogStore + 文件），不挂 console handler。

    和 ``setup_agent()`` 的 logging 部分保持一致的 session 命名与目录结构，
    但不包括 API Recorder 等重型组件。
    """
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(config.get("logging.log_dir", default=".mutagent/logs"))

    root_logger = logging.getLogger()  # root logger，捕获所有库的日志
    root_logger.setLevel(logging.DEBUG)

    # Memory handler
    log_store = LogStore()
    mem_handler = LogStoreHandler(log_store)
    mem_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(mem_handler)

    # File handler
    if config.get("logging.file_log", default=True):
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"{session_ts}.log", encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(SingleLineFormatter(
            "%(asctime)s %(levelname)-8s %(name)s - %(message)s"
        ))
        root_logger.addHandler(file_handler)

    logger.info("Pysandbox logging initialized (session=%s)", session_ts)


def _build_sandbox(config: Any) -> Any:
    """构造 SandboxApp 并注入 MCP/CLI namespaces，返回 sandbox。"""
    from mutagent.sandbox.app import SandboxApp
    from mutagent.sandbox._adapter_mcp import MCPConnection
    from mutagent.sandbox._adapter_cli import build_cli_namespace

    _setup_pysandbox_logging(config)

    sandbox = SandboxApp()
    main_loop = asyncio.get_running_loop()
    mcp_sources = config.get("mcp_sources", default={}) or {}
    for ns_name, server_cfg in mcp_sources.items():
        autostart = bool(server_cfg.get("autostart", True))
        retry_cooldown = float(server_cfg.get("retry_cooldown", 5.0))
        try:
            conn = MCPConnection(
                ns_name, server_cfg, main_loop,
                retry_cooldown=retry_cooldown)
        except Exception as e:
            logger.warning("MCP source '%s' init failed: %s", ns_name, e)
            continue
        sandbox.add_namespace(conn.namespace, on_remove=conn.close)
        if autostart:
            async def _bg(c: MCPConnection = conn, n: str = ns_name) -> None:
                try:
                    await c.ensure_connected()
                except Exception as exc:
                    logger.warning(
                        "MCP source '%s' autostart failed: %s", n, exc)
            asyncio.create_task(_bg())

    cli_sources = config.get("cli_sources", default={}) or {}
    if cli_sources:
        cli_ns = build_cli_namespace(cli_sources)
        sandbox.add_namespace(cli_ns)

    return sandbox


class SandboxConsole(code.InteractiveConsole):
    """交互式 REPL 控制台，将代码执行委托给 SandboxApp。

    复写 ``runsource``：先通过 ``code.compile_command`` 判断输入是否完整，
    完整时通过 ``sandbox.exec_code(source, self.locals)`` 执行，
    利用 ``self.locals`` 保持变量跨步骤存活。
    """

    def __init__(self, sandbox: Any) -> None:
        super().__init__(locals={})
        self.sandbox = sandbox

    def runsource(self, source: str,
                  filename: str = '<pysandbox>',
                  symbol: str = 'single') -> bool:
        try:
            code_obj = code.compile_command(source, filename, symbol)
        except (OverflowError, SyntaxError, ValueError):
            self.showsyntaxerror(filename)
            return False
        if code_obj is None:
            return True  # 输入不完整，提示 ... 继续

        result = self.sandbox.exec_code(source, self.locals)
        text, is_error = self.sandbox.format_result(result)
        if text:
            print(text, file=sys.stderr if is_error else sys.stdout)
        return False


async def _run(config: Any, code: str) -> int:
    """构造 sandbox → 执行 → 清理。返回 exit code。"""
    sandbox = _build_sandbox(config)
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, sandbox.exec_code, code, None)
    finally:
        try:
            await sandbox.close()
        except Exception:
            logger.exception("sandbox.close() failed")

    text, is_error = sandbox.format_result(result)
    stream = sys.stderr if is_error else sys.stdout
    if text:
        print(text, file=stream)
    return 1 if is_error else 0


async def _run_repl(config: Any) -> None:
    """构造 sandbox → 进入交互 REPL → 清理。"""
    sandbox = _build_sandbox(config)
    banner = (
        "Python sandbox (mutagent pysandbox)\n"
        "Type 'help()' to discover available namespaces, Ctrl-D to exit.\n"
    )
    try:
        console = SandboxConsole(sandbox)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, console.interact, banner)
    except asyncio.CancelledError:
        # Ctrl+C 触发的 asyncio task cancel — 静默，走 finally 清理
        pass
    finally:
        try:
            await sandbox.close()
        except Exception:
            logger.exception("sandbox.close() failed")


def dispatch_pysandbox(app: Any, args: argparse.Namespace) -> None:
    """由 main() 调用：app 已 load_config 完毕。"""
    if args.code is not None and args.script is not None:
        print("Error: -c CODE and script file are mutually exclusive", file=sys.stderr)
        sys.exit(2)

    # 无参数 + 交互式终端 → 进入 REPL
    if args.code is None and args.script is None and sys.stdin.isatty():
        try:
            asyncio.run(_run_repl(app.config))
        except KeyboardInterrupt:
            pass  # Ctrl+C 干净退出，不打印 traceback
        return

    code = _read_code(args)
    if not code.strip():
        print("Error: empty code.", file=sys.stderr)
        sys.exit(2)

    try:
        exit_code = asyncio.run(_run(app.config, code))
    except KeyboardInterrupt:
        exit_code = 0  # Ctrl+C 干净退出
    sys.exit(exit_code)
