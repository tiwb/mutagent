"""`mutagent pysandbox` —— 独立执行沙箱代码。

不依赖任何 server 进程，从 config 构造 SandboxApp，连接 mcp_sources / cli_sources，
执行用户代码后关闭并退出。

对齐 python CLI 约定:
  mutagent pysandbox -c "code"      # 单条代码
  mutagent pysandbox script.py      # 脚本文件
  mutagent pysandbox -                # 从 stdin 读
  echo "code" | mutagent pysandbox    # 管道

MSYS2 兼容：含 `/` 的参数（URL/正则/路径）通过 stdin 或 ``--config`` 传入，
避免 Git Bash 自动转换为 Windows 路径。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any


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


async def _run(config: Any, code: str) -> int:
    """构造 sandbox → 执行 → 清理。返回 exit code。

    MCP 连接走与 App.connect_sources 一致的 MCPConnection 路径：
    autostart=true 后台连（不阻塞，调用时会 wait）；
    autostart=false 完全 lazy。
    """
    from mutagent.sandbox.app import SandboxApp
    from mutagent.sandbox._adapter_mcp import MCPConnection
    from mutagent.sandbox._adapter_cli import build_cli_namespace
    import logging
    _logger = logging.getLogger(__name__)

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
            _logger.warning("MCP source '%s' init failed: %s", ns_name, e)
            continue
        sandbox.add_namespace(conn.namespace, on_remove=conn.close)
        if autostart:
            async def _bg(c: MCPConnection = conn, n: str = ns_name) -> None:
                try:
                    await c.ensure_connected()
                except Exception as exc:
                    _logger.warning(
                        "MCP source '%s' autostart failed: %s", n, exc)
            asyncio.create_task(_bg())

    cli_sources = config.get("cli_sources", default={}) or {}
    if cli_sources:
        cli_ns = build_cli_namespace(cli_sources)
        sandbox.add_namespace(cli_ns)

    try:
        # exec_code 是 sync，丢到 executor 避免阻塞 loop（MCP 调用走 loop 回调）
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


def dispatch_pysandbox(app: Any, args: argparse.Namespace) -> None:
    """由 main() 调用：app 已 load_config 完毕。"""
    if args.code is not None and args.script is not None:
        print("Error: -c CODE and script file are mutually exclusive", file=sys.stderr)
        sys.exit(2)

    code = _read_code(args)
    if not code.strip():
        print("Error: empty code.", file=sys.stderr)
        sys.exit(2)

    exit_code = asyncio.run(_run(app.config, code))
    sys.exit(exit_code)
