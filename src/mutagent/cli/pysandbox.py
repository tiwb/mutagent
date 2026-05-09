"""``mutagent pysandbox`` —— 通用 pysandbox MCP 客户端 + 纯沙箱服务。

两种运行模式：

- **Client（默认）**: 连远程 MCP server，调 ``pysandbox`` tool 执行代码。
  适配 agent 一次性脚本调用。本地无 logging、无 config，纯 RPC。

- **Server (``--serve``)**: 启动一个 HTTP MCP server，暴露 ``pysandbox``
  tool。常驻进程，没有 agent loop / WebUI。

对齐 python CLI 约定（仅 client 模式）::

  mutagent pysandbox --port P -c "code"          # 单条代码
  mutagent pysandbox --port P script.py          # 脚本文件
  mutagent pysandbox --port P -                  # 从 stdin 读
  echo "code" | mutagent pysandbox --port P      # 管道

  mutagent pysandbox --url http://host:P/mcp -c "code"   # 显式 URL
  mutagent pysandbox --port P                    # 进入交互 REPL

  mutagent pysandbox --serve --port P            # 纯沙箱服务

MSYS2 兼容：含 ``/`` 的参数（URL/正则/路径）通过 stdin 或 ``--config``
传入，避免 Git Bash 自动转换为 Windows 路径。
"""

from __future__ import annotations

import argparse
import asyncio
import code
import json
import logging
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# CLI 注册
# ---------------------------------------------------------------------------

def add_pysandbox_subcommand(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "pysandbox",
        help="pysandbox MCP client（默认）/ --serve 启动纯沙箱 MCP server",
    )
    # 模式选择
    parser.add_argument(
        "--serve",
        action="store_true",
        help="启动纯沙箱 MCP server（常驻），而非作为 client",
    )

    # 公用 host/port
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="server 地址（默认 127.0.0.1，与 webui 对齐）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="server 端口（必填）。client 模式下若用 --url 则可省",
    )

    # client only
    parser.add_argument(
        "--url",
        default=None,
        help="client 模式：完整 MCP endpoint URL，覆盖 --host/--port",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"client 模式：RPC timeout 秒（默认 {DEFAULT_TIMEOUT}）",
    )

    # 代码源（仅 client 模式有效）
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


# ---------------------------------------------------------------------------
# 输入校验 & 错误信息
# ---------------------------------------------------------------------------

def _has_code_source(args: argparse.Namespace) -> bool:
    """是否提供了任何代码源（-c / script / 非交互 stdin）。"""
    if args.code is not None:
        return True
    if args.script is not None:
        return True
    if not sys.stdin.isatty():
        return True
    return False


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """互斥与必填检查。报错时通过 parser.error 退出。"""
    if args.serve:
        # server 模式
        if args.url is not None:
            parser.error("--url is a client-only option, cannot combine with --serve")
        if args.port is None:
            parser.error(
                "--port is required with --serve\n"
                "\n"
                "Example:\n"
                "  mutagent pysandbox --serve --port 8080"
            )
        if _has_code_source(args):
            parser.error(
                "--serve does not accept code input (-c / script / stdin); "
                "it runs as a pure MCP server."
            )
    else:
        # client 模式
        if args.code is not None and args.script is not None:
            parser.error("-c CODE and script file are mutually exclusive")
        if args.url is None and args.port is None:
            parser.error(
                "--port is required (or use --url for full endpoint)\n"
                "\n"
                "Examples:\n"
                "  mutagent pysandbox --port 8080 -c 'help()'\n"
                "  mutagent pysandbox --url http://host:8080/mcp -c 'help()'\n"
                "\n"
                "To start a sandbox server:\n"
                "  mutagent pysandbox --serve --port 8080"
            )


def _read_code(args: argparse.Namespace) -> str:
    """读取代码源。失败时退出。"""
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
    print('  mutagent pysandbox --port P -c "help()"', file=sys.stderr)
    print("  mutagent pysandbox --port P script.py", file=sys.stderr)
    print('  echo "help()" | mutagent pysandbox --port P', file=sys.stderr)
    sys.exit(2)


def _server_unreachable_message(url: str, reason: str) -> str:
    return (
        f"Error: No pysandbox server at {url} ({reason})\n"
        f"\n"
        f"Start a server:\n"
        f"  mutagent pysandbox --serve --port PORT      # sandbox-only MCP server\n"
        f"\n"
        f"Or point to a different server:\n"
        f"  mutagent pysandbox --port PORT -c '...'\n"
        f"  mutagent pysandbox --url URL    -c '...'\n"
    )


# ---------------------------------------------------------------------------
# Client 模式
# ---------------------------------------------------------------------------

def _format_tool_result(result: dict[str, Any]) -> tuple[str, bool]:
    """把 MCPClient.call_tool 的返回拍成文本。返回 (text, is_error)。"""
    is_error = bool(result.get("isError"))
    content = result.get("content") or []
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    if parts:
        return "\n".join(parts), is_error
    return json.dumps(result, ensure_ascii=False, indent=2), is_error


async def _run_client(args: argparse.Namespace, code: str) -> int:
    """连远程 MCP server，调 pysandbox tool。"""
    from mutio.mcp.client import MCPClient, MCPError

    url = args.url if args.url is not None else f"http://{args.host}:{args.port}/mcp"
    client = MCPClient(
        url=url,
        client_name="mutagent-pysandbox-cli",
        timeout=args.timeout,
    )
    try:
        try:
            await client.connect()
        except (ConnectionError, OSError, TimeoutError) as e:
            print(_server_unreachable_message(url, str(e)), file=sys.stderr)
            return 1
        except Exception as e:
            # connect 内部可能是 httpx / 异常包装；按"连不上"处理
            print(_server_unreachable_message(url, str(e)), file=sys.stderr)
            return 1

        try:
            result = await client.call_tool("pysandbox", code=code)
        except MCPError as e:
            print(f"Error: MCP {e.code}: {e.message}", file=sys.stderr)
            return 1
    finally:
        try:
            await client.close()
        except Exception:
            pass

    text, is_error = _format_tool_result(result)
    stream = sys.stderr if is_error else sys.stdout
    if text:
        print(text, file=stream)
    return 1 if is_error else 0


class _REPLConsole(code.InteractiveConsole):
    """交互式 REPL，通过 MCP 协议逐条发送代码到远程 pysandbox server。

    复写 ``runsource``：用 ``code.compile_command`` 判断输入是否完整，
    完整时通过 MCP client 调用远程 ``pysandbox`` tool 执行。
    跨行状态由 ``code.InteractiveConsole`` 基类管理（buffer）。
    """

    def __init__(self, client: Any, timeout: float, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__(locals={})
        self._client = client
        self._timeout = timeout
        self._loop = loop

    def runsource(
        self, source: str,
        filename: str = "<pysandbox>",
        symbol: str = "single",
    ) -> bool:
        try:
            code_obj = code.compile_command(source, filename, symbol)
        except (OverflowError, SyntaxError, ValueError):
            self.showsyntaxerror(filename)
            return False
        if code_obj is None:
            return True  # 输入不完整，提示 ... 继续

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._client.call_tool("pysandbox", code=source),
                self._loop,
            )
            result = future.result(timeout=self._timeout)
        except TimeoutError:
            print(f"Error: RPC timeout after {self._timeout}s", file=sys.stderr)
            return False
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return False

        text, is_error = _format_tool_result(result)
        if text:
            print(text, file=sys.stderr if is_error else sys.stdout)
        return False


async def _run_repl(args: argparse.Namespace) -> int:
    """进入交互 REPL，持续复用同一条 MCP 连接。"""
    from mutio.mcp.client import MCPClient

    url = args.url if args.url is not None else f"http://{args.host}:{args.port}/mcp"
    client = MCPClient(
        url=url,
        client_name="mutagent-pysandbox-cli",
        timeout=args.timeout,
    )
    try:
        try:
            await client.connect()
        except (ConnectionError, OSError, TimeoutError) as e:
            print(_server_unreachable_message(url, str(e)), file=sys.stderr)
            return 1
        except Exception as e:
            print(_server_unreachable_message(url, str(e)), file=sys.stderr)
            return 1

        banner = (
            "Python sandbox (mutagent pysandbox)\n"
            "Type 'help()' to discover available namespaces, Ctrl-D to exit.\n"
        )
        console = _REPLConsole(client, args.timeout, asyncio.get_running_loop())
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, console.interact, banner)
    finally:
        try:
            await client.close()
        except Exception:
            pass
    return 0


def _dispatch_client(args: argparse.Namespace) -> None:
    """client 模式入口（不依赖 app.config）。

    无代码源 + 交互终端 → REPL；否则一次性执行。
    """
    # 无参数 + 交互式终端 → 进入 REPL
    if not _has_code_source(args) and sys.stdin.isatty():
        try:
            exit_code = asyncio.run(_run_repl(args))
        except KeyboardInterrupt:
            exit_code = 0
        sys.exit(exit_code)

    code = _read_code(args)
    if not code.strip():
        print("Error: empty code.", file=sys.stderr)
        sys.exit(2)
    try:
        exit_code = asyncio.run(_run_client(args, code))
    except KeyboardInterrupt:
        exit_code = 0
    sys.exit(exit_code)


# ---------------------------------------------------------------------------
# Server 模式
# ---------------------------------------------------------------------------

def _setup_pysandbox_logging(config: Any) -> None:
    """配置 server 模式的基础 logging（LogStore + 文件）。

    沿用原独立模式的实现：写 ``.mutagent/logs/<ts>.log``，不挂 console handler。
    """
    from mutagent.runtime.log_store import LogStore, LogStoreHandler, SingleLineFormatter

    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(config.get("logging.log_dir", default=".mutagent/logs"))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    log_store = LogStore()
    mem_handler = LogStoreHandler(log_store)
    mem_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(mem_handler)

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

    logger.info("Pysandbox server logging initialized (session=%s)", session_ts)


def _build_sandbox(config: Any) -> Any:
    """构造 SandboxApp 并注入 mcp_sources / cli_sources，返回 sandbox。

    必须在 server 的 event loop 里调用（MCPConnection 捕获当前 loop）。
    """
    from mutagent.sandbox.app import SandboxApp
    from mutagent.sandbox._adapter_mcp import MCPConnection
    from mutagent.sandbox._adapter_cli import build_cli_namespace

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
        # 回引 sandbox，供 _do_rebuild 同步注册 peer namespaces。
        conn._sandbox = sandbox
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


async def _serve(config: Any, host: str, port: int) -> None:
    """启动纯沙箱 MCP server，阻塞监听直到收到取消信号。"""
    import mutagent
    from mutio.mcp.view import MCPView
    from mutio.net.server import Server
    from mutagent.sandbox.tools import PySandboxTools

    # 1. 构建 sandbox（必须在 server loop 里）
    sandbox = _build_sandbox(config)

    # 2. 注入到 PySandboxTools 类级单例
    PySandboxTools._app = sandbox

    # 3. 定义本进程独占的 MCPView，path=/mcp。
    #    PySandboxTools.path == "/mcp" 会自动挂到这个 view。
    class PySandboxMCPView(MCPView):
        path = "/mcp"
        name = "mutagent-pysandbox"
        version = mutagent.__version__

    # 4. 定义 Server 子类，限制 views 仅本进程的，避免与潜在第三方 View 冲突
    class PySandboxServer(Server):
        views = (PySandboxMCPView,)

    # 5. 监听 socket（统一 IPv4，避免 host="" 造成的 v4/v6 歧义）
    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listen_sock.bind((host, port))
    except OSError as exc:
        listen_sock.close()
        raise SystemExit(f"Error: cannot bind {host}:{port}: {exc}") from exc

    actual_host, actual_port = listen_sock.getsockname()[:2]
    url = f"http://{actual_host}:{actual_port}/mcp"

    server = PySandboxServer(host=actual_host, port=actual_port)
    print(f"mutagent sandbox server: {url}")
    logger.info("Starting pysandbox MCP server at %s", url)

    try:
        await server.start(listen=[listen_sock])
        # start() 立即返回，需要阻塞等待中断信号
        stop_evt = asyncio.Event()
        try:
            await stop_evt.wait()
        except asyncio.CancelledError:
            pass
    finally:
        try:
            await server.stop()
        except Exception:
            logger.exception("server.stop() failed")
        try:
            await sandbox.close()
        except Exception:
            logger.exception("sandbox.close() failed")
        try:
            listen_sock.close()
        except OSError:
            pass


def _dispatch_server(app: Any, args: argparse.Namespace) -> None:
    """server 模式入口（依赖 app.config）。"""
    _setup_pysandbox_logging(app.config)
    try:
        asyncio.run(_serve(app.config, args.host, args.port))
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# 总入口
# ---------------------------------------------------------------------------

def dispatch_pysandbox(app: Any, args: argparse.Namespace) -> None:
    """由 main() 调用。

    Server 模式：``app.config`` 已 load 完毕（main 入口加载）。
    Client 模式：``app.config`` **不依赖**——main 入口已为 client 跳过 load。
    """
    # 互斥/必填校验放在 dispatcher 里以便复用 parser.error 风格
    # （parser 已被 main 消费，这里手写报错走 sys.exit(2)）
    if args.serve:
        _dispatch_server(app, args)
    else:
        _dispatch_client(args)


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """供 main 在 parse 后立即调用的校验入口。"""
    _validate(parser, args)
