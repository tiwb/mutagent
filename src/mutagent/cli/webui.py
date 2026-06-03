"""CLI helpers for the built-in WebUI."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import webbrowser
from typing import Any

from mutagent.app.app import App
from mutagent.app.log_store import SingleLineFormatter

logger = logging.getLogger(__name__)


def add_webui_subcommand(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "webui",
        help="Start the built-in web UI (requires mutagent[webui])",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 = auto-select an available port")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser")
    return parser


def _ensure_console_logging(level_name: str = "WARNING") -> None:
    """Attach a stdout handler for WebUI debugging when not already present."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if getattr(handler, "_mutagent_console_handler", False):
            return

    level = getattr(logging, level_name.upper(), logging.WARNING)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler._mutagent_console_handler = True  # type: ignore[attr-defined]
    console_handler.setLevel(level)
    console_handler.setFormatter(SingleLineFormatter(
        "%(asctime)s %(levelname)-8s %(name)s - %(message)s"
    ))
    root_logger.addHandler(console_handler)
    logger.info("Console logging enabled for WebUI (level=%s)", level_name)


def dispatch_webui(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """启动 WebUI server。"""
    try:
        from mutagent.webui import WebUIServer
    except ImportError as exc:
        raise SystemExit("需要先安装 WebUI 依赖：pip install mutagent[webui]") from exc

    app = App()
    app.load_config(args.config)
    app.setup_agent()
    console_level = str(app.config.get("logging.console_level", default="WARNING"))
    _ensure_console_logging(console_level)

    host = args.host
    port = args.port
    open_browser = not args.no_browser

    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listen_sock.bind((host, port))
    except OSError as exc:
        listen_sock.close()
        raise SystemExit(str(exc)) from exc

    actual_host, actual_port = listen_sock.getsockname()[:2]
    url = f"http://{actual_host}:{actual_port}/"
    os.environ["MUTAGENT_PORT"] = str(actual_port)
    server = WebUIServer(app=app, agent=app.agent, host=actual_host, port=actual_port)
    logger.info("Starting mutagent WebUI server at %s", url)

    print(f"mutagent webui: {url}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            logger.warning("Failed to open browser for %s", url, exc_info=True)

    try:
        server.run(listen=[listen_sock])
    finally:
        try:
            listen_sock.close()
        except OSError:
            pass
