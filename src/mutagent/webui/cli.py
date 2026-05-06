"""CLI helpers for the built-in WebUI."""

from __future__ import annotations

import argparse
from typing import Any


def add_webui_subcommand(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "webui",
        help="Start the built-in web UI (requires mutagent[webui])",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 = auto-select an available port")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser")
    return parser


def dispatch_webui(app: Any, args: argparse.Namespace) -> None:
    app.run_webui(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )
