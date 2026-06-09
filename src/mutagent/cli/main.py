from __future__ import annotations

import mutagent


def main() -> None:
    """Bootstrap mutagent.  Not overridable.
    """
    import argparse
    from mutagent.cli.terminal import add_terminal_subcommand, dispatch_terminal
    from mutagent.cli.webui import add_webui_subcommand, dispatch_webui
    from mutagent.cli.pysandbox import add_pysandbox_subcommand, dispatch_pysandbox

    parser = argparse.ArgumentParser(description="mutagent — AI Agent Framework")
    parser.add_argument("-V", "--version", action="version", version=f"mutagent {mutagent.__version__}")
    parser.add_argument("--config", default=".mutagent/config.json",
                        help="Path to config file (default: .mutagent/config.json)")
    subparsers = parser.add_subparsers(dest="command")
    add_terminal_subcommand(subparsers)
    add_webui_subcommand(subparsers)
    add_pysandbox_subcommand(subparsers)
    args = parser.parse_args()

    # webui / pysandbox 子命令
    if args.command == "webui":
        dispatch_webui(parser, args)
        return

    if args.command == "pysandbox":
        dispatch_pysandbox(parser, args)
        return

    if args.command == "terminal":
        dispatch_terminal(parser, args)
        return

    parser.print_help()