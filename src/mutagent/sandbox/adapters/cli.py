"""CLI 白名单 — 配置允许的 CLI 命令，包装为命名空间函数。"""

import subprocess
from typing import Any

from mutagent.sandbox.namespace import Namespace


def build_cli_namespace(cli_config: dict[str, dict[str, Any]]) -> Namespace:
    """从配置构建 cli 命名空间。

    Args:
        cli_config: {func_name: {"command": ..., "args": [...]}}

    Returns:
        Namespace("cli") 包含所有白名单命令
    """
    ns = Namespace("cli")

    for func_name, cmd_config in cli_config.items():
        command = cmd_config.get("command", func_name)
        base_args = cmd_config.get("args", [])

        fn = _make_cli_func(func_name, command, base_args)
        ns.register(func_name, fn, f"CLI: {command}")

    return ns


def _make_cli_func(func_name: str, command: str,
                   base_args: list[str]) -> Any:
    """为一个 CLI 命令生成包装函数。"""

    def cli_func(*args: str) -> str:
        """执行 CLI 命令。参数按 CLI 风格传递。

        成功返回 stdout 字符串，失败抛出 RuntimeError（含 stderr）。
        """
        full_cmd = [command] + base_args + list(args)
        try:
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Command timed out: {func_name}")
        except FileNotFoundError:
            raise RuntimeError(f"Command not found: {command}")

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"Command failed (exit {result.returncode}): {error_msg}")

        return result.stdout

    cli_func.__name__ = func_name
    cli_func.__doc__ = f"Execute: {command} {' '.join(base_args)} [args...]"
    return cli_func
