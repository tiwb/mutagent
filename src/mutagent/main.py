"""mutagent.main -- Entry point for assembling and running the agent."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from mutagent.agent import Agent
from mutagent.client import LLMClient
from mutagent.essential_tools import EssentialTools
from mutagent.runtime.impl_loader import ImplLoader
from mutagent.runtime.module_manager import ModuleManager
from mutagent.selector import ToolSelector


_builtins_loaded = False

_DEFAULT_MODEL = "claude-sonnet-4-20250514"
_DEFAULT_BASE_URL = "https://api.anthropic.com"


def load_builtins() -> None:
    """Load all builtin .impl.py files. Safe to call multiple times."""
    global _builtins_loaded
    if _builtins_loaded:
        return
    builtins_dir = Path(__file__).parent / "builtins"
    loader = ImplLoader()
    loader.load_all(builtins_dir, "mutagent.builtins")
    _builtins_loaded = True


def load_config() -> dict[str, str]:
    """Load configuration from mutagent.json (fallback) then environment variables (override).

    mutagent.json format (same as Claude Code):
        { "env": { "ANTHROPIC_AUTH_TOKEN": "...", ... } }

    Returns:
        Dict with keys: api_key, model, base_url.

    Raises:
        SystemExit: If no API key is found.
    """
    # 1. Read mutagent.json as fallback
    file_env: dict[str, str] = {}
    config_path = Path("mutagent.json")
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            file_env = data.get("env", {})
        except (json.JSONDecodeError, KeyError):
            pass

    def _get(var_name: str, default: str = "") -> str:
        """Env var > mutagent.json > default."""
        return os.environ.get(var_name) or file_env.get(var_name) or default

    api_key = _get("ANTHROPIC_AUTH_TOKEN")
    if not api_key:
        raise SystemExit(
            "Error: ANTHROPIC_AUTH_TOKEN not set.\n"
            "Set it via environment variable or in mutagent.json under env."
        )

    return {
        "api_key": api_key,
        "model": _get("ANTHROPIC_MODEL", _DEFAULT_MODEL),
        "base_url": _get("ANTHROPIC_BASE_URL", _DEFAULT_BASE_URL),
    }


def create_agent(
    api_key: str,
    model: str = _DEFAULT_MODEL,
    base_url: str = _DEFAULT_BASE_URL,
    system_prompt: str = "",
) -> Agent:
    """Create a fully assembled Agent with all components wired up.

    This loads all builtin .impl.py files (if not already loaded),
    creates the ModuleManager, EssentialTools, ToolSelector, LLMClient,
    and Agent.

    Args:
        api_key: Anthropic API key.
        model: Model identifier.
        base_url: API base URL.
        system_prompt: System prompt for the agent.

    Returns:
        A ready-to-use Agent instance.
    """
    load_builtins()

    # Create components
    module_manager = ModuleManager()
    tools = EssentialTools(module_manager=module_manager)
    selector = ToolSelector(essential_tools=tools)
    client = LLMClient(model=model, api_key=api_key, base_url=base_url)

    if not system_prompt:
        system_prompt = (
            "You are a Python AI Agent with the ability to inspect, modify, "
            "and run Python code at runtime. Use the available tools to help "
            "the user with their tasks."
        )

    agent = Agent(
        client=client,
        tool_selector=selector,
        system_prompt=system_prompt,
        messages=[],
    )

    return agent


async def run_agent(
    api_key: str,
    user_input: str,
    **kwargs,
) -> str:
    """Convenience function to create an agent and run it with user input.

    Args:
        api_key: Anthropic API key.
        user_input: The user's input message.
        **kwargs: Additional arguments passed to create_agent.

    Returns:
        The agent's response.
    """
    agent = create_agent(api_key=api_key, **kwargs)
    return await agent.run(user_input)
