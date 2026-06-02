"""Default implementation for mutagent.app.App methods."""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import sys
from pathlib import Path

import mutobj
from mutagent.app.config import Config
from mutagent.core.agent import Agent
from mutagent.core.context import AgentContext
from mutagent.core.messages import Message, TextBlock
from mutagent.app.app import App
from mutagent.app.log_store import (
    LogStore, LogStoreHandler, SingleLineFormatter,
)
from mutagent.sandbox import SandboxEnv, MCPConnection
from mutagent.sandbox.entry_agent import SandboxToolkit
from mutagent.core.tools import ToolSet
from mutagent.core.llm import LLMApiClient

logger = logging.getLogger(__name__)



SYSTEM_PROMPT = """\
You are mutagent assistant.
- Help users with their tasks using your knowledge and available tools
- Always respond in the user's language
"""


@mutobj.impl(App.load_config)
def app_load_config(self, config_path: str = ".mutagent/config.json") -> None:
    self.config = Config()
    self.config.load(config_path)

    # Set environment variables from config
    for key, value in self.config.get("env", default={}).items():
        os.environ[key] = value

    # Auto-register .mutagent/ directories to sys.path
    for mutagent_dir in [
        str(Path.home() / ".mutagent"),
        str(Path.cwd() / ".mutagent"),
    ]:
        if mutagent_dir not in sys.path:
            sys.path.insert(0, mutagent_dir)

    # Extend sys.path from config
    for p_str in self.config.get("path", default=[]):
        if p_str not in sys.path:
            sys.path.insert(0, p_str)

    # Load extension modules
    for module_name in self.config.get("modules", default=[]):
        importlib.import_module(module_name)


@mutobj.impl(App.setup_agent)
def app_setup_agent(self, system_prompt: str = "") -> Agent:
    from datetime import datetime

    spec = self.config.resolve_model()
    if spec is None:
        raise SystemExit(
            "Error: no models configured.\n"
            "Run the setup wizard or add a 'providers' section to your config."
        )

    # --- Logging setup ---
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(self.config.get("logging.log_dir", default=".mutagent/logs"))

    # 1. Create LogStore (in-memory, no capacity limit)
    log_store = LogStore()

    # 2. Configure Python logging — 用 root logger 捕获所有库的日志
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Memory handler — message only (timestamp stored in LogEntry.timestamp)
    mem_handler = LogStoreHandler(log_store)
    mem_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(mem_handler)

    # 3. File handler (default on)
    if self.config.get("logging.file_log", default=True):
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"{session_ts}.log", encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(SingleLineFormatter(
            "%(asctime)s %(levelname)-8s %(name)s - %(message)s"
        ))
        root_logger.addHandler(file_handler)

    logger.info("Logging initialized (session=%s)", session_ts)

    # --- SandboxEnv (空 registry, MCP/CLI 由 connect_sources 后续注入) ---
    self.sandbox = SandboxEnv()

    # --- ToolSet: 唯一工具 SandboxToolkit ---
    tool_set = ToolSet()
    tool_set.add(SandboxToolkit(_env=self.sandbox, _state={}))

    provider = LLMApiClient.from_spec(spec)

    if not system_prompt:
        system_prompt = SYSTEM_PROMPT
    context = AgentContext()
    context.prompts.append(
        Message(role="system", blocks=[TextBlock(text=system_prompt)], label="base")
    )
    self.agent = Agent(
        llm=provider,
        model=provider.model_id,
        tools=tool_set,
        context=context,
    )
    tool_set.agent = self.agent
    return self.agent


@mutobj.impl(App.connect_sources)
async def app_connect_sources(self) -> None:
    """在 agent 将运行的 event loop 上连接 mcp_sources

    MCP 连接采用「长生命周期代理 + 懒连 + 自动重连」模型：

    - 为每个 mcp source 创建 :class:`MCPConnection` 并常驻注册 namespace；
      连接失败不会丢 namespace，下次调用会重试。
    - ``autostart=true``（默认）：启动后开后台任务异步连，不阻塞 setup。
    - ``autostart=false``：完全 lazy，首次访问 namespace 成员时才连。
    - ``retry_cooldown``（默认 5s，0 禁用）：失败后冷却期内不重试。
    """
    sandbox = getattr(self, "sandbox", None)
    if sandbox is None:
        logger.warning("connect_sources called before setup_agent; skipping")
        return

    mcp_sources = self.config.get("mcp_sources", default={}) or {}
    for ns_name, server_cfg in mcp_sources.items():
        autostart = bool(server_cfg.get("autostart", True))
        try:
            conn = MCPConnection(ns_name, server_cfg)
        except Exception as e:
            logger.warning("MCP source '%s' init failed: %s", ns_name, e)
            continue

        sandbox.connect_source(conn)

        if autostart:
            async def _bg_connect(c: MCPConnection = conn,
                                  n: str = ns_name) -> None:
                try:
                    await c.ensure_connected()
                    logger.info("MCP source '%s' connected (%d functions)",
                                n, len(c.namespace._functions))
                except Exception as exc:
                    logger.warning(
                        "MCP source '%s' autostart failed: %s", n, exc)
            asyncio.create_task(_bg_connect())
        else:
            logger.info(
                "MCP source '%s' registered (lazy, autostart=false)", ns_name)
