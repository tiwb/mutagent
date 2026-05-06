"""Default implementation for mutagent.main.App methods."""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import socket
import sys
import webbrowser
from pathlib import Path
from typing import Any

import mutagent
from mutagent.config import Config, ConfigChangeEvent, ChangeCallback, Disposable
from mutagent.agent import Agent
from mutagent.toolkits.agent_toolkit import AgentToolkit
from mutagent.client import LLMClient
from mutagent.context import AgentContext
from mutagent.messages import Message, TextBlock
from mutagent.toolkits.log_toolkit import LogToolkit
from mutagent.main import App
from mutagent.toolkits.module_toolkit import ModuleToolkit
from mutagent.runtime.module_manager import ModuleManager
from mutagent.runtime.log_store import (
    LogStore, LogStoreHandler, SingleLineFormatter, ToolLogCaptureHandler,
)
from mutagent.runtime.api_recorder import ApiRecorder
from mutagent.tools import ToolSet
from mutagent.userio import UserIO
from mutagent.provider import LLMProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DictConfig — mutagent CLI 的 Config 实现
# ---------------------------------------------------------------------------

def _expand_env(value: Any) -> Any:
    """递归展开配置值中的环境变量引用。"""
    if isinstance(value, str):
        return re.sub(
            r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)',
            lambda m: os.environ.get(m.group(1) or m.group(2), m.group(0)),
            value,
        )
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _resolve_paths_inplace(data: dict, config_dir: Path) -> None:
    """将 data 中的相对 path 条目解析为绝对路径。"""
    raw_paths = data.get("path")
    if not isinstance(raw_paths, list):
        return
    resolved: list[str] = []
    for p in raw_paths:
        pp = Path(p)
        if not pp.is_absolute():
            pp = (config_dir / pp).resolve()
        resolved.append(str(pp))
    data["path"] = resolved


class DictConfig(Config):
    """单 dict 配置。mutagent CLI 用。"""

    _data: dict
    _listeners: list  # list[tuple[str, ChangeCallback]]

    def get(self, name: str, *, default: Any = None) -> Any:
        """点分路径导航 _data，递归展开环境变量。"""
        node = self._data
        for key in name.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return _expand_env(node)

    def set(self, name: str, value: Any, *, source: str = "") -> None:
        """按点分路径写入 _data，触发匹配的 on_change 回调。"""
        node = self._data
        keys = name.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
        event = ConfigChangeEvent(key=name, source=source, config=self)
        for pattern, cb in self._listeners:
            if self.affects(pattern, name):
                cb(event)

    def on_change(self, pattern: str, callback: ChangeCallback) -> Disposable:
        """注册监听。返回 Disposable 用于取消。"""
        entry = (pattern, callback)
        self._listeners.append(entry)
        def dispose() -> None:
            self._listeners.remove(entry)
        return Disposable(dispose=dispose)


def _create_llm_client(
    spec: dict, api_recorder: ApiRecorder | None = None
) -> LLMClient:
    """从模型 spec 创建 LLMClient。

    使用 resolve_class 按 provider 配置自动加载 LLMProvider 子类，
    缺省 provider 为 AnthropicProvider（向后兼容）。
    """
    import mutobj

    # 确保内置 provider 已注册
    import mutagent.builtins.anthropic_provider  # noqa: F401
    import mutagent.builtins.openai_provider  # noqa: F401

    provider_path = spec.get("provider", "AnthropicProvider")
    provider_cls = mutobj.resolve_class(provider_path, base_cls=LLMProvider)
    provider = provider_cls.from_spec(spec)

    return LLMClient(
        provider=provider,
        model=spec.get("model_id", ""),
        api_recorder=api_recorder,
    )


def _ensure_console_logging(config: Config) -> None:
    """Attach a stdout handler for WebUI debugging when not already present."""
    root_logger = logging.getLogger("mutagent")
    for handler in root_logger.handlers:
        if getattr(handler, "_mutagent_console_handler", False):
            return

    level_name = str(config.get("logging.console_level", default="INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler._mutagent_console_handler = True  # type: ignore[attr-defined]
    console_handler.setLevel(level)
    console_handler.setFormatter(SingleLineFormatter(
        "%(asctime)s %(levelname)-8s %(name)s - %(message)s"
    ))
    root_logger.addHandler(console_handler)
    logger.info("Console logging enabled for WebUI (level=%s)", level_name)


SYSTEM_PROMPT = """\
You are **mutagent**, a self-evolving Python AI Agent framework.

## Identity
You are built on the mutobj declaration-implementation separation pattern. \
Your own source code is organized as declarations (.py) with implementations (_impl.py), \
and you can inspect, modify, and hot-reload any of it at runtime — including yourself.

## Core Tools
- **inspect(module_path, depth)** — Browse module structure. Call with no arguments to see unsaved modules.
- **view_source(target)** — Read source code of any module, class, or function.
- **define(module_path, source)** — Define or redefine a Python module in memory (not persisted until saved).
- **save(module_path, level)** — Persist a module to disk. level="project" (default, ./.mutagent/) or "user" (~/.mutagent/).
- **query(pattern, level, limit, tool_capture)** — Search logs or configure logging. \
Use tool_capture="on" to include logs in tool output for debugging.

## Tool Development
To create a new tool, define a Toolkit subclass with `define_module`:

    define_module("my_tools", \"\"\"
    import mutagent

    class MyTools(mutagent.Toolkit):
        def my_tool(self, arg: str, count: int = 1) -> str:
            '''Tool description.

            Args:
                arg: Argument description.
                count: How many times. Default 1.

            Returns:
                Result description.
            '''
            return arg * count
    \"\"\")

The tool is **automatically available** after define — no registration needed. \
Test it by calling it directly. If the result is wrong, redefine the module — changes take effect immediately. \
Once validated, use save to persist.

Rules:
- Every tool method MUST have type annotations and a Google-style docstring with Args section.
- The docstring is shown to you as the tool description — write it clearly.
- Test with diverse inputs before saving.
- Keep one Toolkit class per module for clarity.
- To create test functions, define them as methods on a Toolkit subclass (e.g. test_my_tool).

## Workflow
When modifying code, follow this cycle:
1. **inspect** — Understand the current structure
2. **view_source** — Read the specific code to change
3. **define** — Apply changes in runtime (module is in memory only)
4. **inspect** — Verify the module structure is correct
5. **view_source** — Verify the code was applied as expected
6. **save** — Persist to disk once validated

Do NOT create throwaway test modules. Validate changes by inspecting and viewing source.

## Module Naming
- New modules should use **functional names** based on their purpose (e.g. "web_search", "file_utils", "math_tools").
- Do NOT place new modules under the "mutagent" namespace. The mutagent namespace is for the framework itself.
- NEVER redefine existing mutagent.* modules with define — this replaces the entire module. \
To change a specific behavior, create a new _impl module and use @impl to override just that method.
- Modules are saved to .mutagent/ directories which are automatically in sys.path.

## Key Concepts
- **Declaration (.py)** = stable interface (class + stub methods). Safe to import.
- **Implementation (_impl.py)** = replaceable logic via @impl. Loaded at startup via direct import.
- **define_module = write + restart**: defining a module completely replaces its namespace.
- **DeclarationMeta**: classes that inherit mutagent.Declaration are updated in-place on redefinition (id preserved, isinstance works, @impl survives).
- **Toolkit**: classes that inherit mutagent.Toolkit have their public methods auto-discovered as tools.
- **Module path is first-class**: everything is addressed as `package.module.Class.method`.
- **Namespace packages**: submodules of the same package can live in different .mutagent/ directories (project-level and user-level).

## Debugging
- All internal logs (DEBUG level) are captured in memory.
- Use query() to view recent activity or search for specific events.
- Use query(tool_capture="on") to attach logs to tool results — useful for diagnosing issues.
- API calls are automatically recorded to .mutagent/logs/ for session replay.

## Self-Evolution
You can evolve yourself:
- Override any existing tool implementation: create a NEW module (e.g. "my_agent_impl") with @impl(Agent.run), \
then define + save. Do NOT redefine mutagent.agent or mutagent.builtins.* — later @impl registrations auto-override.
- Create entirely new tool classes: define a new mutagent.Toolkit subclass — its methods become tools automatically.
- Extend ToolSet: add new tools to the Agent's tool set.

## Task Discipline
- Complete the CORE task first. Do NOT create additional versions, documentation modules, \
demos, guides, or summaries unless explicitly requested.
- After completing the core implementation, STOP and report results to the user. \
Let the user decide if further work is needed.
- define is for CODE only. Do NOT use it to create documentation, READMEs, \
guides, or text content. If the user needs documentation, describe it in your response text.
- Keep module names lowercase_with_underscores. Do NOT use ALL_CAPS module names.
- Before calling define, carefully review your source code for:
  - Indentation errors (Python is whitespace-sensitive)
  - Full-width characters in code (use ASCII punctuation only)
  - Import errors (verify the library is available)

## Guidelines
- When redefining declarations, remember DeclarationMeta preserves class identity.
- When redefining implementations, the old @impl is automatically unregistered.
- Use Chinese or English based on the user's language.
"""

@mutagent.impl(App.load_config)
def load_config(self, config_path: str = ".mutagent/config.json") -> None:
    p = Path(config_path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    # 项目级配置不存在时 fallback 到用户级 ~/.mutagent/config.json
    if not p.exists():
        user_p = Path.home() / ".mutagent" / "config.json"
        if user_p.exists():
            p = user_p
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
        _resolve_paths_inplace(data, p.parent)
    else:
        data = {}
    self.config = DictConfig(_data=data, _listeners=[])
    self.config_path = p

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


@mutagent.impl(App.setup_agent)
def setup_agent(self, system_prompt: str = "") -> Agent:
    from pathlib import Path
    from datetime import datetime

    spec = LLMProvider.resolve_model(self.config)
    if spec is None:
        raise SystemExit(
            "Error: no models configured.\n"
            "Run the setup wizard or add a 'providers' section to your config."
        )
    model = spec

    # --- UserIO setup ---
    import mutagent.builtins.block_handlers  # noqa: F401  -- register BlockHandler subclasses
    from mutagent.builtins.userio_impl import discover_block_handlers
    self.userio = UserIO(block_handlers=discover_block_handlers())

    # --- Logging setup ---
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(self.config.get("logging.log_dir", default=".mutagent/logs"))

    # 1. Create LogStore (in-memory, no capacity limit)
    log_store = LogStore()

    # 2. Configure Python logging
    root_logger = logging.getLogger("mutagent")
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

    # 4. Tool log capture handler (always installed, activated via flag)
    capture_handler = ToolLogCaptureHandler()
    capture_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s - %(message)s"
    ))
    root_logger.addHandler(capture_handler)

    logger.info("Logging initialized (session=%s)", session_ts)

    # --- API Recorder ---
    api_recorder = None
    if self.config.get("logging.api_record", default=True):
        log_dir.mkdir(parents=True, exist_ok=True)
        api_mode = self.config.get("logging.api_record_mode", default="incremental")
        api_recorder = ApiRecorder(log_dir, mode=api_mode, session_ts=session_ts)
        logger.info("API recorder started (mode=%s)", api_mode)

    # --- Components ---
    search_dirs: list[str | Path] = [
        Path.home() / ".mutagent",
        Path.cwd() / ".mutagent",
    ]
    module_manager = ModuleManager(search_dirs=search_dirs)
    module_tools = ModuleToolkit(module_manager=module_manager)
    log_tools = LogToolkit(log_store=log_store)
    tool_set = ToolSet(auto_discover=True)
    tool_set.add(module_tools)
    tool_set.add(log_tools)
    client = _create_llm_client(model, api_recorder)

    # --- Sub-Agents & AgentToolkit ---
    agents_config = self.config.get("agents", default={})
    if agents_config:
        sub_agents = {}
        for agent_name, agent_conf in agents_config.items():
            # Create sub-agent ToolSet with specified tools
            sub_tool_set = ToolSet()
            sub_tool_methods = agent_conf.get("tools", [])
            if sub_tool_methods:
                sub_tool_set.add(module_tools, methods=sub_tool_methods)
                # Add log tools if query is requested
                if "query" in sub_tool_methods:
                    sub_tool_set.add(log_tools, methods=["query"])
            else:
                sub_tool_set.add(module_tools)
                sub_tool_set.add(log_tools)

            # Use specified model or share the main client
            sub_model_name = agent_conf.get("model")
            if sub_model_name:
                sub_spec = LLMProvider.resolve_model(self.config, sub_model_name)
                if sub_spec is None:
                    logger.warning("Sub-agent '%s': model '%s' not found, using main client", agent_name, sub_model_name)
                    sub_client = client
                else:
                    sub_client = _create_llm_client(sub_spec, api_recorder)
            else:
                sub_client = client

            sub_prompt = agent_conf.get("system_prompt", f"You are a sub-agent named '{agent_name}'.")
            sub_context = AgentContext()
            sub_context.prompts.append(
                Message(role="system", blocks=[TextBlock(text=sub_prompt)], label="base")
            )
            sub_agent = Agent(
                llm=sub_client,
                tools=sub_tool_set,
                context=sub_context,
                config=self.config,
            )
            sub_tool_set.agent = sub_agent
            sub_agents[agent_name] = sub_agent
            logger.info("Sub-agent '%s' created (tools=%s)", agent_name,
                        [t.name for t in sub_tool_set.get_tools()])

        agent_toolkit = AgentToolkit(agents=sub_agents)
        tool_set.add(agent_toolkit, methods=["delegate"])
        logger.info("AgentToolkit registered with %d sub-agents", len(sub_agents))

    # Record session metadata
    if api_recorder is not None:
        effective_prompt = system_prompt or SYSTEM_PROMPT
        tool_schemas = tool_set.get_tools()
        api_recorder.start_session(
            model=client.model,
            system_prompt=effective_prompt,
            tools=[{"name": t.name, "description": t.description} for t in tool_schemas],
        )

    if not system_prompt:
        system_prompt = (
            "You are a Python AI Agent with the ability to inspect, modify, "
            "and run Python code at runtime. Use the available tools to help "
            "the user with their tasks."
        )
    context = AgentContext()
    context.prompts.append(
        Message(role="system", blocks=[TextBlock(text=system_prompt)], label="base")
    )
    if client.context_window:
        context.context_window = client.context_window
    self.agent = Agent(
        llm=client,
        tools=tool_set,
        context=context,
        config=self.config,
    )
    tool_set.agent = self.agent
    return self.agent


@mutagent.impl(App.run)
def run(self) -> None:
    self.setup_agent(system_prompt=SYSTEM_PROMPT)
    spec = LLMProvider.resolve_model(self.config)
    print(f"mutagent ready  (model: {spec.get('model_id', '?') if spec else '?'})")
    print("Type your message. 'exit' or Ctrl+C to quit.\n")

    import asyncio
    import concurrent.futures
    import queue
    import threading
    from mutagent.messages import StreamEvent, Message, TextBlock
    from mutagent.messages import TurnStartBlock
    from uuid import uuid4

    # 启动 asyncio event loop 线程
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    event_q: queue.Queue[StreamEvent | None] = queue.Queue()
    future: concurrent.futures.Future[None] | None = None

    waiting_for_input = True  # True when blocked on read_input()

    while True:
        try:
            waiting_for_input = True
            user_input = self.userio.read_input()
            waiting_for_input = False

            if not user_input:
                continue

            # exit / /exit 直接退出
            if user_input in ("exit", "/exit"):
                break

            # 构造 async input source（单条消息）
            async def single_input(text=user_input):
                yield Message(
                    role="user",
                    blocks=[TurnStartBlock(turn_id=uuid4().hex[:12]), TextBlock(text=text)],
                )

            # 提交 agent 任务到 asyncio 线程
            async def run_agent(input_gen=single_input()):
                async for event in self.agent.run(input_gen):
                    event_q.put(event)
                event_q.put(None)  # sentinel

            future = asyncio.run_coroutine_threadsafe(run_agent(), loop)

            # 主线程同步消费事件
            for event in iter(event_q.get, None):
                self.userio.render_event(event)

        except KeyboardInterrupt:
            if waiting_for_input:
                # 空输入时 Ctrl+C：询问是否退出
                if self.userio.confirm_exit():
                    break
            else:
                # agent 运行中 Ctrl+C：取消当前任务
                if future is not None and not future.done():
                    future.cancel()
                    # 排空 event_q
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

    # 清理 event loop：取消运行中的任务 → 关闭异步生成器 → 停止循环
    if future is not None and not future.done():
        future.cancel()
    try:
        asyncio.run_coroutine_threadsafe(loop.shutdown_asyncgens(), loop).result(timeout=2)
    except Exception:
        pass
    loop.call_soon_threadsafe(loop.stop)
    loop_thread.join(timeout=2)


@mutagent.impl(App.run_webui)
def run_webui(
    self,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
) -> None:
    try:
        from mutagent.webui.server import WebUIServer
    except ImportError as exc:
        raise SystemExit("需要先安装 WebUI 依赖：pip install mutagent[webui]") from exc

    self.setup_agent(system_prompt=SYSTEM_PROMPT)
    _ensure_console_logging(self.config)

    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listen_sock.bind((host, port))
    except OSError as exc:
        listen_sock.close()
        raise SystemExit(str(exc)) from exc

    actual_host, actual_port = listen_sock.getsockname()[:2]
    url = f"http://{actual_host}:{actual_port}/"
    server = WebUIServer(app=self, agent=self.agent, host=actual_host, port=actual_port)
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
