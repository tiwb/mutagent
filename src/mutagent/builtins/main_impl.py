"""Default implementation for mutagent.main.App methods."""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import socket
import sys
import asyncio
import webbrowser
from pathlib import Path
from typing import Any

import mutagent
from mutagent.config import Config, ConfigChangeEvent, ChangeCallback, Disposable
from mutagent.agent import Agent
from mutagent.client import LLMClient
from mutagent.context import AgentContext
from mutagent.messages import Message, TextBlock
from mutagent.main import App
from mutagent.runtime.log_store import (
    LogStore, LogStoreHandler, SingleLineFormatter,
)
from mutagent.runtime.api_recorder import ApiRecorder
from mutagent.sandbox.app import SandboxApp
from mutagent.sandbox._adapter_mcp import MCPConnection
from mutagent.sandbox._adapter_cli import build_cli_namespace
from mutagent.sandbox.entry_agent import SandboxToolkit
from mutagent.tools import ToolSet
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
You are mutagent assistant.
- Help users with their tasks using your knowledge and available tools
- Always respond in the user's language
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
    from datetime import datetime

    spec = LLMProvider.resolve_model(self.config)
    if spec is None:
        raise SystemExit(
            "Error: no models configured.\n"
            "Run the setup wizard or add a 'providers' section to your config."
        )
    model = spec

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

    # 4. ToolLogCaptureHandler 不再安装：开关只能由已移除的 LogToolkit.query 设置，
    #    安装后无人激活。未来需要时重新设计暴露方式。

    logger.info("Logging initialized (session=%s)", session_ts)

    # --- API Recorder ---
    api_recorder = None
    if self.config.get("logging.api_record", default=True):
        log_dir.mkdir(parents=True, exist_ok=True)
        api_mode = self.config.get("logging.api_record_mode", default="incremental")
        api_recorder = ApiRecorder(log_dir, mode=api_mode, session_ts=session_ts)
        logger.info("API recorder started (mode=%s)", api_mode)

    # --- SandboxApp (空 registry, MCP/CLI 由 connect_sources 后续注入) ---
    self.sandbox = SandboxApp()

    # --- ToolSet: 唯一工具 SandboxToolkit ---
    tool_set = ToolSet()
    tool_set.add(SandboxToolkit(_app=self.sandbox, _state={}))

    client = _create_llm_client(model, api_recorder)

    # Record session metadata (现在只有 1 个 tool: pysandbox)
    if api_recorder is not None:
        effective_prompt = system_prompt or SYSTEM_PROMPT
        tool_schemas = tool_set.get_tools()
        api_recorder.start_session(
            model=client.model,
            system_prompt=effective_prompt,
            tools=[{"name": t.name, "description": t.description} for t in tool_schemas],
        )

    if not system_prompt:
        system_prompt = SYSTEM_PROMPT
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


@mutagent.impl(App.connect_sources)
async def connect_sources(self) -> None:
    """在 agent 将运行的 event loop 上连接 mcp_sources / cli_sources。

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

    main_loop = asyncio.get_running_loop()
    mcp_sources = self.config.get("mcp_sources", default={}) or {}
    for ns_name, server_cfg in mcp_sources.items():
        autostart = bool(server_cfg.get("autostart", True))
        retry_cooldown = float(server_cfg.get("retry_cooldown", 5.0))
        try:
            conn = MCPConnection(
                ns_name, server_cfg, main_loop,
                retry_cooldown=retry_cooldown)
        except Exception as e:
            # 连设置都走不通 — 跳过，但记个 log
            logger.warning("MCP source '%s' init failed: %s", ns_name, e)
            continue

        # namespace 全程注册，cleanup 绑 connection.close（而非单个 client，
        # 避免重连后 cleanup 调到旧 client）
        sandbox.add_namespace(conn.namespace, on_remove=conn.close)
        # 回引 sandbox，供 _do_rebuild 同步注册 peer namespaces。
        # 详见 feature-namespace-multi-provider.md。
        conn._sandbox = sandbox

        if autostart:
            # 后台连，不阻塞 setup；失败只 log，不影响其他 namespace
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

    cli_sources = self.config.get("cli_sources", default={}) or {}
    if cli_sources:
        cli_ns = build_cli_namespace(cli_sources)
        sandbox.add_namespace(cli_ns)
        logger.info("CLI namespace built (%d functions)", len(cli_ns._functions))


@mutagent.impl(App.run)
def run(self) -> None:
    self.setup_agent(system_prompt=SYSTEM_PROMPT)
    spec = LLMProvider.resolve_model(self.config)
    print(f"mutagent ready  (model: {spec.get('model_id', '?') if spec else '?'})")
    print("Type your message. 'exit' or Ctrl+C to quit.\n")

    import asyncio
    import queue
    import threading
    from mutagent.messages import StreamEvent
    from mutagent.cli.terminal import TerminalRenderer

    renderer = TerminalRenderer()

    # 启动 asyncio event loop 线程
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    # 在 loop 上连接 MCP/CLI sources
    try:
        asyncio.run_coroutine_threadsafe(
            self.connect_sources(), loop
        ).result(timeout=120)
    except Exception as e:
        logger.warning("connect_sources failed: %s", e)
        print(f"Warning: failed to connect sources: {e}", file=sys.stderr)

    # 订阅 agent 事件（线程安全的 queue.put）
    event_q: queue.Queue[StreamEvent] = queue.Queue()
    self.agent.subscribe(event_q.put)

    waiting_for_input = True  # True when blocked on read_input()

    while True:
        try:
            waiting_for_input = True
            user_input = renderer.read_input()
            waiting_for_input = False

            if not user_input:
                continue

            # exit / /exit 直接退出
            if user_input in ("exit", "/exit"):
                break

            # 提交 agent 任务到 asyncio 线程
            asyncio.run_coroutine_threadsafe(
                self.agent.submit(user_input), loop
            )

            # 主线程同步消费事件
            while True:
                try:
                    evt = event_q.get(timeout=0.2)
                except queue.Empty:
                    if self.agent.is_busy():
                        continue
                    break
                renderer.render_event(evt)
                if evt.type == "turn_done":
                    break

        except KeyboardInterrupt:
            if waiting_for_input:
                # 空输入时 Ctrl+C：询问是否退出
                if renderer.confirm_exit():
                    break
            else:
                # agent 运行中 Ctrl+C：取消当前任务
                self.agent.cancel()
                # 排空 event_q 中残留事件
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

    # 清理 sandbox + event loop
    sandbox = getattr(self, "sandbox", None)
    if sandbox is not None:
        try:
            asyncio.run_coroutine_threadsafe(sandbox.close(), loop).result(timeout=5)
        except Exception:
            pass
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
    # 曝光端口给子进程（pi agent / mutagent pysandbox client 可自动发现）
    os.environ["MUTAGENT_PORT"] = str(actual_port)
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
