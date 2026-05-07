# Pysandbox 单一工具 + CLI + MCP 配置集成

**状态**：✅ 已完成
**日期**：2026-05-06
**类型**：功能设计

## 需求

1. mutagent 的 Agent 默认只暴露一个 `pysandbox` 工具（与 mutbot 当前状态对齐）
2. mutagent 支持通过配置 `mcp_sources` / `cli_sources` 接入外部 MCP server 和 CLI 工具，通过 pysandbox 调用
3. mutagent CLI 提供 `pysandbox` 子命令（独立执行，不依赖 server）
4. CLI 子命令在 `--help` 中可见（吸取 mutbot 教训）
5. 移除 ModuleToolkit 和 AgentToolkit 在默认 Agent 中的注册（代码保留，未来重新设计）
6. 为后续 mutbot 删除 agent 相关代码、统一入口到 mutagent 做准备

## 关键参考

- `mutagent/toolkits/` — 现有 Toolkit 目录（module_toolkit, log_toolkit, web_toolkit, agent_toolkit）
- `mutagent/sandbox/app.py` — SandboxApp Declaration（MCP/CLI/NamespaceTools 聚合）
- `mutagent/sandbox/_app_impl.py` — SandboxApp 实现（MCP 桥接 + namespace 缓存）
- `mutagent/sandbox/tools.py` — PySandboxTools（MCPToolSet，MCP server 端暴露用，非 Agent tool）
- `mutagent/sandbox/_engine.py` — execute() 执行引擎
- `mutagent/builtins/main_impl.py` — setup_agent() 当前手动组装 ToolSet
- `mutagent/main.py` — App Declaration + CLI main() 
- `mutbot/builtins/pysandbox_toolkit.py` — mutbot 当前的 PySandboxToolkit（参考实现）
- `mutbot/runtime/session_manager.py` — build_default_agent()（mutbot 端 Agent 组装）
- `mutbot/cli/pysandbox.py` — mutbot 的 pysandbox CLI 子命令（参考实现，对比改进）
- `mutbot/__main__.py` — mutbot CLI 入口（if/elif 手动判断，反例）
- `mutagent/config.py` — Config Declaration

## 设计方案

### 总体思路

mutagent 从"多 tool 各司其职"切换到"单一 pysandbox 入口 + 沙箱内能力聚合"，与 mutbot 对齐。

核心变化：
```
之前: Agent → ModuleToolkit / LogToolkit / WebToolkit / AgentToolkit（4+ 个 tool schema）
之后: Agent → SandboxToolkit.pysandbox(code)（1 个 tool schema）
              └── SandboxApp（纯 sync）
                    ├── CLI whitelist（子进程 → namespace 函数）
                    └── NamespaceTools 自动发现（进程内能力）

MCP 管理在 SandboxApp 外部：
  setup_agent()（async）
    ├── SandboxApp()                   ← sync, 空 registry
    ├── await bridge_mcp_server(...)   ← async, 应用控制 loop
    ├── sandbox.add_namespace(ns)      ← sync
    └── tool_set.add(SandboxToolkit)   ← sync
```

### SandboxApp 纯 sync 化

当前 `SandboxApp` 把 MCP 连接、CLI 构建、loop 捕获全部耦合在 `setup()` 一个 async 方法里。
重构后剥离 MCP 管理，变成纯 sync 的 namespace registry + 执行引擎：

```python
class SandboxApp(mutagent.Declaration):
    """Python 沙箱 — namespace registry + 受限代码执行。

    不读 config，不管理 MCP 连接，不创建 event loop。
    namespace 由外部通过 add_namespace() 注入。
    """

    def add_namespace(self, ns: Namespace) -> None: ...
    def remove_namespace(self, name: str) -> None: ...
    def exec_code(self, code: str, state: dict | None = None) -> dict: ...
```

去掉的内容：
- `config` 属性 — 不再读 config
- `setup()` — MCP/CLI 初始化移到 setup_agent
- `reload()` / `shutdown()` — MCP 生命周期由外部管理
- `bridge_mcp_server()` 中的 `main_loop = asyncio.get_running_loop()` 捕获 — 由外部在 `setup_agent()` 的 async 上下文中自然发生

保留的内容：
- `exec_code()` → `execute()` 执行引擎
- NamespaceTools 自动发现（`_build_declaration_namespaces`）
- `_build_namespace_dict` 缓存机制
- `_adapter_mcp.py` 中 `bridge_mcp_server()` 工具函数（供外部调用）
- `_adapter_cli.py` 中 `build_cli_namespace()` 工具函数

### SandboxToolkit

**位置**：`mutagent/toolkits/sandbox_toolkit.py`（新建）

```python
class SandboxToolkit(Toolkit):
    """pysandbox — 安全的 Python 代码执行环境。

    所有能力（MCP server、CLI 工具、内置函数）通过命名空间注入沙箱，
    LLM 在 pysandbox 中编排多步逻辑，减少 tool call 往返。
    """

    _tool_prefix = ""
    _tool_methods = ["pysandbox"]

    _app: SandboxApp
    _state: dict[str, Any]  # 跨步骤共享的 REPL state

    async def pysandbox(self, code: str) -> str:
        """在沙箱中执行 Python 代码。..."""
```

**与 mutbot 的 PySandboxToolkit 对比**：
- 接口完全一致：`pysandbox(code: str) -> str`
- mutbot 版将被删除，统一用 mutagent 版
- SandboxApp 引用来源不同：mutbot 从 server 单例获取，mutagent 由 setup_agent() 创建

### setup_agent() 改造

**签名**：`def` → `async def`

setup_agent 成为唯一的"组装点"，负责创建 SandboxApp、连接 MCP、构建 CLI namespace、组装 Agent。

```python
async def setup_agent(self, system_prompt: str = "") -> Agent:
    # 1. LLMClient
    spec = LLMProvider.resolve_model(self.config)
    client = _create_llm_client(spec, api_recorder)

    # 2. UserIO + 日志基础设施
    ...

    # 3. SandboxApp（纯 sync 构造，无 MCP）
    sandbox = SandboxApp()

    # 4. MCP 连接（async，在调用方的 loop 上）
    for name, cfg in self.config.get("mcp_sources", {}).items():
        try:
            ns, client = await bridge_mcp_server(name, cfg)
            sandbox.add_namespace(ns)
        except Exception:
            logger.warning("MCP '%s' failed", name)

    # 5. CLI 命名空间（sync）
    cli_config = self.config.get("cli_sources", {})
    if cli_config:
        sandbox.add_namespace(build_cli_namespace(cli_config))

    # 6. 唯一工具：SandboxToolkit
    tool_set = ToolSet()
    tool_set.add(SandboxToolkit(_app=sandbox, _state={}))

    # 7. Agent
    agent = Agent(llm=client, tools=tool_set, context=..., config=self.config)
    tool_set.agent = agent
    return agent
```

**移除的代码**：
- ModuleManager 创建 + ModuleToolkit 添加
- LogToolkit 添加
- Sub-agents / AgentToolkit 整段
- API Recorder 的 session metadata 记录
- `SandboxApp(config=...)` + `await sandbox.setup()` → 改为 `SandboxApp()` + 外部注入 namespace

**保留的代码**：
- 日志基础设施（LogStore、FileHandler、MemoryHandler、ToolCaptureHandler）— 与 tool 无关
- API Recorder — 保留但去掉 tool schema 记录部分
- WebToolkit/ModuleToolkit/AgentToolkit 代码文件 — 不动，未来重新设计

**调用方适配**：
- `App.setup_agent()` Declaration 桩 — `def` → `async def`
- `App.run()` — 在创建 event loop 后，通过 `run_coroutine_threadsafe` 调用 async setup_agent
- `App.run_webui()` — 用 `asyncio.run()` 调用 async setup_agent（WebUIServer 有自己的 loop）

### CLI 改造

**当前问题**：`mutagent` CLI 入口在 `main.py:main()`，用 argparse 但子命令判断不完整。`mutbot` 更糟——`__main__.py` 用 if/elif 手动判断，`--help` 看不到子命令。

**改造目标**：

```bash
$ mutagent --help
usage: mutagent [-h] [-V] [--config CONFIG] [--headless]
                {pysandbox,webui} ...

mutagent — AI Agent Framework

positional arguments:
  {pysandbox,webui}
    pysandbox        在沙箱中执行 Python 代码
    webui            启动 Web UI 服务器

options:
  -h, --help         show this help message and exit
  -V, --version      show program's version number and exit
  --config CONFIG    Path to config file
  --headless         使用终端 REPL 界面
```

**pysandbox 子命令**：

```bash
$ mutagent pysandbox --help
usage: mutagent pysandbox [-h] [-c CODE] [script]

在沙箱中执行 Python 代码

positional arguments:
  script              脚本文件路径，或 - 从 stdin 读

options:
  -h, --help          show this help message and exit
  -c CODE             代码字符串（类似 python -c）
```

**使用示例**：
```bash
mutagent pysandbox -c "help()"                          # 单行代码
mutagent pysandbox -c "web.search(query='hello')"       # 调用 MCP/namespace
mutagent pysandbox script.py                            # 脚本文件
echo "help(web)" | mutagent pysandbox -                 # stdin
```

**实现要点**：
- pysandbox 子命令**独立执行**，不依赖 server 进程
- 从 config 创建 SandboxApp → 遍历 mcp_sources 调用 bridge_mcp_server → add_namespace → exec_code → 输出结果 → 退出
- 对齐 python CLI 约定：`-c CODE`、脚本文件、stdin 三种输入方式
- MSYS2 兼容：含 `/` 的正则/URL/路径等参数通过 stdin 或 `--config` 文件传入，避免 Git Bash 路径转换

### MCP 配置解析

**现状**：`bridge_mcp_server()` 和 `build_cli_namespace()` 在 `sandbox/` 中已实现，接受 dict 配置，返回 `Namespace`。不需改动。

**mutagent 需要做的**：setup_agent() 中从 config 读取 `mcp_sources` / `cli_sources`，调用上述函数，注入 SandboxApp。SandboxApp 自身不再读 config。

**配置格式**（与 mutbot 一致）：

```json
{
  "providers": { ... },
  "mcp_sources": {
    "serena": {
      "transport": "http",
      "url": "http://127.0.0.1:8800/mcp",
      "timeout": 60
    },
    "playwright": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@playwright/mcp@latest"],
      "shell": false
    }
  },
  "cli_sources": {
    "git": {
      "command": "git",
      "args": ["--no-pager"],
      "allow": ["status", "log", "diff"]
    }
  }
}
```

**与 mutbot 的关系**：mutbot 的 `mcp_sources` 配置和 mutagent 完全一致（SandboxApp 是同一个实现），迁移后配置无需变动。

### mutbot 迁移路径（后续步骤，不在本次实施范围）

本次只做 mutagent 侧，但以下迁移路径作为设计约束：

| mutbot 组件 | 未来命运 | 替代 |
|------------|---------|------|
| `mutbot/builtins/pysandbox_toolkit.py` | 删除 | `mutagent.toolkits.sandbox_toolkit.SandboxToolkit` |
| `mutbot/cli/pysandbox.py` | 删除 | `mutagent pysandbox` 子命令 |
| `mutbot/web/mcp.py` (MutBotMCP) | 删除 | MCP endpoint 不再需要（能力已通过 mutagent CLI 和 WebUI 覆盖） |
| `build_default_agent()` 中的 PySandboxToolkit | 改用 mutagent 版 | 只保留 mutbot 特有 tool（ConfigToolkit, UIToolkit） |
| `__main__.py` 的 pysandbox 分支 | 删除 | — |



## 设计决策记录

### setup_agent 改为 async

SandboxApp 剥离 MCP 管理后变成纯 sync 类（只含 registry + exec_code，无 async 方法）。
MCP 连接逻辑保留在 `setup_agent()` 中，调用 `await bridge_mcp_server(...)`。
- `run()` 中通过 `asyncio.run_coroutine_threadsafe` 在已有 loop 上调度
- `run_webui()` 中用 `asyncio.run(setup_agent(...))` 创建临时 loop 执行
- mutbot 的 `build_default_agent()` 也改为 async，在 server 的主 loop 上 await

这使得应用完全控制 MCP 连接所在的 event loop，未来可按需将 MCP 连接移到任意位置。

### SandboxApp 纯 sync 化

SandboxApp 不再接受 config，不再管理 MCP/CLI 生命周期。变为纯 namespace registry + 执行引擎：
- `SandboxApp()` — 构造空 registry
- `add_namespace(ns)` / `remove_namespace(name)` — 管理 registry
- `exec_code(code, state)` — 同步执行，可在任意线程调用
- 无 setup / reload / shutdown

MCP 连接和 CLI 构建全部移到 setup_agent() 中，由应用控制时机和 event loop。

## 已确认决策

### system prompt
照搬 mutbot 当前版本，后续自行调整：
```
You are mutagent assistant.
- Help users with their tasks using your knowledge and available tools
- Always respond in the user's language
```

具体使用范式（help() 自省、命名空间调用等）由 `SandboxToolkit.pysandbox` 的 docstring 承载，不放入 system prompt。

### ToolLogCaptureHandler 保留
保留捕获机制（ContextVar buffer + handler），但不通过 LogToolkit.query 暴露开关。
- 当前：保持 handler 安装，pysandbox 执行期间自动捕获日志到 tool output
- 未来：可能作为 pysandbox 参数（如 `pysandbox(code, capture_logs=True)`），或提供独立接口按 tool_call 索引查询日志
- 本次保证功能不丢失，不在本次暴露用户接口

### ModuleToolkit / AgentToolkit 代码保留
文件不动，只从 `setup_agent()` 的默认注册中移除。未来重新设计。

## 消费者场景

| 消费者 | 场景 | 依赖的输出 | 验收标准 |
|--------|------|-----------|---------|
| mutagent CLI | 用户通过 `mutagent pysandbox -c "code"` 执行代码 | SandboxApp 独立启动、代码执行结果 | 能连接 MCP、执行代码、输出结果 |
| mutagent Agent | LLM 通过 pysandbox tool 编排多步逻辑 | 单一 tool schema、命名空间注入 | Agent 只有 pysandbox 一个工具；help() 列出所有 namespace |
| mutbot（未来） | 创建 Agent 时复用 mutagent 的 SandboxToolkit | SandboxToolkit 可直接实例化 | mutbot 可以 `SandboxToolkit(_app=sandbox_app)` 替代自己的 PySandboxToolkit |

## 设计方案变更记录

与用户讨论后（2026-05-07）的关键调整。原设计内容上方保留，下面补充变更点和理由。

### 范围澄清：本次只做"解耦"，不做 MCP 动态化

MCP 配置驱动 + 自动发现的语义保持不变（启动时读 config，遍历 `mcp_sources`，逐个连接），
变化只是：这段逻辑从 `SandboxApp.setup()` 内嵌挪到外部 helper。

**未来方向**（不在本次实施范围）：在 sandbox 内动态 `mutbot.add_mcp(name, cfg)` / `mutbot.remove_mcp(name)`。
此能力天然由本次解耦后的形态支持——为此本次预留 `add_namespace(ns, on_remove=None)` 钩子参数。

### setup_agent 保持 sync，新增 async `connect_sources`（替代"setup_agent 改 async"）

原方案"setup_agent 改 async"的问题：
- 现有测试和子类覆写均假定 sync，async 化破坏面广
- `run_webui()` 用 `asyncio.run(setup_agent(...))` 会让 MCP client 失活
  （`bridge_mcp_server` 内部 `main_loop = get_running_loop()` 捕获临时 loop，
   `asyncio.run` 退出后 loop 关闭，后续 server 主 loop 上的调用全部失败）

**新方案**：拆成两步

```python
class App(mutagent.Declaration):
    def setup_agent(self, system_prompt: str = "") -> Agent:
        """同步：构造 Agent + 空 SandboxApp + SandboxToolkit + LLM。
        不连接 MCP/CLI。"""

    async def connect_sources(self) -> None:
        """异步：连接 mcp_sources / cli_sources 并注入 self.agent 的 sandbox。
        必须在'agent 将要跑的那个 event loop'里 await。"""
```

各 run 模式按需在合适 loop 上调用：
- `App.run()`：先建 loop_thread → sync `setup_agent` → `run_coroutine_threadsafe(connect_sources(), loop)` → REPL
- `App.run_webui()`：sync `setup_agent` → 在 `WebUIServer.on_startup` 内 `await app.connect_sources()`
- `mutagent pysandbox` CLI：不走 setup_agent，独立用 `build_sandbox` 在 `asyncio.run` 内全程完成

### `build_sandbox(config)` — 唯一构造入口

新增 async helper（位置：`mutagent/sandbox/build.py`），封装"读 config → 连 MCP → 注入 namespace"。
所有需要 SandboxApp 的入口（`connect_sources` / CLI / 未来 mutbot）都走这一个函数，避免代码漂移。

### SandboxApp `add_namespace` 增加 `on_remove` 钩子

```python
def add_namespace(
    self, ns: Namespace,
    on_remove: Callable[[], Any] | None = None,
) -> None: ...
```

`build_sandbox` 用 `add_namespace(ns, on_remove=client.close)` 把 MCP client 句柄
托管给 sandbox。调用方 shutdown 时只需 `await sandbox.close()` 即可统一清理，
不需要自己维护 client 列表。未来动态加 MCP 直接复用此机制。

### Cache invalidation 自动化

`add_namespace` / `remove_namespace` 自动调 `_invalidate_cache(self)`，
不再依赖 `setup` / `reload` 显式触发（这两个方法本来就被删了）。

### SandboxApp 提供 `async close()`

替代被删除的 `shutdown()`。批量调用所有注册的 `on_remove` 回调，清空 registry。多次调用幂等。

### SYSTEM_PROMPT 替换（不只是 setup_agent 改造）

`main_impl.SYSTEM_PROMPT` 当前 60+ 行教 LLM 用 `inspect/view_source/define/save/query`，
这些工具改造后全部不再注册。必须同步替换为简版（见上面"已确认决策"段），
否则 agent 会调用不存在的工具。

### ToolLogCaptureHandler 不安装（修订原"保留"决策）

原决策"保留 handler 安装"的问题：开关 `tool_capture_enabled` 只能由 `LogToolkit.query`
设置，LogToolkit 不再注册后无人能开 → 安装等于死代码。

**修订**：本次直接不安装。`agent_impl.py` 内的 `_get_tool_capture_enabled` 检查保留
（永远返回 False，无害）。未来需要时再设计独立的暴露方式。

### `agents` 配置静默忽略

兼容性不考虑（用户确认 mutbot 后续会全部删除/迁移）。
老 config 中的 `agents` 段不会触发任何行为，也不报警。

### mutbot 兼容性放弃

本次实施可能短暂破坏 mutbot 当前的 server 启动路径
（`SandboxApp(config=...).setup()` 不再存在）。
用户已确认 mutbot 相关代码后续会整体删除/迁移。

### WebUIServer 改造

`WebUIServer.on_startup` 中 await `app.connect_sources()`，确保 MCP client 绑定到 server 主 loop。
`WebUIServer.__init__` 签名保持不变（仍接收 app + agent），不破坏现有调用。

## 实施步骤清单

- [x] 更新设计文档：追加变更记录章节，状态从 📝 → 🔄
- [x] 重构 `sandbox/app.py` Declaration（删 setup/reload/shutdown，新增 add_namespace 带 on_remove、remove_namespace、close）
- [x] 重构 `sandbox/_app_impl.py` 实现（add/remove_namespace 自动 invalidate cache，close 调 on_remove）
- [x] 新建 `sandbox/build.py` 实现 `build_sandbox(config)` async helper + `populate_sandbox(sandbox, config)`
- [x] 新建 `sandbox/_format.py`（`format_exec_result` 从 `sandbox/tools.py` 剩出，避免连带加载 `PySandboxTools` 污染全局 MCPToolSet 发现）
- [x] 新建 `toolkits/sandbox_toolkit.py` 实现 `SandboxToolkit`
- [x] 修改 `main.py` Declaration：新增 `connect_sources()` async 方法 + `sandbox` 字段
- [x] 重构 `builtins/main_impl.py`：
  - [x] 替换 SYSTEM_PROMPT 为简版
  - [x] 重写 setup_agent：移除 ModuleToolkit/LogToolkit/AgentToolkit 注册，移除 sub-agents 段，移除 ToolLogCaptureHandler 安装，添加 SandboxToolkit
  - [x] 新增 connect_sources @impl（调 populate_sandbox 把 namespace 注入 self.sandbox）
  - [x] App.run 调用方式：sync setup_agent + `run_coroutine_threadsafe(connect_sources)` + 退出时 `sandbox.close()`
  - [x] App.run_webui 调用方式：sync setup_agent + 依赖 WebUIServer.on_startup 调 connect_sources
- [x] `webui/_server_impl.py` 添加 on_startup @impl 调 `app.connect_sources()`
- [x] 新建 `cli/__init__.py` + `cli/pysandbox.py` 子命令实现（独立 build_sandbox + exec_code + close + 退出）
- [x] 修改 `main.py:main()` 注册 pysandbox subcommand
- [x] 修改 `tests/test_e2e.py` fixture：手动 add ModuleToolkit + LogToolkit 让旧测试继续跑
- [x] 跑 pytest 验证：全部 790 测试通过，0 回归
- [x] 手动验证：`mutagent pysandbox -c "help()"` 输出 namespace 提示，`-c "1/0"` exit code 1 + stderr 输出
- [x] 手动验证：含 `mcp_sources` 的 config（HTTP 连 mutbot server endpoint）调 `mb.pysandbox(code="1+1")` 返回 2，完整 MCP 桥接路径走通

## 迭代 2：入口重命名 + Docstring 去重 + format_result 归属

**日期**：2026-05-07

### 问题

1. **位置混淆**：`toolkits/sandbox_toolkit.py` 和 `sandbox/tools.py` 是两个不同协议的门面（Agent 入口 vs MCP 入口），但位置分散、命名不清晰
2. **Docstring 重复**：两个 `pysandbox` 方法的 docstring 约 85% 相同，未来漂移风险高
3. **format_exec_result 游离**：`sandbox/_format.py` 独立模块，但它和 `SandboxApp.exec_code` 总是成对使用

### 决策

**入口文件命名**：`entry_agent.py` / `entry_mcp.py`（无 `_` 前缀，对外 import 使用）
- `sandbox/entry_agent.py` → `SandboxToolkit`（Agent 侧入口）
- `sandbox/entry_mcp.py` → `PySandboxTools`（MCP 侧入口）
- 废弃 `toolkits/sandbox_toolkit.py`（原位置删除）
- 废弃 `sandbox/tools.py`（重命名）

**共享 Docstring**：提取为 `sandbox/app.py` 模块常量 `PYSANDBOX_DOC`，两个入口各自拼接定制部分
- `SandboxToolkit` 追加：`Variables persist across calls in the same agent session.`
- `PySandboxTools` 追加：MCP 特有的使用说明

**format_exec_result → SandboxApp.format_result**：
- `app.py` Declaration 桩：`def format_result(self, result: dict) -> tuple[str, bool]: ...`
- `_app_impl.py` @impl：原逻辑移入
- 废弃 `sandbox/_format.py`
- 调用改为 `self._app.format_result(result)`

### 实施步骤清单

- [x] `sandbox/app.py`：添加 `PYSANDBOX_DOC` 常量 + `format_result()` Declaration 桩
- [x] `sandbox/_app_impl.py`：实现 `SandboxApp.format_result`（从 `_format.py` 移入）
- [x] 新建 `sandbox/entry_agent.py`（从 `toolkits/sandbox_toolkit.py` 移入，使用 `PYSANDBOX_DOC`）
- [x] `sandbox/tools.py` → `sandbox/entry_mcp.py`（重命名，使用 `PYSANDBOX_DOC`）
- [x] 删除 `toolkits/sandbox_toolkit.py`
- [x] 删除 `sandbox/_format.py`（保留向后兼容 shim）
- [x] 删除 `sandbox/build.py`，逻辑内联到 `main_impl.connect_sources` 和 `cli/pysandbox._run`
- [x] 更新所有 import 引用：`main_impl.py`、`cli/pysandbox.py`、`tests/test_e2e.py`
- [x] 跑 pytest 验证无回归（790 passed, 5 skipped）

### 已知问题

**`mutagent pysandbox` 独立执行语义不对**：当前实现是完全独立启动 SandboxApp + 执行 + 退出，不与 Agent session 交互。正确做法应像 `mutbot pysandbox` 一样连接到正在运行的 agent session 远程执行。后续迭代重构。
