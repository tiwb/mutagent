# pysandbox CLI 重构 — 通用 MCP 客户端 + 纯沙箱服务

**状态**：✅ 已完成
**日期**：2026-05-09
**类型**：功能设计

## 需求

1. `mutagent pysandbox` 默认走 MCP 客户端模式，连到活 server 执行沙箱代码，适配 agent 一次性脚本调用场景
2. 作为通用 pysandbox MCP 客户端，可连接 mutagent、mutbot 或任何实现 pysandbox 协议的 server
3. 新增 `--serve` 模式：启动纯 MCP 沙箱服务，不启动 agent/WebUI
4. CLI 参数与 `mutagent webui` 对齐（`--host`/`--port` 命名一致）

## 关键参考

- `mutagent/src/mutagent/cli/pysandbox.py` — 当前独立 SandboxApp 实现
- `mutagent/src/mutagent/webui/cli.py` — webui 子命令参数（对齐目标）
- `mutagent/src/mutagent/sandbox/entry_mcp.py` — PySandboxTools（MCP tool 注册）
- `mutagent/src/mutagent/sandbox/_adapter_mcp.py` — MCP connection/client 实现
- `mutbot/src/mutbot/cli/pysandbox.py` — mutbot 端 client 模式实现（参考）
- `mutagent/src/mutagent/main.py` — 子命令注册入口

## 消费者场景

| 消费者 | 场景 | 依赖的输出 | 验收标准 |
|--------|------|-----------|---------|
| Agent / Claude Code 调用 | 一次性脚本：`mutagent pysandbox --port P -c "..."` 调到活 server | stdout 仅包含 sandbox 结果，exit code 表达成功/错误 | client 静默无日志污染；server 不可达时给出可操作的错误提示 |
| 用户/脚本启动纯沙箱服务 | `mutagent pysandbox --serve --port P` 起一个 MCP-only 服务给外部 agent 用 | `http://{host}:{port}/mcp` 暴露 `pysandbox` tool | 启动横幅可见 URL；config 中 `mcp_sources`/`cli_sources` 自动注入 namespace |
| REPL 用户 | `mutagent pysandbox --port P`（无代码源 + tty）连到活 server 进入交互 REPL | 逐条执行并显示结果 | 可连 `mutagent webui`、`mutagent --serve` 或 `mutbot` |
| 旧独立模式用户 | 原 `mutagent pysandbox -c "..."` 命令 | 明确的迁移提示 | 在「server 不可达」错误信息中给出 `--serve` 启动指引 |

## 设计方案

### 两种运行模式

| 模式 | 触发 | 行为 | 生命周期 |
|------|------|------|---------|
| **Client**（默认） | 不加 `--serve` | 连远程 MCP server → 调 `pysandbox` tool → 输出结果 | 执行完退出 |
| **Client REPL** | 不加 `--serve` + 无代码源 + tty | 连远程 MCP server → 进入交互 REPL，逐条通过 MCP 发送代码 | 持续到 Ctrl-D 退出 |
| **Server** | `--serve` | 启动 HTTP MCP server → 暴露 `pysandbox` tool | 常驻，直到信号中断 |

Client 模式提供两种子模式：有代码源时一次性执行（适配 agent 脚本调用），无代码源且终端交互时进入 REPL（适配人工调试）。两种子模式复用同一条 MCP 连接建立/关闭逻辑。

Server 模式提供纯沙箱服务——没有 agent loop、没有 WebUI，只是一个把 SandboxApp 暴露为 MCP endpoint 的常驻进程。

### 独立模式废弃

旧 `mutagent pysandbox` 是"独立进程"——不依赖任何 server，自建 SandboxApp、连本机 `mcp_sources` / `cli_sources`、执行后退出。本次重构**彻底删除独立模式**（用户决定），原因：

- 一次性脱机执行场景可用 `--serve` + 另开 client 进程，或直接用 `python -c` + sandbox 库
- 保留独立模式会让"默认行为"在 "有 server" 和 "没 server" 两种环境下不一致，违反最小惊讶
- mutbot CLI 已经验证 client-only 模型可用

**迁移指引**（写入 client 模式连不上 server 的错误信息中）：旧的 `mutagent pysandbox -c "..."` 用户应改为先 `mutagent pysandbox --serve --port PORT &` 启动一个 server，或连到已运行的 mutbot/webui server。

**REPL 保留**：交互 REPL 模式保留，但实现从本地 SandboxApp 改为通过 MCP client 连接远程 server。`SandboxConsole` 替换为 `_REPLConsole`（内部类），逐条通过 `client.call_tool("pysandbox")` 发送代码，复用同一条 MCP 连接。

### CLI 参数

```
mutagent pysandbox [--host HOST] --port PORT [(-c CODE | script | -)]
mutagent pysandbox --url URL [(-c CODE | script | -)]
mutagent pysandbox --serve [--host HOST] --port PORT
```

#### Client 模式参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `127.0.0.1` | 目标 server 地址（与 webui `--host` 一致） |
| `--port` | **必填，无默认值** | 目标 server 端口。除非指定 `--url`（URL 中已含端口）|
| `--url` | 无 | 完整 MCP endpoint URL。设置后忽略 `--host`/`--port` |
| `-c CODE` | — | 代码字符串 |
| `script` | — | 脚本文件路径 |
| `-` | — | 从 stdin 读 |
| 无代码 + tty | — | 进入交互 REPL |

最终 URL 逻辑：`--url` 存在 → 直接使用；否则构造 `http://{host}:{port}/mcp`。

#### Server 模式参数（`--serve`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `127.0.0.1` | 绑定地址（与 webui `--host` 一致） |
| `--port` | **必填，无默认值** | 绑定端口。不指定时报错，避免随机端口带来的发现困难 |
| `-c CODE` | 无 | 可选：server 启动后立即执行一次代码，然后继续监听 |

`--serve` 不接受 stdin/文件/管道。启动后打印 `mutagent sandbox server: http://{host}:{port}/mcp`。

#### `--serve` 不接受代码输入

`--serve` 与 `-c` / 脚本文件 / stdin **互斥**——纯 server，不带初始化脚本（避免 stdout 与后续日志混杂、语义模糊）。如未来真有「启动时自检」需求，再单独加 `--init-script`。

### 错误信息

#### Client 模式 — server 连不上

```
Error: No pysandbox server at http://{host}:{port}/mcp

Start a server:
  mutagent pysandbox --serve --port PORT      # sandbox-only MCP server

Or point to a different server:
  mutagent pysandbox --port PORT ...
  mutagent pysandbox --url URL ...
```

#### Server 模式 — 缺 --port

```
Error: --port is required with --serve

Example:
  mutagent pysandbox --serve --port 8080
```

#### 互斥检查

- `--serve` + 任何代码源（`-c` / 脚本 / stdin / 管道）→ 报错
- `--url` + `--serve` → 报错（`--url` 是 client 端参数）

### 与 webui 的对齐

| 参数 | webui | pysandbox client | pysandbox --serve |
|------|-------|-----------------|-------------------|
| `--host` | `127.0.0.1` | `127.0.0.1` | `127.0.0.1` |
| `--port` | `0`（自动）| 必填，无默认 | 必填，无默认 |
| `--no-browser` | webui 独有 | — | — |
| `--url` | — | client 独有 | — |

- `--host`/`--port` 命名完全统一
- 默认值不同是合理差异：webui 是 server 端用 0 自动选端口；pysandbox client 和 `--serve` 都要求显式指定端口（client 若用 `--url` 则端口含在 URL 中）

### 跨项目通用性

pysandbox 是一个 MCP tool 协议：工具名 `pysandbox`，参数声明 `code: str`。任何实现该协议的 MCP server 均可作为 pysandbox client 的目标：

- **mutbot**：已暴露 `/mcp` + `pysandbox` tool，直接可用
- **mutagent webui**：已暴露 `/mcp` + `pysandbox` tool（webui 启动时注入 `PySandboxTools._app` 到 sandbox 并注册 `MCPView`）
- **mutagent pysandbox --serve**：纯沙箱 MCP 服务（无 agent/WebUI，仅暴露 `/mcp`）
- **任意兼容 server**：只要 `/mcp` 提供了 `pysandbox` tool

### Server 模式实现

`--serve` 需要启动一个 HTTP server 暴露 `PySandboxTools`。核心流程：

1. 构造 `SandboxApp`（同原独立模式的 `_build_sandbox`）
2. 连接 config 中的 `mcp_sources` / `cli_sources`（注入 namespace）
3. `PySandboxTools._app = sandbox_app`（ClassVar 注入，与 mutbot 一致）
4. 启动 `mutio.net.server.Server`，注册 `PySandboxTools` view，listen 在 `host:port`
5. 启动横幅：`mutagent sandbox server: http://{host}:{port}/mcp`
6. 阻塞监听，直到 SIGINT/SIGTERM；退出时 `await sandbox.close()`

复用 `entry_mcp.py` 的 `PySandboxTools`，不重复造轮子。

## 实施步骤清单

- [x] 重写 `mutagent/src/mutagent/cli/pysandbox.py`：
  - [x] argparse 重新设计：`--host`/`--port`/`--url`/`--serve`/`--timeout`/`-c`/`script`，含互斥与必填校验
  - [x] dispatcher 根据 `--serve` 分发到 `_run_server` 或 `_run_client`
  - [x] 删除独立模式的 `_run`（一次性执行合并到 `_run_client`）
  - [x] 将 REPL 从本地 `SandboxApp` 改为 MCP client 模式：`_REPLConsole` 通过 `client.call_tool` 逐条发送代码，复用同一条 MCP 连接
  - [x] `_build_sandbox` / `_setup_pysandbox_logging` 保留供 `--serve` 复用
- [x] Client 模式实现 `_run_client`：
  - [x] 读代码源（-c / script / stdin）、URL 拼接（`--url` 优先）
  - [x] 用 `mutio.mcp.client.MCPClient` 调 `pysandbox` tool，`--timeout` 默认 30s
  - [x] 结果格式化复用 mutbot CLI 的 `_format_result` 逻辑
  - [x] 连不上时输出迁移指引错误信息（带 `--serve` 启动示例）
- [x] Server 模式实现 `_run_server`：
  - [x] 复用原 `_build_sandbox` 逻辑构造 SandboxApp + 注入 mcp_sources / cli_sources
  - [x] `PySandboxTools._app = sandbox_app` 注入
  - [x] 启动 `mutio.net.server.Server` 挂载 `PySandboxMCPView`（`PySandboxTools` 通过 `path="/mcp"` 自动挂接），listen 在 `host:port`，打印启动横幅
  - [x] 保留 server 侧 logging（复用原 `_setup_pysandbox_logging`）
  - [x] KeyboardInterrupt / asyncio.CancelledError 下 `await server.stop()` + `sandbox.close()` 干净退出
- [x] 调整 `mutagent/src/mutagent/main.py`：在 client 模式跳过 `app.load_config()`，并在 parse 后立即调用 `validate_args`
- [x] 测试验证（手动跑 happy path，补足必要的单测）：
  - [x] 静态：`mutagent pysandbox --help` 可见 / import 无错
  - [x] 静态：四项互斥/必填校验（`--serve` 缺 port 、`--serve`+`--url`、`--serve`+`-c`、`-c`+script）报错信息合理
  - [x] 静态：client 连不上时错误信息含 `--serve` 指引，无 logging 污染，exit=1
  - [x] `mutagent pysandbox --serve --port 8090` 启动成功，启动横幅 URL 对
  - [x] 另一进程 `mutagent pysandbox --port 8090 -c "1+1"` 返回 `2`
  - [x] `mutagent pysandbox --port 8090 -c "x=1"; echo $?` exit code = 0
  - [x] `mutagent pysandbox --port 8090 -c "raise ValueError('x')"` exit code != 0，错误走 stderr
  - [x] `mutagent pysandbox --url http://127.0.0.1:8090/mcp -c "1"` 走 url、忽略 host/port
  - [x] mutbot pysandbox client 反向连 `mutagent --serve` 启动的 server 能跳通（跨项目通用性验证）
- [x] 检查是否有文档引用旧独立模式需同步更新：README / docs/design 不涉及 CLI 用法，无需改动；docs/specifications 中 `bugfix-pysandbox-repl-robustness.md` 和 `feature-pysandbox-repl.md` 描述的 REPL 实现细节（本地 SandboxApp）已变更，建议后续 `sdd archive` 处理（本次不改）

### Client 模式实现

Client 模式**完全脱离 config 与本地 logging**：

- 不调用 `app.load_config()`（main 入口需为 pysandbox 子命令跳过 config 加载）
- 不挂任何 logging handler，不写 `.mutagent/logs/`，stdout/stderr 干净只有结果
- 直接基于 `mutio.mcp.client.MCPClient`（参考 `mutbot/src/mutbot/cli/pysandbox.py`），调用 `pysandbox` tool
- 错误格式化复用 `_format_tool_result`（content[].text 拼接，回退 JSON）
- 无代码源 + 交互终端（`sys.stdin.isatty()`）时进入 REPL，复用同一条 MCP 连接逐条发送代码

### 与 mutbot pysandbox CLI 的关系（暂不处理）

`mutbot pysandbox` 默认连本机 mutbot 8741，是"找本机 mutbot"的快捷方式；`mutagent pysandbox` 是通用 client。本次重构两者共存，未来另行评估是否合并。


