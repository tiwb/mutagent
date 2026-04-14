# Sandbox 设计规范

**状态**：✅ 已完成
**日期**：2026-04-14
**类型**：功能设计

## 需求

1. 将独立的 python-sandbox skill 迁入 mutagent，作为 `mutagent.sandbox` 子模块
2. PySandbox 定位为 MutAgent 的核心执行工具（≈ Bash 之于 Claude Code）
3. 统一命名：包名 `mutagent.sandbox`，MCP tool 名 `pysandbox`
4. 保持纯 API 设计，CLI/daemon 作为消费者留在 skill 层

## 关键参考

- python-sandbox skill 目录 — 迁移前的实现（server.py, engine.py, namespace.py, config.py, cli.py, adapters/）
- `src/mutagent/net/mcp.py` — MCPToolSet / MCPView 声明
- `src/mutagent/net/_mcp_impl.py` — MCP tool 自动发现和分发
- `src/mutagent/tools.py` — ToolSet / Toolkit 声明（agent 内部 tool 系统）
- mutobj `docs/specifications/feature-safe-subset.md` — MutScript 规范（TypeSpec 部分，未来 engine 升级路径）
- mutobj `docs/postmortems/2026-04-14-mutscript-boundary-redesign.md` — 设计边界决策记录

## 设计方案

### 定位

PySandbox 是 mutagent 的核心能力模块，为 agent 提供安全的 Python 代码执行环境。agent 通过编写 Python 代码组合多种外部能力（MCP 服务、CLI 工具），完成复杂任务。

类比：
- Claude Code 的 agent 通过 Bash 执行 shell 命令
- MutAgent 的 agent 通过 PySandbox 执行 Python 代码

### 架构层次

```
mutobj           TypeSpec（类型安全声明，未来使用）
                   │
mutagent.sandbox   纯 API 层（库）
├── engine         执行引擎（当前：AST 检查 + 受限 builtins）
├── namespace      函数注册 + 名称路由（纯内部机制，不面向 agent）
├── app            SandboxApp — 组装层，串联 engine + namespace + adapters
├── adapters       外部能力桥接（MCP bridge 统一文件，支持 stdio + HTTP）
└── tools          MCPToolSet 定义（pysandbox tool）

skill 层（消费者，独立于 mutagent 仓库）
├── skill.md                    知识层：告诉 agent 怎么用
├── server.py                   SandboxServer（ASGI 组装，启动 daemon 用）
├── cli.py                      daemon 管理（start/stop/exec）
└── config.py                   daemon 的 config 管理
```

### 已确定的设计决策

**包名与 tool 名分层**：`mutagent.sandbox` 是 Python 包名，`pysandbox` 是 MCP tool 名。Claude Code 配置 MCP 时的 server name（如 "MutAgent"）是使用者侧的配置，与包名无关。

**sandbox 是纯 API**：所有执行能力通过 Python API 暴露（`app.exec_code()` 等）。CLI/daemon/config 是消费者层关注的事，留在 skill 目录。mutagent 作为纯库，不包含 CLI。未来 mutbot 可直接嵌入 sandbox API，不经过 MCP/HTTP。

**namespace 不提供列表发现**：去掉 `list_functions()`。agent 通过 skill 获取"有什么能力、怎么用"的知识，不通过运行时内省发现函数。保留 `help(func)` 用于按需查询具体函数的参数和用法（≈ `man` 命令）。

**namespace 是纯内部路由**：namespace 系统管理函数注册和名称路由（`playwright.navigate` → 实际函数），但这是实现细节，不面向 agent。agent 通过 skill 知道该写什么代码。

**engine 渐进增强**：当前使用简单 AST 检查（禁 import/class + 受限 builtins），实践证明 agent 不会尝试越界。未来需要更严格隔离时，引入 mutobj 的 Script + TypeSpec，对上层透明。

**config 暂时独立**：config 留在 skill 层（daemon 的消费者配置），不进入 mutagent。等 mutbot 消费 sandbox 时再统一。

**不做可选依赖**：MCP bridge（subprocess 连接外部 MCP server）和 CLI adapter 都基于 Python 标准库 + mutagent 已有能力，不引入额外依赖。

**MCP bridge 统一为一个文件**：bridge 的核心逻辑是"从 MCP server 发现 tools → 包装为 namespace 函数"，与 transport 无关。stdio 和 Streamable HTTP 两种 transport 通过 mutagent.net 的 MCPClient 统一处理，bridge 只做 namespace 包装。现有 `StdioMCPClient` 自实现的 MCP 协议处理退役，改用 mutagent.net.client。

### 与 mutobj 的关系

mutobj 提供 TypeSpec（类型安全方法声明），mutagent.sandbox 是消费者。当前 engine 不依赖 TypeSpec，预留升级路径：

```
当前：engine → 简单 AST 检查（禁 import/class, 受限 builtins）
未来：engine → mutobj Script 编译 + TypeSpec 白名单 + guard 变换
```

Script 和 ScriptEnv 的设计归 mutagent.sandbox 所有（见 mutobj 复盘文档）。

### 与 mutagent tool 系统的关系

mutagent 有两套 tool 暴露机制：
- **MCPToolSet**（`mutagent.net.mcp`）— 面向外部消费者（Claude Code、其他 MCP 客户端）
- **Toolkit**（`mutagent.tools`）— 面向 mutagent 内部的 agent

PySandbox 需要同时支持：
- 作为 MCPToolSet 暴露给外部（`tools.py` 中定义 `pysandbox` tool）
- 未来作为 Toolkit 暴露给 mutagent 内部 agent

两者的底层都是 `app.exec_code()`，只是包装层不同。

### 知识层与执行层分离

```
Skill（知识层）    描述可用能力、使用场景、最佳实践、规范
PySandbox（执行层） 接收代码，执行，返回结果
```

PySandbox 的 MCP tool description 极简（"Execute Python code in a sandboxed environment"），不列举可用函数。agent 通过 skill 获取"什么场景用什么函数"的完整知识。

每接入一个新的外部能力（如 playwright、redmine CLI），对应写一个 skill 描述使用方法，而非在 sandbox 内部做运行时发现。

## 推迟的设计决策

以下决策在迁移完成后根据实际需求再做：

- **CLI 入口形式** — `uv tool` 独立命令 / `python -m mutagent` 子命令，待 CLI 整体设计时统一
- **SandboxApp API 精简** — 先保持现有 API，mutbot 消费时再根据实际需求调整
- **daemon 端口和发现** — 保持现有策略（config 控制，默认 `127.0.0.1:8765`），搬迁不影响

## 实施步骤清单

### 迁入 mutagent.sandbox（纯 API）

- [x] 创建 `mutagent/src/mutagent/sandbox/` 包，搬入 engine.py、namespace.py
- [x] 从 server.py 中提取 SandboxApp 到 `sandbox/app.py`（构造函数改为接收配置字典而非路径）
- [x] 搬入 adapters 目录（mcp_bridge.py 暂保留 StdioMCPClient，MCPClient 的 HTTP 支持后续增加）
- [x] 创建 `sandbox/tools.py`，MCPToolSet 定义（tool 名改为 `pysandbox`，用 path 路由）
- [x] namespace.py 去掉 `list_functions()`，保留 `help()`
- [x] 修复所有内部 import 路径（`sandbox.xxx` → `mutagent.sandbox.xxx`）
- [x] 补充 `sandbox/__init__.py` 导出公共 API

### 更新 skill 层

- [x] skill 目录保留 server.py（ASGI 组装）、cli.py（daemon 管理）、config.py
- [x] 更新 skill 层 import（`from mutagent.sandbox import SandboxApp` 等）
- [x] 更新 skill.md（tool 名、调用方式、描述）

### 单元测试

- [x] 迁移 engine 测试（12 个）+ 安全边界测试（14 个）+ namespace 测试（10 个）到 mutagent
- [x] 全部 36 个测试通过

### 验证

- [x] `pip install -e .` 安装 mutagent 后，skill 的 daemon 正常启动
- [x] Claude Code 通过 MCP 调用 pysandbox tool 正常执行代码
- [x] 外部 MCP server 桥接正常工作（stdio 模式）
