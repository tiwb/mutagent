# AI Agent MVP 设计规范

**状态**：🔄 进行中
**日期**：2026-02-13
**类型**：功能设计

## 1. 背景

构建一个基于 Python 的 AI Agent 框架，让大语言模型（LLM）能够通过 Python 调用完成各种工作。核心价值在于为 LLM 提供一个**可运行时自我迭代的 Python 环境**——Agent 可以查看、修改代码并热重载验证，形成高效的开发闭环。

框架基于 [forwardpy](https://github.com/tiwb/forwardpy) 库的声明-实现分离模式，天然支持运行时方法替换和模块热重载。

### 1.1 核心理念

- **Agent 即开发者**：LLM 通过 tools 操作 Python 模块，像开发者一样迭代代码
- **运行时可迭代**：基于 forwardpy 的 `@impl` + `override=True` 机制，无需重启即可替换实现
- **模块即一切**：核心抽象基于 Python 运行时模块（`package.module.function`），而非文件路径
- **运行时优先，文件其次**：Agent 迭代时先在运行时 patch 代码验证，通过后再固化到文件
- **自举能力**：Agent 框架本身也用 forwardpy 编写，Agent 可以迭代改进自身

### 1.2 技术栈

- Python 3.11+
- forwardpy（声明-实现分离、热重载基础）
- asyncio + aiohttp（直接通过 HTTP 请求调用 LLM API，不使用 SDK）

## 2. 设计方案

### 2.1 整体架构

```
┌─────────────────────────────────────────────────┐
│                   Agent Framework                │
│                                                  │
│  ┌───────────┐  ┌───────────┐  ┌─────────────┐  │
│  │ LLM Client│  │   Agent   │  │  Tool System │  │
│  │ (Claude)  │──│   Core    │──│             │  │
│  └───────────┘  └───────────┘  └──────┬──────┘  │
│                                       │          │
│                 ┌─────────────────────┼────┐     │
│                 │     Built-in Tools       │     │
│                 │                          │     │
│                 │  ┌──────────────────┐    │     │
│                 │  │ inspect_module   │    │     │
│                 │  │ (模块结构查看)    │    │     │
│                 │  ├──────────────────┤    │     │
│                 │  │ view_source      │    │     │
│                 │  │ (查看源码)       │    │     │
│                 │  ├──────────────────┤    │     │
│                 │  │ patch_module     │    │     │
│                 │  │ (运行时 patch)   │    │     │
│                 │  ├──────────────────┤    │     │
│                 │  │ save_module      │    │     │
│                 │  │ (固化到文件)     │    │     │
│                 │  ├──────────────────┤    │     │
│                 │  │ execute          │    │     │
│                 │  │ (执行验证)       │    │     │
│                 │  └──────────────────┘    │     │
│                 └─────────────────────────┘     │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │         forwardpy Runtime                 │    │
│  │   (声明-实现分离 / 热重载基础)            │    │
│  └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

### 2.2 模块结构

```
<package>/
├── __init__.py
├── client/                    # LLM 客户端抽象
│   ├── __init__.py
│   ├── base.py               # LLM 客户端声明（forwardpy Object）
│   ├── claude.py             # Claude API 实现（@impl, asyncio + aiohttp）
│   └── messages.py           # 消息模型定义
├── agent/                     # Agent 核心
│   ├── __init__.py
│   ├── core.py               # Agent 声明
│   └── core_impl.py          # Agent 主循环实现
├── tools/                     # Tool 系统
│   ├── __init__.py
│   ├── base.py               # Tool 基类声明
│   ├── registry.py           # Tool 注册表
│   └── builtins/             # 内置 tools
│       ├── __init__.py
│       ├── inspect_module.py  # 模块结构查看
│       ├── view_source.py     # 查看源码
│       ├── patch_module.py    # 运行时 patch（代码直接注入运行时）
│       ├── save_module.py     # 固化到文件
│       └── execute.py         # 执行验证
└── runtime/                   # 运行时环境
    ├── __init__.py
    └── module_manager.py      # Python 模块管理（发现、加载、patch、固化）
```

### 2.3 LLM Client 层

基于 forwardpy 声明-实现分离，便于后续扩展 OpenAI 等其他协议。使用 asyncio + aiohttp 直接发送 HTTP 请求，不依赖任何 LLM SDK。

```python
# client/base.py - 声明
class LLMClient(forwardpy.Object):
    """LLM 客户端接口"""
    model: str
    api_key: str
    base_url: str

    async def send_message(self, messages: list[Message], tools: list[ToolSchema]) -> Response: ...

# client/claude.py - Claude 实现
@forwardpy.impl(LLMClient.send_message)
async def send_message(self: LLMClient, messages, tools):
    # 使用 aiohttp 直接调用 Claude Messages API
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{self.base_url}/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [...],  # 转换为 Claude 格式
                "tools": [...],     # 转换为 Claude tool 格式
            }
        ) as resp:
            data = await resp.json()
            return Response.from_claude(data)
```

**消息模型**（`client/messages.py`）：
- `Message`：统一消息格式（role, content, tool_calls, tool_results）
- `ToolCall`：LLM 发起的工具调用（tool_name, arguments, id）
- `ToolResult`：工具执行结果（tool_call_id, content, is_error）
- `Response`：LLM 响应（message, stop_reason, usage）
- `ToolSchema`：工具的 JSON Schema 描述

### 2.4 Agent 核心

Agent 负责管理对话循环：发送消息 → 接收响应 → 执行 tool calls → 反馈结果 → 继续对话。

```python
# agent/core.py - 声明
class Agent(forwardpy.Object):
    client: LLMClient
    tools: ToolRegistry
    system_prompt: str
    messages: list[Message]

    async def run(self, user_input: str) -> str: ...
    async def step(self) -> Response: ...
    async def handle_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolResult]: ...
```

**Agent 主循环**：
1. 用户输入 → 添加到 messages
2. 调用 `step()` → LLM 返回响应
3. 如果响应包含 tool_calls → `handle_tool_calls()` 执行
4. 将 tool_results 添加到 messages → 回到步骤 2
5. 如果响应是 end_turn → 返回最终文本

### 2.5 Tool 系统

#### Tool 基类

```python
# tools/base.py
class Tool(forwardpy.Object):
    name: str
    description: str

    def get_schema(self) -> ToolSchema: ...       # 返回 JSON Schema 给 LLM
    async def execute(self, **params) -> str: ...  # 执行工具调用
```

#### Tool 注册表

```python
# tools/registry.py
class ToolRegistry(forwardpy.Object):
    def register(self, tool: Tool) -> None: ...
    def get_tool(self, name: str) -> Tool: ...
    def get_all_schemas(self) -> list[ToolSchema]: ...
```

### 2.6 内置 Tools 设计

核心工作流：`inspect_module` → `view_source` → `patch_module` → `execute`（验证）→ `save_module`（固化）

#### 2.6.1 `inspect_module` — 模块结构查看

**功能**：基于 Python 运行时模块体系展示结构，精确到类和函数。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `module_path` | `str` | 可选，模块路径如 `mypackage.tools.base`，不填则从根模块开始 |
| `depth` | `int` | 展开深度，默认 2 |

**输出示例**：
```
mypackage/
  client/
    base [module]
      LLMClient (class)
        .model: str
        .send_message(messages, tools) -> Response
    claude [module]
      (implementations for LLMClient)
    messages [module]
      Message (class)
      ToolCall (class)
      ToolResult (class)
  agent/
    core [module]
      Agent (class)
        .run(user_input) -> str
        .step() -> Response
```

**实现要点**：
- 使用 `importlib` + `inspect` 模块遍历已加载模块
- 通过 `pkgutil.walk_packages()` 发现子模块
- 用 `inspect.getmembers()` 获取类、函数、属性签名

#### 2.6.2 `view_source` — 查看源码

**功能**：查看指定模块、类或函数的源代码。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `target` | `str` | 目标路径，如 `mypackage.agent.core.Agent` 或 `mypackage.tools.base.Tool.execute` |

**实现要点**：
- 使用 `inspect.getsource()` 获取源码
- 支持模块级、类级、函数/方法级查看
- 对于运行时 patch 的代码，从内存中获取源码

#### 2.6.3 `patch_module` — 运行时 patch

**功能**：将 Agent 生成的 Python 代码直接注入运行时，不写文件。这是 Agent 迭代的核心工具。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `module_path` | `str` | 目标模块路径（已有模块则 patch，不存在则创建虚拟模块） |
| `source` | `str` | 完整的 Python 源代码 |

**工作原理**：
1. 将 `source` 编译为代码对象（`compile()`）
2. 在目标模块的命名空间中执行（`exec()`）
3. 如果是新模块，创建 `types.ModuleType` 并注册到 `sys.modules`
4. 如果模块已存在，在其命名空间中执行新代码（覆盖/新增定义）
5. forwardpy 的 `@impl(override=True)` 自动处理方法替换

**关键特性**：
- **零文件 IO**：代码只存在于运行时内存中
- **可叠加**：多次 patch 同一模块，后续 patch 叠加在前面之上
- **可回溯**：维护 patch 历史，支持查看每次 patch 的内容

#### 2.6.4 `save_module` — 固化到文件

**功能**：将运行时验证通过的模块源码写入文件系统，持久化保存。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `module_path` | `str` | 要固化的模块路径 |
| `file_path` | `str` | 可选，目标文件路径。不填则根据模块路径自动推导 |

**实现要点**：
- 从 patch 历史中组装最终源码
- 自动创建必要的包目录和 `__init__.py`
- 写入文件后执行 `importlib.reload()` 确保文件版本与运行时一致

#### 2.6.5 `execute` — 执行验证

**功能**：在当前运行时执行 Python 代码片段，验证修改效果。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `code` | `str` | 要执行的 Python 代码 |

**实现要点**：
- 在受控命名空间中执行代码
- 捕获输出（stdout/stderr）和返回值
- 捕获异常并返回完整 traceback
- 设置执行超时（防止无限循环）

### 2.7 运行时模块管理

`runtime/module_manager.py` 统一负责：
- 维护模块索引（已加载模块 + 运行时 patch 的虚拟模块）
- 模块发现与加载（`importlib` + `pkgutil`）
- 运行时 patch 执行与历史记录
- 模块固化（内存 → 文件）
- 与 forwardpy 注册表交互，追踪声明和实现的对应关系

### 2.8 Claude API 对接细节

直接通过 asyncio + aiohttp 调用 Claude Messages API（`https://api.anthropic.com/v1/messages`）：

- **模型**：`claude-sonnet-4-20250514`（默认，可配置）
- **认证**：`x-api-key` header
- **Tool Use**：将 Tool 的 `get_schema()` 输出转为 Claude 的 tool 格式
- **消息格式映射**：
  - `user` / `assistant` 角色直接映射
  - `tool_use` content block → 内部 `ToolCall`
  - `tool_result` content block → 内部 `ToolResult`
- **MVP 不做流式**：直接使用普通 POST 请求等待完整响应

### 2.9 设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| HTTP 调用方式 | asyncio + aiohttp | 不依赖 SDK，更轻量可控 |
| 代码迭代方式 | 运行时 patch 优先 | 避免频繁文件 IO，快速验证 |
| 安全边界 | 无沙箱，仅超时 | MVP 面向开发者，后续可加 |
| 对话持久化 | 不持久化 | MVP 每次全新会话 |
| forwardpy 使用 | 所有核心类 | 自举能力，但不成为阻碍 |
| 核心抽象 | 模块路径而非文件路径 | `package.module.function` 是第一公民 |

## 3. 待定问题

### Q1: 项目包名

**问题**：Python 包名选择。要求：不含 "py"，能表达核心概念，PyPI 上可用。

**PyPI 可用性调查结果**：

| 名称 | 状态 | 说明 |
|------|------|------|
| `iteragent` | 可用 | iteration + agent，核心概念：可迭代的 agent |
| `selfagent` | 可用 | self + agent，核心概念：自我迭代的 agent |
| `mutagent` | 可用 | mutation + agent，核心概念：可变异/进化的 agent |
| `agentmod` | 可用 | agent + module，核心概念：基于模块的 agent |
| `agentmorph` | 可用 | agent + morph，核心概念：可变形的 agent |
| agentcore | 已占用 | - |
| agentforge | 已占用 | - |
| evoagent | 已占用 | - |
| agentica | 已占用 | - |
| nexagent | 冲突 | nex-agent 已存在，PEP 503 规范化冲突 |

**建议**：`iteragent` — 直接表达"可迭代 agent"的核心概念，简洁（9 字符），无任何命名冲突。备选 `mutagent`（更有创意，8 字符）。

我mutagent很好，用这个名称，迭代时创建一个最小的占位 PyPI 包，通过 twine 发布。

### Q2: patch_module 的源码追踪

**问题**：`patch_module` 将代码注入运行时后，`inspect.getsource()` 无法获取动态生成的代码源码。需要自行维护源码映射。这对 `view_source` 工具的实现有影响——是否需要在 MVP 阶段就实现完整的源码追踪？
**建议**：MVP 阶段在 `module_manager` 中维护一个简单的 `{module_path: source_code}` 字典，`view_source` 优先从这个字典查找，找不到再 fallback 到 `inspect.getsource()`。

同意，如果是在内存中生成的代码，可以想办法把代码直接包含在对应的对象上（比如函数或类）

可以继续详细讨论这个问题，这是这个agent的一个非常重要的设计。

### Q3: Agent 的异步架构

**问题**：由于 LLM 调用使用 asyncio，Agent 主循环也需要是异步的。这意味着 tools 的 `execute` 方法也是 `async def`。但内置 tools 中 `inspect_module`、`view_source` 等本质是同步操作。是否所有 tool 统一用 `async def`，还是区分同步/异步 tool？
**建议**：统一用 `async def`，简化接口。同步操作在 async 函数中直接执行即可（不阻塞事件循环的操作无需 `run_in_executor`）。

同意。

## 4. 实施步骤清单

### 阶段一：项目基础设施 [待开始]

- [ ] **Task 1.1**: 初始化项目结构
  - [ ] 创建包目录结构
  - [ ] 配置 pyproject.toml（依赖：forwardpy, aiohttp）
  - [ ] 创建 `__init__.py` 入口
  - 状态：⏸️ 待开始

- [ ] **Task 1.2**: 消息模型定义
  - [ ] 定义 Message、ToolCall、ToolResult、Response、ToolSchema 数据类
  - [ ] 单元测试
  - 状态：⏸️ 待开始

### 阶段二：LLM Client [待开始]

- [ ] **Task 2.1**: LLM Client 声明
  - [ ] 实现 LLMClient forwardpy 声明（base.py）
  - [ ] 定义 async 接口方法签名
  - 状态：⏸️ 待开始

- [ ] **Task 2.2**: Claude 实现
  - [ ] 使用 aiohttp 直接调用 Claude Messages API
  - [ ] 实现消息格式转换（内部格式 ↔ Claude 格式）
  - [ ] 实现 tool schema 格式转换
  - [ ] 集成测试（需要 API Key）
  - 状态：⏸️ 待开始

### 阶段三：Tool 系统 [待开始]

- [ ] **Task 3.1**: Tool 基类与注册表
  - [ ] 实现 Tool forwardpy 声明
  - [ ] 实现 ToolRegistry
  - [ ] 单元测试
  - 状态：⏸️ 待开始

- [ ] **Task 3.2**: 运行时模块管理器
  - [ ] 实现 ModuleManager（模块发现、索引）
  - [ ] 实现运行时 patch（compile + exec + 虚拟模块注册）
  - [ ] 实现 patch 历史追踪
  - [ ] 实现模块固化（内存 → 文件）
  - [ ] 单元测试
  - 状态：⏸️ 待开始

- [ ] **Task 3.3**: inspect_module 工具
  - [ ] 实现模块结构遍历
  - [ ] 格式化输出
  - [ ] 单元测试
  - 状态：⏸️ 待开始

- [ ] **Task 3.4**: view_source 工具
  - [ ] 实现源码查看（文件 + 运行时 patch 双来源）
  - [ ] 单元测试
  - 状态：⏸️ 待开始

- [ ] **Task 3.5**: patch_module 工具
  - [ ] 基于 ModuleManager 实现运行时 patch
  - [ ] 单元测试
  - 状态：⏸️ 待开始

- [ ] **Task 3.6**: save_module 工具
  - [ ] 基于 ModuleManager 实现模块固化
  - [ ] 单元测试
  - 状态：⏸️ 待开始

- [ ] **Task 3.7**: execute 工具
  - [ ] 实现代码执行（受控命名空间）
  - [ ] 输出捕获 + 超时控制
  - [ ] 单元测试
  - 状态：⏸️ 待开始

### 阶段四：Agent 核心 [待开始]

- [ ] **Task 4.1**: Agent 声明与实现
  - [ ] 实现 Agent forwardpy 声明
  - [ ] 实现 agent 异步主循环（run / step / handle_tool_calls）
  - [ ] 单元测试（mock LLM）
  - 状态：⏸️ 待开始

- [ ] **Task 4.2**: 端到端集成
  - [ ] 组装所有组件
  - [ ] 实现简单的 main.py 入口
  - [ ] 端到端测试：Agent 使用 tools 完成简单任务
  - 状态：⏸️ 待开始

## 5. 测试验证

### 单元测试
- [ ] 消息模型序列化/反序列化
- [ ] Tool schema 生成
- [ ] ToolRegistry 注册/查找
- [ ] ModuleManager: patch / 固化 / 源码追踪
- [ ] inspect_module 模块遍历
- [ ] view_source 源码获取（文件源 + patch 源）
- [ ] patch_module 运行时注入
- [ ] save_module 文件固化
- [ ] execute 代码执行与输出捕获
- [ ] Agent 主循环（mock LLM 响应）

### 集成测试
- [ ] Claude API 实际调用测试（aiohttp 直连）
- [ ] Agent + Tools 端到端：让 Agent 查看模块结构 → patch 代码 → 执行验证 → 固化文件
