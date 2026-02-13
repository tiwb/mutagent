# PyAgent MVP 设计规范

**状态**：🔄 进行中
**日期**：2026-02-13
**类型**：功能设计

## 1. 背景

构建一个基于 Python 的 AI Agent 框架，让大语言模型（LLM）能够通过 Python 调用完成各种工作。核心价值在于为 LLM 提供一个**可运行时自我迭代的 Python 环境**——Agent 可以查看、修改代码并热重载验证，形成高效的开发闭环。

框架基于 [forwardpy](https://github.com/tiwb/forwardpy) 库的声明-实现分离模式，天然支持运行时方法替换和模块热重载。

### 1.1 核心理念

- **Agent 即开发者**：LLM 通过 tools 操作 Python 模块，像开发者一样迭代代码
- **运行时可迭代**：基于 forwardpy 的 `@impl` + `override=True` 机制，无需重启即可替换实现
- **自举能力**：Agent 框架本身也用 forwardpy 编写，Agent 可以迭代改进自身

### 1.2 技术栈

- Python 3.11+
- forwardpy（声明-实现分离、热重载基础）
- anthropic SDK（Claude API 接入）

补充说明：不要使用anthropic SDK，我更倾向使用asyncio直接通过http请求调用。

## 2. 设计方案

### 2.1 整体架构

```
┌─────────────────────────────────────────────────┐
│                   PyAgent                        │
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
│                 │  │ inspect_project  │    │     │
│                 │  │ (工程结构查看)    │    │     │
│                 │  ├──────────────────┤    │     │
│                 │  │ view_source      │    │     │
│                 │  │ (查看源码)       │    │     │
│                 │  ├──────────────────┤    │     │
│                 │  │ edit_module      │    │     │
│                 │  │ (修改模块)       │    │     │
│                 │  ├──────────────────┤    │     │
│                 │  │ hot_reload       │    │     │
│                 │  │ (热重载)         │    │     │
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
pyagent/
├── __init__.py
├── client/                    # LLM 客户端抽象
│   ├── __init__.py
│   ├── base.py               # LLM 客户端声明（forwardpy Object）
│   ├── claude.py             # Claude API 实现（@impl）
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
│       ├── inspect_project.py # 工程结构查看
│       ├── view_source.py     # 查看源码
│       ├── edit_module.py     # 修改模块/函数/类
│       ├── hot_reload.py      # 热重载
│       └── execute.py         # 执行验证
└── runtime/                   # 运行时环境
    ├── __init__.py
    ├── module_manager.py      # Python 模块管理
    └── reload.py              # 热重载引擎
```

### 2.3 LLM Client 层

基于 forwardpy 声明-实现分离，便于后续扩展 OpenAI 等其他协议。

```python
# client/base.py - 声明
class LLMClient(forwardpy.Object):
    """LLM 客户端接口"""
    model: str

    def send_message(self, messages: list[Message], tools: list[ToolSchema]) -> Response: ...
    def stream_message(self, messages: list[Message], tools: list[ToolSchema]) -> Iterator[StreamEvent]: ...

# client/claude.py - Claude 实现
@forwardpy.impl(LLMClient.send_message)
def send_message(self: LLMClient, messages, tools):
    # 使用 anthropic SDK 调用 Claude API
    ...
```

**消息模型**（`client/messages.py`）：
- `Message`：统一消息格式（role, content, tool_calls, tool_results）
- `ToolCall`：LLM 发起的工具调用（tool_name, arguments）
- `ToolResult`：工具执行结果（tool_call_id, content, is_error）
- `Response`：LLM 响应（message, stop_reason, usage）

### 2.4 Agent 核心

Agent 负责管理对话循环：发送消息 → 接收响应 → 执行 tool calls → 反馈结果 → 继续对话。

```python
# agent/core.py - 声明
class Agent(forwardpy.Object):
    client: LLMClient
    tools: ToolRegistry
    system_prompt: str
    messages: list[Message]

    def run(self, user_input: str) -> str: ...
    def step(self) -> Response: ...
    def handle_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolResult]: ...
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

    def get_schema(self) -> dict: ...      # 返回 JSON Schema 给 LLM
    def execute(self, **params) -> str: ... # 执行工具调用
```

#### Tool 注册表

```python
# tools/registry.py
class ToolRegistry(forwardpy.Object):
    def register(self, tool: Tool) -> None: ...
    def get_tool(self, name: str) -> Tool: ...
    def get_all_schemas(self) -> list[dict]: ...
```

### 2.6 内置 Tools 设计

#### 2.6.1 `inspect_project` — 工程结构查看

**功能**：基于 Python 模块体系（非文件目录）展示工程结构，精确到类和函数。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `module_path` | `str` | 可选，模块路径如 `pyagent.tools.base`，不填则从根模块开始 |
| `depth` | `int` | 展开深度，默认 2 |

**输出示例**：
```
pyagent/
  client/
    base [module]
      LLMClient (class)
        .model: str
        .send_message(messages, tools) -> Response
        .stream_message(messages, tools) -> Iterator[StreamEvent]
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
| `target` | `str` | 目标路径，如 `pyagent.agent.core.Agent` 或 `pyagent.tools.base.Tool.execute` |

**实现要点**：
- 使用 `inspect.getsource()` 获取源码
- 支持模块级、类级、函数/方法级查看

#### 2.6.3 `edit_module` — 修改模块/函数/类

**功能**：对 Python 模块进行增加、修改操作。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `target` | `str` | 目标模块路径，如 `pyagent.tools.builtins.my_tool` |
| `action` | `str` | `add_function` / `add_class` / `replace_function` / `replace_class` / `add_impl` |
| `name` | `str` | 函数/类名称 |
| `source` | `str` | 完整的 Python 源代码 |

**关键操作**：
- `add_function`：向模块追加函数定义
- `add_class`：向模块追加类定义
- `replace_function`：替换已有函数
- `replace_class`：替换已有类
- `add_impl`：添加 forwardpy `@impl` 实现（自动处理 import 和 override）

**实现要点**：
- 解析目标模块的 AST 找到插入/替换位置
- 对源码写入文件后触发热重载
- `add_impl` 操作需特殊处理：自动添加 `@forwardpy.impl(Target.method, override=True)` 以支持运行时替换

#### 2.6.4 `hot_reload` — 热重载

**功能**：重新加载指定模块，使修改生效。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `module_path` | `str` | 要重载的模块路径，如 `pyagent.tools.builtins.my_tool` |

**实现要点**：
- 使用 `importlib.reload()` 重新加载模块
- forwardpy 的 `@impl(override=True)` 确保方法替换生效
- 处理模块间依赖：按依赖顺序重载

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

`runtime/module_manager.py` 负责：
- 维护目标项目的模块索引
- 提供模块发现、加载、卸载能力
- 与 forwardpy 注册表交互，追踪声明和实现的对应关系

`runtime/reload.py` 负责：
- 模块热重载的核心逻辑
- 依赖关系分析（哪些模块需要一起重载）
- 重载前后状态校验

### 2.8 Claude API 对接细节

使用 Anthropic Python SDK（`anthropic`），对接 Messages API：

- **模型**：`claude-sonnet-4-20250514`（默认，可配置）
- **Tool Use**：将 Tool 的 `get_schema()` 输出转为 Claude 的 tool 格式
- **消息格式映射**：
  - `user` / `assistant` 角色直接映射
  - `tool_use` content block → 内部 `ToolCall`
  - `tool_result` content block → 内部 `ToolResult`
- **流式支持**：使用 `client.messages.stream()` 实现流式输出

## 3. 待定问题

### Q1: 项目包名

**问题**：Python 包的正式名称是什么？用 `pyagent` 还是其他名称？
**建议**：使用 `pyagent` 作为包名，简洁且能表达核心概念。

 这个包名被使用了，对于名称，我有几点考虑：
 1. 名称中可以不出现py，比如pyagent等，虽然这个项目是基于python的。但不排除agent最后进化出使用自创的更高效的语言来表达的可能性。
 2. 只要是能表达出本项目的核心概念的名称都可以接受，可以给出一些名称建议，注意验证名称的可用性（如 PyPI 上是否已被占用，以及是否有名称相近的库导致创建失败的）。

### Q2: Agent 作用域 — 操作自身 vs 操作外部项目

**问题**：MVP 阶段 Agent 的 tools 操作目标是什么？是操作 Agent 自身的代码（自举），还是操作用户指定的外部 Python 项目，还是两者都要？
**建议**：MVP 先实现操作用户指定的外部 Python 项目（通过配置 target project path）。自举能力作为后续目标——因为框架本身也用 forwardpy 编写，理论上天然支持。

回答：Agent可以操作自身，但是可能会有比较严格的显示，避免Agent把自己搞死。安全的核心迭代可能需要在fork整个环境。但MVP不用考虑。 另外，这个项目的核心理念是已于python的运行时模块，比如package.module.function, 而非文件路径。 内置的工具和agent迭代出来的工具，在运行时都有办法获取到源码的路径。 另外 agent在迭代时，可能不需要写文件，直接生成python实现，然后patch到运行时中。 等验证通过需要固化后，再固化到文件中。

### Q3: edit_module 的粒度

**问题**：`edit_module` 工具需要支持多细粒度的编辑？当前设计支持函数/类级别的增加和替换。是否需要支持更细粒度（如修改类的单个方法）或更粗粒度（如整个文件替换）？
**建议**：MVP 支持函数/类级别即可。对于类中单个方法的修改，通过 forwardpy 的 `add_impl` 操作用 `@impl(override=True)` 来覆盖实现，无需直接修改原始类定义。这正是 forwardpy 的核心优势。

MVP阶段，先用最简单的实现，agent能生成新的模块，迭代之前生成的模块即可。后续让agent自己进化出更高效的迭代方式。

### Q4: 安全边界

**问题**：`execute` 工具在当前进程中执行任意代码，是否需要沙箱机制？
**建议**：MVP 不做沙箱，仅做超时控制。目标用户是开发者本人，Agent 运行在开发环境中。后续版本可考虑 subprocess 隔离。

同意。

### Q5: 对话持久化

**问题**：是否需要持久化对话历史？Agent 重启后是否需要恢复上下文？
**建议**：MVP 不做持久化，每次启动是全新会话。对话历史仅在内存中维护。

确认。

### Q6: forwardpy 框架在 Agent 自身中的应用深度

**问题**：Agent 框架本身哪些部分用 forwardpy 的声明-实现分离模式？是所有核心类都用，还是只在需要运行时替换的关键点使用？
**建议**：所有核心声明类（`LLMClient`、`Agent`、`Tool`、`ToolRegistry`）都使用 forwardpy.Object，实现放在单独的 `_impl.py` 文件中。这样从第一天起就具备自举迭代能力。

同意，但forwardpy不能成为阻碍，这是一个可在开发中迭代的项目，它主要是为了这个agent服务的。

## 4. 实施步骤清单

### 阶段一：项目基础设施 [待开始]

- [ ] **Task 1.1**: 初始化项目结构
  - [ ] 创建 pyagent 包目录结构
  - [ ] 配置 pyproject.toml（依赖：forwardpy, anthropic）
  - [ ] 创建 `__init__.py` 入口
  - 状态：⏸️ 待开始

- [ ] **Task 1.2**: 消息模型定义
  - [ ] 定义 Message、ToolCall、ToolResult、Response 数据类
  - [ ] 单元测试
  - 状态：⏸️ 待开始

### 阶段二：LLM Client [待开始]

- [ ] **Task 2.1**: LLM Client 声明
  - [ ] 实现 LLMClient forwardpy 声明（base.py）
  - [ ] 定义接口方法签名
  - 状态：⏸️ 待开始

- [ ] **Task 2.2**: Claude 实现
  - [ ] 使用 anthropic SDK 实现 `send_message`
  - [ ] 实现 tool schema 格式转换
  - [ ] 实现 `stream_message`（可选，MVP 可先不做流式）
  - [ ] 集成测试（需要 API Key）
  - 状态：⏸️ 待开始

### 阶段三：Tool 系统 [待开始]

- [ ] **Task 3.1**: Tool 基类与注册表
  - [ ] 实现 Tool forwardpy 声明
  - [ ] 实现 ToolRegistry
  - [ ] 单元测试
  - 状态：⏸️ 待开始

- [ ] **Task 3.2**: inspect_project 工具
  - [ ] 实现模块结构遍历
  - [ ] 格式化输出
  - [ ] 单元测试
  - 状态：⏸️ 待开始

- [ ] **Task 3.3**: view_source 工具
  - [ ] 实现源码查看
  - [ ] 单元测试
  - 状态：⏸️ 待开始

- [ ] **Task 3.4**: edit_module 工具
  - [ ] 实现 add_function / add_class
  - [ ] 实现 replace_function / replace_class
  - [ ] 实现 add_impl（forwardpy 集成）
  - [ ] 单元测试
  - 状态：⏸️ 待开始

- [ ] **Task 3.5**: hot_reload 工具
  - [ ] 实现模块热重载
  - [ ] 依赖分析
  - [ ] 单元测试
  - 状态：⏸️ 待开始

- [ ] **Task 3.6**: execute 工具
  - [ ] 实现代码执行（受控命名空间）
  - [ ] 输出捕获 + 超时控制
  - [ ] 单元测试
  - 状态：⏸️ 待开始

### 阶段四：Agent 核心 [待开始]

- [ ] **Task 4.1**: Agent 声明与实现
  - [ ] 实现 Agent forwardpy 声明
  - [ ] 实现 agent 主循环（run / step / handle_tool_calls）
  - [ ] 单元测试（mock LLM）
  - 状态：⏸️ 待开始

- [ ] **Task 4.2**: 端到端集成
  - [ ] 组装所有组件
  - [ ] 实现 CLI 入口或简单的 main.py
  - [ ] 端到端测试：Agent 使用 tools 完成简单任务
  - 状态：⏸️ 待开始

## 5. 测试验证

### 单元测试
- [ ] 消息模型序列化/反序列化
- [ ] Tool schema 生成
- [ ] ToolRegistry 注册/查找
- [ ] inspect_project 模块遍历
- [ ] view_source 源码获取
- [ ] edit_module 代码修改
- [ ] hot_reload 重载验证
- [ ] execute 代码执行与输出捕获
- [ ] Agent 主循环（mock LLM 响应）

### 集成测试
- [ ] Claude API 实际调用测试
- [ ] Agent + Tools 端到端：让 Agent 查看工程结构 → 编写函数 → 热重载 → 执行验证
