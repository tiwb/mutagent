# mutagent MVP 设计规范

**状态**：🔄 进行中
**日期**：2026-02-13
**类型**：功能设计

## 1. 背景

构建一个基于 Python 的 AI Agent 框架（**mutagent**），让大语言模型（LLM）能够通过 Python 调用完成各种工作。核心价值在于为 LLM 提供一个**可运行时自我迭代的 Python 环境**——Agent 可以查看、修改代码并热重载验证，形成高效的开发闭环。

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
│                    mutagent                       │
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
mutagent/
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

Agent 负责管理对话循环：发送消息 → 接收响应 → 执行 tool calls → 反馈结果 → 继续对话。全部使用 async 接口。

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

统一使用 `async def` 接口，简化设计。同步操作在 async 函数中直接执行即可。

#### Tool 基类

```python
# tools/base.py
class Tool(forwardpy.Object):
    name: str
    description: str

    def get_schema(self) -> ToolSchema: ...       # 返回 JSON Schema 给 LLM
    async def execute(self, **params) -> str: ...  # 执行工具调用（统一 async）
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
| `module_path` | `str` | 可选，模块路径如 `mutagent.tools.base`，不填则从根模块开始 |
| `depth` | `int` | 展开深度，默认 2 |

**输出示例**：
```
mutagent/
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
| `target` | `str` | 目标路径，如 `mutagent.agent.core.Agent` 或 `mutagent.tools.base.Tool.execute` |

**实现要点**：
- 直接使用 `inspect.getsource()` 获取源码——运行时 patch 的代码通过 linecache 机制透明支持（见 2.8 节）
- 支持模块级、类级、函数/方法级查看

#### 2.6.3 `patch_module` — 运行时 patch

**功能**：将 Agent 生成的 Python 代码直接注入运行时，不写文件。这是 Agent 迭代的核心工具。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `module_path` | `str` | 目标模块路径（已有模块则 patch，不存在则创建虚拟模块） |
| `source` | `str` | 完整的 Python 源代码 |

**工作原理**（详见 2.8 节）：
1. 生成虚拟文件名 `mutagent://module_path`
2. 将 `source` 注入 `linecache.cache`（使 `inspect.getsource()` 透明工作）
3. 使用 `compile(source, filename, 'exec')` 编译
4. 在目标模块的命名空间中执行
5. 如果是新模块，创建 `types.ModuleType` 并注册到 `sys.modules`
6. forwardpy 的 `@impl(override=True)` 自动处理方法替换

**关键特性**：
- **零文件 IO**：代码只存在于运行时内存中
- **inspect 透明**：`inspect.getsource()` 对 patch 代码中的函数/类正常工作
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
- 写入文件后更新模块的 `__file__` 和 linecache，确保文件版本与运行时一致

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

### 2.8 运行时源码追踪（核心设计）

这是 mutagent 的关键基础设施。Agent 在运行时生成的代码必须对 Python 的 `inspect` 体系完全透明——`inspect.getsource()`、traceback 等应像操作普通文件代码一样工作。

#### 2.8.1 问题背景

Python 的 `inspect.getsource()` 调用链：
```
getsource(obj)
  → getsourcefile(obj)
    → getfile(obj)          # 获取文件名
      - 函数: obj.__code__.co_filename
      - 类: sys.modules[obj.__module__].__file__
      - 模块: obj.__file__
  → linecache.getlines()    # 通过文件名获取源码行
  → getblock()              # 提取函数/类代码块
```

动态生成的代码如果不做特殊处理，`inspect.getsource()` 会因找不到源文件而失败。

#### 2.8.2 解决方案：虚拟文件名 + linecache 注入 + loader 协议

**三层机制协同工作**：

| 层 | 机制 | 作用 |
|---|---|---|
| 1 | `linecache.cache` 注入 | 提供即时源码访问，`mtime=None` 免受 `checkcache()` 清理 |
| 2 | `__loader__` + `get_source()` | 自愈能力——`linecache` 被清空后可重新填充 |
| 3 | `sys.modules` 注册 | 使类的 `inspect.getfile()` 正常工作（类通过 `__module__` → `sys.modules` → `__file__` 路径解析） |

**虚拟文件名格式**：`mutagent://<module_path>`

不使用尖括号格式（如 `<dynamic>`），因为 linecache 的 `_source_unavailable()` 会将 `<...>` 格式视为"源码不可用"，阻止 `lazycache` / `__loader__` 机制工作。

#### 2.8.3 实现设计

```python
import types, sys, linecache, importlib.machinery

class ModuleManager:
    """运行时模块管理器，支持 PEP 302 loader 协议"""

    def __init__(self):
        self._sources: dict[str, str] = {}       # module_path → 当前源码
        self._history: dict[str, list[str]] = {}  # module_path → patch 历史

    def get_source(self, fullname: str) -> str | None:
        """PEP 302 loader 协议：linecache 自愈时调用"""
        return self._sources.get(fullname)

    def patch_module(self, module_path: str, source: str) -> types.ModuleType:
        """将源码注入运行时，使 inspect.getsource() 透明工作"""

        filename = f"mutagent://{module_path}"

        # 1. 存储源码（用于 __loader__.get_source() 自愈）
        self._sources[module_path] = source
        self._history.setdefault(module_path, []).append(source)

        # 2. 注入 linecache（即时可用，mtime=None 防清理）
        lines = [line + '\n' for line in source.splitlines()]
        linecache.cache[filename] = (len(source), None, lines, filename)

        # 3. 创建或获取模块
        if module_path in sys.modules:
            mod = sys.modules[module_path]
        else:
            mod = types.ModuleType(module_path)
            mod.__file__ = filename
            mod.__loader__ = self        # PEP 302: 使 lazycache 自愈生效
            mod.__spec__ = importlib.machinery.ModuleSpec(
                module_path, self, origin=filename
            )
            sys.modules[module_path] = mod  # 必须：类的 inspect 解析依赖此

        # 4. 编译并执行（co_filename 自动传播到所有内部函数/类）
        code = compile(source, filename, 'exec')
        exec(code, mod.__dict__)

        return mod
```

#### 2.8.4 效果验证

patch 后以下操作均正常工作：

| 操作 | 结果 |
|------|------|
| `inspect.getsource(func)` | 返回函数源码 |
| `inspect.getsource(cls)` | 返回类源码 |
| `inspect.getsource(cls.method)` | 返回方法源码 |
| `inspect.getsource(module)` | 返回整个模块源码 |
| `inspect.getfile(obj)` | 返回 `mutagent://module_path` |
| traceback | 显示 `mutagent://module_path` 及正确行号 |
| `help(obj)` | 正常显示文档 |

#### 2.8.5 固化时的过渡

当 `save_module` 将代码写入文件后：
1. 更新 `mod.__file__` 为实际文件路径
2. 更新 `linecache.cache`：移除虚拟文件名条目，让 linecache 从真实文件读取
3. 更新所有函数的 `__code__`：通过 `code.replace(co_filename=real_path)` 替换文件名
4. 从 `_sources` 中移除该模块（不再需要内存中的源码副本）

### 2.9 Claude API 对接细节

直接通过 asyncio + aiohttp 调用 Claude Messages API（`https://api.anthropic.com/v1/messages`）：

- **模型**：`claude-sonnet-4-20250514`（默认，可配置）
- **认证**：`x-api-key` header
- **Tool Use**：将 Tool 的 `get_schema()` 输出转为 Claude 的 tool 格式
- **消息格式映射**：
  - `user` / `assistant` 角色直接映射
  - `tool_use` content block → 内部 `ToolCall`
  - `tool_result` content block → 内部 `ToolResult`
- **MVP 不做流式**：直接使用普通 POST 请求等待完整响应

### 2.10 设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 包名 | `mutagent` | mutation + agent，表达"可变异/进化"核心概念，PyPI 可用 |
| HTTP 调用方式 | asyncio + aiohttp | 不依赖 SDK，更轻量可控 |
| 代码迭代方式 | 运行时 patch 优先 | 避免频繁文件 IO，快速验证 |
| 源码追踪 | linecache 注入 + loader 协议 | 使 `inspect.getsource()` 透明工作，三层自愈 |
| 虚拟文件名 | `mutagent://module_path` | 避免尖括号陷阱，兼容 lazycache 自愈机制 |
| Tool 接口 | 统一 `async def` | 简化设计，同步操作在 async 中直接执行 |
| 安全边界 | 无沙箱，仅超时 | MVP 面向开发者，后续可加 |
| 对话持久化 | 不持久化 | MVP 每次全新会话 |
| forwardpy 使用 | 所有核心类 | 自举能力，但不成为阻碍 |
| 核心抽象 | 模块路径而非文件路径 | `package.module.function` 是第一公民 |

## 3. 待定问题

### Q1: patch_module 的叠加语义

**问题**：多次 patch 同一模块时，语义应该是"增量叠加"还是"完整替换"？
- **增量叠加**：每次 patch 的代码 exec 到已有模块命名空间，新定义覆盖旧定义，未涉及的保持不变
- **完整替换**：每次 patch 清空模块命名空间后重新执行

两种方式影响 Agent 的使用体验：增量叠加更灵活（可以只 patch 一个函数），完整替换更可预测（不会有残留状态）。

**建议**：默认使用增量叠加（exec 到已有命名空间）。如果 Agent 需要完整替换，可以先传一个完整的模块源码。增量叠加更符合"迭代"的理念，Agent 可以小步修改。

回答，Agent必须有迭代一个函数的能力，当然需要有办法能迭代完整的模块。
这引出了一个新的需要细化的点，就是forwardpy在本工程中使用的源码规范是什么？可能是素有的可import 的py文件，都是只包含类型声明的，而具体的实现文件的扩展名可能是xxx.impl.py这种python无法直接导入的。这样可能会更清晰。同时声明和实现文件可能也可以存在不同的目录下，甚至不同的包中。这个设计很有意思，是的agent能够比较安全的patch一些实现，如果需要禁用掉这些path，不加载对应的实现模块就行。

### Q2: 虚拟模块的包层级

**问题**：当 Agent patch 一个深层路径如 `myproject.utils.helpers` 时，父包 `myproject` 和 `myproject.utils` 是否需要存在？如果不存在，是否自动创建虚拟父包？

**建议**：自动创建虚拟父包。当 patch `a.b.c` 时，如果 `a` 和 `a.b` 不在 `sys.modules` 中，自动创建空的 `types.ModuleType` 作为父包，并设置 `__path__` 属性使其成为 package。这样 `import a.b.c` 的语义保持一致。

同意。

### 补充说明：
1. 给这个工程中的核心类增加一个统一的基类，不要每个都从forwardpy.Object继承，实现的annotation也自行定义，为以后预留扩展能力。
2. 这个工程的另外一个核心设计就是tool和Python接口的对应，agent可以根据要处理的问题，自己提出需要哪些工具，然后在工具库了找到符合要求的工具再进行决策和迭代。 而工具本身应该是跟Agent运行时环境中的其他模块一样，也就是说，agent可以自行创建和迭代工具。设计中也请考虑这一点。 我觉得一开始可能只是提供了一些模块和类来完成获取模块结构，获取代码，修改代码，重载和调用等实现，然后有一个实现是工具选择，最开始的版本可以是返回所有核心基础工具（mutagent核心模块下的），随着要做的事情越来越多，ai要负责自行迭代工具选择的实现。 总结下来，就是未来所有的功能，都是基于一个简单可循环可进化的框架开始的。

## 4. 实施步骤清单

### 阶段零：PyPI 占位 [待开始]

- [ ] **Task 0.1**: 发布 mutagent 占位包到 PyPI
  - [ ] 创建最小 pyproject.toml
  - [ ] 通过 twine 发布
  - 状态：⏸️ 待开始

### 阶段一：项目基础设施 [待开始]

- [ ] **Task 1.1**: 初始化项目结构
  - [ ] 创建 mutagent 包目录结构
  - [ ] 配置 pyproject.toml（依赖：forwardpy, aiohttp）
  - [ ] 创建 `__init__.py` 入口
  - 状态：⏸️ 待开始

- [ ] **Task 1.2**: 消息模型定义
  - [ ] 定义 Message、ToolCall、ToolResult、Response、ToolSchema 数据类
  - [ ] 单元测试
  - 状态：⏸️ 待开始

### 阶段二：运行时核心（源码追踪） [待开始]

- [ ] **Task 2.1**: ModuleManager 核心
  - [ ] 实现 linecache 注入 + `__loader__` 协议
  - [ ] 实现 `patch_module()`（虚拟文件名、模块创建、代码编译执行）
  - [ ] 实现 patch 历史追踪
  - [ ] 实现虚拟父包自动创建
  - [ ] 单元测试：验证 `inspect.getsource()` 对 patch 后的函数/类/模块正常工作
  - 状态：⏸️ 待开始

- [ ] **Task 2.2**: 模块固化
  - [ ] 实现 `save_module()`（内存 → 文件）
  - [ ] 实现固化过渡（更新 `__file__`、`co_filename`、linecache）
  - [ ] 单元测试
  - 状态：⏸️ 待开始

### 阶段三：LLM Client [待开始]

- [ ] **Task 3.1**: LLM Client 声明
  - [ ] 实现 LLMClient forwardpy 声明（base.py）
  - [ ] 定义 async 接口方法签名
  - 状态：⏸️ 待开始

- [ ] **Task 3.2**: Claude 实现
  - [ ] 使用 aiohttp 直接调用 Claude Messages API
  - [ ] 实现消息格式转换（内部格式 ↔ Claude 格式）
  - [ ] 实现 tool schema 格式转换
  - [ ] 集成测试（需要 API Key）
  - 状态：⏸️ 待开始

### 阶段四：Tool 系统 [待开始]

- [ ] **Task 4.1**: Tool 基类与注册表
  - [ ] 实现 Tool forwardpy 声明
  - [ ] 实现 ToolRegistry
  - [ ] 单元测试
  - 状态：⏸️ 待开始

- [ ] **Task 4.2**: inspect_module 工具
  - [ ] 实现模块结构遍历（含虚拟模块）
  - [ ] 格式化输出
  - [ ] 单元测试
  - 状态：⏸️ 待开始

- [ ] **Task 4.3**: view_source 工具
  - [ ] 基于 `inspect.getsource()` 实现（自动支持 patch 源码）
  - [ ] 单元测试
  - 状态：⏸️ 待开始

- [ ] **Task 4.4**: patch_module 工具
  - [ ] 封装 ModuleManager.patch_module 为 Tool
  - [ ] 单元测试
  - 状态：⏸️ 待开始

- [ ] **Task 4.5**: save_module 工具
  - [ ] 封装 ModuleManager.save_module 为 Tool
  - [ ] 单元测试
  - 状态：⏸️ 待开始

- [ ] **Task 4.6**: execute 工具
  - [ ] 实现代码执行（受控命名空间）
  - [ ] 输出捕获 + 超时控制
  - [ ] 单元测试
  - 状态：⏸️ 待开始

### 阶段五：Agent 核心 [待开始]

- [ ] **Task 5.1**: Agent 声明与实现
  - [ ] 实现 Agent forwardpy 声明
  - [ ] 实现 agent 异步主循环（run / step / handle_tool_calls）
  - [ ] 单元测试（mock LLM）
  - 状态：⏸️ 待开始

- [ ] **Task 5.2**: 端到端集成
  - [ ] 组装所有组件
  - [ ] 实现简单的 main.py 入口
  - [ ] 端到端测试：Agent 使用 tools 完成简单任务
  - 状态：⏸️ 待开始

## 5. 测试验证

### 单元测试
- [ ] 消息模型序列化/反序列化
- [ ] ModuleManager: patch → inspect.getsource() 验证
- [ ] ModuleManager: 多次 patch 叠加
- [ ] ModuleManager: 固化过渡（虚拟 → 文件）
- [ ] ModuleManager: 虚拟父包创建
- [ ] Tool schema 生成
- [ ] ToolRegistry 注册/查找
- [ ] inspect_module 模块遍历（含虚拟模块）
- [ ] view_source 源码获取（文件源 + patch 源，统一用 inspect）
- [ ] patch_module 运行时注入
- [ ] save_module 文件固化
- [ ] execute 代码执行与输出捕获
- [ ] Agent 主循环（mock LLM 响应）

### 集成测试
- [ ] Claude API 实际调用测试（aiohttp 直连）
- [ ] Agent + Tools 端到端：让 Agent 查看模块结构 → patch 代码 → 执行验证 → 固化文件
