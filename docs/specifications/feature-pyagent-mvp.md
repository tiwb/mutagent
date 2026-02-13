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
- **自进化工具**：工具本身也是运行时模块，Agent 可以创建、迭代和进化工具

### 1.2 技术栈

- Python 3.11+
- forwardpy（声明-实现分离、热重载基础）
- asyncio + aiohttp（直接通过 HTTP 请求调用 LLM API，不使用 SDK）

## 2. 设计方案

### 2.1 整体架构

```
┌────────────────────────────────────────────────────────────┐
│                         mutagent                            │
│                                                            │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐  │
│  │ LLM Client │  │   Agent    │  │    Tool Selector     │  │
│  │  (Claude)  │──│   Core     │──│  (可被 Agent 迭代)   │  │
│  └────────────┘  └────────────┘  └──────────┬───────────┘  │
│                                              │              │
│                        ┌─────────────────────┤              │
│                        │                     │              │
│              ┌─────────┴────────┐  ┌─────────┴────────┐    │
│              │   Core Modules   │  │  Agent-Created   │    │
│              │   (核心原语)      │  │    Modules       │    │
│              │                  │  │  (Agent 创建的)   │    │
│              │  inspect_module  │  │                  │    │
│              │  view_source     │  │  (运行时动态     │    │
│              │  patch_module    │  │   生成和迭代)    │    │
│              │  save_module     │  │                  │    │
│              │  execute         │  │                  │    │
│              └──────────────────┘  └──────────────────┘    │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              声明 (.py) / 实现 (.impl.py)              │  │
│  │         forwardpy Runtime + mutagent Loader           │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### 2.2 声明与实现分离规范

mutagent 建立在 forwardpy 的声明-实现分离之上，并制定更严格的源码规范：

#### 2.2.1 文件规范

| 文件类型 | 扩展名 | 可 import | 内容 |
|---------|--------|-----------|------|
| 声明文件 | `.py` | 是 | 类声明、类型定义、stub 方法、接口契约 |
| 实现文件 | `.impl.py` | 否（需 mutagent loader） | `@impl` 实现、Extension 定义 |

**关键设计**：Python 标准 import 机制不会加载 `.impl.py` 文件（它不是合法的模块名映射），必须通过 mutagent 的 loader 显式加载。这创造了天然的安全边界：

- **声明 = 稳定契约**：`.py` 文件定义接口，可以安全 import，不包含可变逻辑
- **实现 = 可替换**：`.impl.py` 文件包含具体逻辑，可以被 Agent 安全 patch
- **禁用 patch = 不加载 impl**：如果需要回滚 Agent 的修改，只需不加载对应的 `.impl.py`

#### 2.2.2 目录组织

声明和实现文件可以灵活组织：

**方式 A：同目录**
```
mutagent/
├── agent/
│   ├── core.py              # 声明：class Agent(mutagent.Object): ...
│   └── core.impl.py         # 实现：@impl(Agent.run) def run(...): ...
```

**方式 B：分离目录**
```
mutagent/
├── agent/
│   └── core.py              # 声明
├── _impl/
│   └── agent/
│       └── core.impl.py     # 实现（可在不同包中）
```

**方式 C：Agent 运行时生成**
```
# Agent 在运行时 patch 一个实现，无需文件
manager.patch_module("mutagent._impl.agent.core", source="""
import mutagent
from mutagent.agent.core import Agent

@mutagent.impl(Agent.run, override=True)
async def run(self: Agent, user_input: str) -> str:
    # Agent 改进后的实现
    ...
""")
```

#### 2.2.3 Impl Loader

mutagent 提供自定义 loader 来发现和加载 `.impl.py` 文件：

```python
# runtime/impl_loader.py
class ImplLoader:
    """发现并加载 .impl.py 实现文件"""

    def discover(self, package_path: str) -> list[str]:
        """扫描目录，发现所有 .impl.py 文件"""
        ...

    def load(self, impl_path: str) -> None:
        """加载单个 .impl.py，执行其中的 @impl 注册"""
        ...

    def load_all(self, package_path: str) -> None:
        """加载包下所有 .impl.py"""
        ...
```

### 2.3 统一基类

所有 mutagent 核心类继承自统一基类 `mutagent.Object`，而非直接从 `forwardpy.Object` 继承：

```python
# mutagent/base.py
import forwardpy

class Object(forwardpy.Object):
    """mutagent 统一基类，为未来扩展预留能力"""
    pass

# mutagent/__init__.py
from mutagent.base import Object
from forwardpy import impl  # 重新导出，统一入口
```

**设计考虑**：
- 所有核心类（`LLMClient`、`Agent`、`Tool` 等）继承 `mutagent.Object`
- 未来可在 `mutagent.Object` 上添加通用能力（如运行时元数据、序列化等），不影响已有代码
- `mutagent.impl` 重新导出 `forwardpy.impl`，保持统一的使用入口
- forwardpy 作为底层实现细节，对 mutagent 用户透明

### 2.4 模块结构

```
mutagent/
├── __init__.py               # 导出 Object, impl 等核心接口
├── base.py                   # mutagent.Object 统一基类
├── client/
│   ├── __init__.py
│   ├── base.py               # LLMClient 声明
│   ├── messages.py           # 消息模型定义
│   └── claude.impl.py        # Claude API 实现
├── agent/
│   ├── __init__.py
│   ├── core.py               # Agent 声明
│   └── core.impl.py          # Agent 主循环实现
├── tools/
│   ├── __init__.py
│   ├── base.py               # Tool 基类声明
│   ├── selector.py           # ToolSelector 声明
│   ├── selector.impl.py      # 初始工具选择实现（Agent 可迭代）
│   └── builtins/             # 核心原语
│       ├── __init__.py
│       ├── inspect_module.py  # 模块结构查看
│       ├── view_source.py     # 查看源码
│       ├── patch_module.py    # 运行时 patch
│       ├── save_module.py     # 固化到文件
│       └── execute.py         # 执行验证
└── runtime/
    ├── __init__.py
    ├── module_manager.py      # 模块管理（patch、固化、源码追踪）
    └── impl_loader.py         # .impl.py 文件发现与加载
```

### 2.5 LLM Client 层

使用 asyncio + aiohttp 直接发送 HTTP 请求，不依赖任何 LLM SDK。

```python
# client/base.py - 声明
import mutagent

class LLMClient(mutagent.Object):
    """LLM 客户端接口"""
    model: str
    api_key: str
    base_url: str

    async def send_message(self, messages: list[Message], tools: list[ToolSchema]) -> Response: ...

# client/claude.impl.py - Claude 实现
import mutagent
from mutagent.client.base import LLMClient

@mutagent.impl(LLMClient.send_message)
async def send_message(self: LLMClient, messages, tools):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{self.base_url}/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={...}
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

### 2.6 Agent 核心

Agent 负责管理对话循环。全部使用 async 接口。

```python
# agent/core.py - 声明
import mutagent

class Agent(mutagent.Object):
    client: LLMClient
    tool_selector: ToolSelector
    system_prompt: str
    messages: list[Message]

    async def run(self, user_input: str) -> str: ...
    async def step(self) -> Response: ...
    async def handle_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolResult]: ...
```

**Agent 主循环**：
1. 用户输入 → 添加到 messages
2. 调用 `tool_selector.select(context)` 获取当前可用 tools
3. 调用 `step()` → 将 tools 和 messages 发送给 LLM → 返回响应
4. 如果响应包含 tool_calls → `handle_tool_calls()` 执行
5. 将 tool_results 添加到 messages → 回到步骤 2
6. 如果响应是 end_turn → 返回最终文本

### 2.7 Tool 系统（自进化设计）

#### 2.7.1 核心理念

Tool 不是一个封闭的类层级，而是**运行时模块中的 Python 可调用对象**。Agent 可以像操作任何其他模块一样创建和迭代 Tool。

关键区别：
- **传统设计**：Tool 是预定义的类，注册到 Registry
- **mutagent 设计**：Tool 是任何带类型标注的 Python 函数/方法，通过 ToolSelector 动态选择

#### 2.7.2 Tool 基类

```python
# tools/base.py
import mutagent

class Tool(mutagent.Object):
    """Tool 基类 — 将 Python 可调用对象暴露为 LLM tool"""
    name: str
    description: str

    def get_schema(self) -> ToolSchema: ...
    async def execute(self, **params) -> str: ...
```

核心内置 tools（`inspect_module`、`view_source` 等）继承此基类，是 Agent 最底层的操作原语。

#### 2.7.3 ToolSelector（可进化的工具选择器）

```python
# tools/selector.py - 声明
import mutagent

class ToolSelector(mutagent.Object):
    """工具选择器 — 决定哪些 tools 对 LLM 可用"""

    def select(self, context: dict) -> list[Tool]: ...

# tools/selector.impl.py - 初始实现（v0：返回所有核心 tools）
import mutagent
from mutagent.tools.selector import ToolSelector

@mutagent.impl(ToolSelector.select)
def select(self: ToolSelector, context: dict) -> list[Tool]:
    """MVP 版本：返回所有核心内置 tools"""
    return self._get_core_tools()
```

**进化路径**：
1. **v0（MVP）**：返回所有核心 tools（inspect、view、patch、save、execute）
2. **v1（Agent 自行迭代）**：Agent 根据任务上下文过滤 tools，减少 LLM 的认知负担
3. **v2+**：Agent 发现需要新工具时，自行创建新 Tool 模块 → patch 到运行时 → 注册到可选工具列表

#### 2.7.4 Agent 创建 Tool 的流程

```
Agent 遇到需要新工具的场景
  → patch_module("project.tools.file_search", source="...")  # 创建新 tool 模块
  → execute("from project.tools.file_search import ...")      # 验证
  → patch ToolSelector.select 的实现                          # 将新 tool 加入选择器
  → 后续 step 中 LLM 即可使用新 tool
```

这就是"自进化"的完整闭环：Agent 用核心原语（patch/execute）来创建新 tool，再用新 tool 解决更复杂的问题。

### 2.8 内置 Tools（核心原语）

这些是 Agent 的最小操作集，是所有高级能力的基础。

核心工作流：`inspect_module` → `view_source` → `patch_module` → `execute`（验证）→ `save_module`（固化）

#### 2.8.1 `inspect_module` — 模块结构查看

**功能**：基于 Python 运行时模块体系展示结构，精确到类和函数。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `module_path` | `str` | 可选，模块路径如 `mutagent.tools.base`，不填则从根模块开始 |
| `depth` | `int` | 展开深度，默认 2 |

**实现要点**：
- 使用 `importlib` + `inspect` 遍历已加载模块
- 通过 `pkgutil.walk_packages()` 发现子模块
- 用 `inspect.getmembers()` 获取类、函数、属性签名

#### 2.8.2 `view_source` — 查看源码

**功能**：查看指定模块、类或函数的源代码。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `target` | `str` | 目标路径，如 `mutagent.agent.core.Agent` |

**实现要点**：
- 直接使用 `inspect.getsource()` — 运行时 patch 的代码通过 linecache 机制透明支持（见 2.10 节）
- 支持模块级、类级、函数/方法级查看

#### 2.8.3 `patch_module` — 运行时 patch

**功能**：将 Agent 生成的 Python 代码直接注入运行时，不写文件。支持模块级和函数级粒度。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `module_path` | `str` | 目标模块路径（已有模块则增量 patch，不存在则创建虚拟模块） |
| `source` | `str` | Python 源代码（可以是完整模块，也可以是单个函数/类定义） |

**工作原理**（详见 2.10 节）：
1. 生成虚拟文件名 `mutagent://module_path`
2. 将 `source` 注入 `linecache.cache`
3. 使用 `compile(source, filename, 'exec')` 编译
4. 在目标模块的命名空间中执行（增量叠加）
5. forwardpy 的 `@impl(override=True)` 自动处理方法替换

**叠加语义**：默认增量叠加——新定义覆盖旧定义，未涉及的保持不变。Agent 可以只 patch 一个函数，也可以 patch 完整模块。

#### 2.8.4 `save_module` — 固化到文件

**功能**：将运行时验证通过的模块源码写入文件系统。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `module_path` | `str` | 要固化的模块路径 |
| `file_path` | `str` | 可选，目标文件路径。不填则自动推导 |

**实现要点**：
- 从 patch 历史中组装最终源码
- 自动创建必要的包目录和 `__init__.py`
- 写入文件后更新 `__file__`、linecache，确保一致

#### 2.8.5 `execute` — 执行验证

**功能**：在当前运行时执行 Python 代码片段，验证修改效果。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `code` | `str` | 要执行的 Python 代码 |

**实现要点**：
- 在受控命名空间中执行
- 捕获 stdout/stderr 和返回值
- 捕获异常并返回完整 traceback
- 设置执行超时

### 2.9 运行时模块管理

`runtime/module_manager.py` 统一负责：
- 维护模块索引（已加载模块 + 运行时 patch 的虚拟模块）
- 模块发现与加载（`importlib` + `pkgutil`）
- 运行时 patch 执行与历史记录
- 模块固化（内存 → 文件）
- 与 forwardpy 注册表交互，追踪声明和实现的对应关系

### 2.10 运行时源码追踪（核心设计）

Agent 在运行时生成的代码必须对 Python 的 `inspect` 体系完全透明。

#### 2.10.1 问题背景

Python 的 `inspect.getsource()` 调用链：
```
getsource(obj)
  → getfile(obj)          # 获取文件名
    - 函数: obj.__code__.co_filename
    - 类: sys.modules[obj.__module__].__file__
    - 模块: obj.__file__
  → linecache.getlines()  # 通过文件名获取源码行
  → getblock()            # 提取代码块
```

#### 2.10.2 解决方案：三层机制

| 层 | 机制 | 作用 |
|---|---|---|
| 1 | `linecache.cache` 注入 | 即时源码访问，`mtime=None` 免受 `checkcache()` 清理 |
| 2 | `__loader__` + `get_source()` | 自愈能力——`linecache` 被清空后可重新填充 |
| 3 | `sys.modules` 注册 | 使类的 `inspect.getfile()` 正常工作 |

**虚拟文件名格式**：`mutagent://module_path`（不用尖括号 `<>`，避免 linecache 的 `_source_unavailable()` 陷阱）

#### 2.10.3 实现设计

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

        # 1. 存储源码
        self._sources[module_path] = source
        self._history.setdefault(module_path, []).append(source)

        # 2. 注入 linecache（mtime=None 防清理）
        lines = [line + '\n' for line in source.splitlines()]
        linecache.cache[filename] = (len(source), None, lines, filename)

        # 3. 自动创建虚拟父包
        self._ensure_parent_packages(module_path)

        # 4. 创建或获取模块
        if module_path in sys.modules:
            mod = sys.modules[module_path]
        else:
            mod = types.ModuleType(module_path)
            mod.__file__ = filename
            mod.__loader__ = self
            mod.__spec__ = importlib.machinery.ModuleSpec(
                module_path, self, origin=filename
            )
            sys.modules[module_path] = mod

        # 5. 编译并执行（增量叠加到已有命名空间）
        code = compile(source, filename, 'exec')
        exec(code, mod.__dict__)

        return mod

    def _ensure_parent_packages(self, module_path: str) -> None:
        """自动创建虚拟父包"""
        parts = module_path.split('.')
        for i in range(1, len(parts)):
            parent = '.'.join(parts[:i])
            if parent not in sys.modules:
                pkg = types.ModuleType(parent)
                pkg.__path__ = []  # 标记为 package
                pkg.__package__ = parent
                sys.modules[parent] = pkg
```

#### 2.10.4 固化时的过渡

当 `save_module` 将代码写入文件后：
1. 更新 `mod.__file__` 为实际文件路径
2. 更新 `linecache.cache`：移除虚拟文件名条目
3. 更新所有函数的 `__code__`：`code.replace(co_filename=real_path)`
4. 从 `_sources` 中移除该模块

### 2.11 Claude API 对接细节

直接通过 asyncio + aiohttp 调用 Claude Messages API（`https://api.anthropic.com/v1/messages`）：

- **模型**：`claude-sonnet-4-20250514`（默认，可配置）
- **认证**：`x-api-key` header
- **Tool Use**：将 Tool 的 `get_schema()` 输出转为 Claude 的 tool 格式
- **消息格式映射**：
  - `user` / `assistant` 角色直接映射
  - `tool_use` content block → 内部 `ToolCall`
  - `tool_result` content block → 内部 `ToolResult`
- **MVP 不做流式**：直接使用普通 POST 请求等待完整响应

### 2.12 设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 包名 | `mutagent` | mutation + agent，PyPI 可用 |
| 统一基类 | `mutagent.Object` | 封装 forwardpy.Object，预留扩展 |
| 声明/实现规范 | `.py` / `.impl.py` | 声明可 import、实现需 loader，天然安全边界 |
| HTTP 调用 | asyncio + aiohttp | 不依赖 SDK |
| patch 语义 | 增量叠加（默认） | 支持函数级迭代，也可整模块替换 |
| 源码追踪 | linecache + loader 协议 | `inspect.getsource()` 透明工作 |
| 虚拟文件名 | `mutagent://module_path` | 避免尖括号陷阱 |
| Tool 接口 | 统一 `async def` | 简化设计 |
| Tool 选择 | ToolSelector（可进化） | 初始返回全部，Agent 可迭代选择逻辑 |
| 安全边界 | 无沙箱，仅超时 | MVP 面向开发者 |
| 对话持久化 | 不持久化 | MVP 每次全新会话 |
| 核心抽象 | 模块路径 | `package.module.function` 是第一公民 |

## 3. 待定问题

### Q1: .impl.py 的发现与加载策略

**问题**：`.impl.py` 文件的发现和加载时机是什么？有几种选择：

- **方式 A：启动时全量扫描**：mutagent 启动时扫描所有包目录，自动发现并加载所有 `.impl.py`
- **方式 B：按需延迟加载**：当首次访问某个 stub 方法时，搜索对应的 `.impl.py` 并加载
- **方式 C：显式注册**：在 `__init__.py` 或配置中显式列出需要加载的 `.impl.py`

**建议**：MVP 使用方式 A（启动时全量扫描）——简单直接。mutagent 启动时调用 `ImplLoader.load_all(package_path)` 扫描并加载所有 `.impl.py`。延迟加载可作为后续优化。

同意，这块以后可以进化

### Q2: 增量 patch 的 linecache 一致性

**问题**：当 Agent 对同一模块进行多次增量 patch 时，linecache 中存储的源码与模块实际状态会不一致。例如：
- 第一次 patch：定义 `func_a` 和 `func_b`（linecache 记录完整源码）
- 第二次 patch：只重定义 `func_a`（linecache 更新为只含 `func_a` 的新源码）
- 此时 `inspect.getsource(mod.func_b)` 会失败，因为 linecache 中的新源码不包含 `func_b`

**建议**：每次增量 patch 时，将新代码追加到当前源码末尾，形成完整的累积源码存储到 linecache。这样 `inspect.getsource()` 始终能从累积源码中找到所有定义。但这带来一个副作用：源码中可能有同名函数的多个版本（旧版本在前，新版本在后），`inspect` 会返回第一个匹配。需要进一步思考这个问题。

回答：确实这是一个比较复杂的问题，我觉得我们需要保证patch机制的简单和直观。也就是说，我patch了一个模块以后，行为应该是跟我写文件然后重新启动是一样的。所以，模块里的未定义的函数应该是被卸载的。我觉得基于forwardpy的设计这是能实现的，在重新定义一个模块时，之前模块里override的所有函数可以先卸载，然后被override。这可能需要forwardpy拥有对应的能力。没有关系，只要能实现，我们就可以假设这个模块是拥有的。可以同时迭代forwardpy和mutagent。至于声明文件，帮我想想看有没有方案。
最重要的是： 我们不需要能够patch任何python模块，只需要在mutagent框架下的类行能正常工作即可。

### Q3: mutagent.Object 的初始扩展

**问题**：`mutagent.Object` 在 MVP 阶段是否需要在 `forwardpy.Object` 基础上添加额外能力，还是先做纯透传？

**建议**：MVP 先做纯透传（`class Object(forwardpy.Object): pass`）。预留的扩展方向包括：
- 运行时元数据（如创建时间、版本号）
- 自省增强（快速查看声明与实现的对应关系）
- 序列化/反序列化支持

先建立规范，后续按需添加。

同意

## 4. 实施步骤清单

### 阶段零：PyPI 占位 [待开始]

- [ ] **Task 0.1**: 发布 mutagent 占位包到 PyPI
  - [ ] 创建最小 pyproject.toml
  - [ ] 通过 twine 发布
  - 状态：⏸️ 待开始

### 阶段一：项目基础设施 [待开始]

- [ ] **Task 1.1**: 初始化项目结构
  - [ ] 创建 mutagent 包目录结构（按 2.4 节）
  - [ ] 配置 pyproject.toml（依赖：forwardpy, aiohttp）
  - [ ] 实现 `mutagent.Object` 统一基类
  - [ ] 创建 `__init__.py`（导出 Object, impl）
  - 状态：⏸️ 待开始

- [ ] **Task 1.2**: 消息模型定义
  - [ ] 定义 Message、ToolCall、ToolResult、Response、ToolSchema
  - [ ] 单元测试
  - 状态：⏸️ 待开始

### 阶段二：运行时核心 [待开始]

- [ ] **Task 2.1**: ModuleManager 核心
  - [ ] 实现 linecache 注入 + `__loader__` 协议
  - [ ] 实现 `patch_module()`（增量叠加语义）
  - [ ] 实现 patch 历史追踪
  - [ ] 实现虚拟父包自动创建
  - [ ] 单元测试：`inspect.getsource()` 对 patch 后的函数/类/模块正常工作
  - 状态：⏸️ 待开始

- [ ] **Task 2.2**: ImplLoader
  - [ ] 实现 `.impl.py` 文件发现
  - [ ] 实现加载与 `@impl` 注册
  - [ ] 单元测试
  - 状态：⏸️ 待开始

- [ ] **Task 2.3**: 模块固化
  - [ ] 实现 `save_module()`（内存 → 文件）
  - [ ] 实现固化过渡（更新 `__file__`、`co_filename`、linecache）
  - [ ] 单元测试
  - 状态：⏸️ 待开始

### 阶段三：LLM Client [待开始]

- [ ] **Task 3.1**: LLM Client 声明
  - [ ] 实现 LLMClient 声明（base.py）
  - [ ] 定义 async 接口方法签名
  - 状态：⏸️ 待开始

- [ ] **Task 3.2**: Claude 实现
  - [ ] 使用 aiohttp 直接调用 Claude Messages API（claude.impl.py）
  - [ ] 实现消息格式转换
  - [ ] 实现 tool schema 格式转换
  - [ ] 集成测试
  - 状态：⏸️ 待开始

### 阶段四：Tool 系统 [待开始]

- [ ] **Task 4.1**: Tool 基类与 ToolSelector
  - [ ] 实现 Tool 声明（base.py）
  - [ ] 实现 ToolSelector 声明（selector.py）
  - [ ] 实现 ToolSelector 初始实现（selector.impl.py：返回所有核心 tools）
  - [ ] 单元测试
  - 状态：⏸️ 待开始

- [ ] **Task 4.2**: 核心 tools 实现
  - [ ] inspect_module
  - [ ] view_source
  - [ ] patch_module（封装 ModuleManager）
  - [ ] save_module（封装 ModuleManager）
  - [ ] execute
  - [ ] 各工具单元测试
  - 状态：⏸️ 待开始

### 阶段五：Agent 核心 [待开始]

- [ ] **Task 5.1**: Agent 声明与实现
  - [ ] 实现 Agent 声明（core.py）
  - [ ] 实现 agent 异步主循环（core.impl.py）
  - [ ] 单元测试（mock LLM）
  - 状态：⏸️ 待开始

- [ ] **Task 5.2**: 端到端集成
  - [ ] 组装所有组件
  - [ ] ImplLoader 加载所有 .impl.py
  - [ ] 实现 main.py 入口
  - [ ] 端到端测试：Agent 查看模块 → patch 代码 → 执行验证 → 固化文件
  - 状态：⏸️ 待开始

## 5. 测试验证

### 单元测试
- [ ] mutagent.Object 基类继承
- [ ] 消息模型序列化/反序列化
- [ ] ModuleManager: patch → inspect.getsource() 验证
- [ ] ModuleManager: 多次增量 patch
- [ ] ModuleManager: 固化过渡（虚拟 → 文件）
- [ ] ModuleManager: 虚拟父包创建
- [ ] ImplLoader: .impl.py 发现与加载
- [ ] Tool schema 生成
- [ ] ToolSelector: 初始版本返回所有核心 tools
- [ ] 各 builtin tool 功能测试
- [ ] Agent 主循环（mock LLM 响应）

### 集成测试
- [ ] Claude API 实际调用测试（aiohttp 直连）
- [ ] Agent + Tools 端到端：Agent 查看模块 → patch 代码 → 执行验证 → 固化文件
- [ ] 自进化验证：Agent 创建新 tool → 注册到 ToolSelector → 使用新 tool
