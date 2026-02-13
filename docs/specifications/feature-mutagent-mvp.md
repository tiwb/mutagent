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
│              │   Essential      │  │  Agent-Created   │    │
│              │   Tools          │  │    Modules       │    │
│              │  (最小工具集)     │  │  (Agent 创建的)   │    │
│              │  inspect_module  │  │                  │    │
│              │  view_source     │  │  (运行时动态     │    │
│              │  patch_module    │  │   生成和迭代)    │    │
│              │  save_module     │  │                  │    │
│              │  run_code        │  │                  │    │
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

**方式 A：mutagent 的组织方式（声明扁平 + builtins 集中实现）**
```
mutagent/
├── agent.py                # 声明：class Agent(mutagent.Object): ...
├── builtins/
│   └── agent.impl.py      # 实现：@impl(Agent.run) def run(...): ...
```

**方式 B：同目录（用户项目可选）**
```
myproject/
├── processor.py            # 声明
└── processor.impl.py       # 实现
```

**方式 C：Agent 运行时生成**
```
# Agent 在运行时 patch 一个实现，无需文件
manager.patch_module("mutagent.builtins.agent", source="""
import mutagent
from mutagent.agent import Agent

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

### 2.3 统一基类与元类

所有 mutagent 核心类继承自 `mutagent.Object`，使用 `MutagentMeta` 元类支持就地类更新：

```python
# mutagent/base.py
import forwardpy

class MutagentMeta(forwardpy.ObjectMeta):
    """支持类的就地重定义（见 2.9.2 节）"""
    _class_registry: dict[tuple[str, str], type] = {}
    # ... 详见 2.9.2

class Object(forwardpy.Object, metaclass=MutagentMeta):
    """mutagent 统一基类，MVP 阶段纯透传"""
    pass

# mutagent/__init__.py
from mutagent.base import Object, MutagentMeta
from forwardpy import impl  # 重新导出，统一入口
```

**设计考虑**：
- 所有核心类（`LLMClient`、`Agent`、`ToolSelector`、`EssentialTools` 等）继承 `mutagent.Object`
- `MutagentMeta` 扩展 `forwardpy.ObjectMeta`，增加就地类更新能力
- 未来可在 `mutagent.Object` 上添加通用能力，不影响已有代码
- forwardpy 作为底层实现细节，对 mutagent 用户透明

### 2.4 模块结构

```
mutagent/
├── __init__.py              # 导出 Object, impl 等核心接口
├── base.py                  # mutagent.Object + MutagentMeta
├── agent.py                 # Agent 声明
├── selector.py              # ToolSelector 声明
├── essential_tools.py       # EssentialTools 声明（最小工具集）
├── client.py                # LLMClient 声明
├── messages.py              # 消息模型定义（Message, ToolCall, ToolResult, Response, ToolSchema）
├── builtins/                # 所有默认实现（框架的"默认人格"，全部可被 Agent 替换）
│   ├── __init__.py
│   ├── agent.impl.py        # Agent.run, Agent.step, Agent.handle_tool_calls
│   ├── selector.impl.py     # ToolSelector.get_tools, ToolSelector.dispatch
│   ├── claude.impl.py       # LLMClient.send_message (Claude API)
│   ├── inspect_module.impl.py   # EssentialTools.inspect_module
│   ├── view_source.impl.py      # EssentialTools.view_source
│   ├── patch_module.impl.py     # EssentialTools.patch_module
│   ├── save_module.impl.py      # EssentialTools.save_module
│   └── run_code.impl.py         # EssentialTools.run_code
└── runtime/                 # 基础设施（非典型 Agent patch 目标）
    ├── __init__.py
    ├── module_manager.py    # ModuleManager（patch、固化、源码追踪）
    └── impl_loader.py       # ImplLoader（.impl.py 发现与加载）
```

**目录分层**：
- `mutagent/*.py` — 所有声明（Agent 浏览 `inspect_module("mutagent")` 一次看全）
- `mutagent/builtins/` — 所有默认实现（Agent 可逐个或全部替换）
- `mutagent/runtime/` — 基础设施层（ModuleManager, ImplLoader）

### 2.5 LLM Client 层

使用 asyncio + aiohttp 直接发送 HTTP 请求，不依赖任何 LLM SDK。

```python
# mutagent/client.py - 声明
import mutagent

class LLMClient(mutagent.Object):
    """LLM 客户端接口"""
    model: str
    api_key: str
    base_url: str

    async def send_message(self, messages: list[Message], tools: list[ToolSchema]) -> Response: ...

# mutagent/builtins/claude.impl.py - Claude 实现
import mutagent
from mutagent.client import LLMClient

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

**消息模型**（`mutagent/messages.py`）：
- `Message`：统一消息格式（role, content, tool_calls, tool_results）
- `ToolCall`：LLM 发起的工具调用（tool_name, arguments, id）
- `ToolResult`：工具执行结果（tool_call_id, content, is_error）
- `Response`：LLM 响应（message, stop_reason, usage）
- `ToolSchema`：工具的 JSON Schema 描述

### 2.6 Agent 核心

Agent 负责管理对话循环。全部使用 async 接口。

```python
# mutagent/agent.py - 声明
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
2. 调用 `await tool_selector.get_tools(context)` 获取当前可用 tool schemas
3. 调用 `step()` → 将 tools 和 messages 发送给 LLM → 返回响应
4. 如果响应包含 tool_calls → `handle_tool_calls()` 通过 `await tool_selector.dispatch(call)` 逐个执行
5. 将 tool_results 添加到 messages → 回到步骤 2
6. 如果响应是 end_turn → 返回最终文本

### 2.7 Tool 系统（自进化设计）

#### 2.7.1 核心理念

mutagent 不定义"什么是工具"。**ToolSelector 决定如何发现、呈现和调用工具**。但有一个基本约束：

**框架下的一切都是声明+实现，一切都可被 patch。**

这意味着：
- 内置工具是 `mutagent.Object` 子类上的**方法声明**（stub），实现通过 `@impl` 注册
- Agent 关注的是**声明**（类和方法签名），而非具体实现
- 普通函数是实现细节，Agent 知道直接 patch 这些会导致混乱
- 如果外部函数需要被 Agent 使用，需要先包装为框架下的对象（声明+实现）
- Agent 创建新工具 = 创建新的 `mutagent.Object` 子类 + `@impl`

关键区别：
- **传统框架**：定义 Tool 基类 → 实现者继承 → 注册到 Registry → 框架统一调用 `tool.execute()`
- **mutagent 设计**：工具是 mutagent.Object 子类上的方法 → **ToolSelector 决定如何发现、呈现和调用它们** → 没有强制的公共接口

#### 2.7.2 EssentialTools — 内置工具声明

内置工具组织为一个 `mutagent.Object` 子类，每个工具是一个独立的方法声明。"Essential" 表达**当前不可缺少的最小集**，但不暗示永恒不变——Agent 可以进化甚至替换这些工具。

```python
# mutagent/essential_tools.py — 声明
import mutagent

class EssentialTools(mutagent.Object):
    """必要工具原语 — Agent 进化的最小操作集

    每个方法是一个独立的工具声明。Agent 可以通过 @impl override 替换任何工具的实现，
    也可以 patch 这个类来增删工具方法。
    """
    module_manager: 'ModuleManager'

    def inspect_module(self, module_path: str = "", depth: int = 2) -> str: ...
    def view_source(self, target: str) -> str: ...
    def patch_module(self, module_path: str, source: str) -> str: ...
    def save_module(self, module_path: str, file_path: str = "") -> str: ...
    def run_code(self, code: str) -> str: ...
```

**设计要点**：
- **每个方法独立 patchable**：`@impl(EssentialTools.inspect_module, override=True)` 只替换一个工具
- **依赖在实例上**：`module_manager` 作为属性，方法实现通过 `self.module_manager` 访问
- **声明即 API**：Agent 通过 `view_source("mutagent.essential_tools.EssentialTools")` 看到所有可用工具
- **可扩展**：Agent 可以 patch EssentialTools 声明来增加新方法，也可以创建全新的工具类

实现按工具拆分为独立 `.impl.py` 文件，每个工具可独立替换：

```python
# mutagent/builtins/inspect_module.impl.py
import mutagent
from mutagent.essential_tools import EssentialTools

@mutagent.impl(EssentialTools.inspect_module)
def inspect_module(self: EssentialTools, module_path="", depth=2):
    # 使用 importlib + inspect 遍历模块
    ...

# mutagent/builtins/patch_module.impl.py
@mutagent.impl(EssentialTools.patch_module)
def patch_module(self: EssentialTools, module_path: str, source: str):
    return self.module_manager.patch_module(module_path, source)
    # ModuleManager 通过 self 访问，无需外部注入
```

#### 2.7.3 ToolSelector — Agent 与工具的唯一桥梁

ToolSelector 是框架中唯一定义的工具相关抽象。它全部使用 async 接口，因为工具选择本身可能涉及 LLM 推理（分析需要什么工具、查询现有工具集、决定是否创造新工具）。

```python
# mutagent/selector.py - 声明
import mutagent

class ToolSelector(mutagent.Object):
    """工具选择与调度 — Agent 与工具之间的唯一桥梁

    职责：
    1. 决定当前上下文下哪些工具对 LLM 可用（get_tools）
    2. 将 LLM 的工具调用路由到具体实现（dispatch）

    这两个职责的实现方式完全由 impl 决定，Agent 可以迭代进化。
    """

    async def get_tools(self, context: dict) -> list[ToolSchema]: ...
    async def dispatch(self, tool_call: ToolCall) -> ToolResult: ...
```

```python
# mutagent/builtins/selector.impl.py - MVP 初始实现
import mutagent
from mutagent.selector import ToolSelector

@mutagent.impl(ToolSelector.get_tools)
async def get_tools(self: ToolSelector, context: dict) -> list[ToolSchema]:
    """MVP：从 EssentialTools 的方法签名自动生成 schema"""
    return make_schemas_from_methods(self.essential_tools, [
        'inspect_module', 'view_source', 'patch_module', 'save_module', 'run_code'
    ])

@mutagent.impl(ToolSelector.dispatch)
async def dispatch(self: ToolSelector, tool_call: ToolCall) -> ToolResult:
    """MVP：直接调用 EssentialTools 上的对应方法"""
    method = getattr(self.essential_tools, tool_call.name)
    try:
        result = method(**tool_call.arguments)
        if asyncio.iscoroutine(result):
            result = await result
        return ToolResult(tool_call_id=tool_call.id, content=str(result))
    except Exception as e:
        return ToolResult(tool_call_id=tool_call.id, content=str(e), is_error=True)
```

#### 2.7.4 进化路径

```
v0（MVP）
  EssentialTools 声明 5 个必要方法
  ToolSelector.get_tools → 从 EssentialTools 方法签名生成 schema
  ToolSelector.dispatch  → getattr(essential_tools, name)(**args)

v1（Agent 迭代工具实现）
  → @impl(EssentialTools.inspect_module, override=True) 替换某个工具的实现
  → 工具变得更智能，但声明不变

v2（Agent 创建新工具类）
  → patch_module 创建新的 mutagent.Object 子类（有新的方法声明）
  → patch ToolSelector 实现，将新类的方法也纳入调度

v3（Agent 进化 ToolSelector 为 LLM 驱动）
  → get_tools 内部调用 LLM 分析"我需要什么工具？"
  → 搜索运行时中的所有 mutagent.Object 子类，发现可用方法
  → 判断是否需要创造新工具
```

#### 2.7.5 Agent 创建工具的流程

```
Agent 遇到需要新工具的场景
  → patch_module("project.tools.search", source="""
      import mutagent
      class SearchTools(mutagent.Object):
          def grep_modules(self, pattern: str) -> str: ...
    """)                                                       # 创建新工具类声明
  → patch_module("project.tools.search_impl", source="""
      from project.tools.search import SearchTools
      @mutagent.impl(SearchTools.grep_modules)
      def grep_modules(self, pattern): ...
    """)                                                       # 提供实现
  → run_code("from project.tools.search import SearchTools")   # 验证
  → patch ToolSelector 的 get_tools/dispatch 实现               # 将新工具纳入选择器
  → 后续 step 中 LLM 即可使用新工具
```

这就是"自进化"的完整闭环：Agent 用核心原语（patch/run_code）来创建新工具类，再用新工具解决更复杂的问题。

### 2.8 内置工具（核心原语）

这些是 Agent 的最小操作集，是所有高级能力的基础。它们是 `EssentialTools` 类上的**方法声明**，实现通过 `@impl` 注册在 `builtins/` 的独立 `.impl.py` 文件中。

核心工作流：`inspect_module` → `view_source` → `patch_module` → `run_code`（验证）→ `save_module`（固化）

#### 2.8.1 `inspect_module` — 模块结构查看

**功能**：基于 Python 运行时模块体系展示结构，精确到类和函数。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `module_path` | `str` | 可选，模块路径如 `mutagent.essential_tools`，不填则从根模块开始 |
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
| `target` | `str` | 目标路径，如 `mutagent.agent.Agent` |

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
4. 完全替换目标模块命名空间（patch = 写文件 + 重启）
5. forwardpy 的 `@impl(override=True)` 自动处理方法替换

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

#### 2.8.5 `run_code` — 执行验证

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

### 2.9 Patch 语义（核心设计）

**核心原则**：patch 行为等同于"写文件 + 重启"。patch 一个模块后，旧模块的状态被完全替换，而非增量叠加。

只需要 **mutagent 框架下的类**（继承自 `mutagent.Object`）能正常工作，不需要支持任意 Python 模块。

#### 2.9.1 实现模块的 Patch（`.impl.py`）

实现模块包含 `@impl` 注册。Patch 流程：

```
patch_module("pkg.foo.impl", new_source):
  1. forwardpy.unregister_module_impls("pkg.foo.impl")
     → 移除该模块注册的所有 @impl，方法恢复为 stub 或前一个 impl
  2. 清空模块命名空间（保留 __name__, __file__ 等系统属性）
  3. compile(new_source) + exec(code, mod.__dict__)
     → 新的 @impl 装饰器重新注册实现
  4. 更新 linecache 为 new_source
```

**效果**：
- 新源码中定义的 `@impl` → 注册为新实现
- 旧源码中有但新源码中没有的 `@impl` → 被卸载，方法恢复为 stub
- 行为完全等同于删掉旧 `.impl.py`、写入新内容、重新加载

**forwardpy 需要的能力**：
```python
# 需要 forwardpy 新增的 API
forwardpy.unregister_module_impls(module_name: str) -> None
    """移除指定模块注册的所有 @impl，恢复为 stub 或前一个 impl"""

# 内部实现：@impl 注册时记录来源模块
# _method_registry[class][method_name] = [(impl_func, source_module), ...]
# unregister 时按 source_module 过滤移除
```

#### 2.9.2 声明模块的 Patch（`.py`）

声明模块包含 `mutagent.Object` 子类的定义。Patch 声明的挑战：

**问题**：重新 exec 一个包含 `class Foo(mutagent.Object)` 的模块会创建**新的** Foo 类对象。旧的 `@impl` 注册、`isinstance` 检查、已有实例都指向旧类对象，会全部失效。

**解决方案**：通过 `mutagent.Object` 的元类实现**就地类更新**。

```python
class MutagentMeta(forwardpy.ObjectMeta):
    """mutagent 元类：支持类的就地重定义"""
    _class_registry: dict[tuple[str, str], type] = {}

    def __new__(mcs, name, bases, namespace, **kwargs):
        module = namespace.get('__module__', '')
        qualname = namespace.get('__qualname__', name)
        key = (module, qualname)

        existing = mcs._class_registry.get(key)
        if existing is not None:
            # 就地更新已有类，而非创建新对象
            mcs._update_class_inplace(existing, namespace)
            return existing  # 返回同一个类对象

        # 首次定义：正常创建
        cls = super().__new__(mcs, name, bases, namespace)
        mcs._class_registry[key] = cls
        return cls

    @staticmethod
    def _update_class_inplace(cls, new_namespace):
        """就地更新类：删旧增新改变"""
        # 1. 收集旧定义（排除系统属性）
        old_attrs = {k for k in cls.__dict__ if not k.startswith('__')}
        new_attrs = {k for k in new_namespace if not k.startswith('__')}

        # 2. 删除旧定义中不存在于新定义的属性
        for attr in old_attrs - new_attrs:
            delattr(cls, attr)

        # 3. 设置新定义的属性
        for attr in new_attrs:
            setattr(cls, attr, new_namespace[attr])

        # 4. 更新 __annotations__
        if '__annotations__' in new_namespace:
            cls.__annotations__ = new_namespace['__annotations__']

        # 5. 对新的描述符调用 __set_name__
        for attr in new_attrs - old_attrs:
            val = new_namespace[attr]
            if hasattr(val, '__set_name__'):
                val.__set_name__(cls, attr)
```

**效果**：
- **类对象身份不变**：`id(Agent)` 不变，`isinstance` 正常
- **已有实例不受影响**：实例的 `__class__` 仍指向同一个类对象
- **@impl 注册不断裂**：因为类对象没变，之前注册在该类上的 impl 仍然有效
- **可增删方法和属性**：从声明中移除的 stub 方法被 `delattr`，新增的被 `setattr`

**注意事项**：
- `mutagent.Object` 不使用 `__slots__`（slots 无法在类创建后修改）
- 使用 `super()` 的方法需要特殊处理 `__class__` 闭包变量（通过更新现有函数的 `__code__` 而非替换函数对象）

#### 2.9.3 声明模块 Patch 的完整流程

```
patch_module("pkg.foo", new_source):
  1. forwardpy.unregister_module_impls("pkg.foo")  # 以防声明模块中有 impl
  2. 清空模块命名空间（保留系统属性）
  3. compile(new_source) + exec(code, mod.__dict__)
     → MutagentMeta.__new__ 拦截类定义：
       - 已有类：就地更新（删旧增新），返回原对象
       - 新类：正常创建并注册
  4. 清理：移除模块中不再存在的旧定义
  5. 更新 linecache 为 new_source
```

#### 2.9.4 forwardpy 扩展需求汇总

| 需求 | 说明 | 优先级 |
|------|------|--------|
| 模块来源追踪 | `@impl` 注册时记录 `source_module` | MVP 必需 |
| 模块 impl 卸载 | `unregister_module_impls(module_name)` | MVP 必需 |
| impl 回退 | 卸载后恢复为 stub 或前一个 impl | MVP 必需 |
| 就地类更新 | `ObjectMeta` 支持返回已有类（或由 mutagent 扩展元类实现） | MVP 必需 |

这些扩展可以在 forwardpy 中实现，也可以在 mutagent 的扩展元类 `MutagentMeta` 中实现。两个项目可以同步迭代。

### 2.10 运行时模块管理与源码追踪

#### 2.10.1 源码追踪：三层机制

Agent 在运行时生成的代码必须对 Python 的 `inspect` 体系完全透明。

| 层 | 机制 | 作用 |
|---|---|---|
| 1 | `linecache.cache` 注入 | 即时源码访问，`mtime=None` 免受 `checkcache()` 清理 |
| 2 | `__loader__` + `get_source()` | 自愈能力——`linecache` 被清空后可重新填充 |
| 3 | `sys.modules` 注册 | 使类的 `inspect.getfile()` 正常工作 |

**虚拟文件名格式**：`mutagent://module_path`（不用尖括号 `<>`，避免 linecache 的 `_source_unavailable()` 陷阱）

#### 2.10.2 ModuleManager 实现

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
        """patch = 写文件 + 重启。完全替换模块内容。"""

        filename = f"mutagent://{module_path}"

        # 1. 卸载旧模块的所有 @impl 注册
        forwardpy.unregister_module_impls(module_path)

        # 2. 存储源码
        self._sources[module_path] = source
        self._history.setdefault(module_path, []).append(source)

        # 3. 注入 linecache（mtime=None 防清理）
        lines = [line + '\n' for line in source.splitlines()]
        linecache.cache[filename] = (len(source), None, lines, filename)

        # 4. 自动创建虚拟父包
        self._ensure_parent_packages(module_path)

        # 5. 创建或重置模块
        if module_path in sys.modules:
            mod = sys.modules[module_path]
            # 清空命名空间（保留系统属性）
            self._reset_module_namespace(mod)
        else:
            mod = types.ModuleType(module_path)
            mod.__file__ = filename
            mod.__loader__ = self
            mod.__spec__ = importlib.machinery.ModuleSpec(
                module_path, self, origin=filename
            )
            sys.modules[module_path] = mod

        # 6. 编译并执行（全新命名空间）
        code = compile(source, filename, 'exec')
        exec(code, mod.__dict__)
        # MutagentMeta 自动处理类的就地更新

        return mod

    def _reset_module_namespace(self, mod):
        """清空模块命名空间，保留系统属性"""
        keep = {'__name__', '__file__', '__loader__', '__spec__',
                '__path__', '__package__', '__builtins__'}
        for key in list(mod.__dict__):
            if key not in keep:
                del mod.__dict__[key]

    def _ensure_parent_packages(self, module_path: str) -> None:
        """自动创建虚拟父包"""
        parts = module_path.split('.')
        for i in range(1, len(parts)):
            parent = '.'.join(parts[:i])
            if parent not in sys.modules:
                pkg = types.ModuleType(parent)
                pkg.__path__ = []
                pkg.__package__ = parent
                sys.modules[parent] = pkg
```

#### 2.10.3 固化时的过渡

当 `save_module` 将代码写入文件后：
1. 更新 `mod.__file__` 为实际文件路径
2. 更新 `linecache.cache`：移除虚拟文件名条目
3. 更新所有函数的 `__code__`：`code.replace(co_filename=real_path)`
4. 从 `_sources` 中移除该模块

### 2.11 Claude API 对接细节

直接通过 asyncio + aiohttp 调用 Claude Messages API（`https://api.anthropic.com/v1/messages`）：

- **模型**：`claude-sonnet-4-20250514`（默认，可配置）
- **认证**：`x-api-key` header
- **Tool Use**：ToolSelector 提供的 `ToolSchema` 转为 Claude 的 tool 格式
- **消息格式映射**：
  - `user` / `assistant` 角色直接映射
  - `tool_use` content block → 内部 `ToolCall`
  - `tool_result` content block → 内部 `ToolResult`
- **MVP 不做流式**：直接使用普通 POST 请求等待完整响应

### 2.13 设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 包名 | `mutagent` | mutation + agent，PyPI 可用 |
| 统一基类 | `mutagent.Object`（MVP 纯透传） | 封装 forwardpy.Object，预留扩展 |
| 声明/实现规范 | `.py` / `.impl.py` | 声明可 import、实现需 loader，天然安全边界 |
| 目录结构 | 声明扁平（`mutagent/*.py`） + 实现集中（`builtins/`） | Agent 一次 inspect 看全声明，实现统一管理 |
| impl 加载策略 | 启动时全量扫描 builtins/ | 简单直接，后续可进化 |
| HTTP 调用 | asyncio + aiohttp | 不依赖 SDK |
| **patch 语义** | **完全替换（写文件+重启）** | 简单直观，行为可预测 |
| 声明 patch | MutagentMeta 就地更新 | 保持类对象身份，先在 mutagent 实现 |
| 实现 patch | 先卸载旧 impl 再注册新 impl | 需要 forwardpy 扩展 |
| 源码追踪 | linecache + loader 协议 | `inspect.getsource()` 透明工作 |
| 虚拟文件名 | `mutagent://module_path` | 避免尖括号陷阱 |
| Tool 接口 | 无公共基类，工具是 mutagent.Object 子类的方法 | 声明可被 patch，符合框架核心理念 |
| Tool 命名 | `EssentialTools`（非 CoreTools） | "Essential" = 当前必要但可进化替换 |
| Tool 调用 | ToolSelector.dispatch 内部决定 | 调用方式也是可进化的实现细节 |
| Tool 选择 | ToolSelector（async，可进化） | 唯一桥梁，可进化为 LLM 驱动 |
| Schema 生成 | 从方法签名 + docstring 自动生成 | ToolSelector 实现的内部工具，非框架规范 |
| 安全边界 | 无沙箱，仅超时 | MVP 面向开发者 |
| 对话持久化 | 不持久化 | MVP 每次全新会话 |
| 核心抽象 | 模块路径 | `package.module.function` 是第一公民 |
| 适用范围 | 仅 mutagent 框架类 | 不需要 patch 任意 Python 模块 |

## 3. 已解决问题

所有设计问题已澄清并确认，决策已记录在 2.13 节。无待定问题。

## 4. 实施步骤清单

### 阶段零：PyPI 占位 [✅ 已完成]

- [x] **Task 0.1**: 发布 mutagent 占位包到 PyPI
  - [x] 创建最小 pyproject.toml
  - [x] 通过 twine 发布
  - 状态：✅ 已完成

### 阶段一：项目基础设施 [✅ 已完成]

- [x] **Task 1.1**: 初始化项目结构
  - [x] 创建 mutagent 包目录结构（按 2.4 节：扁平声明 + builtins/ + runtime/）
  - [x] 配置 pyproject.toml（依赖：forwardpy, aiohttp）
  - [x] 实现 `mutagent.Object` 统一基类（base.py）
  - [x] 创建 `__init__.py`（导出 Object, impl）
  - 状态：✅ 已完成

- [x] **Task 1.2**: 消息模型定义
  - [x] 定义 Message、ToolCall、ToolResult、Response、ToolSchema（messages.py）
  - [x] 单元测试 (16 tests)
  - 状态：✅ 已完成

### 阶段二：运行时核心 [✅ 已完成]

- [x] **Task 2.1**: forwardpy 扩展（可与 mutagent 并行开发）
  - [x] `@impl` 注册时记录 source_module (`_impl_sources` registry)
  - [x] 实现 `unregister_module_impls(module_name)`
  - [x] impl 卸载后恢复为 stub
  - [x] 单元测试 (11 tests in forwardpy)
  - 状态：✅ 已完成

- [x] **Task 2.2**: MutagentMeta 元类
  - [x] 实现类注册表 `_class_registry`
  - [x] 实现就地类更新 `_update_class_inplace()` + `_migrate_forwardpy_registries()`
  - [x] 确保 `mutagent.Object` 使用 `MutagentMeta` 元类
  - [x] 单元测试：类重定义后 `id(cls)` 不变、isinstance 正常、@impl 不断裂 (11 tests)
  - 状态：✅ 已完成

- [x] **Task 2.3**: ModuleManager 核心
  - [x] 实现 `patch_module()`（完全替换语义：卸载旧 impl → 清空命名空间 → 编译执行）
  - [x] 实现 linecache 注入 + `__loader__` 协议
  - [x] 实现 patch 历史追踪
  - [x] 实现虚拟父包自动创建
  - [x] 单元测试：`inspect.getsource()` 对 patch 后的函数/类/模块正常工作 (18 tests)
  - 状态：✅ 已完成

- [x] **Task 2.4**: ImplLoader
  - [x] 实现 `.impl.py` 文件发现
  - [x] 实现加载与 `@impl` 注册
  - [x] 单元测试 (14 tests)
  - 状态：✅ 已完成

- [x] **Task 2.5**: 模块固化
  - [x] 实现 `save_module()`（内存 → 文件）
  - [x] 实现固化过渡（更新 `__file__`、`co_filename`、linecache）
  - [x] 单元测试 (12 tests)
  - 状态：✅ 已完成

### 阶段三：LLM Client [✅ 已完成]

- [x] **Task 3.1**: LLM Client 声明
  - [x] 实现 LLMClient 声明（client.py）
  - [x] 定义 async 接口方法签名
  - [x] 单元测试 (5 tests)
  - 状态：✅ 已完成

- [x] **Task 3.2**: Claude 实现
  - [x] 使用 aiohttp 直接调用 Claude Messages API（builtins/claude.impl.py）
  - [x] 实现消息格式转换（_messages_to_claude, _tools_to_claude, _response_from_claude）
  - [x] 实现 tool schema 格式转换
  - [x] 单元测试 + mock 集成测试 (17 tests)
  - 状态：✅ 已完成

### 阶段四：Tool 系统 [待开始]

- [ ] **Task 4.1**: EssentialTools 声明与 ToolSelector
  - [ ] 实现 EssentialTools 声明（essential_tools.py：5 个工具方法声明）
  - [ ] 实现 ToolSelector 声明（selector.py：get_tools + dispatch，全部 async）
  - [ ] 实现 schema 自动生成（从方法签名 + docstring → ToolSchema）
  - [ ] 实现 ToolSelector MVP 实现（builtins/selector.impl.py）
  - [ ] 单元测试
  - 状态：⏸️ 待开始

- [ ] **Task 4.2**: EssentialTools 各方法实现
  - [ ] builtins/inspect_module.impl.py
  - [ ] builtins/view_source.impl.py
  - [ ] builtins/patch_module.impl.py（依赖 self.module_manager）
  - [ ] builtins/save_module.impl.py（依赖 self.module_manager）
  - [ ] builtins/run_code.impl.py
  - [ ] 各工具单元测试
  - 状态：⏸️ 待开始

### 阶段五：Agent 核心 [待开始]

- [ ] **Task 5.1**: Agent 声明与实现
  - [ ] 实现 Agent 声明（agent.py）
  - [ ] 实现 Agent 异步主循环（builtins/agent.impl.py）
  - [ ] 单元测试（mock LLM）
  - 状态：⏸️ 待开始

- [ ] **Task 5.2**: 端到端集成
  - [ ] 组装所有组件
  - [ ] ImplLoader 加载 builtins/ 下所有 .impl.py
  - [ ] 实现 main.py 入口
  - [ ] 端到端测试：Agent 查看模块 → patch 代码 → 执行验证 → 固化文件
  - 状态：⏸️ 待开始

## 5. 测试验证

### 单元测试
- [ ] mutagent.Object 基类继承（MutagentMeta 生效）
- [ ] MutagentMeta: 类重定义后 id 不变
- [ ] MutagentMeta: 重定义后 isinstance 正常
- [ ] MutagentMeta: 重定义后 @impl 不断裂
- [ ] MutagentMeta: 增删属性和方法
- [ ] forwardpy 扩展: unregister_module_impls
- [ ] forwardpy 扩展: impl 卸载后恢复 stub
- [ ] 消息模型序列化/反序列化
- [ ] ModuleManager: patch 完全替换语义
- [ ] ModuleManager: patch → inspect.getsource() 验证
- [ ] ModuleManager: 固化过渡（虚拟 → 文件）
- [ ] ModuleManager: 虚拟父包创建
- [ ] ImplLoader: .impl.py 发现与加载
- [ ] Schema 自动生成（方法签名 → ToolSchema，self 排除）
- [ ] ToolSelector: get_tools 从 EssentialTools 生成 schema
- [ ] ToolSelector: dispatch 正确路由到 EssentialTools 方法
- [ ] EssentialTools: 各方法 @impl 功能测试
- [ ] Agent 主循环（mock LLM 响应）

### 集成测试
- [ ] Claude API 实际调用测试（aiohttp 直连）
- [ ] Agent + Tools 端到端：Agent 查看模块 → patch 代码 → run_code 验证 → 固化文件
- [ ] 自进化验证：Agent 创建新工具模块 → patch ToolSelector → 使用新工具
