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
- 所有核心类（`LLMClient`、`Agent`、`Tool` 等）继承 `mutagent.Object`
- `MutagentMeta` 扩展 `forwardpy.ObjectMeta`，增加就地类更新能力
- 未来可在 `mutagent.Object` 上添加通用能力，不影响已有代码
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
    def execute(self, **params) -> str: ...             # 同步接口
    async def execute_async(self, **params) -> str: ... # 异步接口

# tools/base.impl.py — 默认实现
@mutagent.impl(Tool.execute_async)
async def execute_async(self: Tool, **params) -> str:
    """默认：委托给同步 execute"""
    return self.execute(**params)
```

**双接口设计**：
- 同步工具（大多数）→ 实现 `execute`，`execute_async` 自动委托
- 异步工具 → 实现 `execute_async`（override=True）
- 框架统一调用 `await tool.execute_async(**params)`

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
- **Tool Use**：将 Tool 的 `get_schema()` 输出转为 Claude 的 tool 格式
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
| impl 加载策略 | 启动时全量扫描 | 简单直接，后续可进化 |
| HTTP 调用 | asyncio + aiohttp | 不依赖 SDK |
| **patch 语义** | **完全替换（写文件+重启）** | 简单直观，行为可预测 |
| 声明 patch | MutagentMeta 就地更新 | 保持类对象身份，先在 mutagent 实现 |
| 实现 patch | 先卸载旧 impl 再注册新 impl | 需要 forwardpy 扩展 |
| 源码追踪 | linecache + loader 协议 | `inspect.getsource()` 透明工作 |
| 虚拟文件名 | `mutagent://module_path` | 避免尖括号陷阱 |
| Tool 接口 | 双接口 `execute` + `execute_async` | 清晰契约，无隐式魔法，Python 常见模式（Django/ASGI） |
| Tool 实现 | 同步工具实现 `execute`，异步工具实现 `execute_async` | 实现者只需关心一个方法，默认委托透明 |
| Tool 选择 | ToolSelector（可进化） | 初始返回全部，Agent 可迭代 |
| 安全边界 | 无沙箱，仅超时 | MVP 面向开发者 |
| 对话持久化 | 不持久化 | MVP 每次全新会话 |
| 核心抽象 | 模块路径 | `package.module.function` 是第一公民 |
| 适用范围 | 仅 mutagent 框架类 | 不需要 patch 任意 Python 模块 |

## 3. 待定问题

### Q1: Tool 的同步/异步接口设计

**问题**：Tool 接口应如何处理同步/异步实现的差异？

**背景**：
- Agent 内部是完全的异步框架
- 核心工具（`inspect_module`、`view_source`）本质是同步操作
- 未来工具（如调用子 Agent、HTTP 请求）需要真正的异步
- 用户要求：**避免任何不清晰的设计**

**三种方案对比**：

#### 方案 A：统一 `async def execute`

```python
class Tool(mutagent.Object):
    async def execute(self, **params) -> str: ...

# 同步工具也必须写 async def
@mutagent.impl(InspectModuleTool.execute)
async def execute(self, **params) -> str:
    return do_sync_stuff()  # 没有 await，但声明是 async
```

| 维度 | 评价 |
|------|------|
| 清晰度 | ⚠️ 中等 — 声明为 async 但实际无 await，产生"假异步"困惑 |
| 简洁度 | ✅ 高 — 只有一个方法 |
| 实现负担 | ⚠️ 每个同步工具都要写无意义的 `async def` |
| 扩展性 | ✅ 框架只需 `await tool.execute()` |

#### 方案 B：框架自动检测 sync/async

```python
class Tool(mutagent.Object):
    async def execute(self, **params) -> str: ...  # 声明为 async

# 但实现可以是 sync，框架自动检测包装
@mutagent.impl(InspectModuleTool.execute)
def execute(self, **params) -> str:              # 实际是 sync
    return do_sync_stuff()
```

| 维度 | 评价 |
|------|------|
| 清晰度 | ❌ 低 — 声明是 async 但允许 sync 实现，隐式魔法 |
| 简洁度 | ✅ 高 — 只有一个方法 |
| 实现负担 | ✅ 低 — 实现者自由选择 |
| 扩展性 | ⚠️ 框架需要 `iscoroutine` 检测逻辑 |

**这正是用户指出的"不清晰设计"** — 声明契约与实现不一致，依赖隐式行为。

#### 方案 D：双接口 `execute` + `execute_async`（用户提议）

```python
# tools/base.py — 声明
class Tool(mutagent.Object):
    name: str
    description: str

    def get_schema(self) -> ToolSchema: ...
    def execute(self, **params) -> str: ...           # 同步接口
    async def execute_async(self, **params) -> str: ... # 异步接口

# tools/base.impl.py — 默认实现：execute_async 委托给 execute
@mutagent.impl(Tool.execute_async)
async def execute_async(self: Tool, **params) -> str:
    return self.execute(**params)  # 默认：直接调用同步版本
```

**同步工具**（大多数内置工具）— 只需实现 `execute`：
```python
@mutagent.impl(InspectModuleTool.execute)
def execute(self, **params) -> str:
    return inspect_module(params['module_path'])
# execute_async 继承默认实现，自动委托到 execute
```

**异步工具**（需要 await 的工具）— 直接实现 `execute_async`：
```python
@mutagent.impl(SubAgentTool.execute_async, override=True)
async def execute_async(self, **params) -> str:
    return await self.client.send_message(...)
# execute 保持为 stub（不需要同步版本）
```

**框架调用**：统一使用 `await tool.execute_async(**params)`

| 维度 | 评价 |
|------|------|
| 清晰度 | ✅ 高 — `def execute` 就是同步，`async def execute_async` 就是异步，零歧义 |
| 简洁度 | ⚠️ 中等 — 两个方法，但实现者只需关心其中一个 |
| 实现负担 | ✅ 低 — 同步工具实现 `execute`（最自然），异步工具实现 `execute_async` |
| 扩展性 | ✅ 高 — 未来可让默认 `execute_async` 使用 `asyncio.to_thread` 运行同步工具 |
| 与 forwardpy 兼容 | ✅ 两个独立 stub，各自走 `@impl` 注册，无特殊处理 |

**关于事件循环阻塞**：
- MVP 阶段：默认 `execute_async` 直接调用 `self.execute()`（同步执行在事件循环中）
- 核心工具（inspect、view_source）操作很快，不会阻塞
- 未来优化：可将默认实现改为 `await asyncio.to_thread(self.execute, **params)` 运行到线程池

**总结对比**：

| | 方案 A（统一 async） | 方案 B（自动检测） | 方案 D（双接口） |
|---|---|---|---|
| 契约清晰度 | 中 | 低 | **高** |
| 方法数量 | 1 | 1 | 2 |
| 实现者负担 | 高（假 async） | 低（但有隐式魔法） | **低（选一个实现）** |
| 框架复杂度 | 低 | 中（检测逻辑） | **低（默认 impl 委托）** |
| Python 惯例 | 少见 | 不推荐 | **常见**（Django, ASGI） |

**建议**：方案 D（双接口）。Python 生态中 Django（`__call__` / `__acall__`）、ASGI（sync/async views）都采用类似模式，是成熟的惯例。契约最清晰：类型签名即契约，无隐式行为。

### Q2: 内置工具 `execute` 与 `Tool.execute()` 方法的命名冲突

**问题**：内置工具中有一个叫 `execute` 的工具（执行 Python 代码片段），同时 `Tool` 基类有 `execute()` 方法。这会导致：

1. **代码可读性**：`execute_tool.execute(code="print(1)")` — 冗余且混淆
2. **LLM 认知**：tool_name="execute" 语义不够具体（execute what?）
3. **与方案 D 的交叉**：Tool 有 `execute` + `execute_async` 两个方法，再加上一个叫 "execute" 的内置工具，概念容易混淆

**建议**：将内置工具 `execute` 重命名为更具体的名称：

| 候选名 | 含义 | 与 Python 惯例 |
|--------|------|----------------|
| `run_code` | 运行代码片段 | 直观，无歧义 |
| `eval_code` | 求值代码 | 但 eval 在 Python 中有特定含义（表达式求值），这里是 exec |
| `exec_code` | 执行代码 | 贴近 Python `exec()`，但 exec 是关键字 |

**推荐**：`run_code` — 简洁明了，与 Tool 方法名零冲突。

### Q3: 双接口模式是否也适用于其他核心声明？

**问题**：确认双接口模式（sync + async）是否只用于 Tool，还是也推广到其他声明类。

当前各类的接口：
| 类 | 方法 | 当前声明 | 本质 |
|----|------|----------|------|
| `LLMClient` | `send_message` | `async def` | 真正的 I/O 异步 |
| `Agent` | `run`, `step`, `handle_tool_calls` | `async def` | 真正的异步循环 |
| `ToolSelector` | `select` | `def` | 同步选择逻辑 |
| `Tool` | `execute` / `execute_async` | 双接口 | 取决于工具实现 |

**分析**：
- `LLMClient` 和 `Agent` 是纯异步的，单 `async def` 就够了
- `ToolSelector.select` 是纯同步的，单 `def` 就够了
- 只有 `Tool` 存在同步/异步的不确定性，需要双接口

**建议**：双接口仅用于 `Tool`。其他类的方法保持当前的单一声明（纯同步或纯异步），因为它们的同步/异步属性是确定的。


统一回复：我觉得我们是不是跑偏了。

mutagent的核心设计是实现一个可自我进化的Agent。
我们有一个Agent，Agent通过语言模型运行做出决策，Agent选择工具，Python运行时中很多工具。
我们并没有也不需要明确的定义：什么是一个工具。 在框架看来，所有的东西都是Python运行时的接口和实现。
是工具选择器决定了选择以后如何调用工具。
所以，即使是工具上的execute接口，也不是任何规范，只是默认工具选择器的实现。所以不应该纠结任何的接口设计。初始框架实现要做的是让这套思路能运转起来，同时尽量不给任何限制。

请你根据以上的思路，重新思考一下这个问题。上面还有一个关键信息，你认为ToolSelector是同步选择的，我会觉得工具选择是大模型的行为，比如我提出一个问题，模型需要先分析，解决这个问题我需要什么工具？然后去选择工具，选择工具的过程很可能是模型要给出一个工具范围，然后查询现有的工具集，判断是否有满足需要的工具，是否需要创造一个新工具等等。这个过程也一定是异步的。 

我们不应该陷入在具体的实现细节中，核心应该是这一切都是可进化的。


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

- [ ] **Task 2.1**: forwardpy 扩展（可与 mutagent 并行开发）
  - [ ] `@impl` 注册时记录 source_module
  - [ ] 实现 `unregister_module_impls(module_name)`
  - [ ] impl 卸载后恢复为 stub
  - [ ] 单元测试
  - 状态：⏸️ 待开始

- [ ] **Task 2.2**: MutagentMeta 元类
  - [ ] 实现类注册表 `_class_registry`
  - [ ] 实现就地类更新 `_update_class_inplace()`
  - [ ] 确保 `mutagent.Object` 使用 `MutagentMeta` 元类
  - [ ] 单元测试：类重定义后 `id(cls)` 不变、isinstance 正常、@impl 不断裂
  - 状态：⏸️ 待开始

- [ ] **Task 2.3**: ModuleManager 核心
  - [ ] 实现 `patch_module()`（完全替换语义：卸载旧 impl → 清空命名空间 → 编译执行）
  - [ ] 实现 linecache 注入 + `__loader__` 协议
  - [ ] 实现 patch 历史追踪
  - [ ] 实现虚拟父包自动创建
  - [ ] 单元测试：`inspect.getsource()` 对 patch 后的函数/类/模块正常工作
  - 状态：⏸️ 待开始

- [ ] **Task 2.4**: ImplLoader
  - [ ] 实现 `.impl.py` 文件发现
  - [ ] 实现加载与 `@impl` 注册
  - [ ] 单元测试
  - 状态：⏸️ 待开始

- [ ] **Task 2.5**: 模块固化
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
- [ ] Tool schema 生成
- [ ] ToolSelector: 初始版本返回所有核心 tools
- [ ] 各 builtin tool 功能测试
- [ ] Agent 主循环（mock LLM 响应）

### 集成测试
- [ ] Claude API 实际调用测试（aiohttp 直连）
- [ ] Agent + Tools 端到端：Agent 查看模块 → patch 代码 → 执行验证 → 固化文件
- [ ] 自进化验证：Agent 创建新 tool → 注册到 ToolSelector → 使用新 tool
