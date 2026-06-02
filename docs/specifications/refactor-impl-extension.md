# Agent / ToolSet / SandboxEnv 用 Extension 替代 object.__setattr__

**状态**：✅ 已完成
**日期**：2026-05-28
**类型**：重构

## 需求

1. `_agent_impl.py` 有 3 处、`_tools_impl.py` 有 7 处、`_env_impl.py` 有 10 处 `object.__setattr__`，均属惰性缓存 / 运行时上下文追踪模式。
2. 全部适合用 mutobj Extension 模式替代（参照 `mutgui` 的 `MenuRuntime`、`mutagent` 的 `SessionRuntime`）。
3. 目标：消除这 20 处 `object.__setattr__`。

## 关键参考

### mutobj Extension 模式

- `mutobj/src/mutobj/core/_extensions.py` — `Extension[T]`，`get()` / `get_or_create()` API
- `mutgui/src/mutgui/_menu_impl.py` — `MenuRuntime(mutobj.Extension[MenuView])`，最简单的范例
- `mutagent/src/mutagent/core/_session_impl.py` — `SessionRuntime(mutobj.Extension[AgentSession])`，最近完成的重构

### 涉及的文件

- `src/mutagent/core/agent.py` — `Agent` Declaration
- `src/mutagent/core/tools.py` — `ToolSet` Declaration
- `src/mutagent/sandbox/env.py` — `SandboxEnv` Declaration
- `src/mutagent/core/_agent_impl.py` — 3 处 `object.__setattr__`
- `src/mutagent/core/_tools_impl.py` — 7 处 `object.__setattr__`
- `src/mutagent/sandbox/_env_impl.py` — 10 处 `object.__setattr__`

## 设计方案

### 模式一：惰性缓存 → Extension 字段

当前写法：

```python
def _get_entries(self: ToolSet) -> dict[str, ToolEntry]:
    entries = getattr(self, '_entries', None)
    if entries is None:
        entries = {}
        object.__setattr__(self, '_entries', entries)
    return entries
```

改为：

```python
def _get_entries(self: ToolSet) -> dict[str, ToolEntry]:
    rt = ToolSetRuntime.get_or_create(self)
    if rt.entries is None:
        rt.entries = {}
    return rt.entries
```

### 模式二：上下文追踪 → Extension 字段

当前写法：

```python
# dispatch 开始时
object.__setattr__(self, '_current_tool_call', tool_call)
# ...
# finally 中
object.__setattr__(self, '_current_tool_call', None)
```

改为：

```python
rt = ToolSetRuntime.get_or_create(self)
rt.current_tool_call = tool_call
# ...
rt.current_tool_call = None
```

### 新增三个 Extension

#### AgentRuntime

```python
class AgentRuntime(mutobj.Extension[Agent]):
    """Agent 运行时内部状态 — 仅 _agent_impl 内部维护"""

    event_listeners: list[Callable[[StreamEvent], Any]] | None = None
    current_task: asyncio.Task[None] | None = None
```

**现有 Declaration 不做任何改动**——`Agent` 没有声明这些字段，它们本来就是纯内部状态。

#### ToolSetRuntime

```python
class ToolSetRuntime(mutobj.Extension[ToolSet]):
    """ToolSet 运行时内部状态 — 仅 _tools_impl 内部维护"""

    entries: dict[str, ToolEntry] | None = None
    added_classes: set[type] | None = None
    discovered: dict[type, dict] | None = None
    last_registry_generation: int | None = None
    current_tool_call: ToolUseBlock | None = None
    active_ui: UIContext | None = None
```

#### SandboxEnvRuntime

```python
class SandboxEnvRuntime(mutobj.Extension[SandboxEnv]):
    """SandboxEnv 运行时内部状态 — 仅 _env_impl 内部维护"""

    registry: NamespaceRegistry | None = None
    cleanups: dict[int, tuple[Namespace, CleanupCallback]] | None = None
    mcp_conns: dict[str, Any] | None = None
    start_time: float | None = None
    async_loop: asyncio.AbstractEventLoop | None = None
    async_loop_thread_id: int | None = None
    cached_ns: dict | None = None
    cached_gen: int = -1
```

### 变更范围

| 文件 | 改动 |
|------|------|
| `_agent_impl.py` | 新增 `AgentRuntime` Extension；3 处 `object.__setattr__` → Extension 字段赋值 |
| `_tools_impl.py` | 新增 `ToolSetRuntime` Extension；7 处 `object.__setattr__` → Extension 字段赋值 |
| `_env_impl.py` | 新增 `SandboxEnvRuntime` Extension；10 处 `object.__setattr__` → Extension 字段赋值 |
| `agent.py` / `tools.py` / `env.py` | Declaration 无变更 |

### 不改 Declaration 的理由

这些字段只在其对应的 `_*_impl.py` 内部使用，不对外暴露。`Agent` / `ToolSet` / `SandboxEnv` 的公开 API 不受影响，因此不需要在 Declaration 中声明任何字段。Extension 字段去掉 `_` 前缀——它们是 Extension 类的公开字段，类型检查器能看到并做补全。

对比 `SessionRuntime`：当时 `path`、`created_at` 原本是 Declaration 的公开字段，所以才需要从 Declaration 中移除后迁入 Extension。这三个 Extension 是纯增量的——不加到 Declaration，直接从 `object.__setattr__` 挪到 Extension。

### 不保留旧私有字段兼容层

这次重构的目标就是把 impl-only runtime state 彻底收敛到 Extension；因此不再保留 `_current_task` / `_current_tool_call` / `_active_ui` / `_registry` / `_async_loop` 这类旧私有字段访问。内部代码统一改为 runtime helper（如 `_get_current_task(...)`、`_get_registry(...)`、`_require_async_loop(...)`），测试里的 fake 对象也直接初始化对应 Extension runtime，而不是继续写旧私有字段。

## 实施步骤清单

- [x] 为 `Agent` / `ToolSet` / `SandboxEnv` 增加对应的 runtime Extension，承接原先的惰性缓存和运行时上下文字段
- [x] 将 `_agent_impl.py`、`_tools_impl.py`、`_env_impl.py` 中目标 `object.__setattr__` 改为 Extension 字段读写
- [x] 将内部调用方改为通过 runtime helper 访问状态，去掉旧私有字段兼容属性与 fallback 逻辑
- [x] 更新相关测试 fake / 断言写法，直接初始化 Extension runtime，不再依赖旧私有字段

## 测试验证

- `..\.venv\Scripts\python.exe -m pytest tests\core\test_agent.py tests\core\test_tools.py tests\sandbox\test_sandbox_async_namespace.py tests\sandbox\test_namespace_describe_api.py tests\sandbox\test_namespace_multi_provider.py tests\sandbox\test_pysandbox_sharing.py tests\sandbox\test_adapter_mcp.py tests\webui\test_mcp_settings_panel.py -q`
- `..\.venv\Scripts\pyright.exe src\mutagent\core\_agent_impl.py src\mutagent\core\_tools_impl.py src\mutagent\sandbox\_env_impl.py src\mutagent\sandbox\_mcp_impl.py src\mutagent\sandbox\_mcp_impl_sandbox.py src\mutagent\sandbox\_namespace_impl.py src\mutagent\webui\_settings_mcp.py`
