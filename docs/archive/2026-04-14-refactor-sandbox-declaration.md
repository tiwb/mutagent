# Sandbox Declaration 化重构 设计规范

**状态**：✅ 已完成
**日期**：2026-04-14
**类型**：重构

## 需求

1. SandboxApp 改为 Declaration，公开接口稳定化
2. 新增 NamespaceTools Declaration — 本进程能力源的自动发现机制
3. 内部模块下划线命名，减少公开 API 面
4. Toolkit 支持 `_tool_prefix` 类变量，允许控制工具命名
5. `exec_code` 支持外部 state 参数

## 关键参考

- `mutagent/sandbox/app.py` — 当前 SandboxApp（普通类）
- `mutagent/sandbox/namespace.py` — Namespace + NamespaceRegistry
- `mutagent/sandbox/engine.py` — execute() 执行引擎
- `mutagent/sandbox/adapters/` → `mutagent/sandbox/_adapter_mcp.py` + `_adapter_cli.py`
- `mutagent/sandbox/tools.py` — PySandboxTools（MCPToolSet）
- `mutagent/tools.py` — Toolkit + ToolSet Declaration
- `mutagent/builtins/tool_set_impl.py:87-98` — `_get_tool_prefix` / `_get_tool_name`
- `mutagent/agent.py` — Agent Declaration 范式参考
- `mutagent/net/_mcp_impl.py:71-92` — MCPToolProvider generation 懒刷新参考
- `D:/ai_skills/pysandbox/` — 独立 pysandbox（需跟随接口变更）

## 实施步骤清单

- [x] Toolkit `_tool_prefix` 支持（`tools.py` + `tool_set_impl.py`）
- [x] 内部模块重命名：`engine.py` → `_engine.py`，`adapters/` → `_adapters/`
- [x] 拆分 `namespace.py`：内部类 → `_namespace.py`，新建 `namespace.py` 放 NamespaceTools Declaration
- [x] 重写 `app.py` 为 SandboxApp Declaration
- [x] 新建 `_app_impl.py`：SandboxApp 实现（cache+rebuild、generation 懒发现、exec_code state 参数）
- [x] 更新 `sandbox/__init__.py` 导出
- [x] 更新 `sandbox/tools.py`（PySandboxTools）适配新 SandboxApp 接口
- [x] 更新独立 pysandbox（`D:/ai_skills/pysandbox/`）精简为薄壳
- [x] 运行测试，修复问题

## 设计方案

### SandboxApp Declaration 化

SandboxApp 改为 mutagent.Declaration，遵循项目标准范式（属性声明 + 方法桩 + register_module_impls）：

```python
class SandboxApp(mutagent.Declaration):
    """Python 沙箱 — 聚合能力源，提供受限代码执行环境。"""

    config: Config

    async def setup(self) -> None:
        """根据 self.config 初始化能力源连接（MCP/CLI）。"""
        ...

    def exec_code(self, code: str, state: dict | None = None) -> dict:
        """执行 Python 代码。state 为 None 时不保留跨步骤变量。"""
        ...

    async def reload(self) -> dict:
        """从 self.config 重载，重连能力源。"""
        ...

    async def shutdown(self) -> None:
        """关闭所有连接。"""
        ...
```

- config 作为属性注入（与 Agent 一致），setup/reload 直接读 `self.config`
- state 参数：传入时使用外部 state dict，None 时每次执行独立（不保留变量）

### 内部缓存机制

namespace dict 采用 cache → rebuild 模式（不用 add/remove 配对，避免不匹配风险）：

- 内部保存数据源（MCP config、CLI config、NamespaceTools 发现结果）
- `exec_code` 时检查是否需要 rebuild（generation 变化或 config 变化时 invalidate）
- rebuild 从数据源完整重建 namespace dict
- NamespaceRegistry 不再作为独立公开接口，合入 SandboxApp 内部实现

### NamespaceTools Declaration

新增 Declaration 子类，用于本进程 Python 函数的自动发现注册：

```python
class NamespaceTools(mutagent.Declaration):
    """声明一组注入 sandbox 命名空间的函数。

    namespace 名从类名推导（去掉 Tools 后缀），或用 _namespace 显式指定。
    子类的 public 方法自动注册为命名空间函数。
    """
    _namespace: ClassVar[str | None] = None
```

发现机制：
- 利用 mutobj 的 `discover_subclasses(NamespaceTools)` + `get_registry_generation()` 懒刷新
- 与 MCPToolProvider、ToolSet auto_discover 同一范式
- import 即注册 — 上层项目（如 mutbot）只需 import 包含 NamespaceTools 子类的模块

async 方法处理：
- sandbox engine 在工作线程中同步执行
- NamespaceTools 的 async 方法在注册时自动包装为 sync（`run_coroutine_threadsafe`，与 mcp_bridge 中相同模式）

### Toolkit `_tool_prefix` 支持

在 Toolkit 类上新增 `_tool_prefix: ClassVar[str | None] = None`：

```python
class Toolkit(mutagent.Declaration):
    _tool_prefix: ClassVar[str | None] = None  # None = 自动推导
```

`_get_tool_prefix` 逻辑调整：
- 显式指定时优先使用 `_tool_prefix` 值
- 空字符串时 `_get_tool_name` 直接返回方法名（无前缀无连字符）
- None 时保持现有行为（类名去 Toolkit 后缀）

### 模块结构重组

公开接口（不带下划线）：

| 文件 | 内容 |
|------|------|
| `app.py` | SandboxApp Declaration |
| `namespace.py` | NamespaceTools Declaration |
| `tools.py` | PySandboxTools（MCPToolSet，不变） |
| `__init__.py` | 只导出 SandboxApp + NamespaceTools |

内部实现（下划线前缀）：

| 文件 | 内容 |
|------|------|
| `_app_impl.py` | SandboxApp 实现 |
| `_engine.py` | execute() 执行引擎 |
| `_namespace.py` | Namespace + NamespaceRegistry 内部类 |
| `_adapter_mcp.py` | MCP bridge（StdioMCPClient） |
| `_adapter_cli.py` | CLI 白名单适配器 |

### 独立 pysandbox 跟随变更

`D:/ai_skills/pysandbox/` 精简为 SandboxApp 的薄壳：
- 删除 `adapters/`（不再直接依赖内部模块）
- 删除 `engine.py`、`namespace.py`（已在 mutagent 内部）
- 保留 `cli.py`（daemon 管理）、`config.py`（配置加载）、`server.py`（SandboxApp 接线）
- `server.py` 只依赖 `SandboxApp` + `PySandboxTools` 公开接口
