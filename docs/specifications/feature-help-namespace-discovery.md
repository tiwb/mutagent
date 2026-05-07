# help() 命令支持 namespace 发现

**状态**：✅ 已完成
**日期**：2026-05-07
**类型**：功能设计

## 需求

1. `help()` 不带参数时，列出所有已注册的 namespace 及其一行说明。
2. `help(namespace)` 显示该 namespace 的**完整** description + 函数列表（每个函数一行摘要）。
3. `help(ns.fn)` 显示函数的**完整** docstring + 签名（已有行为，补 namespace 前缀）。
4. 三种 namespace 来源（MCP、NamespaceTools、CLI）都要能提供 description：
   - MCP：利用协议 `initialize` 响应中的 `instructions` / `serverInfo.title`
   - NamespaceTools：类 docstring 第一段
   - CLI：本轮先不处理（设计可能迭代）

## 关键参考

- `src/mutagent/sandbox/_namespace.py` — `Namespace` / `NamespaceRegistry` / `_make_help`
- `src/mutagent/sandbox/app.py:36-38` — 沙箱 docstring 已承诺 `help()` 列所有 namespace（文档先行、实现欠账）
- `src/mutagent/sandbox/_adapter_mcp.py` — `bridge_mcp_server` 拿到 `connect()` 结果后丢弃了 `instructions`
- `src/mutagent/sandbox/_app_impl.py:70-113` — `_build_declaration_namespaces` 从 `NamespaceTools.__doc__` 可取类说明
- `mutio/src/mutio/mcp/client.py` — `MCPClient` 当前只存 `server_info` / `server_capabilities`，未存 `instructions`
- MCP 协议 `initialize` 响应字段：`serverInfo.name/version/title`、顶层 `instructions`（均可选）

## 设计方案

### 分层显示规则

**焦点对象完整文本，列表成员首行摘要**。

| 场景 | namespace 文案 | function 文案 |
|---|---|---|
| `help()` | 首行 | — |
| `help(ns)` | 完整 | 首行 |
| `help(ns.fn)` / `help(fn)` | — | 完整 + 签名 |

### Layer 1 — `help()` 输出格式

```
Available namespaces:

  playwright — Browser automation and web interaction via Playwright (23 functions)
  web        — 网页搜索与抓取工具 (2 functions)
  fs         — 文件系统只读访问 (8 functions)
  mutbot     (15 functions)

Use help(<namespace>) for details, e.g. help(playwright).
```

- 有 description：`name — desc_first_line (N functions)`
- 空 description：`name (N functions)`，不显示 `—`
- namespace 名左对齐，按最长名 padding
- 列表按 namespace 名字母序

### Layer 2 — `help(ns)` 输出格式

**有 description**：

```
Namespace: playwright

Browser automation and web interaction via Playwright.

This server exposes the page accessibility tree as the primary perception
channel. Prefer browser_snapshot over browser_take_screenshot...
(保留原始段落换行)

23 Functions:

  browser_click              Perform click on a web page
  browser_close              Close the page
  browser_navigate           Navigate to a URL
  browser_snapshot           Capture accessibility snapshot of current page
  ...

Use help(playwright.<function>) for function details.
```

**空 description**：

```
Namespace: mutbot

15 Functions:

  status            查看服务器全局状态
  workspaces        列出所有 workspace
  ...

Use help(mutbot.<function>) for function details.
```

- `N Functions:` 作为函数列表小标题（不加括号，数量前置）
- description 保留原始换行，首尾 strip
- 函数名左对齐，按最长函数名 padding；描述取 docstring 首行

### Layer 3 — `help(ns.fn)` / `help(fn)` 输出格式

```
playwright.browser_navigate(*, url)

Navigate to a URL.

Args:
    url: string (required) — The URL to navigate to
```

- 标题补 `namespace.` 前缀（当前只有裸函数名）
- 签名和 docstring 保持现状

### 数据模型改动

**`Namespace` 新增 `description` 字段**：

```python
class Namespace:
    def __init__(self, name: str, description: str = ""):
        self._name = name
        self._description = description  # 完整文本，可多行
        self._functions: dict[str, Callable] = {}
        self._descriptions: dict[str, str] = {}  # 函数完整 description
```

- `_descriptions` 存**完整**函数描述（当前 MCP 分支已经存完整，只是展示时没截首行；本次明确语义：一律存原文）
- 展示时才调用 `_first_line()` 截首行

### 工具函数

抽取单行提取逻辑，供三处复用（list 所有 ns、list 某 ns 内函数、未来其他地方）：

```python
def _first_line(text: str) -> str:
    """提取文本首行（非空、strip）。空文本返回空串。"""
    if not text:
        return ""
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s
    return ""
```

放在 `_namespace.py` 模块级，私有。

### 三处 description 填充

#### NamespaceTools（本地 Declaration）

`_app_impl.py::_build_declaration_namespaces`：

```python
cls_doc = inspect.getdoc(cls) or ""
ns = Namespace(ns_name, description=cls_doc)
```

`inspect.getdoc` 自动处理缩进去除和继承。

#### MCP server（协议字段）

需要跨两个仓库协作：

**mutio 仓库** (`mutio/src/mutio/mcp/client.py`)：

- `MCPClient` 新增字段：`server_instructions: str = ""`
- `connect()` 实现里从 initialize 响应的顶层 `instructions` 字段（可选）填充

**mutagent 仓库** (`_adapter_mcp.py`)：

- `StdioMCPClient` 的 `connect()` 已返回原始 initialize result，取 `result.get("instructions", "")`
- `HTTPMCPClient.connect()` 读取 instructions 时用 `getattr(self._mcp, 'server_instructions', '')` **防御式访问**（解耦 mutio/mutagent 发版顺序，允许 mutagent 先于 mutio 发版）
- description 来源优先级（退化链）：
  1. `instructions`（MCP 协议标准字段，协议上就是"server 用法说明"）
  2. `serverInfo.title`（可选显示名）
  3. `""`（不使用 `serverInfo.name`，通常等于 namespace 名，信息冗余）

#### CLI 白名单

本轮不改，`Namespace("cli")` 继续用默认空 description。

### `help(ns.fn)` 的 namespace 前缀来源

给 register 出去的函数附加 `__namespace__` 属性（O(1) 查询，侵入性小）：

```python
def register(self, func_name: str, func: Callable, description: str = "") -> None:
    self._functions[func_name] = func
    self._descriptions[func_name] = description
    try:
        func.__namespace__ = self._name  # best-effort，失败忽略（如 builtin）
    except (AttributeError, TypeError):
        pass
```

`_make_help` 的 callable 分支读 `getattr(func, '__namespace__', None)` 拼前缀。

### 显示细节

- 列表 padding：`max_name = max(len(n) for n in names)`，条目用 `f"  {name:<{max_name}}"`
- `Namespace: xxx` 标题行下空一行再接 description，和 `N Functions:` 之间空一行
- 空 description 时直接跳过 description 段落（紧接 `N Functions:`）
- 函数描述为空时只显示函数名，不显示尾部两空格

### 测试覆盖

- `test_namespace.py`（新增或扩充）：
  - `help()` 无 namespace 时输出
  - `help()` 多 namespace 时对齐、排序、空 description 处理
  - `help(ns)` 有/无 description 两种形态
  - `help(ns)` 含函数描述对齐
  - `help(ns.fn_name_str)` 字符串形式
  - `help(fn)` callable 形式带 `__namespace__` 的前缀拼接
- MCP instructions 透传（`test_adapter_mcp.py` 或类似）：
  - Stdio 分支：mock initialize 响应含 `instructions`，断言 `ns._description` 已填充
  - HTTP 分支：mock `MCPClient.server_instructions`，同上
  - 退化链：无 `instructions` 时取 `serverInfo.title`，都没有则为空
- mutio 侧 `MCPClient.server_instructions` 填充（`mutio/tests/...`）

### 版本兼容

- mutio 的 `MCPClient` 新增字段 `server_instructions: str = ""` 是**加字段**，老代码不访问就不受影响
- mutagent 对 mutio 的版本约束：当前 `mutio ~= 0.x`，改动落地后需要两边同步发版（遵循 AGENTS.md 发布流程）

## 实施步骤清单

- [x] `Namespace` 数据模型扩展（`_namespace.py`）：加 `description` 字段、`register` 给函数附加 `__namespace__` 属性
- [x] 新增 `_first_line(text)` 工具函数（`_namespace.py` 模块级）
- [x] 重写 `_make_help` 三层分支：
  - 无参 → Layer 1 列所有 namespace
  - Namespace 对象 → Layer 2 完整 description + 函数首行列表
  - callable / `"ns.fn"` 字符串 → Layer 3 补 `namespace.` 前缀
- [x] `_build_declaration_namespaces` 构造 `Namespace` 时传入 `inspect.getdoc(cls)`（`_app_impl.py`）
- [x] MCP 适配层填充 description（`_adapter_mcp.py`）：
  - `bridge_mcp_server` 从 `connect()` 结果取 `instructions` / `serverInfo.title`
  - `HTTPMCPClient.connect()` 防御式返回 `server_instructions`
  - 将 description 传给新的 `Namespace(...)` 构造
- [x] mutio 侧 `MCPClient` 新增 `server_instructions: str = ""` 字段及 `connect()` 填充逻辑（`mutio/src/mutio/mcp/client.py`）
- [x] 更新沙箱 docstring（`app.py`）与实际显示对齐
- [x] 测试：`test_sandbox.py` 扩充三层显示用例（空/非空 description、对齐、排序、函数前缀）
- [x] 测试：`test_adapter_mcp.py` 覆盖 instructions 透传与退化链（stdio / http / title 退化 / 旧版 mutio 防御）
- [x] 本地跑 `pytest`：mutagent 811 passed，mutio 146 passed
