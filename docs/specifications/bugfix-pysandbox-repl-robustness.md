# Pysandbox REPL 稳定性修复 — logging / namespace 命名 / Ctrl+C 退出

**状态**：✅ 已完成
**日期**：2026-05-08
**类型**：Bug修复

## 需求

1. pysandbox REPL 路径缺少 logging 配置 → MCP 后台连接失败的 WARNING 经 lastResort 冲 stderr，和 `>>> ` 提示符混在一起
2. 含空格的 MCP namespace 名（如 `"My MCP"`）注入 sandbox globals 后无法在 Python 代码中直接访问（非合法标识符）
3. pysandbox REPL 中 Ctrl+C 退出时打印 asyncio traceback，不符合标准 Python REPL 行为（应干净退出或继续）

## 关键参考

- `mutagent/src/mutagent/cli/pysandbox.py` — pysandbox CLI 入口 + `_build_sandbox()` + `_run_repl()` + `dispatch_pysandbox()`
- `mutagent/src/mutagent/builtins/main_impl.py` — `setup_agent()` 中的 logging 配置（LogStore + 文件 handler）、`_ensure_console_logging()`
- `mutagent/src/mutagent/sandbox/_adapter_mcp.py` — `MCPConnection.__init__()` 创建 `Namespace(ns_name, ...)`、`make_client()`
- `mutagent/src/mutagent/sandbox/_namespace.py` — `Namespace.__init__(name, ...)` 用 `self._name = name`
- `mutagent/src/mutagent/sandbox/_app_impl.py` — `_build_namespace_dict()` 把 `ns.name` 作为 dict key 注入 `exec()` globals
- `mutagent/src/mutagent/sandbox/_engine.py` — `execute(code, namespace, state)` 把 namespace dict 注入 `globals_dict`
- `mutagent/docs/specifications/feature-pysandbox-repl.md` — REPL 设计参考

## 设计方案

### 问题 1：pysandbox REPL 缺少 logging 配置

**根因**：`_build_sandbox()` 只构造 SandboxApp + 注册 MCP namespace，不配置 Python logging。`mutagent.*` 子 logger（如 `mutagent.cli.pysandbox`）的 WARNING 经 propagate 到达 root logger，root 无 handler → Python `logging.lastResort` 接管 → 打印到 stderr。

**发现**：最初实现挂 handler 到 `logging.getLogger("mutagent")`，但 mutio 的 logger（`mutio.mcp._client_impl`）不归属 mutagent，直接 propagate 到 root → root 无 handler → lastResort 接管 → 打印到 stderr。`setup_agent()` 存在同样问题。

**方案**：handler 挂到 `logging.getLogger()`（root logger），所有库的日志全部进入 LogStore + 文件，不会经 lastResort 漏到控制台。具体做法：

- 从 `setup_agent()` 提取 session log 目录初始化 + LogStore + 文件 handler 的逻辑为一个独立函数 `_setup_pysandbox_logging(config)`，供两个路径共用
- `_build_sandbox()` 开头调用该函数
- 不挂 console handler（终端模式下不希望日志冲交互流）

**注意**：pysandbox 的生命周期很短（REPL 退出即结束），不需要 `setup_agent()` 中的 API Recorder、session metadata 等重型组件，只需基础 logging 管线。

### 问题 2：MCP namespace 名含空格

**根因**：配置 key 原样作为 `Namespace._name` → `_build_namespace_dict` 作为 dict key → `execute()` 注入 globals → `My MCP.search(...)` 是 Python 语法错误。

**方案**：在 `MCPConnection.__init__()` 中对 `ns_name` 做 sanitize：空格、连字符等非 Python 标识符字符替换为下划线，连续的多个下划线折叠为一个。

具体规则：
- 字母、数字、下划线保留
- 其他字符替换为 `_`
- 连续 `_` 折叠为一个
- 首尾 `_` 去掉

例：`"My MCP"` → `"My_MCP"`，`"my-srv"` → `"my_srv"`。

**影响范围**：
- `help()` 列表中的显示名会变成 sanitized 后的名字
- 不影响原配置 key（只改 runtime namespace name）
- 如果 sanitize 后出现重名（两个不同 key 映射到同一名），后者覆盖前者（和当前 `add_namespace` 同名替换行为一致）

### 问题 3：Ctrl+C traceback

**根因**：`asyncio.run(_run_repl(config))` 在 `dispatch_pysandbox()` 中，KeyboardInterrupt 先 hit `code.InteractiveConsole.interact()` 内部（线程池线程中），打印 `KeyboardInterrupt` 继续。但信号同时中断 asyncio 主线程的 `await loop.run_in_executor()` → `CancelledError` → `_run_repl` 不捕获 → `asyncio.run()` 不捕获 → `dispatch_pysandbox` 不捕获 → Python 打印 traceback 退出。

**方案**：在 `_run_repl()` 中捕获 `asyncio.CancelledError`（asyncio 收到 Ctrl+C 后会 cancel 当前任务），在 `dispatch_pysandbox()` 中捕获 `KeyboardInterrupt`，两者都走正常退出路径（不打印 traceback）。

```
Ctl+C 按下
  → 线程池: interact() 捕获 KeyboardInterrupt → 打印 "KeyboardInterrupt" → 继续
  → 主线程: asyncio.run() cancel 所有 task → CancelledError
  → _run_repl() except CancelledError → 走 finally 清理 → 静默返回
  → dispatch_pysandbox() except KeyboardInterrupt → 不打印 traceback → return
```

**同时修复**：`_run()`（非 REPL 单次执行模式）同样应该捕获 `KeyboardInterrupt`，避免单条代码执行被 Ctrl+C 中断时打印 traceback。

## 决策记录

- **Q1 — sanitize 力度**：只处理空格和 ASCII 标点 → 下划线，不引入 unicode 规范化
- **Q2 — logging session**：和 `setup_agent()` 一致，每次运行创建新 session（时间戳），写入 `.mutagent/logs/`

## 消费者场景

| 消费者 | 场景 | 验收标准 |
|--------|------|---------|
| `mutagent pysandbox` REPL | 用户输入代码，后台 MCP 连接失败 | 控制台干净，无 WARNING 日志冲入；`help()` 可查看 namespace 失败状态 |
| `mutagent pysandbox -c "print(My_MCP)"` | sandbox 代码访问带空格 MCP namespace | 可通过 sanitized 名称正常访问 |
| `mutagent pysandbox` REPL | Ctrl+C 中断代码执行 | 打印 `KeyboardInterrupt` 后继续到 `>>>` 提示符；连续多按几次后退出不打印 traceback |
| `mutagent pysandbox -c "import time; time.sleep(10)"` | Ctrl+C 中断单次执行 | 退出不打印 traceback |

## 实施步骤清单

- [x] `cli/pysandbox.py` — 在 `_build_sandbox()` 中添加 logging 配置（LogStore + 文件 handler）
- [x] `sandbox/_adapter_mcp.py` — 添加 `_sanitize_ns_name()` 并在 `MCPConnection.__init__()` 中使用
- [x] `cli/pysandbox.py` — `_run_repl()` 捕获 `asyncio.CancelledError`，`dispatch_pysandbox()` 捕获 `KeyboardInterrupt`
- [x] `cli/pysandbox.py` — `_setup_pysandbox_logging` handler 挂 root logger 而非 `mutagent` logger（避免 mutio 的日志经 lastResort 漏到 stderr）
- [x] `builtins/main_impl.py` — `setup_agent()` 同样改为 root logger（同样的问题）
