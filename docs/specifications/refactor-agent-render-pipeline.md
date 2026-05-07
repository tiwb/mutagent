# Agent 事件渲染管道统一 设计规范

**状态**：✅ 已完成
**日期**：2026-05-07
**类型**：重构

## 需求

1. Agent 输出事件 → 用户可见展示的管道在 CLI 和 WebUI 两条路径上完全独立，无共享代码
2. CLI 通过 `UserIO.render_event()` 渲染，内置流式 block 检测状态机（````mutagent:type`）和 BlockHandler 扩展机制
3. WebUI 通过 `Conversation._handle_agent_event()` 渲染，直接操作 ViewItem（UserTextItem / AssistantTextItem / ToolCallItem 等），不使用 UserIO，不支持 block 类型解析
4. `agent.submit()` + `agent.subscribe()` 路径完全绕过 UserIO
5. UserIO 将流式 block 解析和终端渲染耦合在一起，无法复用于 Web
6. WebUI 有一套自己的 item 模型（UserTextItem、AssistantTextItem、ToolCallItem、TurnSeparatorItem 等），语义层面更贴近 Agent 实际产出，但只存在于 webui 目录内
7. 如果要加新的 block 类型（如 `mutagent:thinking`），目前需要 CLI 和 WebUI 各写一份处理逻辑

### 事实澄清（2026-05-07 修正）

原需求把症状当成了问题。重新核对后实际情况是：

- **UserIO 名为抽象层，实际只服务 CLI**。WebUI 从未用 UserIO，Conversation 直接订阅 Agent 事件构造 item。所谓"CLI/Web 渲染抽象"只是外壳命名，生产代码零双端复用。
- **真正的不统一在入口流程，不在渲染层**：CLI 手动驱动 `agent.run(input_gen)` + queue + sentinel；Web 走 `agent.submit() + agent.subscribe()`。Agent 已内建 pub/sub 契约，CLI 没用。
- **交互 block（ask/confirm）是 mutbot 早期测试遗留**，CLI 自己也没真支持（BlockHandler 只登 `_pending_interaction`，UserIO 并未实现后续反馈回路）。属于过渡设计，应删。
- **BlockHandler 全系死代码**：系统 prompt 未指示 LLM 产出 `mutagent:` 块，CLI 的流式 block 检测状态机永远处于 NORMAL 状态，所有 BlockHandler 子类（tasks/status/code/thinking/ask/confirm/default）从未被触发。WebUI 有自己的 BlockRenderer 同样不会触发。两端独立实现的 block 渲染基础设施均是死代码，全删。
- **死代码清单**（生产零调用）：`UserIO.present()` / `UserIO.input_stream()` / `BlockHandler.render(content)` / `mutagent.messages.Content` / `AskHandler` / `ConfirmHandler` / `_pending_interaction` / 所有 BlockHandler 子类 / `discover_block_handlers()` / `_colorize_task_line`。
- **rich 可选依赖**：`mutagent.extras.rich` 源码已不存在（只剩 mypy cache），`tests/test_rich_extras.py` 引用的源码路径也已失效。一并清理。

基于上述澄清，需求从"消除 CLI/Web 双端渲染重复"调整为：**撤除名不副实的 UserIO 抽象，让 CLI/Web 共享 Agent 的 `submit/subscribe` 入口契约**。

## 关键参考

- `src/mutagent/userio.py` — UserIO 和 BlockHandler 声明（本次删除）
- `src/mutagent/builtins/userio_impl.py` — UserIO 实现（本次搬迁后删除）
- `src/mutagent/builtins/block_handlers.py` — BlockHandler 子类（全系死代码，本次删除）
- `src/mutagent/builtins/agent_impl.py` — Agent.run() 核心循环 + submit() 封装
- `src/mutagent/builtins/main_impl.py` — CLI App.run() 入口，使用 UserIO
- `src/mutagent/webui/_conversation_impl.py` — WebUI Conversation 实现，`_handle_agent_event()` 自建渲染
- `src/mutagent/webui/messages.py` — WebUI 的 ViewItem 类定义
- `src/mutagent/agent.py` — Agent 声明，提供 run() 和 submit() 两个入口
- `src/mutagent/builtins/agent_impl.py:55-60` — `_emit_event` 已兼容同步 callback，`agent.subscribe(queue.put)` 可直接用
- `src/mutagent/builtins/agent_impl.py:298-370` — `submit()` 内 `_drive()` 在 CancelledError / Exception 分支兜底发 turn_done
- `src/mutagent/cli/` — CLI 模块目录（log_query.py / pysandbox.py），TerminalRenderer 搬迁目标
- `src/mutagent/builtins/main_impl.py:321-400` — CLI `App.run()` 线程模型实现（保留不动）
- `src/mutagent/main.py:34` — `App.userio: UserIO` 公开字段（重构后撤除）
- `src/mutagent/runtime/ansi.py` — ANSI 色彩工具（移到 `cli/ansi.py`）
- `tests/test_ansi.py` — ANSI + 格式化函数测试（import 路径更新为 `cli.ansi`）
- `tests/test_rich_extras.py` — 引用不存在的 `mutagent.extras.rich` 源码（本次删除）

## 设计方案

### 核心思路

承认 UserIO 只服务 CLI，把它从"抽象层"降格为 CLI 内部实现。CLI 和 Web 共享 **Agent 的 `submit + subscribe + cancel` 契约**，不共享渲染器。渲染各做各的：CLI 流式打印 + Markdown 高亮，Web 操作 ViewItem。

```
            ┌──────────────────────────────┐
            │ Agent （唯一共享契约层）      │
            │  submit / subscribe / cancel │
            └──────────────┬───────────────┘
           StreamEvent     │       StreamEvent
                ▼                       ▼
    ┌───────────────────┐     ┌───────────────────┐
    │ CLI (cli/terminal)│     │ Web (Conversation)│
    │ TerminalRenderer  │     │ ViewItem 维护器   │
    │ + Markdown 高亮   │     │ + shell 状态      │
    └───────────────────┘     └───────────────────┘
```

### 线程模型决策（必保留）

**CLI 继续使用"主线程阻塞 input + 后台 asyncio loop 线程 + queue 桥接"，不切换到 asyncio 主循环。**

原因（已踩坑）：

1. **SIGINT 送达路径**：Python 把 SIGINT 送到主线程。主线程阻塞在 `input()` 上，C 层立即抛 KeyboardInterrupt，无 asyncio 中间层。
2. **Windows ProactorEventLoop 的 Ctrl+C 历史坑**：Python 3.8 以前不响应，3.8/3.11 逐步改进但平台差异仍存。
3. **`aioconsole.ainput` 不解决问题**：底层 `run_in_executor` 的线程不可强制终止，"agent 已退出但 stdin 被僵尸线程占着"。
4. **两种 Ctrl+C 场景的区分**：`waiting_for_input` 布尔标志清晰区分"等输入时 Ctrl+C"和"agent 运行中 Ctrl+C"，分别走 confirm_exit / agent.cancel。

**结论**：CLI 线程模型 100% 保留，重构只换"事件消费入口"。

### CLI 入口改造（契约对齐，线程模型不动）

**启动时订阅一次**（`queue.put` 线程安全）：
```python
self.agent.subscribe(event_q.put)
```

**提交**（删掉 `single_input()` async gen / `run_agent()` 包装 / sentinel / `iter(q.get, None)`）**：
```python
asyncio.run_coroutine_threadsafe(self.agent.submit(text), loop)
while True:
    try:
        evt = event_q.get(timeout=0.2)
    except queue.Empty:
        # 双保险：若 agent 已非 busy 且 queue 空，说明本轮结束（防 turn_done 丢失）
        if self.agent.is_busy():
            continue
        break
    renderer.render_event(evt)
    if evt.type == "turn_done":
        break
```

**Ctrl+C / Ctrl+D 冒泡约定**（重要）：
`TerminalRenderer.read_input()` 必须**裸 `input()`**，不捕获 `KeyboardInterrupt` / `EOFError`。两类异常由 `App.run()` 的顶层 except 分支处理——`KeyboardInterrupt` 走 `waiting_for_input ? confirm_exit() : agent.cancel()` 二分，`EOFError` 直接 break。在 renderer 内部捕获会把两条退出路径全部屏蔽（Ctrl+C 变成按回车、Ctrl+D 进入死循环）。

**Ctrl+C 处理**（agent 运行时改走 `agent.cancel()`）：
```python
except KeyboardInterrupt:
    if waiting_for_input:
        if renderer.confirm_exit(): break
    else:
        self.agent.cancel()
        # _drive() 捕 CancelledError 后兜底 emit turn_done，
        # 主循环在下一次 get 拿到 turn_done 自然结束本轮
```

### CLI 模块结构

搬迁后 `src/mutagent/cli/` 目录结构：

```
cli/
├── __init__.py
├── log_query.py
├── pysandbox.py
├── ansi.py          ← runtime/ansi.py + 格式化函数合入
└── terminal.py      ← 新建，TerminalRenderer
```

**`cli/ansi.py`**（`runtime/ansi.py` + 格式化函数合入）：
- ANSI 色彩函数（`dim`、`green`、`bold_red`、`cyan` 等）+ 终端检测
- `highlight_markdown_line` + `_apply_inline_patterns` + 相关正则
- `_format_tool_call` / `_format_tool_result` / `_format_value` + 常量（`_MAX_VALUE_LEN`、`_MAX_SINGLE_LINE` 等）
- 理由：都是 CLI 专用辅助函数，放一起避免文件碎片化

**`cli/terminal.py`**（新建）：
- `TerminalRenderer` class（普通 class，不走 Declaration/`@impl`）
- `render_event()`、`read_input()`、`confirm_exit()`（从 `userio_impl.py` 移入）
- 渲染逻辑简化：`text_delta` 不再走流式 block 状态机，改为直接 `print(event.text, end="", flush=True)`
- 从 `cli.ansi` 导入色彩 + 格式化函数

### 删除清单

| 删除项 | 位置 | 原因 |
|--------|------|------|
| `Content` dataclass | `mutagent.messages` | 零调用 |
| `UserIO` Declaration | `mutagent.userio` | 撤除 |
| `userio_impl.py` | `mutagent.builtins` | 搬迁后删除 |
| `block_handlers.py` | `mutagent.builtins` | 全系死代码 |
| `ansi.py` | `mutagent.runtime` | 移到 `cli/ansi.py` 后删除 |
| `AskHandler` / `ConfirmHandler` / `DefaultHandler` / 所有 Handler 子类 | `mutagent.builtins.block_handlers` | 永不触发 |
| `discover_block_handlers()` | `mutagent.builtins.userio_impl` | 随文件一起删 |
| `_pending_interaction` / `_transfer_pending_interaction` | `mutagent.builtins.userio_impl` | 交互 block 死代码 |
| 流式 block 状态机（`_process_text` / `_process_complete_line` / `_get_parse_state` / `_reset_parse_state` / `_BLOCK_OPEN_RE` / `_BLOCK_CLOSE_RE` / `_could_be_block_start`） | `mutagent.builtins.userio_impl` | BlockHandler 删除后无意义 |
| `UserIO.present()` / `UserIO.input_stream()` | `mutagent.builtins.userio_impl` | 零调用 |
| `test_rich_extras.py` | `tests/` | 引用不存在的源码 |
| `mypy_cache` 中 `extras/rich/` | `.mypy_cache/` | 缓存残留 |
| 相关 test 用例 | `tests/test_userio.py`、`tests/test_ansi.py` | 覆盖已删除符号的用例 |

### `main_impl.py` 改动

- `setup_agent()`：删除 UserIO 创建、`block_handlers` import 和 `discover_block_handlers()` 调用
- `App.run()`：创建 `TerminalRenderer` 实例 + 事件消费从 `agent.run()` 改为 `agent.subscribe()` + `agent.submit()`
- `App.run_webui()`：零改动

## 非目标

- **不动 WebUI**：Conversation 的 item 模型保持原样；WebUI 自有的 BlockRenderer 保留不动
- **不动 CLI 线程模型**：主线程阻塞 input + 后台 asyncio loop + queue 桥接保持
- **不引入共享渲染层/共享 item 模型**：两端通过 Agent 的 StreamEvent 契约统一，渲染各做各的
- **不考虑下游兼容**：本次为内部重构，不保留旧 import 路径

## 验收标准

- CLI 交互行为与重构前完全一致（流式渲染、Markdown 高亮、Ctrl+C 语义、exit/confirm 等）
- WebUI 交互与重构前完全一致（Conversation 代码可以零改动）
- `from mutagent.userio import` 在 grep 中零命中
- `from mutagent.builtins.block_handlers import` 在 grep 中零命中
- `App.userio` 属性在 grep 中零命中
- `mutagent.extras.rich` 在 grep 中零命中
- 实施前 grep `D:/ai/mutbot`、`D:/ai/mutbot.ai`、`D:/ai/mutgui` 确认无 `app.userio` / `from mutagent.userio` / `from mutagent.builtins.block_handlers` 引用
- 没有测试因删除失败

## 实施步骤清单
- [x] 步骤 1：创建 `cli/ansi.py`，合并 `runtime/ansi.py` + 格式化函数 + `_colorize_task_line`
- [x] 步骤 2：创建 `cli/terminal.py`，新建 `TerminalRenderer` 类（render_event / read_input / confirm_exit，无 block 状态机）
- [x] 步骤 3：修改 `main_impl.py` — `setup_agent()` 删除 UserIO 创建 + `App.run()` 改用 `agent.subscribe/submit`
- [x] 步骤 4：修改 `main.py` — 删除 `userio` 属性声明和相关 import
- [x] 步骤 5：删除 `Content` 类（messages.py）
- [x] 步骤 6：删除死代码文件（userio.py / userio_impl.py / block_handlers.py / runtime/ansi.py / test_rich_extras.py）
- [x] 步骤 7：更新 `test_ansi.py` import 路径 + 重写 `test_userio.py`（TerminalRenderer 测试）
- [x] 步骤 8：全量测试（721 passed, 4 skipped），grep 验收零命中，mypy cache 清理

## 实施后 bugfix（2026-05-07）

**问题**：初版 `TerminalRenderer.read_input()` 捕获并吞掉了 `KeyboardInterrupt` 与 `EOFError`（各 `return ""`），导致：

1. 等输入时 Ctrl+C → 返回 `""` → `if not user_input: continue` → `App.run()` 的 `except KeyboardInterrupt` 永不触发，`confirm_exit()` 形同虚设，Ctrl+C 变成按回车。
2. Ctrl+D / Ctrl+Z → 同样被吞，顶层 `except EOFError: break` 不触发，进入"打印提示 + 等输入" 死循环。

违反验收标准「CLI Ctrl+C 语义、exit/confirm 与重构前完全一致」。原 `userio_impl.read_input()` 是裸 `input()`，异常天然往上冒。

**修复**：`cli/terminal.py:read_input()` 改回裸 `input()`，不 try/except；补测试 `test_read_input_ctrl_c_bubbles` / `test_read_input_ctrl_d_bubbles` 断言两类异常向上抛。

- [x] 修复 `cli/terminal.py:read_input()` 去掉 KeyboardInterrupt / EOFError 吞噬
- [x] `test_userio.py` 补两条冒泡测试（723 passed, 4 skipped）
