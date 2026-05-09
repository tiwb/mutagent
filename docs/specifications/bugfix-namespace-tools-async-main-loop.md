# NamespaceTools async 方法未回主 loop 执行修复设计规范

**状态**：✅ 已完成
**日期**：2026-04-28（初版）、2026-05-09（适配无 setup 架构）
**类型**：Bug修复

## 需求

1. `mutagent.sandbox` 中 `NamespaceTools` 的 async 方法被同步 wrapper 调用时，不能在 pysandbox worker 线程里新建 event loop 执行。
2. async 方法应投递回主 asyncio loop 执行，符合 UI/编辑器类宿主的线程亲和性要求。
3. AI 调用 sandbox 时仍保持同步函数心智模型，不需要用户写 `await`。
4. 主 loop 引用未注入时必须明确报错；不再 fallback 到 `asyncio.run()`。
5. 避免 `get_running_loop()` 拿到当前线程 loop 后再 `future.result()` 可能导致的同线程死锁。

## 关键参考

- `src/mutagent/sandbox/_app_impl.py:_wrap_async()` — 当前错误实现：优先尝试 `get_running_loop()`，worker 线程无 loop 时 `asyncio.run()` 新建临时 loop。
- `src/mutagent/sandbox/entry_mcp.py` — `PySandboxTools.pysandbox()` 使用 `loop.run_in_executor(None, self._app.exec_code, code)`，导致 sandbox 代码在线程池执行。
- `src/mutagent/sandbox/entry_agent.py` — `SandboxToolkit.pysandbox()` 相同模式。
- `src/mutbot/builtins/pysandbox_toolkit.py` — mutbot 侧的 PySandboxToolkit @impl，相同模式。
- `src/mutagent/sandbox/_adapter_mcp.py` — 已有相同模式：MCP tool function 通过 `run_coroutine_threadsafe(main_loop)` 跨线程调度。

## 设计方案

### 架构背景

当前主线 SandboxApp 已无 `setup()` 生命周期方法（在 `7de60c8` 重构中移除）。
主 loop 引用由调用方在执行前通过 ``SandboxApp.bind_main_loop()`` 注入：

```python
# 在每个调用 exec_code 的入口点（均位于主 event loop 上）：
self._app.bind_main_loop()
result = await loop.run_in_executor(None, self._app.exec_code, code)
```

``bind_main_loop()`` 是挂在 ``SandboxApp`` 类上的辅助方法（参考已有的
``SandboxApp.remove_provider``），内部捕获 ``asyncio.get_running_loop()``
和 ``threading.get_ident()`` 写入 ``_async_loop`` / ``_async_loop_thread_id``。
抽成 helper 后，新增 entry 只需一行调用，避免 3 处重复模板代码。

涉及的入口点：
- `PySandboxTools.pysandbox()` — MCP 入口
- `SandboxToolkit.pysandbox()` — Agent tool 入口
- `PySandboxToolkit.pysandbox()` @impl（mutbot 侧）

### async wrapper 调度语义

`_build_declaration_namespaces(self)` 在发现 async 方法时，将 `self`（SandboxApp）传给 `_wrap_async(self, bound)`：

1. 如果 `app._async_loop` 已设置：
   - 非目标 loop 线程调用：使用 `asyncio.run_coroutine_threadsafe(coro, loop)` 投递到主 loop，同步 `future.result(timeout=120)`。
   - 目标 loop 同线程调用：抛出 `RuntimeError`，防止同步等待自己造成死锁。
2. 如果 `app._async_loop` 未设置：
   - 抛出 `RuntimeError`，提示调用方需先 `app.bind_main_loop()`。
   - **不再 fallback 到 `asyncio.run()`**，因为临时 loop 会让 async tool 在调用线程执行，async tool 内部发起的 I/O（如 `mutbot.status()` 中的 `urllib.request.urlopen` / `mutbot.restart()` / `mutbot.exec_frontend()` 等）无法回到主 loop 完成。

### timeout

沿用 `future.result(timeout=120)`。

### 兼容性

- `pysandbox` 仍在线程池执行 `exec_code`，避免阻塞 MCP server / agent loop。
- 同步 sandbox 代码仍通过普通函数调用 async `NamespaceTools`。
- async 实际执行线程变为主 loop 线程，满足 Qt/编辑器宿主线程亲和性。
- 对未注入 `_async_loop` 直接调用 async `NamespaceTools` 的场景，明确报错。

## 消费者场景

| 消费者 | 场景 | 依赖的输出 | 验收标准 |
|--------|------|-----------|---------|
| mutagent pysandbox REPL | 通过 MCP 连到 mutbot / mutagent --serve，调用 `mutbot.status()` 等 async 函数 | async tool 在目标 mutbot/mutagent 主 loop 线程执行 | mutbot 不卡死；pysandbox 不超时 |
| AI Agent | 通过 SandboxToolkit 调用 async NamespaceTools 方法 | wrapper 返回 async 方法结果 | async 方法观察到的 thread id 等于主 loop thread id |
| 直接调用方 | 未 `bind_main_loop()` 时调用 async wrapper | 明确生命周期错误 | 抛出 `RuntimeError` |

## 实施步骤清单

- [x] 修改 `_wrap_async(app, coro_fn)`：使用 `app._async_loop` 投递到主 loop，添加同线程死锁保护，移除 `asyncio.run()` fallback。
- [x] 在 3 个入口点（`PySandboxTools.pysandbox` / `SandboxToolkit.pysandbox` / mutbot `PySandboxToolkit.pysandbox`）注入 `_async_loop` + `_async_loop_thread_id`。
- [x] 抽 `SandboxApp.bind_main_loop()` helper，3 个 entry 改为单行调用。
- [x] 补回归测试：worker 线程同步调用的主 loop 路由、未 bind 报错、同线程同步调用报错、bind_main_loop 幂等。
- [x] 编写本规范文档。
