# Namespace Call 事件循环死锁修复

**状态**：✅ 已完成
**日期**：2026-05-09
**类型**：Bug 修复

## 需求

1. 通过 MCP peer client 调用 chrome-cdp daemon 的 `playwright.browser_navigate()` 时，daemon 在 120s 后返回 `TimeoutError`，且 daemon 进程后续不可用。
2. 直接通过 `chrome-cdp pysandbox -c 'playwright.browser_navigate(...)'` 调用正常（秒级返回）。
3. 死锁发生在 `pysandbox/namespaces.call` 处理路径中，影响所有通过 MCPConnection 桥接的 namespace 函数（`playwright.*` 等）。

## 关键参考

- `mutagent/src/mutagent/sandbox/share.py:195-205` — `_handle_call`，namespace call 的 server 端处理
- `mutagent/src/mutagent/sandbox/_adapter_mcp.py:780-795` — `_make_tool_func`，MCPConnection tool 的 sync wrapper
- `mutagent/src/mutagent/sandbox/_adapter_pysandbox.py:94-112` — `_make_namespace_func`，peer namespace 的 sync wrapper
- `mutagent/src/mutagent/sandbox/_app_impl.py:121-173` — `_wrap_async`，已有的 `_async_original` 模式（NamespaceTools 用）
- `mutagent/src/mutagent/sandbox/entry_mcp.py:23-29` — `PySandboxTools.pysandbox`，用 `run_in_executor` 避免死锁
- `mutagent/docs/specifications/feature-pysandbox-namespace-sharing.md` — namespace sharing 设计规范

## 根因分析

### 调用链对比

**正常路径**（`chrome-cdp pysandbox -c 'playwright.browser_navigate(...)'`）：

```
PySandboxTools.pysandbox(code=...)
  → await loop.run_in_executor(None, sandbox.exec_code, code)   ← 线程池执行
    → exec_code → playwright.browser_navigate()
      → sync tool_func() → future.result(timeout=120)           ← 阻塞执行线程
        事件循环线程空闲，正常调度 call_with_retry coroutine
        → ✅ 正常返回
```

**死锁路径**（peer MCP client 通过 `pysandbox/namespaces.call` 调用）：

```
share.py:_handle_call(namespace="playwright", function="browser_navigate", ...)
  → fn(**arguments)                                              ← 事件循环线程同步调用
    → sync tool_func() → future.result(timeout=120)              ← 阻塞事件循环线程
      call_with_retry coroutine 需要事件循环调度执行
      → ❌ 事件循环被阻塞 → 死锁 → 120s 后 TimeoutError
```

### 根因

`_handle_call`（`share.py:195-205`）对 sync 函数走 `fn(**arguments)` 直接调用（else 分支），没有用 `run_in_executor`。当 `fn` 是 MCPConnection 的 sync wrapper（`_make_tool_func` / `_make_namespace_func`）时，其内部通过 `asyncio.run_coroutine_threadsafe` + `future.result()` 需要事件循环调度——但事件循环线程正被 `_handle_call` 占用，形成同线程死锁。

### 为什么已有的 `_async_original` 模式未覆盖

`_wrap_async`（NamespaceTools 的 async 方法包装）会在 sync wrapper 上挂 `_async_original` 属性，`_handle_call` 检测到后直接 `await async_original(...)`，避免死锁。

但 `_make_tool_func` 和 `_make_namespace_func` 没有挂 `_async_original`，所以 `_handle_call` 走了 else 分支，触发死锁。

## 设计方案

### 方案：为 MCPConnection namespace 函数添加 `_async_original` 支持

复用已有的 `_async_original` 模式，`_handle_call` 无需修改。

**`_make_tool_func`（`_adapter_mcp.py`）**：在生成的 `tool_func` 上挂 `_async_original`，指向一个接受 `**kwargs` 的 async 函数，内部复用 `call_with_retry` 的逻辑。

**`_make_namespace_func`（`_adapter_pysandbox.py`）**：同时修复——它的 `ns_func` 也是同一个 sync pattern，当 peer namespace 被再次 share 到第三方时会发生同样的死锁。

### 具体做法

两个 `_make_*` 函数内部已有 `call_with_retry` 闭包，将其提取为独立的 async 函数（接受 `**kwargs`），同时挂到 wrapper 上：

```python
async def call_async(**kwargs):
    await conn.ensure_connected()
    ...
    return await client.call_tool(tool_name, kwargs)

def tool_func(**kwargs):
    future = asyncio.run_coroutine_threadsafe(
        call_async(**kwargs), conn.main_loop)
    return future.result(timeout=120)

tool_func._async_original = call_async
```

`_handle_call` 的现有逻辑自动生效：检测到 `_async_original` 后 `await async_original(**arguments)`，不再走 sync wrapper → 不死锁。

### 不采用的方案

**方案 B：在 `_handle_call` 中对 sync 函数统一用 `run_in_executor`**

```python
result = await loop.run_in_executor(None, lambda: fn(**arguments))
```

问题：`run_in_executor` 对所有 sync 函数生效，包括真正轻量的纯 CPU 函数，增加不必要的线程池调度开销。`_async_original` 方案更精准——只对需要事件循环的函数生效。

### 影响范围

- `_adapter_mcp.py` — `_make_tool_func`
- `_adapter_pysandbox.py` — `_make_namespace_func`
- `share.py` — 无需修改（已有 `_async_original` 检测逻辑）
- 测试：需要验证 peer client → daemon 的死锁场景

## 待定问题

### QUEST Q1: `_async_original` 签名的统一性

**问题**：`_wrap_async` 的 `_async_original` 是原始 coroutine 函数（`coro_fn(**kwargs)`），而 MCPConnection 内部的 `call_with_retry` 接受 `kwargs: dict` 而非 `**kwargs`。需要确认统一为 `**kwargs` 签名是否有副作用。

**建议**：统一为 `**kwargs` 签名——`_handle_call` 已经用 `**arguments` 展开调用，签名统一后无需适配。`call_with_retry` 作为闭包只在一处使用，签名的微小调整无影响。

## 消费者场景

| 消费者 | 场景 | 依赖的输出 | 验收标准 |
|--------|------|-----------|---------|
| chrome-cdp daemon | peer client 通过 namespace fusion 调用 `playwright.browser_navigate()` | `pysandbox/namespaces.call` 处理不再死锁 | `mutagent pysandbox --port 8700 -c 'playwright.browser_navigate(url="https://example.com")'` 正常返回 |
| 任意 mutagent server | 自身 namespace 被下游通过 `namespaces.call` 调用 | MCPConnection 函数不阻塞事件循环 | 同场景复现死锁后验证修复 |

## 实施步骤清单

- [x] `_adapter_mcp.py` — `_make_tool_func`：挂 `_async_original`，指向接受 `**kwargs` 的 async 版本
- [x] `_adapter_pysandbox.py` — `_make_namespace_func`：同上
- [x] 复现验证：`python -m mutagent pysandbox --port 8700 -c 'playwright.browser_navigate(url="https://example.com")'` 正常返回
