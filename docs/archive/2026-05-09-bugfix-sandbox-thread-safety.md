# Sandbox 并发执行线程安全修复

**状态**：✅ 已完成
**日期**：2026-05-09
**类型**：Bug修复

## 背景

多个 MCP 客户端并发调用 `pysandbox` 工具时，各 `exec_code` 在不同 worker 线程中执行。原实现使用 `contextlib.redirect_stdout` 捕获 print 输出，该方法替换全局 `sys.stdout`，导致并发线程的输出互相污染。

### 问题现象

客户端 A 的 tool response 中混入客户端 B 的输出内容，导致 AI 误解执行结果。

### 根因

`redirect_stdout` 的实现是替换进程级全局变量 `sys.stdout`。当多个 worker 线程同时处于 `with redirect_stdout(buf)` 块中时，后进入的线程覆盖前者设置的 `sys.stdout`，所有线程的 `print()` 都写入同一个 buffer。

```
t=0  线程 A: sys.stdout = buf_A
t=1  线程 B: sys.stdout = buf_B  ← 覆盖
t=2  线程 A print(...) → 写入 buf_B（错误目标）
```

## 设计方案

### 修复 1：线程安全的输出捕获（`_engine.py`）

移除 `redirect_stdout` / `redirect_stderr`，改为在 sandbox builtins 中注入闭包 `_safe_print`：

```python
stdout_buf = io.StringIO()

def _safe_print(*args, **kwargs):
    kwargs.setdefault('file', stdout_buf)
    builtins.print(*args, **kwargs)

safe_builtins['print'] = _safe_print
```

每次 `execute()` 调用创建独立的 `stdout_buf` 和 `_safe_print` 闭包。sandbox 代码中的 `print()` 通过闭包直接写入私有 buffer，不碰全局 `sys.stdout`，线程间零共享。

**安全性**：sandbox 禁止 `import`（AST 检查），用户代码无法访问 `sys.stdout`，唯一输出途径是 `print()`。详见下文"已知前提"一节。

### stderr 字段处理

返回 dict 中 `stderr` 字段保留键以维持 caller 契约，但当前 sandbox 配置下用户无法写入 stderr，该字段始终为 `""`。未在 `_engine.py` 中创建 `stderr_buf`，避免死代码麻痹未来 reviewer 的警觉（"哦 stderr 已经在捕获了"）。

触发复活条件见下文"已知前提"。

### 修复 2：`_wrap_async` 可配置超时（`_app_impl.py`）

为支持长任务轮询机制，将 `_wrap_async` 的超时行为改为可配置：

- `SandboxApp._wrap_async_timeout`: 超时秒数（默认 120s）
- `SandboxApp._on_wrap_async_timeout`: 超时回调，签名 `(fn_name: str, future: Future) -> Any`

同时修复 TimeoutError 歧义：`future.result(timeout=N)` 抛出的 `TimeoutError` 可能是等待超时（future 未完成），也可能是协程内部抛出。通过 `future.done()` 区分：

```python
except TimeoutError:
    if future.done():
        # 协程内部抛出的 TimeoutError，直接传播
        return future.result()
    if on_timeout is not None:
        return on_timeout(fn_name, future)
    raise
```

> **移植说明**：main 分支的 `_wrap_async` 已通过 `bind_main_loop()` 重构了 event loop 注入方式，不再使用 `asyncio.run()` fallback。超时配置部分逻辑不变，已直接合入。

## 关键参考

### 源码
- `src/mutagent/sandbox/_engine.py:40-54` — `execute()` 函数
- `src/mutagent/sandbox/_app_impl.py:131-168` — `_wrap_async()` 函数
- `src/mutagent/sandbox/tools.py:77` — `run_in_executor` 调用入口

### 相关规范
- `bugfix-pysandbox-client-timeout.md` — 上游轮询机制设计（依赖本修复）

## 实施步骤清单

### Phase 1: 线程安全输出捕获 [✅ 已完成]

- [x] **Task 1.1**: 移除 `redirect_stdout` / `redirect_stderr`
  - 删除 `from contextlib import redirect_stdout, redirect_stderr`
  - 移除 `with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):` 块
  - 状态：✅ 已完成

- [x] **Task 1.2**: 注入 `_safe_print` 闭包
  - 每次 `execute()` 创建私有 `_safe_print`，绑定本次调用的 `stdout_buf`
  - 注入到 `safe_builtins['print']`
  - 状态：✅ 已完成

### Phase 2: `_wrap_async` 可配置超时 [✅ 已完成]

- [x] **Task 2.1**: 添加超时配置读取
  - 从 SandboxApp 读取 `_wrap_async_timeout`（默认 120s）和 `_on_wrap_async_timeout`
  - 状态：✅ 已完成

- [x] **Task 2.2**: 修复 TimeoutError 歧义
  - `except TimeoutError` 中检查 `future.done()` 区分等待超时和协程内部异常
  - 状态：✅ 已完成

## 测试验证

- 两个 MCP 客户端并发执行长任务，各自输出不再互相污染
- 短任务（<超时阈值）行为不变，直接返回结果
- 协程内部抛出 TimeoutError 时正确传播，不会被误判为等待超时

## 已知前提

本修复的"线程安全 print 拦截"覆盖完整性依赖 sandbox 当前的封闭配置：

- AST 禁止 `import` / `from ... import`
- `_SAFE_BUILTINS` 不暴露 `sys` / `open` / 任何 file-like 构造器
- `namespace` 注入不包含会写 stderr 的对象（如默认 logger、`print`-to-stderr 的调试工具）

满足以上前提时，`print()` 是用户代码唯一可达的输出途径，闭包拦截即 100% 覆盖。

### 触发重审条件

未来若引入以下任一变化，`stderr`（甚至 `stdout` 的非 print 写入路径）将变为可达通道，需重新评估输出捕获策略：

- 放开 `import` 限制（用户可拿到 `sys.stderr` / `warnings` / `logging` 等）
- 在 `_SAFE_BUILTINS` 暴露 `sys` 或任何 file-like 对象
- 在 `namespace` 注入 `logger` 或其他默认写 stderr 的工具
- 暴露 `os.write` / 直接 fd 操作

届时建议升级为更彻底的捕获方案（例如代理 `sys.stdout`/`sys.stderr` + `ContextVar` 实现线程隔离），并同步复活 stderr 捕获：

```python
stderr_buf = io.StringIO()
# ... 通过相应机制（namespace 注入 / sys 代理 / logger handler）将写入导向 stderr_buf
return {..., "stderr": stderr_buf.getvalue()}
```

搜索关键词：grep `stderr_buf` 与本节双向定位，避免静默 regression。
