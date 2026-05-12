# pysandbox 错误信息泄露与 builtins 白名单调整

**状态**：✅ 已完成
**日期**：2026-04-14
**类型**：Bug修复

## 需求

1. sandbox 执行出错时，完整的内部 traceback（含引擎文件路径、行号、内部函数名）被原样返回给 agent，agent 无法利用这些信息，反而被误导去分析内部实现
2. `_SAFE_BUILTINS` 白名单缺少常用内省函数和异常类，导致 agent 无法用常规 Python 手段探索 sandbox 环境

## 问题分析

### 错误信息泄露

**错误传播链**：

1. `_engine.py:execute()` (L109-114) — 捕获异常，`traceback.format_exc()` 生成完整 traceback
2. `_app_impl.py:_exec_code()` (L215-219) — 透传 error dict
3. `tools.py:pysandbox()` (L31-35) — 将 traceback 拼接到 error 文本，返回给 agent

**泄露示例**（来自实际 session 9f6aa99d）：

```
TypeError: _wrap_async.<locals>.wrapper() takes 0 positional arguments but 1 was given
Traceback:
  File "D:\ai\mutagent\src\mutagent\sandbox\_engine.py", line 87, in execute
    exec(compile(ast.Module(body=tree.body, type_ignores=[]),
  File "<pysandbox>", line 4, in <module>
```

```
RuntimeError: no running event loop
  File "D:\ai\mutagent\src\mutagent\sandbox\_app_impl.py", line 121, in wrapper
    loop = asyncio.get_running_loop()
```

Agent 看到 `_wrap_async` 和 `_app_impl.py` 后开始猜测"这是异步函数"，偏离了正确方向。

### builtins 白名单

**当前 `_SAFE_BUILTINS` 设计**（`_engine.py` L10-32）基于 mutobj safe-subset 早期设计的严格限制。经过安全分析（见 `docs/design/sandbox.md` Builtin 函数安全分析），证明大部分被排除的 builtin 不存在逃逸路线，可以安全放开。

**必须排除的 builtin**（存在逃逸路线或实际危害）：
- `eval`、`exec`、`compile` — 绕过 AST 检查
- `__import__` — 加载外部模块
- `open` — 文件系统副作用
- `breakpoint` — 调试器注入
- `input` — 执行阻塞
- `exit`、`quit` — 终止进程

**新增 builtin**（安全分析证明无逃逸路线）：

| 分类 | 新增 | 说明 |
|------|------|------|
| 内省 | `dir`, `hasattr`, `getattr`, `setattr`, `delattr` | 当前无属性守卫，等价于已有的 `.attr` 语法 |
| 环境 | `globals`, `locals`, `vars`（包装版） | `globals` 等价于 `f.__globals__`；`vars` 包装为 `getattr(obj, '__dict__')` |
| 类型 | `object`, `frozenset`, `bytearray`, `complex`, `memoryview`, `slice` | 纯数据类型 |
| 数学/格式 | `pow`, `divmod`, `format`, `ascii` | 纯计算 |
| 类型检查 | `issubclass` | 返回 bool |
| 异步 | `aiter`, `anext` | iter/next 的异步版本 |
| 异常类 | `NameError`, `AssertionError`, `OSError`, `FileNotFoundError`, `PermissionError`, `TimeoutError`, `UnicodeError`, `UnicodeDecodeError`, `UnicodeEncodeError`, `ArithmeticError`, `LookupError`, `EOFError`, `StopAsyncIteration`, `GeneratorExit`, `RecursionError`, `BufferError`, `SystemError`, `FloatingPointError`, `BlockingIOError`, `ConnectionError`, `ConnectionAbortedError`, `ConnectionRefusedError`, `ConnectionResetError`, `BrokenPipeError`, `ChildProcessError`, `IsADirectoryError`, `NotADirectoryError`, `ProcessLookupError`, `InterruptedError` | 异常类不引入能力 |
| 描述符 | `property`, `classmethod`, `staticmethod`, `super` | 无 class 时惰性 |

**受限 builtin**（需包装）：
- `type` — 仅允许单参数 `type(obj)`（已有 `_safe_type`）
- `vars` — 路由到 `getattr(obj, '__dict__')` 而非 C 层直接访问

**实际影响**：`dir`/`hasattr`/`getattr` + `NameError` 直接解决 agent 花 8 轮盲猜的问题——agent 可以 `dir(web)` 列出可用接口、`hasattr(web, 'search')` 检查方法、`try/except NameError` 捕获错误。

## 设计方案

### builtins：黑名单替代白名单

不维护白名单，改为从 `builtins` 模块取全集，只排除必须禁止的项（黑名单）：

`eval`、`exec`、`compile`、`__import__`、`open`、`breakpoint`、`input`、`exit`、`quit`、`help`（sandbox 有自定义 help）

去掉 `_safe_type` 包装，`type`（含三参数动态建类）直接暴露。同时移除 AST 中 `ClassDef` 的禁止——允许 `class` 语句，标准 Python 行为。

### 错误信息清洗

`_engine.py` 的 `except` 分支过滤 traceback：去掉引擎入口帧，保留 `<pysandbox>` 及其下游所有帧。外部函数帧标记为 `<external>` 并保留函数名，agent 能看到完整调用链。

## 实施步骤清单

- [x] `_engine.py`：白名单改黑名单——从 `builtins` 模块取全集，排除危险项，删除 `_safe_type`
- [x] `_engine.py`：AST 检查移除 `ClassDef` 禁止，只保留 `import` 禁止
- [x] `_engine.py`：`execute()` 的异常处理增加 traceback 过滤，只保留 `<pysandbox>` 帧
- [x] `test_sandbox.py`：更新安全测试——反转 `dir`/`getattr`/`globals` 为可用；`type` 三参数改为允许；class 定义改为允许；新增 traceback 过滤测试
- [x] 运行测试验证

## 关键参考

- 沙箱安全模型与 Builtin 分析：`mutagent/docs/design/sandbox.md`
- 引擎执行与错误捕获：`mutagent/src/mutagent/sandbox/_engine.py` L47-114
- `_SAFE_BUILTINS` 白名单定义：`mutagent/src/mutagent/sandbox/_engine.py` L10-32
- async wrapper：`mutagent/src/mutagent/sandbox/_app_impl.py` L121, `_wrap_async`
- tool result 拼接：`mutagent/src/mutagent/sandbox/tools.py` L31-35
- namespace 构建：`mutagent/src/mutagent/sandbox/_app_impl.py:_build_namespace_dict()` L133-171
- 现有测试：`mutagent/tests/test_sandbox.py` L110-123（验证 dir/getattr 不可用 — 需更新）
- 触发问题的 session：`~/.mutbot/logs/session-20260412_041759-9f6aa99d9a92-api.jsonl`
