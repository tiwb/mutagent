# Pysandbox 交互式 REPL — 不带参数时进入交互环境

**状态**：✅ 已完成
**日期**：2026-05-08
**类型**：功能设计

## 需求

1. `mutagent pysandbox` 不带任何参数时，进入交互式 REPL 环境（类似 `python` 不带参数的行为）
2. REPL 保持变量跨行存活（`x=1` 后 `print(x)` 能打印 `1`）
3. 支持多行输入（函数定义、for 循环等），用 `>>>` / `...` 提示符区分
4. `help()` 可用，列出所有注入的 namespace
5. 错误不退出 REPL，只打印 traceback 继续等待输入
6. ^D / `exit()` 退出，行为和 Python 一致

## 关键参考

- `mutagent/src/mutagent/cli/pysandbox.py` — pysandbox CLI 入口（`dispatch_pysandbox`, `_read_code`, `_run`）
- `mutagent/src/mutagent/sandbox/_engine.py` — `execute(code, namespace, state)` 执行引擎，`state` 参数即 REPL 状态
- `mutagent/src/mutagent/sandbox/_app_impl.py` — `exec_code(self, code, state)` 沙箱方法
- `mutagent/src/mutagent/sandbox/app.py` — SandboxApp Declaration
- Python `code` 模块 — `code.InteractiveConsole`、`code.compile_command`（标准库，无需安装）

## 设计方案

### 核心思路

子类化 Python 标准库 `code.InteractiveConsole`，重写 `runsource()` 把执行路径从 Python 原生 `exec()` 替换为沙箱 `sandbox.exec_code(source, self.locals)`。

**选择 `code.InteractiveConsole` 的理由**：
- CPython 自带的标准库，零依赖
- 内置多行输入判断（`codeop.compile_command` 返回 `None` = 不完整）
- 自动集成 readline（有历史记录和行编辑）
- 正确显示 `sys.ps1` / `sys.ps2` 提示符（`>>> ` / `... `）
- 处理 ^D (EOFError) / ^C (KeyboardInterrupt) 退出

### REPL 生命周期

```
dispatch_pysandbox(app, args)
  └─ 检测无参数 → _run_repl(app.config)
       └─ 构造 SandboxApp
       └─ 启动 MCP/CLI 连接（后台 async task）
       └─ 创建 SandboxConsole(locals={})
       └─ console.interact(banner="...")
            ├─ 每行输入 → runsource(source)
            ├─ compile_command 判断完整性
            ├─ 完整 → sandbox.exec_code(source, self.locals)
            ├─ sandbox.format_result() → 打印输出
            └─ 不完整 → 返回 True（提示 ... 继续）
       └─ finally: sandbox.close()
```

### SandboxConsole 实现要点

```python
class SandboxConsole(code.InteractiveConsole):
    def __init__(self, sandbox):
        super().__init__(locals={})
        self.sandbox = sandbox

    def runsource(self, source, filename='<pysandbox>', symbol='single'):
        # 1. 判断输入完整性（compile_command 返回 None = 需要更多行）
        try:
            code_obj = code.compile_command(source, filename, symbol)
        except (OverflowError, SyntaxError, ValueError):
            self.showsyntaxerror(filename)
            return False
        if code_obj is None:
            return True  # 需要继续输入

        # 2. 通过沙箱执行完整代码块
        result = self.sandbox.exec_code(source, self.locals)
        text, is_error = self.sandbox.format_result(result)
        if text:
            print(text, file=sys.stderr if is_error else sys.stdout)
        return False
```

### 异步处理

`_run()` 已使用 `asyncio.run()` 启动 event loop + `run_in_executor` 执行 sync 代码。REPL 模式保持同样模式：

- `asyncio.run(_run_repl(config))` 启动
- 同步 `input()` 在主线程阻塞，`exec_code` 通过 `run_in_executor` 执行
- MCP 连接在 background task 中运行，不影响 REPL 输入

### exit() 处理

沙箱引擎的 `_BLOCKED_BUILTINS` 已包含 `exit` 和 `quit`，沙箱内代码无法调用真正的 `exit()`。REPL 通过以下方式退出：
- ^D (EOFError) — `InteractiveConsole.interact()` 自动处理
- ^C (KeyboardInterrupt) — 打断当前输入

如需支持 `exit()` 退出 REPL，可在 namespace 中注入一个 `exit()` 函数，调用后 raise `SystemExit` 由 REPL 循环捕获。

### 改动范围

只改一个文件：`mutagent/src/mutagent/cli/pysandbox.py`

- 新增 `SandboxConsole` 类（~20 行）
- 新增 `_run_repl()` async 函数（~25 行）
- 修改 `dispatch_pysandbox()` 的无参数分支（从报错退出改为走 REPL）

## 待定问题

## 实施步骤清单

- [x] 添加 `import code` 到模块导入
- [x] 提取 `_build_sandbox(config)` 共用沙箱构造逻辑
- [x] 新增 `SandboxConsole(code.InteractiveConsole)` 子类
- [x] 新增 `_run_repl(config)` REPL 入口函数
- [x] 修改 `dispatch_pysandbox`：无参数 + tty → 进入 REPL
- [x] 手动测试：启动 REPL、多行输入、变量存活、^D 退出
