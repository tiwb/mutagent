# pysandbox 端口环境变量 设计规范

**状态**：✅ 已完成
**日期**：2026-05-11
**类型**：功能设计

## 需求

1. mutagent pysandbox 的 `--port` 参数在不显式指定时，从环境变量 `MUTAGENT_PORT` 读取默认值
2. mutagent 启动后设置 `os.environ['MUTAGENT_PORT']` = 当前服务端口，子进程可继承
3. 环境变量适用于绑定端口的服务模式（`pysandbox --serve`、`webui`）。主 REPL 不绑端口，无需写入

## 消费者场景

| 消费者 | 场景 | 依赖的输出 | 验收标准 |
|--------|------|-----------|---------|
| pi agent | 调用 mutagent pysandbox 执行代码 | 无需指定 `--port` | `mutagent pysandbox -c "code"` 自动找到端口 |
| 子进程 | mutagent agent 内启动外部工具 | 继承 `MUTAGENT_PORT` | `os.environ['MUTAGENT_PORT']` 为父进程端口 |

## 关键参考

- `mutagent/src/mutagent/cli/pysandbox.py` — `add_pysandbox_subcommand()` / `PysandboxClient.resolve_url()` / `_validate()`
- `mutagent/src/mutagent/main.py` — `main()` CLI 入口
- `mutagent/src/mutagent/webui/cli.py` — webui 子命令
- `mutagent/src/mutagent/builtins/main_impl.py` — `App.run()` / `App.run_webui()` 实现

## 设计方案

### 环境变量命名

`MUTAGENT_PORT` — 所有 mutagent 服务模式共用同一端口。

### 读取：pysandbox --port 默认值

在 `add_pysandbox_subcommand()` 中，`--port` 的 `default` 从 `os.environ.get("MUTAGENT_PORT")` 读取 int。

优先级：`--port 显式值` > `MUTAGENT_PORT 环境变量` > 校验报错。

`PysandboxClient.resolve_url()` 无需修改——`args.port` 已有值时自然进入现有分支。

### 写入：mutagent 启动时设置

需要设置的两个位置（都在 socket bind 成功后）：

1. **`_serve()`** (`pysandbox.py`) — pysandbox server 模式，bind 后 `actual_port` 已知
2. **`run_webui()`** (`main_impl.py`) — webui 模式，bind 后 `actual_port` 已知

`run()` (REPL) 不绑定端口，无需设置。

写入方式：`os.environ["MUTAGENT_PORT"] = str(actual_port)`，子进程通过 `os.environ` 自动继承。

### 非法环境变量值的处理

`MUTAGENT_PORT` 由用户/父进程提供，可能不是合法整数。argparse 的 `type=int` 只对用户**显式输入**生效，对 `default` 不调用。因此在 `add_pysandbox_subcommand()` 内部读 env 时手动 `int(...)`：

- 转换成功 → 作为 `--port` 默认值
- 转换失败 → 打 warning 到 stderr，fallback 为 `None`（与未设置 env 等价），让后续校验流程统一报错

避免拼出 `http://host:abc/mcp` 这种非法 URL 后才在 RPC 层失败。

### 连接失败错误提示

两处缺失端口的错误提示均追加 `MUTAGENT_PORT` 环境变量 hint：

- `_validate()` 中的 `parser.error()` 调用（端口缺失时报 `argparse` 错误）
- `server_unreachable_message()` 的默认提示（端口有值但连接失败时）

让用户在通过环境变量发现端口失败时更易反查。下游通过 `unreachable_hint=...` 覆盖默认提示的不受影响。

## 实施步骤清单

- [x] `cli/pysandbox.py`：`add_pysandbox_subcommand()` 中实现 `MUTAGENT_PORT` → `--port default` 的读取与非法值降级
- [x] `cli/pysandbox.py`：`_serve()` 在 bind 成功后写入 `os.environ["MUTAGENT_PORT"]`
- [x] `builtins/main_impl.py`：`run_webui()` 在 bind 成功后写入 `os.environ["MUTAGENT_PORT"]`
- [x] `cli/pysandbox.py`：`server_unreachable_message()` 默认提示中追加 `MUTAGENT_PORT` 相关 hint
- [x] `cli/pysandbox.py`：`_validate()` 中 `parser.error()` 端口缺失提示追加 `MUTAGENT_PORT` 相关 hint
- [x] 手工验证：`mutagent pysandbox --serve --port 0` 启动后 env 写入；子进程 `mutagent pysandbox -c "1+1"` 自动连上
