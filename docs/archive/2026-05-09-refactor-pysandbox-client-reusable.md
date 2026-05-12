# pysandbox Client 复用重构 — `PysandboxClient` 类

**状态**：✅ 已完成
**日期**：2026-05-09
**类型**：重构

## 需求

1. mutbot 长期维护一份 ~146 行的 `cli/pysandbox.py`，与 mutagent 的 client 模式代码 80% 重复（read_code / format_result / run / 错误提示）
2. 未来基于 mutagent 的下游项目（不止 mutbot）都可能想提供自己的 `pysandbox` 子命令 — 应有官方复用机制，而非每家拷贝一份
3. 原架构里 mutbot 的 `web/server.py` 还在桥接 `mcp_sources` / `cli_sources` 进 SandboxApp（与 mutagent 的 `_build_sandbox` 重复 ~32 行），职责应该归一到 mutagent
4. `mutbot pysandbox` 命令必须保留 — 用户高频使用，且默认端口 8741 是关键便利

## 关键参考

- `mutagent/src/mutagent/cli/pysandbox.py` — 通用 CLI + `PysandboxClient` 类（重构产出）
- `mutbot/src/mutbot/__main__.py` — `_pysandbox_command` 走 `PysandboxClient(...)`（重构后 ~33 行胶水）
- `mutbot/src/mutbot/web/server.py` — 删除 mcp_sources/cli_sources 桥接段
- `mutagent/docs/specifications/feature-pysandbox-cli-refactor.md` — 上一轮拆分（独立模式废弃 + `--serve` 引入）的前置规范

## 消费者场景

| 消费者 | 场景 | 依赖的输出 | 验收标准 |
|--------|------|-----------|---------|
| Agent 通过 `mutbot pysandbox` 调代码 | `mutbot pysandbox -c "mutbot.status()"` 连本机 8741 | stdout 仅含执行结果，exit code 0/1 | 与重构前行为等价；server 不可达时给 mutbot 风格提示（`python -m mutbot`） |
| 用户在 mutbot 终端调试 | `mutbot pysandbox`（无代码源 + tty）→ 进入 REPL | 逐条 RPC 执行 | 重构后**新增**能力（原 mutbot CLI 没有 REPL）；banner 显示 `(mutbot pysandbox)` |
| 第三方下游基于 mutagent 做 CLI | 自己的 `myapp pysandbox` 子命令 | `from mutagent.cli.pysandbox import PysandboxClient`，构造时传 `prog` / `default_url` / `unreachable_hint` | 不需子类化也能完成"换品牌+换默认 server"；进阶定制可子类化覆盖任意方法 |
| `mutagent pysandbox` 自身 | 既有命令完全保留 | 行为/输出与重构前一致 | 默认走 `PysandboxClient()`（无参） |

## 设计方案

### 复用机制：构造函数参数为主，子类化为辅

把 mutbot 与 mutagent 的差异梳理成下表：

| 差异项 | 性质 | 解决手段 |
|---|---|---|
| **默认 URL = `http://127.0.0.1:8741/mcp`** | **核心功能差异** | `default_url` 构造参数 |
| 不暴露 `--serve` / `--url` / `--host` | 阉割（mutbot 只连本地） | mutbot 自己的 argparse 不注册这些选项 |
| 错误提示叫 mutbot、给 `python -m mutbot` | 品牌文案 | `prog` + `unreachable_hint` 构造参数 |

**实质差异只有"默认 URL + 品牌文案"两类**，所以 `PysandboxClient` 设计为**构造函数参数驱动**，三个 kwarg 覆盖最常见场景：

```python
PysandboxClient(
    prog="mutbot",                                    # 命令名（错误提示/示例都用它）
    default_url="http://127.0.0.1:8741/mcp",          # args 无 url/port 时兜底
    unreachable_hint="To start: python -m mutbot",   # 替代 mutagent 默认建议段
).dispatch(args)
```

### 为什么不只用子类化（演进过程）

最初设计是子类化（继承 `PysandboxClient` 覆盖 `prog` 类属性 + 几个方法）。Review 时发现：mutbot 的实际差异本质就是「URL 默认值 + 文案」，子类化 ≈ 对 1-2 个钩子做覆盖，构造函数参数更直观。

**保留子类化路径**：所有 hook 方法（`server_unreachable_message` / `repl_banner` / `resolve_url` / `read_code` / `run_client` / `run_repl`）仍可被子类覆盖。下游若想改主流程行为（如 connect 前先 ping `/health`），子类化路径完整可用 — 构造参数和子类化是互补的扩展点。

### `resolve_url` 的优先级

```python
def resolve_url(self, args):
    if getattr(args, "url", None):       # 1. 显式 --url
        return args.url
    if getattr(args, "port", None) is not None:    # 2. --port (+ --host)
        host = getattr(args, "host", "127.0.0.1")
        return f"http://{host}:{args.port}/mcp"
    if self._default_url:                # 3. 构造时的兜底
        return self._default_url
    raise SystemExit("Error: ...")        # 4. 都没有
```

用 `getattr(..., None)` 而非 `args.url` 直接访问 — 下游 argparse 不注册 `--url` / `--host` 时 args 没这些字段，`getattr` 安全回落，**下游不必给 args 注入冗余字段**。

### `unreachable_hint` 的语义

- `unreachable_hint=None`（默认）— 输出 mutagent 完整建议段（`Start a server` + `Or point to a different server`）
- `unreachable_hint="..."` — 第一行换成 `Error: {prog} server not reachable at {url} ({reason})`，建议段被 hint 完全替代

后者刻意去掉 `Or point to a different server`，避免对"只连一个固定 server"的下游产生误导（mutbot 没有 `--url` 选项，提示用户传 `--url` 是 bug）。

### 桥接职责归一到 mutagent

mutbot `web/server.py` 原本会：

```python
mcp_sources = config.get("mcp_sources", default={}) or {}
for ns_name, server_cfg in mcp_sources.items():
    conn = MCPConnection(ns_name, server_cfg, main_loop)
    sandbox_app.add_namespace(conn.namespace, on_remove=conn.close)
    await conn.reconnect()
# 同样逻辑处理 cli_sources
```

这段与 mutagent 的 `_build_sandbox` / `connect_sources` 重复。本次一并删除：

- mutbot 的 `SandboxApp` 仅承载 `mutbot.*` namespace（`MutbotTools`，提供 `mutbot.status()` 等内省函数）
- 外部 MCP/CLI 源桥接由 mutagent 在 agent 模式（`connect_sources()`）或 `--serve` 模式（`_build_sandbox()`）统一处理
- 用户的 `~/.mutbot/config.json` 同步移除 `mcp_sources` 段

## 实施步骤清单

- [x] mutagent 端：在 `cli/pysandbox.py` 引入 `PysandboxClient` 类，把原模块级 `_run_client` / `_run_repl` / `_read_code` / `_server_unreachable_message` / `_dispatch_client` 全部搬进类内方法
- [x] mutagent 端：`PysandboxClient.__init__` 接受 `prog` / `default_url` / `unreachable_hint` 三个 kwarg
- [x] mutagent 端：`resolve_url` 用 `getattr(args, ..., None)` 兼容下游 args 字段缺失，并按 url > port > default_url 优先级回落
- [x] mutagent 端：`server_unreachable_message` 在 `unreachable_hint` 不为 None 时切换为「短提示」格式
- [x] mutagent 端：`dispatch_pysandbox` 内部走 `PysandboxClient().dispatch(args)`，对外 API 签名不变
- [x] mutagent 端：模块/类 docstring 更新，说明构造函数参数用法 + 子类化进阶路径
- [x] mutbot 端：删除 `src/mutbot/cli/pysandbox.py`（146 行）
- [x] mutbot 端：`__main__.py` 的 `_pysandbox_command` 改为构造 `PysandboxClient(prog="mutbot", default_url=..., unreachable_hint=...)` 调 `dispatch(args)`
- [x] mutbot 端：删除 `web/server.py` 的 `mcp_sources` / `cli_sources` 桥接段（~32 行），保留 `SandboxApp` 初始化与 `MutbotTools` 自动发现
- [x] 用户配置：从 `~/.mutbot/config.json` 移除 `mcp_sources` 配置段
- [x] 烟雾测试：mutagent 默认 client 输出与重构前等价；mutbot 客户端 `--help`、连接失败错误、`-c "..."` 均符合预期

## 验证

### 行为对照

| 场景 | 命令 | 期望 |
|---|---|---|
| mutagent 默认 client 错误提示 | （无 server）`mutagent pysandbox --port 65530 -c "1"` | 完整 `Start a server` + `Or point to a different server` 段，命令名全是 `mutagent` |
| mutbot 错误提示 | （无 server）`mutbot pysandbox --port 65530 -c "1"` | `Error: mutbot server not reachable at ...` + `To start: python -m mutbot` + `Read logs offline: tail -100 ~/.mutbot/logs/server-*.log`；**不出现** `--url` / `--serve` 字样 |
| mutbot REPL（新能力） | `mutbot pysandbox` | banner `Python sandbox (mutbot pysandbox)`，可逐条 RPC |
| `resolve_url` 优先级 | 构造 `default_url` 同时给 `args.port` | port 优先（args 显式优先于构造时默认） |

实测全部通过（见会话烟雾测试）。

### 代码量变化

| 文件 | 行数变化 |
|---|---|
| `mutagent/src/mutagent/cli/pysandbox.py` | +246 / -127 净增 ~119 行（含完整 docstring + `PysandboxClient` 类） |
| `mutbot/src/mutbot/cli/pysandbox.py` | **删除 146 行** |
| `mutbot/src/mutbot/__main__.py` | +33 / -3 净增 ~30 行 |
| `mutbot/src/mutbot/web/server.py` | -32 行（mcp_sources/cli_sources 桥接） |

mutbot 净减 ~115 行；mutagent 净增 ~119 行（其中约 60 行是 docstring/示例）。**总和**接近持平，但**重复消除**，下游再增加新项目时零成本复用。

## 遗留问题

- mutbot 错误提示第一行从「`mutbot server not running at ...`」变为「`mutbot server not reachable at ...`」（`running` → `reachable`）。后者更准确（端口冲突/防火墙等也算），可接受；如未来用户反馈"running"更直观，可子类化覆盖 `server_unreachable_message`
- mutagent 后续若给 `PysandboxClient.__init__` 加新 kwarg，下游无需修改（构造函数参数向后兼容）；若新增主流程方法或修改既有方法签名，子类化下游需跟进
