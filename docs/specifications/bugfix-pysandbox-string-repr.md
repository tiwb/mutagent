# pysandbox 返回结果对字符串做 repr 的行为修正

**状态**：✅ 已完成
**日期**：2026-04-23
**类型**：Bug修复

## 需求

### 问题现象

`pysandbox` tool 执行代码后，如果代码的**最后一个表达式**返回字符串，返回值会被 `repr()` 包装一层。调用端拿到的是形如 `"'multi-line\\nvalue'"` 的结果：外层有引号、换行变成字面量 `\n`。

实际命中：

```bash
$ mutbot pysandbox -c "mutbot.status()"
'mutbot v0.8.999\nUptime:      1h3m13s\nWorkspaces:  3\n...'
```

`mutbot.status()` 返回的是一个多行字符串（换行分隔的状态报告），用户期望看到原文。repr 之后变成单行 + `\n` 字面量，不便于阅读。

### 根本原因

在 `mutagent/src/mutagent/sandbox/tools.py::PySandboxTools.pysandbox`（第 62–63 行）：

```python
if result.get("result") is not None:
    parts.append(repr(result["result"]))
```

`repr()` 对所有返回值一视同仁。这个设计借鉴的是 REPL：REPL 把表达式结果 `repr()` 显示能区分 `"a"` 和 `'a'`、区分 `"a\nb"` 和 `"a\\nb"`。

但 pysandbox 不是 REPL。调用者都是一次性执行：

- **MCP agent**（Claude Code 通过 `mcp__mutbot__pysandbox`）— 拿到的是给 LLM 看的文本，repr 让 traceback、日志内容变得难读
- **CLI 一次性调用**（`mutbot pysandbox -c`）— 用户期望的是原文
- **内部调用**（agent 框架自己调 `PySandboxToolkit`）— 同理

REPL 式的 repr 行为在这些场景都是**负向效果**。

### 重复代码

同样的逻辑在 `mutbot/src/mutbot/builtins/pysandbox_toolkit.py::_pysandbox`（第 60–61 行）里完全重复：

```python
if result.get("result") is not None:
    parts.append(repr(result["result"]))
```

这是 mutagent 的 `tools.py` 被 mutbot 的 `pysandbox_toolkit.py` 抄过去的 —— 两者都要改。正确做法是**基础逻辑放在 mutagent**，mutbot 的 `PySandboxToolkit`（Agent Toolkit 场景）直接复用。

### 影响

- **MCP 入口**（`mcp__mutbot__pysandbox`）— Claude Code 调 `mutbot.logs()` / `mutbot.status()` / `mutbot.session_messages()` 等返回字符串的函数，都拿到 repr 后的结果，多浪费 token、可读性差
- **CLI 入口**（`mutbot pysandbox -c`）— 终端用户看到带转义的单行字符串
- **Agent tool 入口**（内部 agent 调 `PySandboxToolkit.pysandbox`）— 同样影响

所有现有下游（agent 训练数据、CLI 用例、MCP 调用日志）都在 "repr 包装的字符串"基础上工作，改完行为会直接变好，不需要迁移。

## 关键参考

- `mutagent/src/mutagent/sandbox/tools.py:62-63` — 缺陷点（MCP 入口）
- `mutbot/src/mutbot/builtins/pysandbox_toolkit.py:60-61` — 重复代码（Agent tool 入口）
- `mutagent/src/mutagent/sandbox/_engine.py` — `execute()` 返回 `{"result": Any, "stdout": str, "stderr": str}`
- `mutagent/src/mutagent/sandbox/app.py:43` — `exec_code` 契约

## 设计方案

### 修复策略

只对**字符串类型**做特殊处理，其他类型保持 `repr()`（dict / list / 对象等仍需要 repr 才能可读）。

```python
# 推荐：只对 str 类型直接输出原文
value = result["result"]
if isinstance(value, str):
    parts.append(value)
else:
    parts.append(repr(value))
```

### 备选考虑

1. **打印式**（`print(x)` 语义）：对所有类型走 `str()` 而非 `repr()`
   - 优点：统一规则
   - 缺点：`str(dict)` / `str(list)` 对复杂对象输出和 `repr` 一致，但对自定义类会拿到 `<X object at 0x...>` 而不是可读形式，反而更糟
2. **可选 flag**：加个 `raw: bool = False` 参数让调用方决定
   - 缺点：REPL 场景不存在（pysandbox 不是 REPL），徒增 API 噪音
3. **仅改 MCP 入口，不改 Agent tool**：两处行为不一致反而加剧混乱

**选 "仅对 str 特化"**。理由：str 是"已经是可读文本"的信号，repr 只会破坏可读性；其他类型的 repr 仍有价值（dict/list 的 repr 基本等于 str）。

### 去重

`mutbot/builtins/pysandbox_toolkit.py::_pysandbox` 应当直接复用 mutagent 的序列化逻辑，而不是各自维护一份。具体手段：

- 在 `mutagent/sandbox/` 抽出一个 helper `format_exec_result(result: dict) -> str`（或 `ToolResult`）
- `tools.py::PySandboxTools.pysandbox` 调用它
- `mutbot/builtins/pysandbox_toolkit.py::_pysandbox` 也调用它（通过 `from mutagent.sandbox.tools import format_exec_result`）

两份实现合并，后续再修只需要改一处。

## 待定问题

### QUEST Q1: 是否同步改 `error` + `traceback` 分支

**问题**：当前 error 分支把 `traceback` 用 `"\n"` 拼到 `error` 后面，拼出来的是**纯字符串**（不走 repr），所以 error 路径本来就没问题。是否需要额外调整？

**建议**：无需调整。error 路径已经是裸文本输出，符合预期。

### QUEST Q2: 旧数据是否需要迁移

**问题**：agent 对话历史、API 录制日志里可能存了一大堆 repr 后的字符串，改完之后历史和新数据风格不一致。

**建议**：不迁移。历史只用来回看，不再消费；新数据直接改善。

## 消费者场景

| 消费者 | 场景 | 当前行为 | 修复后期望 |
|---|---|---|---|
| Claude Code agent | `mcp__mutbot__pysandbox(code="mutbot.logs(...)")` | 拿到 `'[10:23:45] ERROR ...\\nTraceback...\\n'` | 拿到多行原文 traceback，可直接读 |
| 终端用户 | `mutbot pysandbox -c "mutbot.status()"` | 外层引号 + `\n` 字面量 | 多行原文 |
| Agent 框架内部 | `PySandboxToolkit.pysandbox(code=...)` 返回给 agent reasoning | 同上 | 同上 |
| 返回 dict / list 的场景 | `mutbot pysandbox -c "mutbot.workspaces()"` | 走 repr（无变化） | 保持 repr（可读） |

## 测试验证

- [x] `mutbot pysandbox -c "mutbot.status()"` 输出多行原文，无外层引号
- [x] `mcp__mutbot__pysandbox(code="mutbot.logs(level='ERROR', last_n=5)")` 返回多行 traceback 原文
- [x] `mutbot pysandbox -c "{'a': 1, 'b': 2}"` 仍返回 `{'a': 1, 'b': 2}`（dict 保持 repr）
- [x] `mutbot pysandbox -c "[1, 2, 3]"` 仍返回 `[1, 2, 3]`
- [x] `mutbot pysandbox -c "raise ValueError('bad')"` error 路径不受影响，traceback 仍原样输出
- [x] mutbot 的 `PySandboxToolkit`（agent 内部调用）和 MCP 入口行为一致（共用 `format_exec_result`）