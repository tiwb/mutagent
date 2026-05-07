# pysandbox tool description 丢失限制说明 设计规范

**状态**：✅ 已完成
**日期**：2026-05-07
**类型**：Bug修复

## 需求

1. Agent 调用 pysandbox 时经常尝试 `import` 语句，每次都报错回退——LLM 不知道这个限制
2. 根因：`parse_docstring()` 只取第一行作为 tool description，`SandboxToolkit.pysandbox` 的 docstring 中 "NOT supported: import, eval, exec, ..." 在第 20+ 行，被丢弃
3. `mutbot.builtins.pysandbox_toolkit.py` 也有同样问题（`import is not supported` 同样不在第一行）

## 关键参考

- `mutagent/builtins/schema.py` — `parse_docstring()` (L36) + `make_schema()` (L149) — 只取第一行
- `mutagent/sandbox/entry_agent.py` — `SandboxToolkit.pysandbox` docstring，用 `PYSANDBOX_DOC` 插值
- `mutagent/sandbox/app.py` — `PYSANDBOX_DOC` 常量
- `mutbot/builtins/pysandbox_toolkit.py` — mutbot 版，同样问题
- `mutagent/tests/test_schema.py` — parse_docstring 测试，需同步更新预期

## 设计方案

### 修改 `parse_docstring`：description 取节标题前的全部文本

当前只取第一行：

```python
# 当前
for line in lines:
    stripped = line.strip()
    if stripped:
        description = stripped
        break
```

改为取 `Args:/Arguments:/Parameters:/Returns:/Raises:...` 等节标题前的所有行：

```python
SECTION_HEADERS = ("Args", "Arguments", "Parameters", "Returns", "Return",
                   "Raises", "Raise", "Yields", "Yield", "Note", "Notes",
                   "Example", "Examples", "Attributes")
desc_lines = []
for line in lines:
    stripped = line.strip()
    if re.match(r"^(%s)\s*:" % "|".join(SECTION_HEADERS), stripped):
        break
    desc_lines.append(line.rstrip())
description = "\n".join(desc_lines).strip()
```

### 影响面评估

`parse_docstring` 只被 `make_schema()` 一处调用，后者只用于 Agent 内部 tool schema 生成。影响：

| 工具 | 当前 description | 改后 description |
|------|-----------------|-----------------|
| pysandbox | "Execute Python code in a sandboxed environment." | 完整 PYSANDBOX_DOC + 限制说明 |
| Module-inspect | "Inspect the structure of a Python module." | 同（该 docstring 首行后直接是 Args:，无额外描述行） |
| Module-define | "Define or redefine a Python module at runtime." | 同 |
| Log-query | "Query log entries or configure logging." | 同 |
| 所有带 Args: 的工具 | 不变 | 不变 |

**结论**：改动安全——大部分工具 docstring 首行后直接是 Args: 节，description 不变。只有 pysandbox 这种长描述无 Args: 的场景会改变。

### 测试更新

`test_schema.py` 中 `test_description_and_args` 和 `test_no_args_section` 的预期需更新：
- `test_no_args_section`：docstring 无 Args 节，description 改为全部文本
- `test_description_and_args`：首行后有空行才到 Args，description 应包含空行前的文本

## 待定问题

无。

## 实施步骤清单

- [x] 修改 `parse_docstring`：description 取节标题前的全部文本
- [x] 修复 `entry_agent.py` / `entry_mcp.py`：`%s` 取模表达式不被 Python 认作 docstring，改为显式 `__doc__` 赋值
- [x] 更新 `test_schema.py` 中受影响的测试用例预期（实际无测试需要改——现有 docstring 首行后紧接 Args:，description 不变）
- [x] 跑 pytest 验证无回归（790 passed）

| 消费者 | 场景 | 依赖的输出 | 验收标准 |
|--------|------|-----------|---------|
| Agent LLM | 收到 pysandbox tool schema | description 含 `import` 限制 | schema.description 中出现 "import" / "NOT supported" |
| 其他 Toolkit (Module/Log/Agent) | tool schema 生成 | description 语义不变 | 现有 schema 测试通过 |
