# Terminal --resume 增强：无值取最新 + 历史回显

**状态**：✅ 已完成
**日期**：2026-06-02
**类型**：功能设计

## 需求

1. `mutagent terminal --resume` 不带 session_id 时，自动恢复最新日期的 session
2. resume 后在终端打印之前的会话历史，输出格式与实时流程完全一致

## 关键参考

- `src/mutagent/cli/terminal.py` — `--resume` 参数定义、`TerminalRenderer`、`dispatch_terminal` 主流程
- `src/mutagent/core/_session_impl.py` — `_resolve_resume_path()` session 解析逻辑
- `src/mutagent/core/messages.py` — `StreamEvent`、`TextBlock`、`ToolUseBlock`、`ToolResultBlock`
- `tests/cli/test_terminal_session.py` — 终端 session 测试
- `tests/core/test_session.py` — session 持久化测试
- `docs/specifications/feature-session.md` — Session 持久化设计文档

## 设计方案

### 一、`--resume` 无值取最新

**argparse 改造**：`nargs="?"` + `const=""`

```
--resume            → args.resume = ""      → 最新 session
--resume abc123     → args.resume = "abc123" → 指定 session
（不带 --resume）    → args.resume = None     → 新 session
```

**`_resolve_resume_path` 空字符串分支**：

- 在 session 目录下 `sorted(glob("*.jsonl"))[-1]` 取最新
- 文件名格式 `YYYY-MM-DDTHH-MM-SSZ_<id>.jsonl`，字母序即时间序
- session 目录不存在或无文件时抛出 `FileNotFoundError`

### 二、历史回显

**核心原则**：复用 `TerminalRenderer.render_event()`，将加载的 `Message` → `ContentBlock` 转为 `StreamEvent`，保证输出与实时流程像素级一致。

**转换规则**：

| Message role | ContentBlock 类型 | 转为 StreamEvent | 备注 |
|-------------|-------------------|-----------------|------|
| user | TextBlock | `print("> text")` | 模拟 `> ` 输入提示符 |
| assistant | TextBlock | `type="text_delta", text=...` | 走现有 markdown 高亮 |
| assistant | ToolUseBlock | `type="tool_exec_start", tool_call=...` | 走 `_format_tool_call` |
| assistant | ToolResultBlock | `type="tool_exec_end", tool_call=...` | 走 `_format_tool_result` |
| assistant | （每轮末） | `type="turn_done"` | 模拟轮次结束换行 |

**ThinkingBlock** 不渲染（实时流程也不渲染）。

**调用时机**：`_build_agent_session` 之后、`"mutagent ready"` 之前。新 session 时 `context.messages` 为空，自动跳过。

**修改文件**：
- `terminal.py` — 新增 `TerminalRenderer.render_history()` 方法
- `terminal.py` — import 新增 `StreamEvent`（运行时）、`TextBlock`

### 三、不受影响的路径

- `_build_agent_session` 仅调整 resume 判定条件：`start_new()` + `resume()` 的调用顺序不变
- `AgentSession.resume` 声明不改：空字符串仍是 `str | Path`
- WebUI 不受影响：不涉及 `TerminalRenderer`

## 实施步骤清单

- [x] 让 `terminal --resume` 无值时在 CLI 主流程中正确恢复最新 session
- [x] 为 `TerminalRenderer` 增加历史回显，并复用实时事件渲染格式
- [x] 补充终端 session 与渲染测试，覆盖无值 resume 和历史回显

## 测试验证

- `pytest tests/cli/test_userio.py tests/cli/test_terminal_session.py tests/core/test_session.py`
- `pyright src\mutagent\cli\terminal.py tests\cli\test_userio.py tests\cli\test_terminal_session.py`
