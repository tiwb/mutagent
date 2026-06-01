# Session 持久化与会话生命周期

**状态**：✅ 已完成
**日期**：2026-06-01
**类型**：功能设计

## 概述

mutagent session 子系统提供三层能力：

1. **Transcript 层**：`ToolUseBlock` / `ToolResultBlock` 分离模型，append-only 对话记录
2. **持久化层**：JSONL 格式的 session 文件读写，支持 lazy create、增量写入、恢复
3. **运行时层**：`AgentSession` 公开接口，供 CLI / WebUI 统一消费

## 关键参考

- `src/mutagent/core/session.py` — `AgentSession` 声明
- `src/mutagent/core/_session_impl.py` — Session 持久化实现、`SessionRuntime` Extension、JSONL codec
- `src/mutagent/core/messages.py` — `ToolUseBlock`（请求）、`ToolResultBlock`（结果）block 类型
- `src/mutagent/core/_agent_impl.py` — Agent 主循环，append-only transcript 流程
- `src/mutagent/core/_tools_impl.py` — 工具调度返回 `ToolResultBlock`
- `src/mutagent/core/_llm_impl_anthropic.py` / `_llm_impl_openai.py` — 基于新 block 模型的 provider 转换
- `src/mutagent/cli/terminal.py` — CLI `--resume`、事件驱动增量持久化
- `src/mutagent/webui/_conversation_impl.py` — WebUI 适配新 block 类型
- `tests/core/test_session.py` — session 持久化行为测试
- `tests/cli/test_terminal_session.py` — CLI session 集成测试

## 设计方案

### 一、Transcript 层：ToolUseBlock / ToolResultBlock 分离

`Message` 只表达可持久化事实，不再承载执行中的临时状态。

```python
@dataclass
class ToolUseBlock(ContentBlock):
    """工具调用请求块——assistant 发出，创建后不可变。"""
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolResultBlock(ContentBlock):
    """工具执行结果块——放在独立 user message 中。"""
    type: str = "tool_result"
    tool_use_id: str = ""
    tool_name: str = ""
    content: str = ""
    is_error: bool = False
    duration: float = 0.0
```

**Transcript 示例**：

```
m1  user       [TurnStartBlock, TextBlock("读一下 x.py")]
m2  assistant  [ThinkingBlock(...), ToolUseBlock(id=t1, name=read_file, input={...})]
m3  user       [ToolResultBlock(tool_use_id=t1, content="...")]
m4  assistant  [TextBlock("我已经读完。")]
```

- `ToolUseBlock.status/result/is_error/duration` 字段已删除
- 工具结果通过独立 `role="user"` message 中的 `ToolResultBlock` 表达
- `TurnStartBlock` / `TurnEndBlock` 仅用于运行时输入触发，不进入持久化 transcript

### 二、Agent 主循环 append-only 流程

```
1. 追加 user message（含 TurnStartBlock + text）
2. LLM 返回 assistant message（含 ToolUseBlock）
3. 框架执行工具 → 得到 ToolResultBlock
4. 追加 user message（含 ToolResultBlock）
5. 继续下一轮 LLM 调用（回到步骤 2）
6. 最后追加 assistant message（最终文本回复）
7. turn_done 事件发送，不修改 transcript
```

`agent_run()` 的核心变更：

- 工具执行从"原地修改 `ToolUseBlock`"改为"返回 `ToolResultBlock`，追加新 user message"
- 不再追加 `TurnEndBlock` 到最后一条 assistant message
- 异常中断时不再原地标记工具块，而是丢弃未完成的工具消息

### 三、AgentSession 公开接口

```python
class AgentSession(mutobj.Declaration):
    """Runtime session lifecycle manager."""

    id: str = ""               # session 标识
    dir: Path | None = None    # session 文件目录
    cwd: str = ""              # 工作目录
    model: str = ""            # 当前模型

    def start_new(self, *, session_dir, cwd, model, session_id="") -> None:
        """初始化惰性会话。不创建文件，首次 sync() 时才落盘。"""

    def resume(self, value: str | Path, context: AgentContext) -> Path:
        """加载已有会话到 context，返回解析后的文件路径。"""

    def sync(self, context: AgentContext) -> None:
        """将 context 中新增消息写入会话文件。"""
```

### 四、SessionRuntime Extension

内部运行态簿记全部承载在 Extension 中：

```python
class SessionRuntime(mutobj.Extension[AgentSession]):
    """AgentSession internal runtime bookkeeping."""

    path: Path | None = None
    persisted_model: str = ""
    created_at: float = 0.0
    head_entry_id: str = ""
    persisted_message_count: int = 0
    is_persisted: bool = False
```

`AgentSession` Declaration 只保留外部需要读取的 4 个字段（`id`、`dir`、`cwd`、`model`），6 个内部簿记字段迁入 Extension。`_session_impl.py` 内部通过 `SessionRuntime.get_or_create(session)` 访问。

### 五、Lazy Create

`start_new()` 只准备内存态 session，不创建文件。首次 `sync()` 检测到 context 有新消息时才：

1. 创建 `~/.mutagent/sessions/` 目录
2. 生成 `YYYY-MM-DDTHH-MM-SSZ_<session_id>.jsonl` 文件
3. 写入 `session` header（含 model、cwd、created_at）
4. 追加当前 `context.prompts` 和 `model_change` entry
5. 追加新消息

CLI 启动后未发送任何消息即退出 → 不产生 session 文件。

### 六、增量写入

`sync(context)` 只追加 `persisted_message_count` 之后的新消息：

- 维护 `persisted_message_count` 追踪已持久化数量
- 维护 `persisted_model` 追踪模型变化
- 模型变更时自动插入 `model_change` entry
- `TurnStartBlock` / `TurnEndBlock` 不落盘（`_filter_persisted_blocks()` 过滤）

### 七、Resume

`mutagent terminal --resume <value>` 参数解析规则：

1. 含路径分隔符或以 `.jsonl` 结尾 → 按文件路径解析
2. 否则按 `session_id` 解析，在 `~/.mutagent/sessions/` 中匹配 `*_<id>.jsonl`

resume 后：

1. `_load()` 读取 JSONL，回填 `prompts` / `messages` 到 `AgentContext`
2. `AgentSession` 恢复 id / dir / model 及内部簿记
3. 后续新消息继续追加到同一文件

### 八、JSONL 持久化格式

文件为 append-only JSONL，首行 `session` header，后续每条 entry 带 `id` + `parentId` + `timestamp`。

**Entry 类型**：

| type | payload | 说明 |
|------|---------|------|
| `session` | `id`, `version`, `timestamp`, `cwd`, `title`, `model`, `meta` | 文件头 |
| `system_prompt` | `message`（标准 Message 结构） | 系统提示词 |
| `message` | `message`（标准 Message 结构） | 对话消息 |
| `model_change` | `model`, `meta` | 模型切换记录 |

**Message JSON 映射**：

- `Message.blocks` → `content`（JSON 字段，过滤掉 TurnStartBlock / TurnEndBlock）
- `Message.id` → `id`
- `Message.label`, `sender`, `model`, `timestamp`, `duration`, `input_tokens`, `output_tokens`, `cacheable`, `priority` → 逐个映射

**内部模块级函数**：

| 函数 | 职责 |
|------|------|
| `_write_header(path, meta)` | 写入 session header |
| `_append_prompt(path, prompt, ...)` | 追加 system_prompt entry |
| `_append_message(path, message, ...)` | 追加 message entry |
| `_append_model_change(path, model, ...)` | 追加 model_change entry |
| `_save(path, data)` | 完整保存 SessionData |
| `_load(path)` | 完整加载为 SessionData |

### 九、Provider 适配

**Anthropic**: `_messages_to_claude()` 直接映射 `ToolUseBlock` → `tool_use`、`ToolResultBlock` → `tool_result`，不再依赖 `status` 字段拆分完成/未完成状态。

**OpenAI**: `_messages_to_openai()` 直接映射 `ToolUseBlock` → `tool_calls`、`ToolResultBlock` → `role: "tool"` 消息。

### 十、消费者接入

**CLI** (`cli/terminal.py`)：

- `--resume` 参数
- `_build_agent_session()` 创建或恢复 session
- `_make_session_persist_callback()` 注册事件回调，每次事件后调用 `session.sync(context)`
- 退出时最后一次 `sync()`
- 渲染适配：`tool_exec_start` 读取 `ToolUseBlock.name/input`，`tool_exec_end` 读取 `ToolResultBlock.content/is_error`

**WebUI** (`webui/_conversation_impl.py`)：

- `tool_exec_start` 通过 `isinstance(event.tool_call, ToolUseBlock)` 识别
- `tool_exec_end` 通过 `isinstance(event.tool_call, ToolResultBlock)` 识别
- 使用 `tool_call.tool_use_id` 匹配工具项，读取 `tool_call.content/is_error/duration`

### 十一、当前限制

- WebUI 尚未接入 `AgentSession` 增量持久化
- 持久层不单独记录"只移动 head、不产生新消息"的结构性操作
- Turn separator 由 runtime 事件推导，不额外落盘 turn entry

## 测试覆盖

| 测试文件 | 覆盖内容 |
|----------|---------|
| `tests/core/test_session.py` | session roundtrip、entry append、lazy create、resume by path/id |
| `tests/cli/test_terminal_session.py` | terminal `--resume` 参数解析、session id 查找 |
| `tests/core/test_messages.py` | `ToolUseBlock` / `ToolResultBlock` 块定义 |
| `tests/core/test_agent.py` | append-only 流程、tool dispatch 返回值、TurnEndBlock 移除 |
| `tests/core/test_openai_provider.py` / `test_claude_impl.py` | 新 block 模型的 provider 转换 |
| `tests/core/test_tools.py` | `dispatch()` 返回 `ToolResultBlock` |

## 实施步骤清单

- [x] 拆分 `ToolUseBlock` / `ToolResultBlock`，移除 `status/result/is_error/duration`
- [x] 改造 `ToolSet.dispatch()` 返回 `ToolResultBlock`
- [x] 改造 Agent 主循环为 append-only，追加独立 tool_result message
- [x] 移除 `TurnEndBlock` append 逻辑
- [x] 适配 Anthropic / OpenAI provider 消息转换
- [x] 新增 `AgentSession` Declaration 与 `SessionRuntime` Extension
- [x] 实现 JSONL 持久化（lazy create、增量写入、resume）
- [x] CLI `--resume` 接入与事件驱动持久化
- [x] WebUI 适配新 block 类型渲染
- [x] 更新全部相关测试
- [x] `pyright` + `mutobj-lint` 通过
