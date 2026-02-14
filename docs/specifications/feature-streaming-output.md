# 交互模式流式输出 设计规范

**状态**：✅ 已完成
**日期**：2026-02-14
**类型**：功能设计

## 1. 背景

当前交互模式（`python -m mutagent`）中，Agent 调用 Claude API 时使用非流式请求，用户必须等待完整响应返回后才能看到输出。对于长回复或包含多轮工具调用的场景，用户体验较差——没有任何中间反馈，看起来像是卡住了。

需要在三个层次引入流式支持：
1. **LLMClient 层**：使用 Claude API 的 SSE streaming 协议获取增量响应
2. **Agent 层**：将流式事件从 LLM 客户端传递到上层调用者
3. **REPL 层**：实时将文本 token 输出到终端，并显示工具调用等中间状态

### 当前调用链

```
REPL ──await──▸ Agent.run() ──while──▸ Agent.step() ──await──▸ LLMClient.send_message()
                                │                                      │
                                │                                      └── aiohttp POST, await resp.json()
                                └── handle_tool_calls()                     （阻塞等待完整响应）
```

### 目标调用链

```
REPL ──async for──▸ Agent.run() ──async for──▸ Agent.step() ──async for──▸ LLMClient.send_message()
  │                    │                           │                              │
  │                    │                           │                              └── stream=True:  SSE 逐事件 yield
  │                    │                           │                                  stream=False: 整包请求，包装为事件 yield
  │                    │                           └── 透传 StreamEvent + 收集 response_done 驱动循环
  │                    └── step 循环 + handle_tool_calls + yield 工具执行事件
  └── 按事件类型渲染到终端
```

## 2. 设计方案

### 2.1 核心设计决策

#### 统一接口模式

**所有层级使用统一的 `AsyncIterator[StreamEvent]` 返回类型**，通过 `stream` 参数控制底层 HTTP 行为：

- `stream=True`（默认）：LLM 客户端走 SSE 流式协议，逐 token yield 事件
- `stream=False`：LLM 客户端走普通 HTTP 请求，将完整响应包装为少量事件 yield

调用方始终使用 `async for event in ...` 消费，无需关心底层是否真正流式。

**选择理由**：
- 单一接口——Agent 层只有一套循环逻辑，不存在 `run`/`run_stream` 代码重复
- 非流式包装开销可忽略（仅多创建 2-3 个 Python 对象 + yield）
- 所有调用方使用统一模式，不需要在两套 API 间抉择

#### SSE 手动解析

Claude API 的 SSE 格式固定（`event: xxx\ndata: {json}\n\n`），手动解析，不引入额外依赖。

#### 工具调用显示（中等详细度）

REPL 中显示工具名称 + 参数摘要 + 执行结果摘要。

#### 错误处理（MVP）

不做自动重试，通过 `StreamEvent(type="error")` 上报错误，由用户决定是否重新发送。

### 2.2 Claude API Streaming 协议

Claude Messages API 支持 `stream: true` 参数，返回 SSE（Server-Sent Events）流。关键事件类型：

| 事件类型 | 含义 | 关键字段 |
|---------|------|---------|
| `message_start` | 消息开始 | `message.usage.input_tokens` |
| `content_block_start` | 内容块开始（text 或 tool_use） | `content_block.type`, `content_block.id` |
| `content_block_delta` | 增量内容 | `delta.text` 或 `delta.partial_json` |
| `content_block_stop` | 内容块结束 | — |
| `message_delta` | 消息级别更新 | `delta.stop_reason`, `usage.output_tokens` |
| `message_stop` | 消息结束 | — |

### 2.3 流式事件模型（`messages.py` 新增）

```python
@dataclass
class StreamEvent:
    """流式响应中的单个事件。"""
    type: str          # "text_delta" | "tool_use_start" | "tool_use_delta" | "tool_use_end"
                       # | "tool_exec_start" | "tool_exec_end" | "response_done" | "error"
    text: str = ""                          # type="text_delta" 时的文本片段
    tool_call: ToolCall | None = None       # type="tool_use_start" / "tool_exec_start" 时
    tool_json_delta: str = ""               # type="tool_use_delta" 时，partial JSON 片段
    tool_result: ToolResult | None = None   # type="tool_exec_end" 时，工具执行结果
    response: Response | None = None        # type="response_done" 时，完整的 Response 对象
    error: str = ""                         # type="error" 时的错误信息
```

事件类型说明：
- `text_delta` — LLM 输出的文本增量
- `tool_use_start` — LLM 开始构造一个工具调用（来自 SSE `content_block_start`）
- `tool_use_delta` — 工具调用参数的 JSON 增量（来自 SSE `content_block_delta`）
- `tool_use_end` — LLM 完成一个工具调用块（来自 SSE `content_block_stop`）
- `tool_exec_start` — Agent 开始执行一个工具（Agent 层发出）
- `tool_exec_end` — Agent 工具执行完成，携带结果（Agent 层发出）
- `response_done` — 一次 LLM 调用完成，携带完整 Response
- `error` — 错误事件

### 2.4 LLMClient 层：修改 `send_message`

**声明变更**（`client.py`）：

```python
async def send_message(
    self,
    messages: list[Message],
    tools: list[ToolSchema],
    system_prompt: str = "",
    stream: bool = True,
) -> AsyncIterator[StreamEvent]:
    ...
```

**实现逻辑**（`claude.impl.py`）：

`stream=True` 路径：
1. payload 增加 `"stream": true`
2. 使用 `resp.content` 逐行读取 SSE
3. 解析 `event:` / `data:` 行，转换为 `StreamEvent` 并 yield
4. 内部累积文本和工具调用，在 `message_stop` 时组装完整 `Response`
5. yield `StreamEvent(type="response_done", response=response)`

`stream=False` 路径：
1. 与当前实现相同的普通 HTTP 请求
2. 将完整响应包装为事件序列 yield：
   - 若有文本 → `StreamEvent(type="text_delta", text=full_text)`
   - 若有工具调用 → 每个 `StreamEvent(type="tool_use_start", tool_call=tc)`
   - 最终 → `StreamEvent(type="response_done", response=response)`

### 2.5 Agent 层：修改 `run` 和 `step`

**声明变更**（`agent.py`）：

```python
async def run(self, user_input: str, stream: bool = True) -> AsyncIterator[StreamEvent]:
    ...

async def step(self, stream: bool = True) -> AsyncIterator[StreamEvent]:
    ...
```

**`step` 实现**：透传 `send_message` 的事件流。

**`run` 实现**：
```python
async def run(self, user_input, stream=True):
    self.messages.append(Message(role="user", content=user_input))
    while True:
        response = None
        async for event in self.step(stream=stream):
            yield event
            if event.type == "response_done":
                response = event.response

        self.messages.append(response.message)

        if response.stop_reason == "tool_use" and response.message.tool_calls:
            # yield 工具执行事件
            for call in response.message.tool_calls:
                yield StreamEvent(type="tool_exec_start", tool_call=call)
                result = await self.tool_selector.dispatch(call)
                yield StreamEvent(type="tool_exec_end", tool_call=call, tool_result=result)
                results.append(result)
            self.messages.append(Message(role="user", tool_results=results))
        else:
            return  # async generator 结束
```

### 2.6 REPL 层：流式渲染（`__main__.py`）

```python
async for event in agent.run(user_input):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
    elif event.type == "tool_exec_start":
        print(f"\n[调用工具: {event.tool_call.name}({_summarize_args(event.tool_call.arguments)})]",
              flush=True)
    elif event.type == "tool_exec_end":
        summary = event.tool_result.content[:100]
        status = "错误" if event.tool_result.is_error else "完成"
        print(f"  → [{status}] {summary}", flush=True)
    elif event.type == "error":
        print(f"\n[错误: {event.error}]", file=sys.stderr, flush=True)
    elif event.type == "response_done":
        print()  # 最后换行
```

## 3. 已确认决策

- **统一接口模式**：所有层级返回 `AsyncIterator[StreamEvent]`，`stream` 参数控制底层 HTTP 行为
- **`stream` 默认 `True`**：交互模式是主要使用场景，程序化调用方可传 `stream=False`
- **SSE 手动解析**：不引入额外依赖
- **工具调用中等详细度**：工具名 + 参数摘要 + 结果摘要
- **MVP 不自动重试**：错误通过 `StreamEvent(type="error")` 上报
- **破坏性变更可接受**：项目 0.1.0 早期阶段，`main.py` 中 `run_agent()` 同步修改

## 4. 实施步骤清单

### 阶段一：数据模型与 LLMClient 流式支持 [✅ 已完成]
- [x] **Task 1.1**: 在 `messages.py` 中新增 `StreamEvent` 数据类
  - [x] 定义 `StreamEvent` dataclass 及所有字段
  - [x] 确保类型注解完备
  - 状态：✅ 已完成

- [x] **Task 1.2**: 修改 `client.py` 中 `send_message` 声明
  - [x] 添加 `stream: bool = True` 参数
  - [x] 返回类型改为 `AsyncIterator[StreamEvent]`
  - [x] 添加必要的导入
  - 状态：✅ 已完成

- [x] **Task 1.3**: 在 `claude.impl.py` 中实现流式 + 非流式双路径
  - [x] 实现 SSE 行解析逻辑（手动解析）
  - [x] `stream=True`：逐事件 yield StreamEvent
  - [x] `stream=False`：包装完整响应为事件序列 yield
  - [x] 内部累积状态，流结束时组装完整 Response
  - [x] 错误处理（HTTP 错误、连接中断 → error 事件）
  - 状态：✅ 已完成

### 阶段二：Agent 流式循环 [✅ 已完成]
- [x] **Task 2.1**: 修改 `agent.py` 中 `run` 和 `step` 声明
  - [x] `run` 添加 `stream` 参数，返回类型改为 `AsyncIterator[StreamEvent]`
  - [x] `step` 添加 `stream` 参数，返回类型改为 `AsyncIterator[StreamEvent]`
  - 状态：✅ 已完成

- [x] **Task 2.2**: 修改 `agent.impl.py` 实现
  - [x] `step`：透传 `send_message` 事件流
  - [x] `run`：流式循环 + 从 `response_done` 事件提取 Response 驱动循环
  - [x] `run`：工具调用阶段 yield `tool_exec_start` / `tool_exec_end` 事件
  - [x] 正确维护 `self.messages` 历史
  - 状态：✅ 已完成

### 阶段三：REPL 流式渲染 [✅ 已完成]
- [x] **Task 3.1**: 修改 `__main__.py` 使用流式输出
  - [x] 将 `await agent.run()` 替换为 `async for event in agent.run()`
  - [x] 实现各事件类型的终端渲染（文本增量、工具执行、错误）
  - [x] 保持 Ctrl+C 优雅退出
  - 状态：✅ 已完成

- [x] **Task 3.2**: 同步修改 `main.py` 中的 `run_agent()`
  - [x] 适配 `Agent.run()` 新返回类型（消费事件流，提取文本）
  - 状态：✅ 已完成

### 阶段四：测试验证 [✅ 已完成]
- [x] **Task 4.1**: Agent 流式循环测试（`test_agent.py`）
  - [x] 适配原有 5 个测试用例到流式接口
  - [x] 新增 4 个流式事件序列测试
  - [x] 覆盖：简单响应、工具调用、多工具、错误事件、stream=False
  - 状态：✅ 已完成

- [x] **Task 4.2**: LLMClient 集成测试（`test_claude_impl.py`）
  - [x] 适配 3 个 send_message 集成测试到流式接口
  - 状态：✅ 已完成

- [x] **Task 4.3**: E2E 测试（`test_e2e.py`）
  - [x] 适配 4 个 e2e 测试到流式接口
  - 状态：✅ 已完成

---

### 实施进度总结
- ✅ **阶段一：数据模型与 LLMClient** — 100% 完成 (3/3 任务)
- ✅ **阶段二：Agent 流式循环** — 100% 完成 (2/2 任务)
- ✅ **阶段三：REPL 流式渲染** — 100% 完成 (2/2 任务)
- ✅ **阶段四：测试验证** — 100% 完成 (3/3 任务)

**测试结果：164/165 通过，1 个 pre-existing 失败（版本号断言），2 个跳过（需 API key）**

## 5. 测试验证

### 单元测试
- [x] `StreamEvent` 组装：text_delta、tool_use 系列事件、response_done
- [x] 非流式包装：完整响应正确转为事件序列
- [x] 错误场景：error 事件正确生成和传递
- 执行结果：12/12 通过

### 集成测试
- [x] LLMClient 非流式路径：成功、工具调用、API 错误
- [x] 完整 e2e 流程：inspect → patch → run → save
- [x] 工具调用流式事件序列正确性
- [x] 自我演化流程（创建新工具并使用）
- 执行结果：164/165 通过（1 个 pre-existing 版本号断言失败）
