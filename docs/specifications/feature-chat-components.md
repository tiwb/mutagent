# 对话前端组件化：ThinkingBlock + ToolCard → mutagent.* 控件

**状态**：✅ 已完成
**日期**：2026-06-17
**类型**：功能设计

## 需求

1. 把 `_messages.py` 中 `ThinkingBlockView` 和 `ToolCallCard` 的 `html.div` / `html.pre` + inline style 替换为 `mutagent.*` 命名空间的前端 React 组件
2. ToolCall 的参数面板做成前端控件，以 Python 调用风格展示（`tool_name(key=value)`），自动判断一行/折行
3. 保持现有 VirtualList item 粒度、ChatItem 类型体系和流式更新逻辑不变（此 spec 不改变数据模型）
4. `_blocks.py`（Paragraph/Heading/Quote/CodeBlock）和 `_toolbar.py`（StatusDetail）后续 spec 再迭代

## 当前状态

### 已有前端组件

| 组件 | 说明 |
|------|------|
| `mutagent.ChatInput` | 聊天输入框（antd Input.TextArea 封装） |

### 本 spec 范围 — 仍用 html.* 的控件

| 所在文件 | 控件 | html.* 方式 |
|----------|------|------------|
| `_messages.py` | UserMessage | `html.div` 气泡壳（bubble_shell + bubble_style）+ meta 行 |
| `_messages.py` | AssistantMessage | `html.div` 气泡壳 + meta 行 + BlockRenderer（内部 html.*） |
| `_messages.py` | AssistantError | `html.div` 气泡壳 + meta 行 + `html.pre` 错误文本 |
| `_messages.py` | TurnSeparator | `html.div` 分割线 + token 信息 |
| `_messages.py` | ThinkingBlockView | `html.div` 外壳 + `html.pre` 思考文本 + `antd.Button` 展开 |
| `_messages.py` | ToolCallCard | `html.div` 卡片 + `html.pre` 参数/结果 + `antd.Button` 展开 |

### 本 spec 不处理（后续迭代）

| 所在文件 | 控件 | 说明 |
|----------|------|------|
| `_blocks.py` | Paragraph / Heading / Quote / CodeBlock | BlockRenderer 的 markdown 渲染，后续重新迭代 |
| `_toolbar.py` | AgentStatusBar 详情 | Popover 内的 label-value 行，后续 spec |
| `_messages.py` | UserMessage / AssistantMessage / AssistantError / TurnSeparator | 气泡壳 + meta + 分割线，暂保持 html.div |

### 当前 ChatItem 类型（不变）

```python
UserTextItem          # 用户消息
AssistantTextItem     # 助手回复（含 BlockRenderer 的 markdown 解析）
AssistantErrorItem    # 运行时错误
ThinkingBlockItem     # 思考过程（可折叠）
ToolCallItem          # 工具调用（pending/success/error）
TurnSeparatorItem     # turn 分隔
```

## 关键参考

- `mutagent/frontend/src/index.tsx` — mutagent 前端入口，注册组件到 `mutagent` 命名空间
- `mutagent/frontend/src/components/ChatInput.tsx` — 已有 mutagent 前端组件参考
- `mutagent/src/mutagent/webui/_messages.py` — ThinkingBlockView / ToolCallCard 渲染
- `mutagent/src/mutagent/webui/_conversation.py` — rebuild_items_from_messages、handle_agent_event
- `mutagent/src/mutagent/core/messages.py` — ToolUseBlock.input 类型为 `JsonObject`（`dict[str, JsonValue]`）
- `mutgui/frontend/src/integrations/antd.ts` — antd 全量注册参考
- `mutgui/frontend/src/integrations/html.ts` — html 命名空间注册参考

## 设计方案

### 一、新增前端 React 组件

在 `mutagent/frontend/src/components/` 下新建文件，在 `index.tsx` 中注册：

| 组件 | 用途 | 替换的 html.* | Python 端传参 |
|------|------|--------------|-------------|
| `mutagent.UserMessage` | 用户消息（气泡壳 + meta + 文本） | `UserMessage.render` | `{role: str, timestamp: float, text: str}` |
| `mutagent.AssistantMessage` | 助手消息（气泡壳 + meta + BlockRenderer） | `AssistantMessage.render` 外壳 | `{role: str, model: str, timestamp: float}` + children（BlockRenderer） |
| `mutagent.AssistantError` | 错误消息（气泡壳 + meta + 错误文本） | `AssistantError.render` | `{role: str, timestamp: float, error: str}` |
| `mutagent.TurnSeparator` | turn 分割线 | `TurnSeparator.render` | `{detail: str}` |
| `mutagent.ThinkingBlock` | 思考过程卡片 | `ThinkingBlockView.render` | `{thinking: str, expanded: bool}` |
| `mutagent.ToolCallCard` | 工具调用卡片（容器） | `ToolCallCard.render` 外壳 | `{name, status, expanded}` + children |
| `mutagent.ToolParameter` | 工具参数面板 | ToolCallCard 内 `html.pre` | `{name: str, input: JsonObject}` |
| `mutagent.ToolResult` | 工具结果面板 | ToolCallCard 内 `html.pre` | `{result: str, isError: bool}` |

共 8 个组件。其中 `AssistantMessage` 的 children 包含 BlockRenderer（内部 markdown 渲染暂维持 html.*，后续 spec 迭代）。

### 二、ThinkingBlock 组件

```
┌─ 思考过程 ──── [展开] ─┐
│  (展开后显示思考文本)    │
└────────────────────────┘
```

Props：`{thinking: string, expanded: boolean}`。默认收起，点击切换。展开/收起沿用当前 `Callback` 模式（通过 mutgui 事件回传）。

### 三、ToolCallCard + ToolParameter + ToolResult

ToolCall 拆为三层：卡片外壳 + 参数面板 + 结果面板。

```
┌─ ToolCallCard (border=statusColor) ─────────────────┐
│  name                                   statusText   │
│  ┌─ ToolParameter ──────────────────────────────┐   │
│  │  get_weather(                                  │   │
│  │      city="北京",                              │   │
│  │      date="2026-06-17",                        │   │
│  │  )                                             │   │
│  └───────────────────────────────────────────────┘   │
│  ┌─ ToolResult ──────────────────────────────────┐   │
│  │  北京 2026-06-17: 晴，25°C ~ 35°C              │   │
│  └───────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

#### ToolCallCard

Props：`{name: string, status: "pending"|"success"|"error"|"cancelled", expanded: boolean}`。只做容器：状态色左边框、名称、状态文字、展开/收起按钮。参数和结果作为 children 传入。

#### ToolParameter

Props：`{name: string, input: JsonObject}`。`input` 是结构化的 `dict[str, JsonValue]`（`ToolUseBlock.input` 原样），不在 Python 端做 `str()` 拍扁。前端渲染为 Python 调用风格：

- 单行：`tool_name(key1=value1, key2=value2)`
- 多行（嵌套对象 / 多参数 / 总长度超阈值自动折行）：

```python
tool_name(
    key1=value1,
    key2=value2,
    nested={
        "sub": "value",
    },
)
```

"一行/多行"判断逻辑在前端：参数个数 > 3 或任意参数值含嵌套结构或总长度 > 80 字符时折行。这是纯渲染层逻辑，不影响数据模型。

#### ToolResult

Props：`{result: string, isError: boolean}`。首版简单实现——等宽字体 pre-wrap 显示文本，isError 时加红色标记。后续扩展支持流式追加和不同工具的自定义结果类型。

### 四、UserMessage / AssistantMessage / AssistantError / TurnSeparator

这四个控件目前用 `html.div` + inline style 拼外壳。替换为 `mutagent.*` 组件后，Python 端只传内容数据，样式全部由前端控制：

#### UserMessage

Props：`{role: string, timestamp: float, text: string}`。前端渲染为左对齐气泡（accent 背景 + meta 行 + 文本）。meta 行由前端用 `role` + `timestamp` 自行排版（如 `你 · 18:13`），不再由 Python 端拼接字符串。

#### AssistantMessage

Props：`{role: string, model: string, timestamp: float}` + children。前端渲染为左对齐气泡（surface 背景 + meta 行），children 传入 `BlockRenderer` View。meta 行由前端拼 `role · model · timestamp`。

#### AssistantError

Props：`{role: string, timestamp: float, error: string}`。前端渲染为左对齐气泡（红色边框 + meta 行 + 等宽错误文本）。

#### TurnSeparator

Props：`{detail: string}`。渲染为分割线 + 居中 token/耗时信息。

### 五、Python 端改动

| 文件 | 改动 |
|------|------|
| `_messages.py` | 全部 6 个 ChatItemView 的 render 改为 `mutagent.*` 组件；删除 `_bubble_shell`、`_bubble_style`、`_meta_style`、`_role_meta`、`_format_clock` 等辅助函数；`ToolCallItem.input_text` 类型从 `str` 改为 `JsonObject` |
| `_conversation.py` | `rebuild_items_from_messages` 和 `handle_agent_event` 中 `str(tool_call.input)` → 直接存 `tool_call.input`（JsonObject） |

### 五、消息气泡保留，只做左对齐

- `_bubble_shell` 已经改为统一 `flex-start`（用户消息左对齐），保留气泡壳和 meta 行
- UserMessage / AssistantMessage / AssistantError / TurnSeparator 的 html.div 保持现状，本 spec 不处理

### 七、实施步骤清单

- [x] 新建 `MessageShell.tsx` + `.css`：UserMessage / AssistantMessage / AssistantError / TurnSeparator 四个组件
- [x] 新建 `ThinkingBlock.tsx` + `.css`：思考过程可折叠卡片
- [x] 新建 `ToolCallCard.tsx` + `ToolResult.tsx` + `.css`：ToolCallCard 自包含（参数渲染 + 分隔线 + ToolResult），ToolResult 独立组件
- [x] `_messages.py`：6 个 ChatItemView render 改用 `mutagent.*` 组件；删除 9 个辅助函数/导入；`input_text` → `input_kwargs`
- [x] `_conversation.py`：`str(tool_call.input)` → `tool_call.input`（3 处）；`_appended_thinking_ids` → `appended_thinking_ids`
- [x] `index.tsx`：注册 7 个新组件到 `mutagent` 命名空间
- [x] 前端构建通过、pyright 0 errors

### 八、不做的事

- **不改变 VirtualList item 粒度** — 此 spec 只替换渲染控件，不改 item 拆分方式
- **不升级 markdown 解析** — 仍由 BlockRenderer 用现有正则分段（后续 spec）
- **不动 `_blocks.py`** — Paragraph / Heading / Quote / CodeBlock 后续重新迭代
- **不动 `_toolbar.py`** — StatusDetail 后续 spec 处理
- **不动设置页面** — `_settings_*.py` 暂不处理
- **不组件化 BlockRenderer 内部** — AssistantMessage 内的 Paragraph/Heading/Quote/CodeBlock 保持 html.*，后续 `_blocks.py` spec 处理
