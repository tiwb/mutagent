# mutagent 内置 WebUI 设计规范

**状态**：✅ 已完成
**日期**：2026-04-27（初版）→ 2026-05-06（合并终版）
**类型**：功能设计

## 需求

1. mutagent 当前是"中间库"形态，无独立产品体验。需要让 `pip install mutagent` 后能直接得到完整可用的 agent 产品，对标 Claude Code / pi 的"单包即产品"形态。
2. mutagent 启动方式新增 `mutagent webui` 子命令，本地起 web 服务，浏览器即用。
3. UI 层不依赖任何 web 框架第三方包（不用 starlette/uvicorn/fastapi），完全基于 mutio + mutgui。
4. WebUI 所有代码（控件、Action、Server、CLI）统一放在 `mutagent.webui` 下，直接基于 web 技术栈（mutgui antd 组件）实现。不追求跨渲染后端抽象——经验表明 web 与 TUI 的交互模型（事件绑定、布局、滚动、组件注册）差异过大，强行共享 Declaration 反而两面不讨好。
5. 功能对齐 mutbot 当前 agent 面板（消息列表、流式渲染、工具调用展示、模型切换、状态栏、Markdown、围栏块），作为"未来替换 mutbot 前端"的可行性验证。
6. 不影响 mutagent 现有默认 `App.run()` 的流式 stdout/stdin 体验（headless 模式继续可用）。
7. 视觉对齐 mutbot AgentPanel 的 80%+（暗色主题、panel 色背景、无外框、滚动条贴边）。
8. 顶部工具栏改为 ActionToolbar 驱动，支持主菜单（LLM 设置、模型刷新）；输入区按钮区作为扩展挂载点，通过 Action 扩展。
9. 提供基于 mutgui 的 LLM 设置界面（右侧抽屉），支持 provider 新增/编辑/删除、模型发现、默认模型选择、运行时生效。

## 关键参考

### mutagent 现有 UI 抽象

- `src/mutagent/main.py` — `App` Declaration，`load_config / setup_agent / run / run_webui` 入口
- `src/mutagent/agent.py` — `Agent` Declaration，`submit / cancel / subscribe / select_model / list_models / is_busy` 运行时接口
- `src/mutagent/builtins/main_impl.py` — `run_webui` 实现，WebUI server 启动逻辑
- `src/mutagent/builtins/agent_impl.py` — `Agent.submit / cancel / subscribe / select_model / list_models` 实现
- `src/mutagent/builtins/provider_impl.py` — provider-based `default_model` / `providers` 解析
- `src/mutagent/messages.py` — `StreamEvent` / `Content` / `Message` 数据结构

### mutagent WebUI 源码

- `src/mutagent/webui/conversation.py` — `Conversation` 根 View Declaration
- `src/mutagent/webui/_conversation_impl.py` — Conversation @impl + Agent↔View 适配器
- `src/mutagent/webui/chat_input.py` — `ChatInput` Declaration
- `src/mutagent/webui/_chat_input_impl.py` — ChatInput @impl（`mutagent.ChatInput` 前端组件协议）
- `src/mutagent/webui/messages.py` — `MessageList` + 消息 item 类型
- `src/mutagent/webui/_messages_impl.py` — MessageList + 各 item 控件 @impl
- `src/mutagent/webui/toolbar.py` — `AgentStatusBar` Declaration
- `src/mutagent/webui/_toolbar_impl.py` — AgentStatusBar @impl
- `src/mutagent/webui/tool_call.py` — `ToolCallCard` Declaration
- `src/mutagent/webui/_tool_call_impl.py` — ToolCallCard @impl
- `src/mutagent/webui/blocks.py` — `BlockRenderer` / `ThinkingBlock` Declaration
- `src/mutagent/webui/_blocks_impl.py` — BlockRenderer / ThinkingBlock @impl
- `src/mutagent/webui/settings.py` — `LLMSettingsPanel` Declaration
- `src/mutagent/webui/_settings_impl.py` — LLMSettingsPanel @impl（provider 列表/编辑/模型发现/保存）
- `src/mutagent/webui/_actions_impl.py` — Action 定义（ModelSelector / Status / Send / Cancel / MainMenu 等）
- `src/mutagent/webui/server.py` — `WebUIServer` Declaration
- `src/mutagent/webui/_server_impl.py` — WebUIServer @impl + import-map 协议 + WebSocketChannel + 根 HTML
- `src/mutagent/webui/cli.py` — `mutagent webui` 子命令解析与路由
- `frontend/src/components/ChatInput.tsx` — `mutagent.ChatInput` 前端 React 组件
- `frontend/src/index.tsx` — `@mutagent/ui` 运行时注册入口

### mutgui 框架

- `mutgui/src/mutgui/view.py` — `View` Declaration
- `mutgui/src/mutgui/action.py` — `Action` / `ActionMenu` / `ActionToolbar` / `ActionContext`
- `mutgui/src/mutgui/modules.py` — `ModuleRegistry`（import-map 聚合）
- `mutgui/src/mutgui/virtual_list.py` — `VirtualList` View
- `mutgui/docs/design/framework-capabilities.md` — 渲染模型 / 事件协议 / IME 处理
- `mutgui/docs/specifications/feature-virtual-list-streaming.md` — VirtualList 流式能力

### mutbot 参考（对齐目标）

- `mutbot/frontend/src/panels/AgentPanel.tsx` — 主面板
- `mutbot/frontend/src/components/ChatInput.tsx` — 输入框（统一容器 + split button 视觉模型）
- `mutbot/frontend/src/components/ToolCallCard.tsx` — 工具调用卡片

## 设计方案

### 整体分层

```
mutagent.main.App
    └─ App.run() / App.run_webui()        双入口
           ↓
mutagent.webui  ◀═══════════════════════ 全套 WebUI（控件 + Action + Server + CLI）
   ├─ WebUIServer(mutio.Server)           只在 `mutagent webui` 子命令下使用
   ├─ webui 子命令解析
   ├─ 根 HTML 模板 + import-map 协议
   ├─ 控件：Conversation / MessageList / ChatInput / ToolCallCard / ...
   ├─ Action 定义（ModelSelector / Send / Cancel / MainMenu / Settings）
   └─ LLM 设置面板
           ↓
mutgui                                    通用 UI 框架（已有）
   ├─ View / ViewPort / Channel / VirtualList
   ├─ Action / ActionToolbar / ActionContext
   └─ antd 组件白名单
           ↓
mutio.net                                 web 传输层（已有）
   ├─ Server / View / WebSocketView
   └─ ASGI / HTTP / WebSocket
```

依赖方向单向：`App → mutagent.webui → mutgui → mutio → mutobj`。

### 包结构

所有 WebUI 代码统一放在 `mutagent/webui/`，不再拆 `ui/` + `webui/` 两层：

```
mutagent/src/mutagent/
    webui/                       # WebUI 全部代码（控件 + Server + Action + CLI）
        __init__.py              # 公开 API 集中导出
        cli.py                   # `mutagent webui` 子命令解析与路由
        server.py                # WebUIServer(mutio.Server) 声明
        _server_impl.py          # WebUIServer @impl + WebSocketChannel + 根 HTML + import-map 协议
        conversation.py          # Conversation 根 View + adapter（Agent ↔ View 胶水）
        _conversation_impl.py
        toolbar.py               # AgentStatusBar
        _toolbar_impl.py
        messages.py              # MessageList + UserTextItem / AssistantTextItem / AssistantErrorItem / TurnSeparatorItem
        _messages_impl.py
        tool_call.py             # ToolCallCard
        _tool_call_impl.py
        blocks.py                # BlockRenderer + ThinkingBlock
        _blocks_impl.py
        chat_input.py            # ChatInput Declaration
        _chat_input_impl.py      # ChatInput @impl（前端组件协议 + ActionToolbar）
        settings.py              # LLMSettingsPanel Declaration
        _settings_impl.py        # LLMSettingsPanel @impl（provider 管理 / 模型发现 / 保存）
        _actions_impl.py         # Action 定义（ModelSelector / Send / Cancel / MainMenu / Settings）
    static/                      # 前端构建产物（@mutagent/ui runtime lib）
        manifest.json
        libs/mutagent-ui.js
        libs/mutagent-ui.css
frontend/                        # 前端源码（复用 mutgui build-preset）
    build.mjs
    mutagent.build.mjs
    src/
        index.tsx                # @mutagent/ui 运行时注册入口
        components/
            ChatInput.tsx        # mutagent.ChatInput React 组件
            ChatInput.css
```

### 入口设计

```bash
mutagent                  # 默认 stdout REPL（保留）
mutagent --headless       # 显式声明 headless（与默认行为等价）
mutagent webui            # 子命令：启 web 服务，自动开浏览器
mutagent webui --no-browser --port 8080 --host 127.0.0.1
```

- `App.run()` 保留，行为零变化
- `App.run_webui(host, port, open_browser)` 新增，`main()` 在 subparser 路由到 `webui` 后调用
- argparse subparsers 体系：主命令兼容（无子命令 = headless），`webui` 子命令独立持有自身参数集合
- `--headless` 与 `webui` 子命令互斥，同时出现报错退出

### Agent 运行时接口

为支持 WebUI 的非阻塞交互，`Agent` 新增以下公开 API（`agent.py` Declaration + `agent_impl.py` @impl）：

| 方法 | 说明 |
|---|---|
| `async submit(text)` | 提交用户文本，后台 asyncio.Task 驱动 run()，期间 emit StreamEvent |
| `subscribe(callback) → Disposable` | 订阅 StreamEvent 流，返回可释放订阅 |
| `cancel() → bool` | 取消当前运行中的 turn |
| `select_model(name)` | 切换下次 turn 使用的 LLM model（运行时生效） |
| `list_models() → list[dict]` | 列出所有已配置模型（供 UI render） |
| `is_busy() → bool` | 当前是否有运行中的 turn |

`submit()` 内部用 `asyncio.create_task` 驱动 agent.run() 的 async generator，逐个 yield StreamEvent 后 `_emit_event` 分发给所有订阅回调。cancel 时 `task.cancel()` + 补发 `turn_done` 事件。异常时补发 `error` + `turn_done` 事件。

### 控件清单

| widget | 归属文件 | 来源数据 | 说明 |
|---|---|---|---|
| **Conversation** | conversation.py | Agent | 根 View；组合 toolbar + MessageList + ChatInput + settings drawer；订阅 Agent 的 StreamEvent |
| **MessageList** | messages.py | `list[ChatItem]` | VirtualList 适配；流式追加；stick-to-bottom |
| **UserTextItem** | messages.py | `user.text` | 角色标签（meta 档）+ 满宽文本气泡，**无头像** |
| **AssistantTextItem** | messages.py | `assistant.text` | 角色·模型·耗时（meta 档）+ Markdown 满宽气泡，**无头像** |
| **AssistantErrorItem** | messages.py | `assistant.error` | 错误样式包裹的文本气泡 |
| **TurnSeparatorItem** | messages.py | `turn_done` | 回合分隔线 + 耗时与 token 统计 |
| **ToolCallCard** | tool_call.py | `assistant.tool_group` | 工具名 + 参数 + 结果 + 状态（pending/success/error/cancelled）+ 折叠 |
| **BlockRenderer** | blocks.py | 围栏块 type+content | thinking / code / status / fallback CodeBlock |
| **ThinkingBlock** | blocks.py | thinking 围栏 | 可折叠思考块 |
| **ChatInput** | chat_input.py | 用户输入 | 前端 `mutagent.ChatInput` 组件（React）+ 后端 ActionToolbar |
| **AgentStatusBar** | toolbar.py | Agent 状态 | idle/thinking/tool_calling + token 用量显示 + context/cache 统计 |
| **LLMSettingsPanel** | settings.py | App + Agent config | Provider 管理 + 模型发现 + 默认模型选择（右侧抽屉） |

**Declaration 公开接口约束（硬性）**：控件 Declaration 上暴露的属性、事件、方法必须用 **agent 业务语义** 命名。✅ 如 `messages: list[ChatItem]`、`disabled: bool`、`on_send(text: str)`；❌ 禁止 `dom_id`、`class_name`、antd component 名作为属性。`render()` 内部自由使用 antd 组件。

**首版不包含**：会话切换、workspace、auth、自定义编辑器、富输入（@ 文件/图片拖拽）、多 Agent/Sub-Agent 展开、亮色主题/主题切换、跨 item 文本选择复制（受 VirtualList 能力限制，用 AssistantMessage「复制」按钮兜底）、持久化历史。

### Agent ↔ View 适配

`Conversation.__init__` 中 `agent.subscribe(callback)` 订阅 StreamEvent，事件处理如下：

| Agent 事件 | widget 动作 |
|---|---|
| `response_start` | 追加 AssistantTextItem，status="thinking" |
| `text_delta` | 当前 AssistantTextItem 追文本 → invalidate_item |
| `tool_exec_start` | 追加 ToolCallItem（pending） |
| `tool_exec_end` | 更新 ToolCallItem 状态/结果/耗时 |
| `response_done` | 更新 AssistantTextItem 的 model/duration/tokens |
| `error` | 追加 AssistantErrorItem |
| `turn_done` | 追加 TurnSeparatorItem；status="idle"；is_busy=False；cancel 时补 `[interrupted]` |

ChatInput 的 `on_send` 触发：`agent.submit(text)` → UI 进入 busy 状态（ChatInput.disabled=True，StatusBar 显示 thinking）。

### 动作系统

顶部工具栏和输入区按钮区统一由 mutgui `ActionToolbar` 驱动，不再手工拼装普通 View。

#### 顶部工具栏

- Category：`mutagent.conversation.toolbar`
- 默认动作：
  - `ModelSelectorAction` — variant="dropdown"，动态 label 显示当前模型名，菜单列出所有模型
  - `AgentStatusAction` — variant="widget"，渲染 AgentStatusBar
  - `MainMenuAction` — variant="dropdown"，☰ 三道杠菜单

#### 主菜单

`MainMenuAction` 菜单项：

- `OpenLLMSettingsAction` — "LLM API Settings"，打开右侧设置抽屉
- `RefreshModelsAction` — "Refresh Models"，从配置重新计算模型列表

#### 输入区工具栏

- Category：`mutagent.chat_input.toolbar`
- 默认动作：
  - `SendMessageAction` — variant="split"，主按钮发送 + ▴ 上弹菜单切换 send mode
  - `CancelMessageAction` — 仅 busy 时可见，Stop 按钮

后续如果要加 `@`、`/` 命令、图片插入等动作，只需往同一 category 投放 Action。

### 设置界面

**形态**：右侧 `antd.Drawer`，与主菜单入口天然配套。修改 provider 后可立即回到当前会话，不打断消息列表上下文。

**内容**：

1. **Provider 列表页**（默认首页）：
   - 已有 provider 卡片列表（名称 + 类型 + 模型摘要），点击进入编辑
   - "Add Anthropic" / "Add OpenAI" 按钮新增 provider
   - 删除按钮（带确认）

2. **Provider 编辑页**：
   - Provider Name
   - Provider Type（Anthropic / OpenAI / OpenAI-compatible）
   - API Base URL
   - API Token
   - Models（可发现 + 可手输，Discover Models 按钮 inline 在字段旁）
   - Save / Cancel 按钮

3. **默认模型选择**：在首页顶部，基于所有 provider 的模型汇总做单行 select

**Provider 配置格式**（沿用 mutagent 现有 `providers` + `default_model`）：

```json
{
  "default_model": "claude-sonnet-4",
  "providers": {
    "anthropic": {
      "provider": "AnthropicProvider",
      "base_url": "https://api.anthropic.com",
      "auth_token": "$ANTHROPIC_API_KEY",
      "models": ["claude-sonnet-4", "claude-haiku-4.5"]
    }
  }
}
```

**模型发现**：OpenAI / OpenAI-compatible provider 调用 `{base_url}/models`（fallback `/v1/models`）发现模型，结果与手工输入合并。

**运行时生效**：保存时 → 更新 `Config` 内存 → 写回 `config.json` → 重新计算模型列表 → 切换到 `default_model` 或第一个可用模型 → 刷新 toolbar / 输入区 / 设置页。

### 视觉规范

#### 颜色与 token

全部使用 mutgui token，不硬编码颜色：

| mutbot CSS 变量 | mutagent 使用的 mutgui token |
|---|---|
| `--bg` / `--bg-panel` | `var(--mutgui-bg)` |
| `--bg-input` | `var(--mutgui-surface)` |
| `--accent` | `var(--mutgui-accent)` |
| `--text` | `var(--mutgui-text)` |
| `--text-dim` | `var(--mutgui-text-dim)` |
| `--border` | `var(--mutgui-border)` |

- 宿主 HTML 不写 `body { background }`，让 mutgui dark plugin 的 `--mutgui-bg = #1F1F1F` 接管
- 暗色主题首版唯一，亮色/主题切换为后续增强项

#### 字号两档

| 档位 | 字号 | token | 用途 |
|---|---|---|---|
| **正文** | 13px | `--mutagent-font-size-base` | 消息正文、Markdown、代码块、输入框文字 |
| **辅助** | 12px | `--mutagent-font-size-meta` | 消息元信息、TurnSeparator 统计、状态栏 |

正文内字号严格一致，不存在 h1>h2>h3 的传统排版阶梯。Markdown 标题层次通过 **粗细（bold） + `#` 前缀 + 间距** 表达。辅助档统一比正文小一号，用 `--mutgui-text-dim` 色。

token 挂在 Conversation 根容器 style 上，不污染 mutgui 全局。

#### 布局

- 根 div：`height: 100vh` + flex 列布局，`gap: 0`，无 padding / background
- messages-shell：无 padding / border / borderRadius / background，让 VirtualList 滚动条自然贴边
- toolbar-shell：`padding: 8px 12px`
- 消息气泡保留左右对齐 + 圆角（baseline 来自 mutgui virtual_list_chat demo），无头像，顶部以 meta 档显示角色/模型/时间

#### Markdown 渲染约束

- `h1~h6` 全部强制 `font-size: inherit; font-weight: bold`
- `code` inline 与 `pre` 块字号 inherit，仅切字体族
- `blockquote` 用左侧色条 + 缩进，不变字号
- `table` 表头 bold + 浅背景，单元格字号 inherit

#### 输入区视觉

输入区采用 mutbot 风格的"统一容器"视觉模型：

```
┌──────────────────────────────────────────┐
│  textarea（透明背景，borderless，自适应高度）│
├──────────────────────────────────────────┤
│                   [Send ▴]               │
└──────────────────────────────────────────┘
   ↑ 整个容器背景 var(--mutgui-surface)，圆角 8px
     focus-within 时 border: var(--mutgui-accent)
```

- textarea `antd.Input.TextArea` variant="borderless" + `autoSize={{ minRows: 1, maxRows: 8 }}`
- 工具栏 split button：主发送按钮 + ▴ 上弹菜单切换 Enter / Ctrl+Enter 发送模式
- CSS 注入到宿主 HTML `<style>`（`.mutagent-chat-input-shell` / `.mutagent-chat-input-toolbar` 等 class）

### ChatInput 前端组件

首版前端化的组件为 `mutagent.ChatInput`（React 组件，注册到 mutgui registry 的 `mutagent` 命名空间）。它承担：

- textarea 本地键盘处理 + IME 组合输入
- 自适应高度
- 发送模式（Enter 发送 / Ctrl+Enter 发送）
- Enter 提交时 `preventDefault()` + 触发发送按钮 click（避免 round-trip 延迟 + textarea 残留文本）
- split send 按钮 + 上弹菜单的视觉渲染

后端 `ChatInput.render()` 收敛为单组件协议：

```python
{
    "$component": "mutagent.ChatInput",
    "value": self.text,
    "sendMode": self.send_mode,
    "disabled": self.disabled,
    "placeholder": ...,
    "$children": [self.toolbar],  # ActionToolbar
    "onChange": Bind(self, "text", "$0"),
    "onSubmit": Callback(_submit, view="@view"),
}
```

### 前端构建与运行时

`mutagent/frontend/` 复用 mutgui 的 `@mutgui/core/build-preset`：

```text
frontend/
  build.mjs
  mutagent.build.mjs
  src/
    index.tsx              # registerComponents({ __name__: 'mutagent', ChatInput })
    components/
      ChatInput.tsx
      ChatInput.css
```

构建产物输出到 `src/mutagent/static/`：
- `manifest.json` — exports `@mutagent/ui`
- `libs/mutagent-ui.js` / `libs/mutagent-ui.css`

WebUI HTML 使用 import map + `boot.js` 协议：
- `<script type="importmap">` — 由 `ModuleRegistry.add_from_package("mutgui")` + `add_from_package("mutagent")` 聚合
- `<script src="/static/modules/mutgui/boot.js">` — mutgui 统一启动器
- 运行时装配消息：`runtime.css` → `runtime.import` → `runtime.install` → `runtime.mount`

### MessageList 与 VirtualList

直接基于 mutgui `VirtualList` 实现，应对长会话数千轮消息场景。能力依赖全部已落地：

| 能力点 | mutgui VirtualList 状态 |
|---|---|
| 大量 item 虚拟滚动 | ✅ |
| 可变高度 item | ✅（`feature-virtual-list-streaming.md`） |
| 追加新 item 到尾部 | ✅ |
| 单 item 内容变化重测高度 | ✅（ResizeObserver） |
| stick-to-bottom 跟随策略 | ✅（FOLLOWING/DETACHED 状态机） |
| 跨 item 文本选择 | ❌ 已知限制，用 AssistantMessage「复制」按钮兜底 |

MessageList 的 item 分发：按 `ChatItem` 类型（kind）派发到 UserTextItem / AssistantTextItem / ToolCallItem / TurnSeparatorItem / AssistantErrorItem 等子 View。

### 流式刷新策略

复用 mutgui 现有 `invalidate()` + `asyncio.call_soon()` 去重机制，同一 tick 内的多次文本增量只触发一次 push。

### 错误处理

- 启 webui 时若未装 mutgui，捕获 ImportError 提示 `pip install mutagent[webui]`
- 端口冲突时（指定了非 0 端口）报错退出
- WebSocket 断开时 ViewPort detach，不影响 Agent 运行（重连自动重放最新 view）
- Agent 抛异常时通过 AssistantErrorItem 展示，不让 web 服务挂掉

### 依赖声明

```toml
[project.optional-dependencies]
webui = ["mutgui~=0.1.0"]
```

## 消费者场景

| 消费者 | 场景 | 依赖的输出 | 验收标准 |
|---|---|---|---|
| 终端用户 | `mutagent webui` 启动后浏览器聊天 | webui 完整可用 | 能聊天、看流式输出、看工具调用、切模型、取消 |
| 终端用户 | 默认 `mutagent`（无子命令） | headless 模式不被破坏 | 现有 stdout REPL 行为零变化 |
| WebUI 用户 | 在主菜单中修改 LLM provider 与默认模型 | 主菜单、设置抽屉、配置落盘、运行时刷新 | 保存后模型选择器立即反映新配置 |
| 第三方扩展者 | 基于 mutagent 写自己的 agent 产品 | App + UI 控件库可继承/覆盖 | `App.run_webui` 可被 @impl 覆盖；UI 控件可被子类化 |
| 扩展者 | 往输入区挂接 @ 命令、图片等动作 | `mutagent.chat_input.toolbar` category | 不改 ChatInput 主体也能新增输入动作 |

## 验收标准

- `pip install mutagent[webui]` 后 `mutagent webui` 可启动
- 浏览器进入后能完成一轮：输入 → 流式回答 → 看到工具调用 → 输出完成 → 状态回 idle
- 至少一种工具调用能正确展示 input/result
- thinking 围栏块能折叠展开
- Markdown 渲染正常（含代码高亮）
- 切换模型后下一轮生效
- 流式输出过程可点取消
- 关闭浏览器再开新窗口能看到完整历史
- 消息列表基于 VirtualList，流式输出时默认 stick-to-bottom，用户上滚后停止跟随
- UI 为暗色主题，背景与 mutgui dock demo 一致（VS Code 深灰 `#1F1F1F`）
- 消息区无头像；用户右对齐 + AI 左对齐气泡保留，顶部以 meta 档字号显示角色/模型/时间
- Markdown 渲染中 h1~h6 字号与正文一致，靠 bold + `#` 前缀区分层次
- 输入区与 mutbot AgentPanel 视觉对齐：textarea 与 Send 按钮在同一个圆角矩形内，focused 时整体边框变 accent 蓝
- 输入框自适应高度（1 行起、最多 8 行），Enter 立即提交（无 round-trip 延迟），Shift+Enter 换行，IME 候选词期间 Enter 不误提交
- Send 按钮右侧 ▴ 上弹菜单切换 enter / ctrl-enter 发送模式
- 主菜单（☰）包含 LLM API Settings 和 Refresh Models 入口
- LLM 设置抽屉可新增/编辑/删除 provider，Discover Models 可用，保存后模型选择器立即刷新
- 页面使用 import-map + boot.js 协议加载，不依赖旧的 IIFE 三件套
- `mutagent.ChatInput` 作为前端组件正常渲染
- 默认 `mutagent`（无子命令）行为零变化
- mutagent 测试集全部通过

## 实施步骤清单

- [x] 补齐 `App` / CLI / `Agent` 的 WebUI 运行时接口（`submit/cancel/subscribe/select_model/list_models/is_busy`），确保默认 headless 行为不变
- [x] 实现 `mutagent.webui` 控件与 Conversation 事件适配，完成消息列表、工具卡片、输入框、状态栏和模型选择器
- [x] 实现 `mutagent.webui` server、根 HTML（import-map + boot.js 协议）、WebSocket Channel 与浏览器启动逻辑
- [x] 视觉对齐：去硬编码背景色、收紧根布局（去外框/去 padding）、统一输入区容器（borderless + split button + 前端键盘）
- [x] 新增 `mutagent/frontend` 构建目录，复用 mutgui build-preset 产出 `@mutagent/ui`
- [x] 前端化 `mutagent.ChatInput`，后端 `ChatInput.render()` 改为单组件协议
- [x] 重构 `Conversation` 顶部为 `ActionToolbar`（ModelSelector + StatusBar + MainMenu）
- [x] 重构 `ChatInput` 为带 `toolbar` 槽位的复合控件，接入 `mutagent.chat_input.toolbar` 动作
- [x] 实现主菜单（☰）与 LLM 设置抽屉（provider 管理 + 模型发现 + 默认模型选择）
- [x] 实现 provider 配置读写、模型发现与运行时刷新（保存后无需重启 WebUI）
- [x] 更新打包入口与可选依赖（`webui` extra），删除旧 `web` extra
- [x] 补充并更新测试，覆盖 CLI 路由、运行时接口、WebUI 核心适配流程与设置面板

## 不在首版范围

- 多客户端协同
- 亮色主题 / 主题切换
- 跨 item 文本选择复制（用「复制」按钮兜底）
- 文件 / 图片输入
- 持久化历史
- 多 Agent / Sub-Agent 展开
- 会话切换 / workspace / auth
- 发送模式 `localStorage` 持久化
- ToolCallCard 风格平直化对齐 mutbot
- ModelSelector 改为 mutbot 风格轻量 dropdown
- AgentStatusBar 移到 MessageList 与 ChatInput 之间
- 字号可调、字体族切换
