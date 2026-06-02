# WebUI Session 恢复

**状态**:✅ 已完成
**日期**:2026-06-02
**类型**:功能设计

## 需求

1. WebUI 聊天过程中自动保存 session(增量持久化)
2. 提供 `#/resume` 顶级路由,列出历史会话
3. 点击历史会话 → 恢复对话内容到当前窗口(替换现有对话)
4. Session 标题自动取第一条 user 消息,去换行保证单行显示,超长 CSS ellipsis 截断
5. 工具里菜单里增加Resume Session和New Session入口

## 关键参考

- `src/mutagent/core/session.py` - `AgentSession` 声明(`start_new` / `resume` / `sync`)
- `src/mutagent/core/_session_impl.py` - Session 持久化实现、JSONL codec、`_load` / `_save`
- `src/mutagent/cli/terminal.py` - CLI 参考实现(`_build_agent_session` / `_make_session_persist_callback`)
- `src/mutagent/webui/conversation.py` - `Conversation` 声明,路由权威 `current_route`
- `src/mutagent/webui/_conversation_impl.py` - Conversation 实现,路由分发 `render()` / `_apply_route()`
- `src/mutagent/webui/_session_page.py` - ResumeSessionPage、session 生命周期与 main menu Actions 实现
- `src/mutagent/webui/_toolbar_impl.py` - MainMenuAction 通过 `mutagent.main_menu` 类别自动发现菜单项
- `src/mutagent/core/messages.py` - `Message` / `TextBlock`,用于从消息提取标题
- `docs/specifications/feature-session.md` - Session 持久化总体设计

## 设计方案

### 一、自动保存(增量 sync)

在 `ConversationExt` 中新增 `session: AgentSession` 字段,`conversation_init__` 中创建:

```python
self.session = AgentSession()
self.session.start_new(
    session_dir=Path.home() / ".mutagent" / "sessions",
    cwd=str(Path.cwd()),
    model=agent.model,
)
```

在 `handle_agent_event` 的 `turn_done` 分支末尾调用 `self.session.sync(agent.context)`。

不需要在退出时额外 sync--WebUI 是长连接,`turn_done` 时已落地。会话结束后如果用户刷新页面,当前 session 自动保存过,新会话重新 `start_new`。

本次实施采用以下默认决策:每次进入 resume 页面都重新扫描 session 目录;恢复历史 session 时不弹确认框;恢复后的新消息继续追加到原 JSONL 文件。

### 二、ResumeSessionPage 组件

单文件 `src/mutagent/webui/_session_page.py`,直接继承 `View`(不采用 Declaration/Extension 分离模式)。

```
ResumeSessionPage (View)
├── 构造参数: conversation: Conversation | None
│   └── 通过 conversation 直调 navigate_to("") 离开页面
├── 扫描 ~/.mutagent/sessions/*.jsonl
├── 解析每个文件的 session header 拿到元数据
├── 标题提取: 扫描文件找到第一条 role="user" 的消息
│   └── 取 TextBlock.text,换行替换为空格,单行
└── 列表渲染: 每项显示标题、日期、模型
    └── 标题超长 CSS textOverflow: ellipsis
    └── 点击 → resume_session(conversation, session_path)
```

与 SettingsPage 同一风格:持有 `conversation` 引用而非回调注入,需要导航时直接 `self.conversation.navigate_to(route)`。

#### 标题提取逻辑

```python
def _extract_title(path: Path) -> str:
    """从 session JSONL 提取第一条 user 消息作为标题。"""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("type") != "message":
                continue
            msg = entry.get("message", {})
            if msg.get("role") != "user":
                continue
            blocks = msg.get("content", [])
            text = "".join(
                b.get("text", "") for b in blocks if b.get("type") == "text"
            )
            if text:
                # 去换行,保证单行
                return text.replace("\n", " ").replace("\r", " ").strip()
    return "(empty)"
```

注意:这里做轻量 JSON 解析(不依赖 `_session_impl._load`),避免导入循环,也避免完整加载全部消息只为拿标题。

#### Session 列表排序

按文件修改时间倒序(最新的在上)。

### 三、路由集成

`Conversation` 新增顶级路由 `"resume"`,与 `""`(对话)和 `"settings"`(设置)平级。

**`_apply_route()` 变更**:

当前只处理 settings 进/出的 panel 生命周期。resume 无子 panel,只需 show/hide:

```python
prev_in_resume = prev.startswith("resume")
new_in_resume = route.startswith("resume")

# 进入 resume:触发 refresh 扫描最新 session 列表
# 离开 resume:无特殊生命周期
```

resume 路由不支持子路由(无 `resume/<id>`),`"resume"` 即为唯一合法值。

**`render()` 变更**:

```python
if in_settings:
    children = [ext.settings_page]
elif self.current_route == "resume":
    children = [ext.resume_page]
else:
    children = [toolbar, messages, chat_input]
```

### 四、恢复流程

用户点击 ResumeSessionPage 中的某条记录 → 调用 `resume_session`：

1. `AgentSession.resume(path, agent.context)` — 加载历史消息到 context
2. 清空 `ConversationExt.items`
3. 从 `agent.context.messages` 重建 UI item 列表（`_rebuild_items_from_messages`）
4. `session` 切换到恢复的 session（后续新消息追加到同一文件）
5. `navigate_to("")` — 切回对话主页

恢复时是**替换**,不是追加。用户的未保存对话会丢失(可在恢复前弹出确认提示,v1 先不加,后续迭代)。

### 五、Toolbar 入口

`MainMenuAction.menu_actions()` 通过 `ActionRef(category="mutagent.main_menu")` 自动发现所有注册到该类别的 Action:

- `ResumeSessionAction` - 执行 `conversation.navigate_to("resume")`,仅 idle 时可用
- `NewSessionAction` - 清空消息 + 重置状态 + 创建新 `AgentSession`,仅 idle 时可用
- `OpenSettingsAction` - 同步改为 `mutagent.main_menu` 类别注册(原为动态构造)

各 Action 直接定义为 Action 子类,声明 `categories = ("mutagent.main_menu",)`,无需在 `MainMenuAction` 中手写菜单列表。

### 六、消息历史重建(context → UI items)

`context.messages` 是 `Message` 列表,需要转换成 WebUI 的 item 类型(`UserTextItem` / `AssistantTextItem` / `ToolCallItem` / `TurnSeparatorItem`)。新建辅助函数:

```python
def _rebuild_items_from_messages(messages: list[Message]) -> list[Any]:
    """从 AgentContext.messages 重建 WebUI item 列表。"""
```

重建逻辑:
- `role="user"` + `TextBlock` → `UserTextItem`
- `role="assistant"` + `TextBlock` → `AssistantTextItem`
- `role="user"` + `ToolResultBlock` → 更新对应 `ToolCallItem` 的 status/result
- 每个 assistant message 后跟 `TurnSeparatorItem`(如果后一条是 user 或列表结束)

Note: ToolUseBlock 在 `role="assistant"` 消息中,ToolResultBlock 在 `role="user"` 消息中。重建时需要配对。

## 实施步骤清单

- [x] 新增 `src/mutagent/webui/_session_page.py`,实现 `ResumeSessionPage`(View 子类)、session 生命周期函数(`_start_session` / `start_new_session` / `resume_session`)和 `ResumeSessionAction` / `NewSessionAction` 菜单 Action
- [x] 更新 `src/mutagent/webui/_conversation_impl.py` / `src/mutagent/webui/conversation.py`,接入 `AgentSession` 自动保存、`#/resume` 路由、`_rebuild_items_from_messages` 历史重建;`SettingsPage` 同步简化为 `conversation` 直传(去回调注入)
- [x] 更新 `src/mutagent/webui/_toolbar_impl.py` / `src/mutagent/webui/__init__.py`,`MainMenuAction` 改为 `mutagent.main_menu` 类别自动发现;导出 `ResumeSessionPage`
- [x] 更新 `tests/webui/test_webui_routing.py`、`tests/webui/test_webui_ui.py`,覆盖 resume 路由、渲染模式和相关 UI 测试

## 测试验证

- `pytest` - 771 passed, 4 skipped
- `pyright` - 0 errors
- `mutobj-lint` - 0 errors,现有 1 条 warning 位于 `src/mutagent/core/context.py`
