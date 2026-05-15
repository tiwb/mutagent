# Settings 全页面 + Conversation Hash 路由

**状态**：✅ 已完成
**日期**：2026-05-14
**类型**：功能设计

## 需求

1. 设置从 `antd.Drawer` 浮层改为「全页面替换式视图」，不再因点击遮罩 / Esc 误关闭，编辑状态不丢失。
2. URL hash（`#/`、`#/settings`、`#/settings/<panel>`）反映当前所在页面，支持：
   - 浏览器 back / 前进按钮在对话页与设置页之间切换，**不**整页刷新（WebSocket 不断连）。
   - 复制带 hash 的 URL 在新标签页打开，直接进入对应页面。
   - 在设置页 F5 刷新仍停留在设置页（编辑中的未保存内容丢失，预期）。
3. 设置页内有多个 SettingsPanel（当前 LLM、MCP）时支持左侧菜单切换，每次切换更新 URL。
4. 实现依赖 mutgui 提供的 `mutgui.setHash` 命令 + `$hashchange` 事件通道（见 mutgui `feature-system-events-and-hash-nav.md`），mutagent 只负责"路由解析 + 状态切换"的应用层逻辑。

## 关键参考

### mutagent 改造目标

- `src/mutagent/webui/settings.py` — `SettingsDrawer` Declaration（要重命名 `SettingsPage`，去 `is_open`，新增 `activate`/`deactivate`/`close`）
- `src/mutagent/webui/_settings_drawer_impl.py` — `SettingsDrawer` 实现（要重命名 `_settings_page_impl.py`，render 改全页面布局，`open`/`close`/`switch_to` 删除）
- `src/mutagent/webui/_settings_llm.py` 1571 行 — `LLMSettingsPanel` 实现（不改），其中 `_save_all_settings` 调用 `await view.drawer.close()` —— SettingsPage 需保留 `close()` 方法以维持兼容
- `src/mutagent/webui/_settings_mcp.py` — `MCPSettingsPanel` 实现（不改）
- `src/mutagent/webui/conversation.py` — `Conversation` Declaration（新增 `current_route` / `navigate_to` / `on_hash_change`）
- `src/mutagent/webui/_conversation_impl.py` — `Conversation` 实现（双模式 render、`on_event` 拦截 `$hashchange`、`SettingsPage` 注入 `on_request_close`）
- `src/mutagent/webui/_toolbar_impl.py` `OpenSettingsAction` —— 改为调 `conv.navigate_to(f"settings/{panel_id}")`
- `src/mutagent/webui/_server_impl.py` — root viewport 仍然绑定 `Conversation`，此设计**不改**协议层

### mutgui 依赖（已设计，未实施）

- `mutgui/docs/specifications/feature-system-events-and-hash-nav.md` — 依赖其提供的 `mutgui.setHash` 命令与 `$hashchange` 事件
- `frontend/src/core/system-events.ts`（即将由 mutgui 引入）— 前端在 WS 连上后向后端补发初始 hash
- `mutgui/src/mutgui/_view_impl.py` `_route_event` — `source: []` 路由到根 View `on_event`，`event.component_id == ""`

## 设计方案

### 路由权威集中在 Conversation（不抽 RouterView）

`Conversation` 作为 root View，新增字段 `current_route: str` 作为**单一真相源**。`SettingsPage`（重命名后）不持有 `is_open` —— 它的"当前 panel"由外部驱动。

不引入 mutgui 级 RouterView 抽象（YAGNI）。当前 mutagent 只有 conversation 与 settings 两类页面，未来若新增 history、export、profile 等顶级页面，再考虑提升路由层。

`current_route` 是裸字符串，遵循 `<page>` 或 `<page>/<sub>` 的扁平约定：

```
""               → 对话主页
"settings"       → 设置页（默认 panel）
"settings/llm"   → 设置页，LLM panel
"settings/mcp"   → 设置页，MCP panel
```

URL hash 与 route 的映射（前缀 `#/`，给未来 `#/history`、`#/export` 留扩展空间）：

```
#/                  ⇄ ""
#/settings          ⇄ "settings"
#/settings/llm      ⇄ "settings/llm"
```

解析与构造各一个工具函数：

```python
def _parse_hash(hash_value: str) -> str:
    return hash_value.lstrip("#").lstrip("/")

def _hash_for_route(route: str) -> str:
    return f"#/{route}" if route else "#/"
```

### URL 与 State 的关系

URL 是**输入**，state 是**真相**。

- 用户改 URL（back/前进/手输/新 tab）→ 浏览器触发 `$hashchange` → 后端 `Conversation.on_hash_change()` 解析 → 更新 `current_route` + 调用 panel 的 on_open/on_close → invalidate。**不**回发 setHash。
- 后端编程式导航（点设置按钮、保存后关闭）→ `Conversation.navigate_to()` 更新 `current_route` + 调 `send_command("mutgui.setHash", ...)` + invalidate。
- `render()` 永远只看 `current_route`，与 URL 解耦。

### 防循环：依赖 W3C 规范，零标记位

`pushState` / `replaceState` 不触发 `hashchange`、不触发 `popstate`（W3C 规定）。所以：

- `navigate_to` 发 `mutgui.setHash` 命令 → 前端 `pushState` → URL 变了但**没有事件触发** → 不回传后端 → 不循环。
- `on_hash_change` 处理浏览器侧事件时**不**回发 setHash，因为 URL 已经在前端是新值了。

无任何防循环标记位、setTimeout、内部方法分流等机制。

### Conversation 改造

#### Declaration（`conversation.py`）

```python
class Conversation(View):
    current_route: str  # "" | "settings" | "settings/<panel_id>"

    def __init__(self, *, agent: Agent, app: App | None = None) -> None: ...
    def render(self) -> ViewBlock: ...

    async def navigate_to(self, route: str) -> None: ...
    async def on_hash_change(self, hash_value: str) -> None: ...
```

#### 实现要点（`_conversation_impl.py`）

- `__init__` 初始化 `self.current_route = ""`，构造 `SettingsPage` 时通过两个回调注入路由能力：
  - `on_models_changed` — 现有，保留
  - `on_request_close` — 新增，`SettingsPage` 内部"返回对话"按钮、面板保存后等场景调它，等价于 `await conv.navigate_to("")`
  - `on_request_navigate` — 新增，`SettingsPage` 左侧菜单点击切换面板时调它，等价于 `await conv.navigate_to(f"settings/{panel_id}")`
- `navigate_to(route)`：若 route 与当前相同直接 return；否则调 `_apply_route(route)` 同步 panel 状态 → `send_command("mutgui.setHash", hash=...)` → `invalidate()`。
- `on_hash_change(hash_value)`：解析 hash → 若 route 与当前相同直接 return；否则 `_apply_route(route)` → `invalidate()`。**不**发 setHash 命令。
- `_apply_route(route)`：核心状态切换逻辑：
  - 若 prev 是 `"settings*"` 而 new 不是 → `await settings_page.deactivate()`（触发当前 panel 的 `on_close`）。
  - 若 new 是 `"settings*"` → 解析出 `panel_id`（无则用默认）→ `await settings_page.activate(panel_id)`（触发新 panel 的 `on_open`）。
  - 同在 settings 内换 panel（`"settings/llm"` → `"settings/mcp"`）→ `activate(new_panel_id)` 内部处理"先 close 旧 panel 再 open 新 panel"。
  - 最后 `self.current_route = route`。
- `on_event` override：拦截 `event.component_id == "" and event.name == "$hashchange"`，调 `on_hash_change(data["hash"])` 返回 True；其他事件走默认 `view_on_event`。

#### Render 双模式（最小侵入）

```python
@mutagent.impl(Conversation.render)
def render(self: Conversation) -> ViewBlock:
    _refresh_shell(self)
    in_settings = self.current_route.startswith("settings")
    children = (
        [self.settings_page]
        if in_settings else
        [
            {"$component": "div", "$id": "toolbar-shell", ..., "$children": [self.toolbar]},
            {"$component": "div", "$id": "messages-shell", ..., "$children": [self.message_list]},
            self.chat_input,
        ]
    )
    return ViewBlock([{
        "$component": "div", "$id": "conversation-root",
        "style": {... 现有 ...},
        "$children": children,
    }])
```

**关键**：每种模式只渲染本模式需要的子节点。settings 模式下 `chat_input` / `message_list` / `toolbar` **不**进 wire tree，前端 React 子树不挂载。但 `self.message_list` 等字段还是 Conversation 的属性，View 实例不销毁，下次切回来自动 reconcile 恢复。

settings 模式故意**不**渲染 toolbar（toolbar 上的 ☰ 菜单是「打开设置」入口、settings 已在其中重复；Status / Model selector 与对话强相关、设置上下文无意义；右侧 header 的「← 返回对话」按钮已提供离开入口）。如未来反馈需要在设置页看到 status，再独立做 settings 顶部 status 条。

### SettingsDrawer → SettingsPage 重命名 + 字段精简

#### Declaration（`settings.py`）

```python
class SettingsPage(View):
    """全页面设置容器。route 由 Conversation 驱动；本类不持有 is_open。"""

    active_panel_id: str  # 派生自 conv.current_route，由 activate() 写入

    def __init__(
        self, *, app: App, agent: Agent,
        on_models_changed=None,
        on_request_close=None,      # () → awaitable，请求 Conversation 退出 settings
        on_request_navigate=None,   # (route: str) → awaitable，请求 Conversation 切到指定 route
    ) -> None: ...

    def render(self) -> ViewBlock: ...

    async def activate(self, panel_id: str) -> None:
        """由 Conversation 调用：切到指定 panel（空字符串 = 默认）+ 触发 on_close/on_open。"""
    async def deactivate(self) -> None:
        """由 Conversation 调用：离开 settings → 当前 panel.on_close。"""
    async def close(self) -> None:
        """兼容方法：转发给 on_request_close。LLMSettingsPanel/_save_all_settings 仍在调用。"""

    def list_panels(self) -> list[SettingsPanel]: ...
    async def notify_models_changed(self, preferred_model: str = "") -> None: ...

# 不留 alias；SettingsDrawer 是内部类，趁本次一次改干净
```

`is_open` 字段彻底删除。`render()` 通过 `bool(self._panels.get(self.active_panel_id))` 判断有无 active panel；上层 Conversation 通过 `current_route.startswith("settings")` 判断是否进入 settings 模式。两份判断各管各的语义，不再有"两份状态需要同步"的隐患。

#### 全页面布局（左侧菜单 + 右侧内容）

```
┌────────────────────────────────────────────────────────┐
│ ┌──────────┐ ┌────────────────────────────────────┐   │
│ │           │ │ ← 返回  面板标题 ─────────────────│   │
│ │ LLM API  │ ├────────────────────────────────────│   │
│ │ MCP 连接  │ │                                    │   │
│ │           │ │  当前面板内容                       │   │
│ │           │ │  (scrollable)                       │   │
│ │           │ │                                    │   │
│ └──────────┘ └────────────────────────────────────┘   │
└────────────────────────────────────────────────────────┘
```

- 左侧 `antd.Menu mode="inline"`，宽度 220px。`selectedKeys=[active_panel_id]`，items 来自 `_ordered_panel_ids`。左侧菜单**纯粹做 panel 切换**，不放其他控件。
- 右侧 header 最左侧渲染"← 返回对话"按钮触发 `on_request_close`（与现有 ← Back 按钮风格延续，便于与面板标题并排显示，职责清晰）。
- 菜单项点击触发 `on_request_navigate(f"settings/{key}")`，**不**直接改 `active_panel_id` —— 必须走 Conversation 让 URL 同步。
- 右侧 header 显示当前 panel.title；右侧 body 渲染 `[active_panel]`。

#### activate / deactivate / close 实现

```python
async def activate(self, panel_id: str) -> None:
    target = panel_id or (self._ordered_panel_ids[0] if self._ordered_panel_ids else "")
    if target == self.active_panel_id:
        return
    # 先 close 旧 panel
    prev = self._panels.get(self.active_panel_id)
    if prev is not None:
        await _maybe_await(prev.on_close())
    # 切换 + open 新 panel
    self.active_panel_id = target
    new = self._panels.get(target)
    if new is not None:
        await _maybe_await(new.on_open())
    self.invalidate()

async def deactivate(self) -> None:
    prev = self._panels.get(self.active_panel_id)
    if prev is not None:
        await _maybe_await(prev.on_close())
    # active_panel_id 保留：再次进入 settings 时如果 hash 没指定 panel，
    # 走默认（first panel），不依赖上次的 active_panel_id。
    self.invalidate()

async def close(self) -> None:
    if self._on_request_close is not None:
        await _maybe_await(self._on_request_close())
```

注意：`deactivate` **不**清空 `active_panel_id`。理由：再次进入 settings 时由 Conversation 的 `_apply_route` 显式 `activate(panel_id)`，panel_id 来自 hash 解析，状态权威。

### OpenSettingsAction 改造

`_settings_drawer_impl.py` 中 `OpenSettingsAction.execute`：

```python
class OpenSettingsAction(Action):
    async def execute(self, context: ActionContext) -> None:
        conv = context.get("conversation")
        if conv is not None:
            await conv.navigate_to(f"settings/{self._panel_id}")
```

不再依赖 `context.get("settings_drawer")`。`MainMenuAction.menu_actions` 仍可通过 `conv.settings_page.list_panels()` 拿 panel 列表用于菜单渲染。

### LLMSettingsPanel / MCPSettingsPanel 不动

它们目前调用 `await view.drawer.close()` 和 `await view.drawer.notify_models_changed(...)`。

- `view.drawer` 字段名保留（其实指向 SettingsPage 实例，仅是属性名延续历史）。
- SettingsPage 提供 `close()` 兼容方法（转发到 `on_request_close`）和 `notify_models_changed()`（保留）。

**panel 文件零改动**。这是本次设计的兼容性核心收益。

### 影响范围

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `webui/settings.py` | 修改 | `SettingsDrawer` → `SettingsPage`；删除 `is_open` 字段、`open` / `switch_to` 方法；新增 `activate` / `deactivate` |
| `webui/_settings_drawer_impl.py` → `_settings_page_impl.py` | 重命名 + 重写 | 全页面布局；`open`/`close`/`switch_to` 删除；新增 `activate`/`deactivate`/`close`；`OpenSettingsAction` 改用 conv |
| `webui/conversation.py` | 修改 | 新增 `current_route` 字段、`navigate_to` / `on_hash_change` 方法 |
| `webui/_conversation_impl.py` | 修改 | 双模式 render；`on_event` 拦截 `$hashchange`；构造 SettingsPage 时注入两个 request 回调 |
| `webui/_toolbar_impl.py` | 微调 | `MainMenuAction` 内部 `from ._settings_drawer_impl` import 路径改为 `_settings_page_impl` |
| `webui/_settings_llm.py` / `_settings_mcp.py` | **不改** | drawer 字段名 + close()/notify_models_changed() 兼容 |
| `tests/test_webui_ui.py` | 修改 | 适配 SettingsPage 渲染结构（无 antd.Drawer 浮层）；新增 navigate_to / on_hash_change 测试 |
| `tests/` 新增 | 新增 | route 解析、双模式 render、hash 同步集成测试 |

### 状态保持验收

| 场景 | 期望行为 |
|------|---------|
| 在 LLM 设置中编辑了 API key（未保存）→ 切到 MCP → 切回 LLM | API key 编辑保留（panel View 实例存活，`_drafts` 字典不丢） |
| 在 LLM 设置 → 点"← 返回对话" → 再点 ☰ 菜单进 LLM 设置 | LLMSettingsPanel.on_open 重新触发 `_load_from_config`，未保存编辑会被重置（按现有 on_open 行为，预期） |
| 在 LLM 设置 → 浏览器 back | 同上：on_close 触发，回 conversation 模式，message_list 完整 |
| 在 conversation → back | 浏览器走出本应用（或回到上一页面）。本设计不阻拦。 |
| 在 settings 页 F5 刷新 | hash 保留 → setupSystemEvents 补发 `$hashchange` → Conversation `_apply_route` 进入 settings → on_open 重新加载 config |
| 复制 `http://localhost:8741/#/settings/mcp` 在新 tab 打开 | 同上 F5 路径，直接进入 MCP 面板 |

### 兼容性

- `SettingsPanel` 子类的 `panel_id` / `panel_title` / `panel_placement` ClassVar 不变，`discover_subclasses` 自动发现机制不变。
- `on_open` / `on_close` 调用时机改变：
  - 旧：`SettingsDrawer.open(panel_id)` / `close()` 直接触发。
  - 新：通过 `Conversation.navigate_to` 间接触发（`_apply_route` → `settings_page.activate/deactivate`）。
  - 第三方 SettingsPanel 子类**不感知**这层变化。
- `OpenSettingsAction` 的导入路径改变（`_settings_drawer_impl` → `_settings_page_impl`）。`MainMenuAction` 已用 `from ... import OpenSettingsAction` 局部导入，跟随改即可。
- Drawer 浮层视觉效果消失，UI 测试需要重新对齐。
- `on_open` / `on_close` **不**区分"用户点菜单" vs "hash 切换"触发源——panel 子类只关心"我被打开/关闭了"。如未来真有按来源差异化的需求（如 hash 切换时不弹"未保存"提示），给 `on_open` 加可选参数 `*, source: str = "user"`（值 `"user"` / `"hash"` / `"initial"`），本期先简单。

## 未来嵌入兼容性注记

本设计隐含 3 个"Conversation 是 root"的假设。未来 mutbot 如果演化为基于 mutgui 的多应用 shell 容器（详见 `mutgui/docs/specifications/feature-multi-app-shell-container.md`），这些假设会不再成立。将这些假设显式记录在这里作为预定改造点，避免成为被遗忘的隐式契约。

**隐含假设 1：Conversation 是 root View**

这使得它可以直接 override `on_event` 拦截 `$hashchange`（`source: []` 才到得了它）。嵌入场景下 root 是 shell 的 `AppContainer`，Conversation 是某个 sub-app 的子 View，拿不到 `source: []` 事件。

- **预定改造点**：`_conversation_impl.py` 的 `on_event` override（全文 1 处）。
- **预期改造方式**：改为由父 View（`AppContainer` / `RouterView`）注入回调调用 `on_hash_change`，或订阅 mutgui 未来提供的"子路由事件"接口。

**隐含假设 2：Conversation 拥有整个 hash 的所有权**

这使得 `_parse_hash` / `_hash_for_route` 是裸 hash 与裸 route 的双向直映射。嵌入场景下 hash 是 `#/<app_id>/settings/llm`，mutagent 只拥有 `settings/llm` 这段，前缀 `<app_id>/` 属于 shell。

- **预定改造点**：`conversation.py` 中的 `_parse_hash` / `_hash_for_route` 两个工具函数。
- **预期改造方式**：改为"接收/产出挂载点相对路径"，由嵌入层（shell 的 `RouterView`）负责加/去前缀。
- **关键好品味**：`current_route = "settings/llm"` 这种字符串形式本身就**已经是"相对于 mutagent 挂载点"的路径**——它没有 `#/` 也没有应用 ID，恰好就是未来嵌入模式下 mutagent 应该看到的内部路径。所以应用内部所有 `navigate_to` / `_apply_route` / 双模式 render 等核心逻辑，**未来一行不用改**。

**隐含假设 3：`navigate_to` 直接调 `mutgui.setHash` 交付裸 hash**

嵌入场景下这会覆盖 shell 的 hash 命名空间。

- **预定改造点**：`navigate_to` 内部的 `send_command("mutgui.setHash", ...)` 调用（1 处）。
- **预期改造方式**：改为调父 View 注入的回调（独立运行时该回调仍可走原命令，嵌入运行时由 shell 拼接完整 hash）。

**总计改造面**

3 个文件、5 个触点，没有任何"散落在多个 panel 里的耦合"。LLMSettingsPanel / MCPSettingsPanel 完全是 panel 内部状态，跟 hash 无关，不受影响。

**本期不为此作任何预防性抽象**（YAGNI），仅记录这 3 个改造点，避免未来重新探索。

## 遗留问题

### 多 ViewPort 共享 View 时的 hash 行为

mutgui 支持同一 View 被多个 ViewPort 观察。如果未来 mutagent 演化为多 tab 共享同一个 Conversation View，tab A 进设置 → `current_route = "settings/llm"` 会被广播到 tab B 的 wire tree，tab B 的 URL 也会被 setHash 改写。

**当前不构成问题**：mutagent WebUI 每个 WebSocket 连接独立创建 Conversation View 实例（`WebUIServer.__init__` 中 `self.conversation = Conversation(...)`），多 tab 是多 instance，hash 互不影响。

**未来若改为共享 View**：需要 mutgui 层面提供 `send_command` 的"指定 viewport"能力（让 setHash 只发给触发导航的那个 ViewPort），超出本期范围。

## 消费者场景

| 消费者 | 场景 | 依赖的输出 | 验收标准 |
|--------|------|-----------|---------|
| 终端用户 | 打开 LLM 设置编辑 → 浏览器 back | URL hash 双向同步 + 全页面 | 回到对话页，消息历史完整，WebSocket 不断 |
| 终端用户 | 复制 `#/settings/mcp` URL 新 tab 打开 | `$hashchange` 初始事件 + Conversation 路由 | 新 tab 直接进入 MCP 设置面板 |
| 终端用户 | 设置页内 F5 刷新 | hash 保留 + 重新连后补发事件 | 仍在设置页，编辑中未保存内容丢失（预期） |
| 终端用户 | 在 LLM 与 MCP 之间切换 | 左侧菜单 → on_request_navigate → URL 同步 | 每次切换 URL 变 `#/settings/llm` ↔ `#/settings/mcp`，back 一次回上一个 panel |
| 第三方 SettingsPanel 子类作者 | 新增一个 panel | `discover_subclasses` 不变 | 新 panel 自动出现在左侧菜单中，URL `#/settings/<panel_id>` 自动可用 |
| LLMSettingsPanel.save_all | `await view.drawer.close()` | SettingsPage.close() 兼容方法 | 保存后回到 conversation 页面，URL 变 `#/`，消息历史完整 |

## 实施步骤清单

### Settings 子系统重命名 + 字段精简

- [x] `git mv` 重命名 `_settings_drawer_impl.py` → `_settings_page_impl.py`（保留 git 历史）
- [x] `webui/settings.py`：`SettingsDrawer` → `SettingsPage` 重命名；删除 `is_open` 字段、`open` / `switch_to` 方法；新增 `activate(panel_id)` / `deactivate()` 方法；`__init__` 增加 `on_request_close` / `on_request_navigate` 两个回调参数；保留 `close()` / `notify_models_changed()` / `list_panels()` 兼容方法（不留 SettingsDrawer alias，一次改干净）
- [x] `_settings_page_impl.py`：实现 `activate` / `deactivate` / `close` 新语义（activate 内部先 close 旧 panel 再 open 新 panel；deactivate 不清空 `active_panel_id`）
- [x] `_settings_page_impl.py`：render 改为全页面布局——左侧 `antd.Menu mode="inline"` 宽 220px（`selectedKeys=[active_panel_id]`、items 按 `_ordered_panel_ids`）+ 右侧 header（最左渲染「← 返回对话」按钮触发 `on_request_close`，右侧渲染当前 panel.title）+ 右侧 body 渲染 `[active_panel]`；删除 `antd.Drawer` 浮层结构
- [x] `_settings_page_impl.py`：菜单项 onClick 触发 `on_request_navigate(f"settings/{panel_id}")`，**不**直接修改 `active_panel_id`（必须走 Conversation 让 URL 同步）
- [x] `_settings_page_impl.py`：`OpenSettingsAction.execute` 改为 `await context.get("conversation").navigate_to(f"settings/{self._panel_id}")`，不再读 `context.get("settings_drawer")`；`RefreshModelsAction` 不动

### Conversation 路由层

- [x] `webui/conversation.py`：Declaration 新增 `current_route: str` 字段、`navigate_to(route)` / `on_hash_change(hash_value)` 方法签名
- [x] `_conversation_impl.py`：`__init__` 初始化 `self.current_route = ""`；`SettingsDrawer` 引用改为 `SettingsPage`；属性名 `self.settings_drawer` 改为 `self.settings_page`（语义对齐）；构造 `SettingsPage` 时注入 `on_request_close` / `on_request_navigate` 两个回调（除既有 `on_models_changed`）
- [x] `_conversation_impl.py`：实现 `_parse_hash` / `_hash_for_route` 工具函数（裸字符串映射 `#/<page>/<sub>` ⇄ `<page>/<sub>`）
- [x] `_conversation_impl.py`：实现 `navigate_to(route)`——同 route 早返回 → `_apply_route` → `send_command("mutgui.setHash", hash=...)` → `invalidate()`
- [x] `_conversation_impl.py`：实现 `on_hash_change(hash_value)`——解析 → 同 route 早返回 → `_apply_route` → `invalidate()`；**不**回发 setHash 命令（防循环靠 W3C 天然行为）
- [x] `_conversation_impl.py`：实现 `_apply_route(route)`——按 prev/new 是否以 `settings` 起头分四象限处理 deactivate / activate / 同 settings 内换 panel；最后写入 `self.current_route = route`
- [x] `_conversation_impl.py`：override `on_event(event)` 拦截 `event.component_id == "" and event.name == "$hashchange"` → `await on_hash_change(data["hash"])` → return True；其他事件 `return await super().on_event(event)` 保留默认子组件分发
- [x] `_conversation_impl.py`：`render` 改双模式——`in_settings = current_route.startswith("settings")` 时 children 为 `[self.settings_page]`；否则 children 为 toolbar-shell + messages-shell + chat_input（**不**含 settings_page）；root div style 保持现状
- [x] `_conversation_impl.py`：`ActionContext.data` 中删除 `settings_drawer` key（OpenSettingsAction 已改用 `conversation` key），保留 `conversation`

### Toolbar 适配

- [x] `_toolbar_impl.py::MainMenuAction.menu_actions`：import 路径 `_settings_drawer_impl` → `_settings_page_impl`；改为通过 `context.get("conversation").settings_page.list_panels()` 拿 panel 列表生成菜单项

### Panel 文件零改动验证

- [x] 验证 `_settings_llm.py:564` `await view.drawer.notify_models_changed(default_model)` 在新结构下仍工作（依赖 SettingsPage 保留 `notify_models_changed` + `setattr(panel, "drawer", self)` 字段名延续）
- [x] 验证 `_settings_llm.py:565,572` 与 `_settings_mcp.py:686` `await view.drawer.close()` 在新结构下走 SettingsPage.close() → on_request_close → conv.navigate_to("")

### 测试

- [x] 重写 `tests/test_webui_ui.py::test_settings_drawer_renders_inline_in_conversation` 为 `test_settings_page_excluded_from_conversation_mode_render`：在 `current_route == ""` 下 root.$children 不含 settings_page
- [x] 更新 `tests/test_webui_ui.py::test_conversation_child_views_have_stable_ids`：`settings_drawer.id` 引用改为 `settings_page.id`
- [x] 新增 route 解析单测：`_parse_hash` / `_hash_for_route` 三向（裸串、`#/`、`#/settings/llm`）+ 空串
- [x] 新增 `navigate_to` / `on_hash_change` 状态机单测：四种状态切换 + 同 route 早返回 + activate/deactivate 触发 panel.on_open/on_close 钩子
- [x] 新增双模式 render 单测：settings 模式 children 只含 settings_page、不含 toolbar/message_list/chat_input；conversation 模式相反
- [x] 新增 on_event 路由单测：`source=[], $hashchange` 走 on_hash_change；其他事件走 super().on_event 默认分发
- [x] 新增防循环单测：`navigate_to` 调 `send_command("mutgui.setHash", ...)` 一次；`on_hash_change` 不调 send_command
- [x] 全量 `pytest` 通过（1003 passed, 4 skipped）

### 人工验收（与「消费者场景」表逐条对应，浏览器肉眼跑通）

- [x] 打开 LLM 设置 → 浏览器 back → 回对话页，消息历史完整，WebSocket 未断（`channel_id` 不变）
- [x] 复制 `http://localhost:8741/#/settings/mcp` 在新 tab 打开 → 首屏直接进入 MCP 面板（依赖 mutgui `mount.attach.client.hash` 握手）
- [x] 设置页内 F5 → 仍在设置页（编辑中未保存内容丢失，预期）
- [x] 左侧菜单 LLM ↔ MCP 切换 → URL 同步变 `#/settings/llm` ↔ `#/settings/mcp`，back 一次回上一个 panel
- [x] LLMSettingsPanel 「Save All」 → 回到对话页，URL 变 `#/`（验证 SettingsPage.close() 兼容方法链路）
- [x] 在 LLM 编辑 API key（未保存）→ 切到 MCP → 切回 LLM → 编辑保留（panel View 实例存活，`_drafts` 不丢）
