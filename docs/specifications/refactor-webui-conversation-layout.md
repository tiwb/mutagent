# WebUI 架构整理 — Conversation 职责过重与设置面板分层

**状态**：✅ 已完成
**日期**：2026-05-11
**类型**：重构

## 需求

1. Conversation 承担了页面布局、设置面板托管等与对话无关的职责，增加新设置（如 MCP Settings）会让混乱加剧
2. `_settings_impl.py`（916 行 / 33KB）名不副实——全是 LLM Settings 的实现，文件名暗示通用，但实际不通用
3. 新增 MCP Settings 需要一个清晰的文件结构，而不是往已有大文件里塞

## 关键参考

### 文件清单与行数

| 文件 | 行数 | 实际内容 |
|------|------|---------|
| `webui/conversation.py` | 27 | `Conversation` View Declaration |
| `webui/_conversation_impl.py` | 447 | ①对话逻辑 ②Agent 桥接 ③页面布局 ④设置面板宿主 ⑤模型管理 |
| `webui/settings.py` | 27 | `LLMSettingsPanel` View Declaration |
| `webui/_settings_impl.py` | 916 | LLM Settings 全套实现（列表页、编辑页、模型发现、持久化） |
| `webui/_actions_impl.py` | 186 | Action 注册（`OpenLLMSettingsAction` 等） |
| `webui/_toolbar_impl.py` | 199 | `AgentStatusBar` 实现 |
| `webui/_chat_input_impl.py` | 89 | `ChatInput` 实现 |

### Conversation 的 6 类职责（`_conversation_impl.py`）

```
① Agent 桥接（核心）
    __init__: agent.subscribe(...)
    _handle_agent_event()        — StreamEvent 分发（text_delta, tool_exec_start...）
    _handle_send()               — agent.submit(text)
    _handle_cancel()             — agent.cancel()
    _ensure_current_assistant()  — 流式响应追踪

② 消息列表管理
    __init__: self.items = []
    _append_item() / _touch_item() / _find_item()

③ 运行时状态
    is_busy / status / _cancel_requested
    _turn_input_tokens / _turn_output_tokens / _total_cost

④ 子组件组装
    status_bar / chat_input / toolbar
    _refresh_shell()  — 同步所有子组件状态

⑤ 设置面板宿主（与对话无关）
    settings_open / settings_panel (LLMSettingsPanel)
    _open_settings() / _close_settings()
    _settings_saved()             — 保存后刷新模型列表
    _refresh_models_from_config()

⑥ 模型管理
    models / current_model
    _handle_model_change()
```

### 设置面板交互链路（当前全嵌在 Conversation 里）

```
☰ 菜单 (MainMenuAction)
  └─ OpenLLMSettingsAction.execute()
       └─ context.get("conversation")._open_settings_action  ← Action 引用了 Conversation
            └─ self.settings_open = True
            └─ self.invalidate()
            └─ render() → antd.Drawer → LLMSettingsPanel      ← Drawer 硬编码在 Conversation.render() 里
```

### LLM Settings 内部结构（`_settings_impl.py` 916 行）

```
30+ 个函数，分为这些组：

  Provider 预设          _PROVIDER_PRESETS, _provider_label_from_path, _provider_protocol...
  Draft 管理             _draft_from_config, _make_provider_draft, _apply_draft...
  模型发现               _discover_remote_models, _prioritize_models, _model_family...
  页面渲染               _render_list (199 行), _render_edit (156 行), _render_message
  用户操作回调           _edit_provider, _start_add_provider, _save_provider_edits,
                        _delete_provider, _discover_models, _save_all_settings...
  配置持久化             _write_config
```

### _actions_impl.py 中与设置相关的 Action

```
OpenLLMSettingsAction    → conversation._open_settings_action
RefreshModelsAction      → conversation._refresh_models_action
MainMenuAction           → menu_actions() 返回 [OpenLLMSettingsAction, RefreshModelsAction]
```

Action 通过 `ActionContext` 获取 `conversation` 对象，然后调用其方法——这要求 Conversation 必须提供这些方法。

## 设计方案

### 总体思路

核心是消除 Conversation 里的「⑤ 设置面板宿主」职责，把它整体下放给 SettingsDrawer 自管。Conversation 仍作为根 View（既是对话也是页面布局），只把 SettingsDrawer 作为子组件持有，不操作其状态（不引入额外的 AppShell 层 — 单对话应用 Conversation 即页面，YAGNI；未来若需 sidebar / 多对话 tab 再升级）。

```
Conversation (根 View — 页面布局 + 对话)
├── ActionToolbar                 顶部工具栏（含 ☰ 菜单）
├── MessageList + ChatInput       对话主体
└── SettingsDrawer (新)           self-contained，自管 is_open / active_panel_id
        ├── LLMSettingsPanel
        ├── McpSettingsPanel  (未来)
        └── ...其他 SettingsPanel 子类
```

职责切割：

| 组件 | 职责 | 不该做的事 |
|------|------|-----------|
| `Conversation` | Agent 桥接、消息列表、ChatInput、StatusBar、模型选择 callback、页面布局、装配 SettingsDrawer 子组件、注入 ActionContext | 操作 Drawer 的 open/close、托管具体面板生命周期 |
| `SettingsDrawer` | `is_open`、`active_panel_id`、Drawer 渲染、面板路由、关闭回调、面板生命周期 | 任何具体面板的业务逻辑 |
| `SettingsPanel`（基类） | 声明 `panel_id` / `panel_title` / `panel_placement`，统一打开/关闭/保存契约 | — |
| `LLMSettingsPanel` | LLM provider 配置（继承 `SettingsPanel`） | — |

### 文件结构

扁平化在 `webui/` 下，不开子目录。设置子系统的公开契约（基类 + Drawer Declaration）放在 `settings.py`，让具体面板文件**显式依赖**它（一眼可见层次关系）。每个具体面板独占一个 `_settings_<name>.py`，**Declaration 与 `@impl` 合写在同一文件**（属于内部实现细节，不再分离）。

```
webui/
  __init__.py
  cli.py

  # ── 顶层 View（公开 API，保留 decl/impl 双文件惯例） ──
  server.py / _server_impl.py
  conversation.py / _conversation_impl.py   减肥到 ~280 行（仍是根 View）

  # ── 对话子组件（保留现状） ─────────────────────────
  chat_input.py / _chat_input_impl.py
  messages.py / _messages_impl.py
  toolbar.py / _toolbar_impl.py
  tool_call.py / _tool_call_impl.py
  blocks.py / _blocks_impl.py

  # ── Actions ─────────────────────────────────────────
  _actions_impl.py

  # ── 设置子系统 ───────────────────────────────────────
  settings.py                  NEW — 公开契约：SettingsPanel 基类 + SettingsDrawer Declaration
  _settings_drawer_impl.py     NEW — SettingsDrawer 的 @impl
  _settings_llm.py             NEW — LLMSettingsPanel（decl + 全部 impl，替代旧 settings.py + _settings_impl.py）
  # 未来：
  # _settings_mcp.py           — McpSettingsPanel
```

命名约定：

| 角色 | 文件名 | 例 |
|------|--------|----|
| 公开契约（被外部 import） | `xxx.py` | `settings.py`、`conversation.py` |
| 公开 View 的实现 | `_xxx_impl.py` | `_conversation_impl.py`、`_settings_drawer_impl.py` |
| 内部独立面板（decl+impl 合一） | `_settings_<name>.py` | `_settings_llm.py`、`_settings_mcp.py` |

旧文件去向：
- `settings.py`（旧 LLMSettingsPanel Declaration）→ **删除**，被新的 `settings.py`（SettingsPanel/SettingsDrawer）替代
- `_settings_impl.py`（916 行 LLM 实现）→ **删除**，内容整体迁入 `_settings_llm.py`

`_settings_llm.py` 仍然较长（~900 行），但这是单一面板的完整自包含实现，符合「一个面板一个文件」约定。如未来真有拆分需求再讨论。

### `settings.py` 公开契约

```python
# webui/settings.py
from mutgui import View, ViewBlock

class SettingsPanel(View):
    """所有设置面板基类。子类只声明元数据，由 SettingsDrawer 自动发现。"""
    panel_id: str = ""           # 唯一 id，用于路由（如 "llm", "mcp"）
    panel_title: str = ""        # Drawer 标题
    panel_placement: str = ""    # 在 ☰ 菜单里的位置（如 "settings:10/10"）
    panel_width: int = 560

    def render(self) -> ViewBlock: ...

    # 统一生命周期钩子（子类按需 override）
    def on_open(self) -> None: ...     # Drawer 打开本面板时调用（reload 配置等）
    def on_close(self) -> None: ...

class SettingsDrawer(View):
    """Drawer 容器，按 active_panel_id 路由到 SettingsPanel 子类。"""
    is_open: bool
    active_panel_id: str

    def __init__(self, *, app, agent) -> None: ...
    def render(self) -> ViewBlock: ...

    async def open(self, panel_id: str) -> None: ...      # 打开并切到指定面板
    async def close(self) -> None: ...
    async def switch_to(self, panel_id: str) -> None: ... # 已打开状态下切换（预留给未来"统一入口"形态）
    def list_panels(self) -> list[SettingsPanel]: ...     # 给菜单/未来设置首页生成用

from . import _settings_drawer_impl  # noqa
```

### SettingsPanel 子类自动发现

通过 mutobj 的 `discover_subclasses(SettingsPanel)` 在 `SettingsDrawer.__init__` 时收集所有面板实例（按 `panel_placement` 排序）。

```python
# _settings_llm.py
from mutagent.webui.settings import SettingsPanel
import mutagent

class LLMSettingsPanel(SettingsPanel):
    panel_id = "llm"
    panel_title = "LLM API 设置"
    panel_placement = "settings:10/10"
    # ...其它字段

@mutagent.impl(LLMSettingsPanel.__init__)
def __init__(self, *, app, agent, on_saved=None): ...

@mutagent.impl(LLMSettingsPanel.render)
def render(self) -> ViewBlock: ...
```

未来新增 MCP Settings：仅新建 `_settings_mcp.py` 一个文件，`Conversation` / `AppShell` / `SettingsDrawer` / `_actions_impl.py` 全部不动。

### 入口策略

采用「每个面板独立菜单项 + 直达打开」（最简形态，未来可平滑升级到「统一设置首页」）。

```
☰ 菜单（MainMenuAction.menu_actions 动态生成）
 ├─ LLM API 设置        → OpenSettingsAction("llm")     → drawer.open("llm")
 ├─ MCP Servers (未来)  → OpenSettingsAction("mcp")     → drawer.open("mcp")
 └─ Refresh Models     → RefreshModelsAction
```

选 A 的理由：
- 当前面板数 1，近期 2，独立入口少一次点击、意图清晰
- LLM Settings 内部已有 list↔edit 两层导航，外面再叠左侧栏会让信息架构变 4 层
- `SettingsDrawer.switch_to()` 已预留切换能力，未来若加「设置首页」面板（`_settings_index.py`），仅追加文件 + 菜单加一项 "All Settings…"，不破坏现有结构

### 通用 OpenSettingsAction

替代手写的 `OpenLLMSettingsAction`。一个类支撑所有面板：

```python
class OpenSettingsAction(Action):
    def __init__(self, panel_id: str, label: str, placement: str):
        super().__init__()
        self._panel_id = panel_id
        self.label = label
        self.placement = placement

    def resolved_action_id(self) -> str:
        return f"mutagent.menu.settings.{self._panel_id}"

    async def execute(self, context: ActionContext) -> None:
        drawer = context.get("settings_drawer")
        await drawer.open(self._panel_id)
```

`MainMenuAction.menu_actions` 改为遍历 `drawer.list_panels()` 动态出菜单项（追加 `RefreshModelsAction` 等非面板项）。

### ActionContext 注入约定

由 `Conversation` 装配 Toolbar 时统一注入：

```python
ActionContext(owner=self, data={
    "conversation": self,                     # 模型选择、状态栏
    "settings_drawer": self.settings_drawer,  # 打开/切换/关闭设置面板
})
```

各 Action 走哪个 key：

| Action | context key |
|--------|-------------|
| `ModelSelectorAction` / `SelectModelAction` | `conversation` |
| `AgentStatusAction` | `conversation` |
| `MainMenuAction` | `settings_drawer`（用于 list_panels） |
| `OpenSettingsAction(panel_id)` | `settings_drawer` |
| `RefreshModelsAction` | `conversation` |

### Conversation 减肥后保留什么

仅去掉 ⑤ 设置宿主，其余全部保留（toolbar 仍在 Conversation 装配，因为它是页面布局的一部分）：

```
① Agent 桥接           保留
② 消息列表管理         保留
③ 运行时状态           保留
④ 子组件组装           保留 status_bar / chat_input / toolbar；新增持有 settings_drawer
✗ 设置面板宿主          删除（→ SettingsDrawer 自管 open/close/路由/生命周期）
⑥ 模型管理             保留 _handle_model_change（属 Agent 行为）
                       _refresh_models_from_config 改名为 public refresh_models()
```

### 「设置保存后刷新模型」的解耦

旧链路：
```
LLMSettingsPanel.on_saved → conversation._settings_saved → _refresh_models_from_config
```

新链路（解除 LLMSettingsPanel 对 Conversation 的反向依赖）：
```
Conversation.__init__:
  self.settings_drawer = SettingsDrawer(
      app=app, agent=agent,
      on_models_changed=self.refresh_models,   # 唯一注入的回调
  )

保存时：
  LLMSettingsPanel.on_saved(preferred_model)
    → SettingsDrawer 调用 on_models_changed(preferred_model)
    → Conversation.refresh_models(preferred_model)
    → SettingsDrawer.close()
```

`SettingsDrawer` 通过构造时注入的 callback 通知外部，不直接持有 Conversation 引用；具体面板（LLMSettingsPanel）只通知 Drawer，不知道 Conversation 存在。

### Server 装配

`_server_impl.py` 不变，根 View 仍是 `Conversation`。

### 不做的事

- 不引入「设置首页」面板（方案 C），等面板数 ≥ 4 时再加
- 不在 Drawer 头部加面板切换下拉，保持 Drawer 头部干净（Drawer 内部已有 list↔edit 导航）
- 不把 LLMSettingsPanel 内部按 provider/draft/discovery/list/edit 拆成多个文件，保持「一个面板一个文件」

### 关键实施决策（新 session 实施时必读）

以下 4 项是讨论后确定的实施细节，避免新 session 实施时重新决策。

**1. SettingsPanel 子类发现时机**

`discover_subclasses(SettingsPanel)` 只能发现已 import 的子类。在 `webui/__init__.py` 顶部显式 import 所有 `_settings_*.py` 模块（如 `from . import _settings_llm`），确保 `SettingsDrawer` 实例化时所有面板已注册。新增面板时同步在 `__init__.py` 加一行 import。

**2. SettingsPanel 统一构造签名**

基类约定 `__init__(*, app, agent)`。`SettingsDrawer` 实例化所有子类时统一传这两个参数。子类如需更多依赖，从 `app` / `agent` 取，不扩展构造签名。

**3. 面板 → Conversation 通信链路**

面板不直接持有 Conversation 引用，链路三跳：
```
LLMSettingsPanel._save_all_settings
  → self.drawer.notify_models_changed(preferred_model)   # 基类提供 self.drawer 反向引用
  → Drawer 调用构造时注入的 on_models_changed callback
  → Conversation.refresh_models(preferred_model)         # Conversation 创建 Drawer 时注入 self.refresh_models
```

基类 `SettingsPanel` 在 `Drawer.__init__` 实例化子类后被 Drawer 反向 set `panel.drawer = self`。

**4. Drawer 渲染 active panel 的方式**

Drawer 在 `__init__` 一次性实例化所有面板（构造 `dict[panel_id, SettingsPanel]` 映射）。`render()` 根据 `active_panel_id` 从映射取对应面板作为 `antd.Drawer` 的 child。未激活面板不渲染但保留实例（`destroyOnHidden=False`，复用现有约定保持面板内部状态）。

## 实施步骤清单

> 实施前必读：
> - 「关键参考」章节定位旧代码
> - 「设计方案 → settings.py 公开契约」明确接口
> - 「设计方案 → 关键实施决策」明确 4 个实施细节

- [x] 新建 `webui/settings.py` — SettingsPanel 基类（panel_id/panel_title/panel_placement/panel_width + render + on_open/on_close 钩子 + drawer 反向引用属性）+ SettingsDrawer Declaration（is_open/active_panel_id + open/close/switch_to/list_panels/notify_models_changed）
- [x] 实现 `webui/_settings_drawer_impl.py` — SettingsDrawer 的 @impl：构造时 `discover_subclasses(SettingsPanel)` 收集子类、统一构造签名实例化、按 placement 排序、反向 set `panel.drawer = self`；render() 用 antd.Drawer 包裹 active panel；open/close/switch_to 维护状态并触发面板 on_open/on_close 钩子；notify_models_changed 转发外部 callback
- [x] 迁移 LLM 面板到 `webui/_settings_llm.py` — 旧 `_settings_impl.py` 内容整体搬入；LLMSettingsPanel 改继承 SettingsPanel；声明 `panel_id="llm"` / `panel_title="LLM API 设置"` / `panel_placement="settings:10/10"`；__init__ 改用统一签名 `(*, app, agent)`；`_save_all_settings` 改为调用 `self.drawer.notify_models_changed(preferred_model)`；`_close_panel` 改为调用 `self.drawer.close()`
- [x] 删除旧 `webui/settings.py`（旧 LLMSettingsPanel Declaration）和 `webui/_settings_impl.py`（916 行旧实现）
- [x] `webui/__init__.py` 显式 import 所有 `_settings_*` 模块，保证子类发现
- [x] Conversation 减肥（`_conversation_impl.py`）— 删除 settings_open / settings_panel / _open_settings / _close_settings / _settings_saved / _open_settings_action / _close_settings_action / _refresh_models_action 全部成员；新增 `self.settings_drawer = SettingsDrawer(app=app, agent=agent)` 并通过参数注入 `on_models_changed=self.refresh_models`；`_refresh_models_from_config` 改名为 public `refresh_models`；render() 把内联 antd.Drawer 块替换为渲染 `self.settings_drawer`
- [x] ActionContext 注入调整 — Conversation 装配 toolbar 时 context.data 加 `"settings_drawer": self.settings_drawer`
- [x] Action 重构（`_actions_impl.py`）— 删除 `OpenLLMSettingsAction`；新增通用 `OpenSettingsAction(panel_id, label, placement)` 走 `context.get("settings_drawer").open(panel_id)`；`MainMenuAction.menu_actions()` 改为遍历 `drawer.list_panels()` 动态生成 + 追加 `RefreshModelsAction`
- [x] 测试适配 — 修复 `tests/test_webui_ui.py` 中所有引用已删成员（settings_open / settings_panel / _open_settings_action 等）的断言；运行 `pytest tests/test_webui_ui.py` 全绿，运行完整 `pytest` 无回归
- [x] 手工验证 — `python -m mutagent` 启动 webui，验证完整闭环：☰ 菜单显示 LLM 项 → 点击打开 Drawer → 编辑/添加/删除 provider → 保存后模型列表刷新且 Drawer 关闭 → 模型选择器仍能切换 → ☰ 菜单 Refresh Models 仍工作
