# Actions 分散到各业务模块 — 设计规范

**状态**：✅ 已完成
**日期**：2026-05-12
**类型**：重构

## 需求

1. `_actions_impl.py` 集中了全部 9 个 Action 子类，但 Action 在逻辑上分属 toolbar、chat_input、settings 三个不同业务域
2. 当前 `_actions_impl.py` 是一个"万能文件"——不属于任何一个具体模块，却引用了所有模块的 View 和上下文
3. 将 Action 按业务域归属分散到各自的 `_xxx_impl.py` 文件中，每个模块自己 Actions + Views 一体，消除跨模块耦合

## 关键参考

- `src/mutagent/webui/_actions_impl.py` — 当前所有 Action 集中地（9 个 Action 子类）
- `src/mutagent/webui/_toolbar_impl.py` — 工具栏 View 实现，应接收 Toolbar 相关 Action
- `src/mutagent/webui/_chat_input_impl.py` — 输入框 View 实现，应接收 ChatInput 相关 Action
- `src/mutagent/webui/_settings_drawer_impl.py` — 设置抽屉 View 实现，应接收 Settings 相关 Action
- `src/mutagent/webui/__init__.py` — 当前通过 `from . import _actions_impl` 触发注册
- `src/mutgui/src/mutgui/action.py` — Action 基类 + ActionRegistry（通过 `discover_subclasses` 全局发现）
- `src/mutagent/webui/_conversation_impl.py` — Conversation 实现，`ActionToolbar` 实例创建处

## 设计方案

### Action 注册机制

Action 子类不通过 `@mutagent.impl()` 绑定到 Declaration，而是通过 `mutobj.discover_subclasses(Action)` **全局自动发现**。只要 Action 类所在的模块被 import 过（类定义被执行过），`ActionRegistry` 就能找到它。

当前 `__init__.py` 末尾：
```python
from . import _actions_impl  # noqa
```
拆散后各 impl 文件已被 `__init__.py` 间接 import（通过 `from . import _toolbar_impl` 等），Action 类定义在其中自然会被发现，**不需要额外的注册代码**。

### Action 归属分配

| Action 类 | 目标文件 | 所属 domains |
|---|---|---|
| `ModelSelectorAction` | `_toolbar_impl.py` | toolbar — 模型选择下拉 |
| `SelectModelAction` | `_toolbar_impl.py` | toolbar — 子菜单模型项 |
| `AgentStatusAction` | `_toolbar_impl.py` | toolbar — 状态 Widget |
| `MainMenuAction` | `_toolbar_impl.py` | toolbar — 主菜单（☰） |
| `OpenSettingsAction` | `_settings_drawer_impl.py` | settings — 面板入口 |
| `RefreshModelsAction` | `_settings_drawer_impl.py` | settings — 刷新模型 |
| `SendMessageAction` | `_chat_input_impl.py` | chat_input — 发送按钮 |
| `CancelMessageAction` | `_chat_input_impl.py` | chat_input — 停止按钮 |
| `SetSendModeChoiceAction` | `_chat_input_impl.py` | chat_input — 发送模式 |

### 辅助函数处理

`_actions_impl.py` 中有 3 个模块级辅助函数：

- `_conversation(context)` — 从 ActionContext 提取 Conversation View → 被 toolbar 和 settings Action 使用，需分别复制（或单独提取到一个小共助模块）
- `_chat_input(context)` — 从 ActionContext 提取 ChatInput View → 仅 chat_input Action 使用，直接移到 `_chat_input_impl.py`
- `_call_action(handler, *args)` — 通用 async 调用包装 → 三个目标文件各需一份

**建议**：不额外引入共助模块。三个辅助函数都很短（2-3 行），在每个目标文件各自定义一份私有版本即可，无实质维护成本。

### 清理动作

拆分完成后必须执行以下两项清理，否则会留下重复定义或死文件：

1. 删除 `src/mutagent/webui/__init__.py` 末尾的 `from . import _actions_impl  # noqa: E402,F401`。
2. 删除 `src/mutagent/webui/_actions_impl.py` 文件本身（不保留空 stub，避免误导）。

### 注册时机不变量

参考 `_settings_llm` 旁边那行注释——"ensure SettingsPanel subclasses are registered before SettingsDrawer instantiates"——Action 注册同样依赖"类定义先于 ActionRegistry 查询"。当前是通过 `webui/__init__.py` 末尾兜底 import `_actions_impl` 保障；拆分后改为依赖以下 import 链：

```
webui/__init__.py
  → from .toolbar import ...      → toolbar.py 末尾 from . import _toolbar_impl
  → from .chat_input import ...   → chat_input.py 末尾 from . import _chat_input_impl
  → from .settings import ...     → settings.py 末尾 from . import _settings_drawer_impl
```

**不变量**：Action 注册依赖 `webui` 包被 import。任何直接 `from mutgui import ActionRegistry` 而未 import `mutagent.webui` 的代码都拿不到这些 Action（现状如此，拆分不改变此行为）。后续若有人想精简各 `xxx.py` 末尾的 impl import，需先确认不会破坏此链路。

### 复杂度考量

- **MainMenuAction** 引用了 `SettingsDrawer` 的 `list_panels()` 方法，这意味着 `_settings_drawer_impl.py` 需要能从 Action 中访问到 SettingsDrawer 实例。当前依赖链是：`context.get("settings_drawer")` —— 由 `_conversation_impl.py` 在 `ActionContext.data` 中注入。所以 `MainMenuAction` 移到 `_toolbar_impl.py` 不产生额外依赖。
- **OpenSettingsAction** 同样通过 `context.get("settings_drawer")` 访问 drawer。移到 `_settings_drawer_impl.py` 后，它与 `SettingsDrawer` 在同一文件中，逻辑更内聚。

## 验收标准

1. **行为对等**：`python -m mutagent` 启动后，工具栏（模型选择、状态、☰ 菜单）、聊天输入框（发送 split 按钮+发送模式子菜单、停止按钮）、设置面板入口与刷新模型菜单项，全部出现且行为与拆分前一致。
2. **注册数量对等**：拆分前后 `ActionRegistry` 中 Action 类数量相同（9 个 Action 类，其中 `SelectModelAction` / `OpenSettingsAction` / `SetSendModeChoiceAction` 是动态实例化、不进 registry 的辅助类，只有 6 个静态 `action_id` 进 registry）。
3. **无残留**：`grep -rn "_actions_impl" mutagent/src` 仅命中无关的 egg-info / pycache（或全部清空）。
4. **既有测试通过**：`pytest mutagent/tests/test_webui_ui.py` 及全量测试无回归。

## 实施步骤清单
- [x] 将 4 个 Toolbar 相关 Action（ModelSelectorAction、SelectModelAction、AgentStatusAction、MainMenuAction）连同辅助函数 `_conversation`/`_call_action` 移到 `_toolbar_impl.py`
- [x] 将 3 个 ChatInput 相关 Action（SendMessageAction、CancelMessageAction、SetSendModeChoiceAction）连同辅助函数 `_chat_input`/`_call_action` 移到 `_chat_input_impl.py`
- [x] 将 2 个 Settings 相关 Action（OpenSettingsAction、RefreshModelsAction）连同辅助函数 `_conversation`/`_call_action` 移到 `_settings_drawer_impl.py`
- [x] 删除 `__init__.py` 末尾的 `from . import _actions_impl` 导入
- [x] 删除 `_actions_impl.py` 文件
- [x] 验证行为对等：工具栏/输入框/设置面板功能正常
- [x] 验证无残留：grep 确认 `_actions_impl` 引用已全部清空
- [x] 验证既有测试：`pytest mutagent/tests/test_webui_ui.py` 无回归

## 待定问题

（无——拆分方案简单明确，不涉及接口变更）
