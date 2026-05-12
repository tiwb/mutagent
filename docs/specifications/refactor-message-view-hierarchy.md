# 消息体系重构：ChatItemView 基类 + 类型驱动映射 — 设计规范

**状态**：✅ 已完成
**日期**：2026-05-12
**类型**：重构

## 需求

1. `ToolCallCard` 和其他消息项 View（`UserMessage`、`AssistantMessage` 等）是同层级的 ChatItem 渲染器，但声明文件独立在 `tool_call.py` 中，与其他消息项不一致
2. 没有统一的 "ChatItem 渲染器" 基类，`_MessageListAdapter.create_item_view()` 靠手动 `isinstance` 链分派，每加一种消息类型都要改一处中心代码
3. 重构目标：
   - 声明集中到 `messages.py`
   - 实现集中到 `_messages_impl.py`（一个文件足够）
   - 加 `ChatItemView` 基类，**用 dataclass 类型而非 kind 字符串**做 ChatItem→View 自动映射
   - 上层项目新增消息类型时，只需新增 `ChatItem` 子类 + `ChatItemView` 子类，无需改 mutagent 代码

## 关键参考

- `src/mutagent/webui/messages.py` — 当前 ChatItem 数据类 + MessageList/UserMessage 等 View 声明
- `src/mutagent/webui/tool_call.py` — ToolCallCard 声明文件（合并到 `messages.py` 后删除）
- `src/mutagent/webui/_messages_impl.py` — messages View 实现 + `_MessageListAdapter`
- `src/mutagent/webui/_tool_call_impl.py` — ToolCallCard 实现（合并到 `_messages_impl.py` 后删除）
- `src/mutagent/webui/_blocks_impl.py` — `BlockRenderer`/`ThinkingBlock` 实现（不变，参见下文"blocks 模块不变"）
- `src/mutagent/webui/__init__.py` — 公开导出
- `src/mutobj/core.py` — `discover_subclasses` / `get_registry_generation` 机制
- `src/mutgui/src/mutgui/action.py` — `discover_subclasses` 使用范例

## 设计方案

### 新组织方式

```
messages.py（声明集中）           _messages_impl.py（实现集中）
─────────────────────────        ────────────────────────────
ChatItemView(View) 基类           ← @impl ─
  ├── UserMessage                 ← @impl ─
  ├── AssistantMessage            ← @impl ─
  ├── AssistantError              ← @impl ─
  ├── TurnSeparator               ← @impl ─
  └── ToolCallCard                ← @impl ─

MessageList(View)                 ← @impl ─
```

文件总数从 4 减到 2。

### ChatItemView 基类（核心）

```python
from typing import ClassVar

class ChatItemView(View):
    """所有 ChatItem 渲染器的统一基类。

    子类声明 ``item_type`` 绑定到具体 ChatItem 数据类，
    ``ChatItemView.for_item()`` 据此自动分派 View 类。
    """
    item: ChatItem
    item_type: ClassVar[type[ChatItem]] = ChatItem  # 子类必须覆盖

    def __init__(self, *, item: ChatItem) -> None: ...

    def render(self) -> ViewBlock: ...

    @classmethod
    def for_item(cls, item: ChatItem) -> "ChatItemView": ...
```

### 类型驱动的映射机制

**键是 `type[ChatItem]` 而非字符串**，IDE 可跳转、mypy 能查、不可能拼写错。

```python
# _messages_impl.py 模块级缓存
_view_map_cache: dict[type[ChatItem], type[ChatItemView]] = {}
_view_map_generation: int = -1

def _resolve_view_class(item_type: type[ChatItem]) -> type[ChatItemView]:
    """按 ChatItem 类型查 ChatItemView 子类，按 registry generation 失效缓存。"""
    global _view_map_generation
    gen = mutobj.get_registry_generation()
    if gen != _view_map_generation:
        _view_map_cache.clear()
        for view_cls in mutobj.discover_subclasses(ChatItemView):
            registered = view_cls.item_type
            if registered is ChatItem:
                continue  # 未覆盖默认值的子类跳过
            if registered in _view_map_cache:
                raise RuntimeError(
                    f"Duplicate ChatItemView for {registered.__name__}: "
                    f"{_view_map_cache[registered].__name__} vs {view_cls.__name__}"
                )
            _view_map_cache[registered] = view_cls
        _view_map_generation = gen
    try:
        return _view_map_cache[item_type]
    except KeyError:
        raise TypeError(f"No ChatItemView registered for {item_type.__name__}") from None
```

要点：
- 用 `mutobj.get_registry_generation()` 失效缓存，**首次访问 + 注册表变更后**重建一次，其余调用 O(1)
- 重复绑定（两个 View 都声明同一 `item_type`）在重建时即报错
- 上层项目（mutbot 等）通过新增 `ChatItem` + `ChatItemView` 子类即可扩展，零改动接入

### `_MessageListAdapter` 简化

```python
class _MessageListAdapter(VirtualListItemAdapter):
    items: list[ChatItem] = mutobj.field(default_factory=list)

    def create_item_view(self, index: int) -> View:
        return ChatItemView.for_item(self.items[index])
```

不再有 `isinstance` 链，新增消息类型完全不需要改这里。

### item_type 分配

| ChatItem dataclass | ChatItemView 子类 |
|---|---|
| `UserTextItem` | `UserMessage` |
| `AssistantTextItem` | `AssistantMessage` |
| `AssistantErrorItem` | `AssistantError` |
| `TurnSeparatorItem` | `TurnSeparator` |
| `ToolCallItem` | `ToolCallCard` |

每个子类声明形如：
```python
class UserMessage(ChatItemView):
    item_type: ClassVar[type[ChatItem]] = UserTextItem
    item: UserTextItem  # 覆盖父类字段，更精确的类型标注

    def __init__(self, *, item: UserTextItem) -> None: ...
    def render(self) -> ViewBlock: ...
```

### `ChatItem.kind` 字段的去留

保留。`kind`（`"user.text"` 等字符串）服务于**前端识别 / 持久化序列化**，与 Python 内部的 View 分派职责正交。本次重构不动它。

### 文件变更清单

| 文件 | 操作 |
|---|---|
| `messages.py` | 新增 `ChatItemView` 基类 + 各子类声明；`ToolCallCard` 声明合并入；各 View 子类改为继承 `ChatItemView` 并声明 `item_type` |
| `_messages_impl.py` | 合并 `_tool_call_impl.py` 全部内容；新增 `_resolve_view_class` 类型驱动映射缓存 + `ChatItemView.for_item` 实现；`_MessageListAdapter.create_item_view` 改为单行 `ChatItemView.for_item(...)`；移除原 `from .tool_call import ToolCallCard` |
| `tool_call.py` | **删除** |
| `_tool_call_impl.py` | **删除**（内容合并到 `_messages_impl.py`） |
| `__init__.py` | `ToolCallCard` 改为从 `.messages` 导出；同时导出新 `ChatItemView` 基类 |

### blocks 模块不变

`BlockRenderer` 和 `ThinkingBlock` 是**内容渲染器**（嵌套在 `AssistantMessage` 内部渲染 markdown / 思考块），不参与 ChatItem→View 映射。它们不继承 `ChatItemView`，层级不同，保持独立文件合理。

## 与现有 mutobj 理念的契合

- **Declaration-Implementation 分离**：`ChatItemView` 及子类都是 Declaration，实现走 `@impl` ✓
- **子类发现，零注册**：用 `discover_subclasses(ChatItemView)`，无手写注册表 ✓
- **可选功能 = Declaration 子类**：上层项目新增消息类型 = 新增 `ChatItem` + `ChatItemView` 子类，import 即生效 ✓
- **依赖方向单向**：mutagent 提供基类，mutbot 等可扩展，但 mutagent 不感知 ✓

## 待定问题

- **泛型化（`class ChatItemView[T: ChatItem]`）**：理想上 `item: T` 可由泛型参数推断，子类无需重复声明 `item: UserTextItem`。但 mutobj 的 `DeclarationMeta` 与 `Generic` 叠加历史上容易踩坑，本次**不引入**；若后续在 mutobj 侧验证可行，再单独提规范。

## 实施步骤清单

- [x] 在 `messages.py` 新增 `ChatItemView` 基类与类型驱动映射机制
  - [x] 声明 `ChatItemView(View)`：`item_type: ClassVar[type[ChatItem]]`、`item: ChatItem`、`__init__`、`render`、`for_item` classmethod
  - [x] 新增模块级 `_resolve_view_class`（基于 `get_registry_generation` 失效缓存，重复绑定时报错）

- [x] 合并 `tool_call.py` 到 `messages.py`
  - [x] `ToolCallCard` 声明并入 `messages.py`，继承 `ChatItemView`，声明 `item_type = ToolCallItem`
  - [x] 现有 `UserMessage`/`AssistantMessage`/`AssistantError`/`TurnSeparator` 改继承 `ChatItemView`，各自声明 `item_type`
  - [x] 删除 `tool_call.py`

- [x] 合并 `_tool_call_impl.py` 到 `_messages_impl.py`
  - [x] `_tool_call_impl.py` 全部内容（包括辅助函数、样式常量）合并入 `_messages_impl.py`
  - [x] 实现 `ChatItemView.__init__` / `render` / `for_item`（`for_item` 委托给 `_resolve_view_class`）
  - [x] `_MessageListAdapter.create_item_view` 简化为 `return ChatItemView.for_item(self.items[index])`，移除 `isinstance` 链
  - [x] 移除 `from mutagent.webui.tool_call import ToolCallCard` 导入
  - [x] 删除 `_tool_call_impl.py`

- [x] 更新 `webui/__init__.py` 公开导出
  - [x] `ToolCallCard` 改为从 `.messages` 导出
  - [x] 新增导出 `ChatItemView`
  - [x] 移除 `from .tool_call import ToolCallCard`

- [x] 验证
  - [x] `pytest`（mutagent 现有测试全过：788 passed, 4 skipped）
  - [x] 用户手动启动 `python -m mutagent` 走 web demo，验证：用户/助手气泡、错误气泡、轮次分隔线、工具卡片、thinking 折叠块均正常渲染
  - [x] 故意删掉某 `ChatItemView` 子类的 `item_type`，确认 `_resolve_view_class` 在该类型 item 出现时抛 `TypeError`（已用临时 `UnknownItem` 验证 `No ChatItemView registered for UnknownItem`）

## 实施过程踩坑

- **基类 `__init__` 必填参数与子类 super 调用冲突**：原子类 `__init__` 写法是 `super(SubClass, self).__init__()`（不传参，直接走 `View.__init__()`）+ `self.item = item`。新增 `ChatItemView` 基类后，基类 `__init__` 实现要求 `item` 必填，子类不传就 `TypeError`。
  - 修复：基类 `__init__` 实现负责 `super(View) + self.item = item`，所有子类 `__init__` 实现改为 `super(SubClass, self).__init__(item=item)`，不再各自写 `self.item = item`。基类承担公共赋值职责，子类只补自己的额外初始化（如 `AssistantMessage._renderer`）。
