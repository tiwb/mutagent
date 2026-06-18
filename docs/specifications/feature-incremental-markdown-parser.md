# 增量 Markdown 解析：标题拆分 + Fenced Code Block 支持

**状态**：✅ 已完成
**日期**：2026-06-17
**类型**：功能设计

## 需求

1. 流式场景下对 LLM 输出的 markdown 文本做增量解析，按标题（`#` 系列）和 fenced code block（` ``` `）切分为多个 VirtualList item
2. 标题拆分后支持按级别折叠 section（后端控制 item 可见性）
3. Fenced code block 作为独立 item，后续可扩展语法高亮、复制按钮等专用组件
4. 表格、段落内格式（粗体/斜体/链接/行内代码）留给前端 markdown 渲染器处理
5. 保持现有 `MessageList.items` 为扁平 list 不变

## 关键参考

- `mutagent/src/mutagent/webui/_messages.py` — ChatItem / ChatItemView 体系、MessageList
- `mutagent/src/mutagent/webui/_conversation.py` — handle_agent_event 流式处理、rebuild_items_from_messages
- `mutagent/src/mutagent/webui/_blocks.py` — BlockRenderer 当前 markdown 解析（将被替换）
- `mutagent/src/mutagent/core/messages.py` — Message / ContentBlock / StreamEvent 模型
- `mutgui/src/mutgui/virtual_list.py` — VirtualList / VirtualListItemAdapter 接口
- `docs/specifications/feature-chat-components.md` — 前端组件化（前置依赖，已完成）

## 设计方案

### 一、解析粒度

只识别两种 block-level 断点。其他全部留给前端 markdown 渲染器。

| 模式 | 行为 | 理由 |
|------|------|------|
| `^#{1,6}\s` 行首 | **拆分** → 创建 SectionHeadingItem | 支持折叠/跳转；缩短流式 item 长度 |
| ` ``` ` 行首 | **拆分** → 创建 CodeBlockItem | 独立样式控制，后续可扩展专用组件 |
| 表格 `|...|` | **不拆分** | 前端 markdown 渲染器 + `remark-gfm` 处理 |
| `\n\n` 段落边界 | **不拆分** | CSS margin 间距足够 |
| `>` 引用、`-` 列表 | **不拆分** | 属于文本段落内部 |

> 所有级别标题（`#` ~ `######`）同等对待，全部拆分。

### 二、新增 ChatItem 类型

AssistantTextItem 当前承载全部文本，BlockRenderer 内部做简单分段。替换为三种新 item：

```python
@dataclass(slots=True)
class SectionHeadingItem(ChatItem):
    """标题 item。text 为标题文本（不含 # 前缀），level 为 1~6。"""
    text: str
    level: int
    collapsed: bool = False
    turn_id: str = ""

@dataclass(slots=True)
class MarkdownItem(ChatItem):
    """文本内容块，由前端 markdown 渲染器处理。"""
    text: str

@dataclass(slots=True)
class CodeBlockItem(ChatItem):
    """Fenced code block 内容块。"""
    code: str
    language: str = ""
```

### 三、增量解析器

```python
class IncrementalMarkdownParser:
    """流式增量 markdown 解析器。

    逐行消费，状态机跟踪当前是否在 code fence 内。
    产出 Section 列表（内部使用，不直接暴露为 ChatItem）。
    """

    def feed(self, new_text: str) -> list[ChatItem]:
        """追加文本，返回当前可见 ChatItem 列表（供 diff 用）。"""
```

#### 状态机

```
状态: PARAGRAPH | FENCE
初始: PARAGRAPH

逐行处理：
  PARAGRAPH:
    行匹配 ^#{1,6}\s
      → flush 当前 buffer 为 MarkdownItem
      → 创建新 section，记录标题及 level
    行匹配 ^```
      → flush 当前 buffer
      → 状态 → FENCE
    其他行
      → 追加到 buffer

  FENCE:
    行匹配 ^```
      → flush fence_buffer 为 CodeBlockItem
      → 状态 → PARAGRAPH
    其他行
      → 追加到 fence_buffer
```

#### `_flush_text` 优化

去头尾空行，纯空白内容跳过不产出 MarkdownItem：

```python
def _flush_text(self) -> None:
    if not self._text_buffer:
        return
    text = "\n".join(self._text_buffer).strip()
    if not text:
        self._text_buffer = []
        return
    # ... 产出 MarkdownItem
```

`.strip()` 仅去除头尾空行，段落间的 `\n\n` 保留，不影响多段落语义。

#### 末闭合处理

流式结束时（`response_done` 之后）调用 `finalize()`：
- 在 PARAGRAPH 状态且 buffer 非空 → 产出最后一个 MarkdownItem
- 在 FENCE 状态 → fence 内容合并到 buffer（当作未闭合的代码块，降级为文本）

### 四、Section 扁平结构

Section 是解析器内部结构，不暴露为 ChatItem。它记录标题间的层级关系，用于折叠逻辑。

```python
@dataclass
class _Section:
    id: str
    title: str             # 标题文本
    level: int             # 1 ~ 6
    parent_id: str | None  # 父 section id
    collapsed: bool = False
```

`parent_id` 构建规则：新增 section 时，往已存在的 section 列表中找 level 小于当前 level 的最近一个 section。

#### 可见性计算

折叠一个 section 后，其所有子孙 section 不出现：

```python
def visible_items(self) -> list[ChatItem]:
    result = []
    skip_above: int | None = None
    for s in self.sections:
        if skip_above is not None and s.level > skip_above:
            continue
        skip_above = None
        result.append(SectionHeadingItem(...))
        result.extend(s.chunks)
        if s.collapsed:
            skip_above = s.level
    return result
```

### 五、流式路径改动

`handle_agent_event` 中文本处理从单 item 追加改为解析器驱动的多 item 管理：

```python
# text_delta 时
elif event.type == "text_delta" and event.text:
    new_items = self.parser.feed(event.text)
    _sync_items(self.message_list, new_items)

# response_done 时
elif event.type == "response_done":
    new_items = self.parser.finalize()
    _sync_items(self.message_list, new_items)
```

`_sync_items` 对比新旧列表，新增项 `append_item`，末尾项内容变化则 `invalidate_item`。ToolUse/ToolResult/ThinkingBlock 路径不变。

### 六、折叠交互（已废弃）

> **决策**：折叠按钮已在实施中移除。解析器基础架构（`toggle_collapse`、`visible_items` 的
> `skip_above` 逻辑）仍保留在代码中，后续需要时只需恢复前端按钮即可启用。

折叠是后端控制的操作，不是前端 CSS 隐藏。因为折叠会移除子 item（可能跨多个 VirtualList item），前端无法感知层级关系。

### 七、前端渲染

三个新组件各有一个配套 CSS 文件，统一样式管理。所有组件使用 className + CSS 文件模式（与 ThinkingBlock / ToolCallCard 一致），不设内联 style。

样式通过一系列 `--mutagent-*` CSS 变量暴露定制入口：

| 变量 | 默认值 | 用途 |
|------|--------|------|
| `--mutagent-heading-color` | `#f0e68c` | 标题颜色 |
| `--mutagent-content-color` | `var(--mutgui-text)` | 正文颜色 |
| `--mutagent-content-border` | `var(--mutgui-border)` | blockquote 左侧竖线颜色 |
| `--mutagent-code-color` | `var(--mutgui-text)` | 行内代码 / 代码块文字颜色 |
| `--mutagent-code-inline-color` | `oklch(0.75 0.15 160)` | 行内代码颜色（浅绿色） |
| `--mutagent-code-fence-color` | `var(--mutgui-text-dim)` | ` ``` ` 标记颜色 |
| `--mutagent-hr-color` | `var(--mutgui-border)` | 分割线颜色 |

#### SectionHeading 组件（`SectionHeading.tsx` + `SectionHeading.css`）

接收 `text`（标题文本）和 `level`（1~6）两个 prop。

- 字号统一 `--mutagent-font-size-base`，与正文一致
- 各级标题仅靠 `font-weight` 区分（L1~L2: 700, L3~L6: 600）
- `#` 前缀由 CSS `::before` 生成（`content: '##'`），`margin-right: 0.4em` 间隔
- 缩进统一 `padding: 6px 12px 2px`（左右对称 12px）
- 颜色使用 `--mutagent-heading-color`（默认淡黄色 `#f0e68c`），可全局覆盖或按 `--levelN` class 单独覆盖
- 不包含折叠按钮、不做行内 markdown 渲染

#### MarkdownContent 组件（`MarkdownContent.tsx` + `MarkdownContent.css`）

使用 `react-markdown` + `remark-gfm` 做 GFM markdown 渲染。

- 缩进统一 `padding: 0 12px`（左右对称 12px）
- 颜色使用 `--mutagent-content-color`（默认 `var(--mutgui-text)`）
- 不使用 `white-space: pre-wrap`，避免元素间空白被渲染为可见空行
- 内联 code 和 pre 样式全部通过 CSS 后代选择器控制：
  - `code`：等宽字体 + `--mutagent-code-inline-color` 浅绿色
  - `blockquote`：`margin: 0`，左侧 `border-left: 2px` 竖线（颜色 `--mutagent-content-border`），竖线位置与标题文字对齐
  - `ol/ul`：缩进 `padding-left: 1.2em`，间距 `margin: 0.4em 0`
  - `p`：间距 `margin: 0.3em 0`
  - `table/th/td`：`border-collapse`，1px 边框，表头浅色背景
  - `hr`：单行 1px 细线（颜色 `--mutagent-hr-color`），间距 `margin: 0.8em 0`
  - `pre`：作为 fenced code block fallback，保留 `white-space: pre-wrap` 和边框背景

#### CodeBlock 组件（`CodeBlock.tsx` + `CodeBlock.css`）

纯文本风格，无边框无背景。

- 缩进统一 `padding: 0 12px`
- 上下 ``` 标记和语言名是独立 `<span>` 元素：
  - `mutagent-code-block__fence-mark`：` ``` ` 符号
  - `mutagent-code-block__fence-lang`：语言名（如 `python`）
- fence 行颜色 `--mutagent-code-fence-color`（默认 `--mutgui-text-dim`），间距 `margin: 4px 0`
- 代码正文颜色 `--mutagent-code-color`（默认 `--mutgui-text`），等宽字体，`white-space: pre-wrap`

### 八、Python 端文件改动范围

| 文件 | 改动 |
|------|------|
| `_messages.py` | 新增 `SectionHeadingItem`、`MarkdownItem`、`CodeBlockItem` 三个 dataclass；新增对应的三个 View 类（`SectionHeadingView`、`MarkdownView`、`CodeBlockView`）；`AssistantTextItem` 和 `AssistantMessage` 保留（兼容历史 session），但流式路径不再创建；`SectionHeadingView.render` 不含折叠按钮 |
| `_conversation.py` | `handle_agent_event` 文本处理分支改用 `IncrementalMarkdownParser`；`rebuild_items_from_messages` 非流式路径调用 `IncrementalMarkdownParser.parse_complete()` |
| `_markdown_parser.py`（新文件） | `IncrementalMarkdownParser` 类：状态机、Section 管理、可见性计算；`_flush_text` 增加 `.strip()` 去头尾空行、跳过纯空白 MarkdownItem |

### 九、前端改动范围

| 文件 | 改动 |
|------|------|
| `SectionHeading.tsx`（新） | 标题组件：按 level 渲染 className，`#` 前缀由 CSS `::before` 生成 |
| `SectionHeading.css`（新） | 标题样式：统一字号、统一缩进、per-level font-weight + `::before` |
| `MarkdownContent.tsx`（新） | 文本内容组件：`react-markdown` + `remark-gfm` 渲染 GFM markdown |
| `MarkdownContent.css`（新） | 正文样式：blockquote 竖线、table/list/p/hr 全套，变量体系 |
| `CodeBlock.tsx`（新） | 代码块组件：fence span 元素渲染 ` ``` ` 和语言名 |
| `CodeBlock.css`（新） | 纯文本风格：fence 行 + 代码，无边框无背景 |
| `index.tsx` | 注册三个新组件（`SectionHeading`、`MarkdownContent`、`CodeBlock`）到 `mutagent` 命名空间 |

新增依赖：`react-markdown`、`remark-gfm`。

### 十、不做的事

- **不改变 VirtualList 接口** — adapter 和 VirtualList 不变
- **不做完整 CommonMark 解析** — 只识别行首 `#{1,6}` 和 ` ``` `
- **折叠 UI 暂不启用** — 解析器折叠基础设施（`toggle_collapse` / `skip_above`）已实现，前端按钮未暴露，后续按需恢复
- **SectionHeading 不做行内 markdown 渲染** — 标题文本中的 `**粗体**` 等不做解析，纯文本显示
- **不做 TOC 侧边栏** — 折叠功能是本 spec 的范围，目录导航后续迭代
- **CodeBlockItem 不做语法高亮** — 仅语言名 + 等宽字体，高亮后续迭代

## 决策记录

### 非流式路径（rebuild_items_from_messages）

对完整 text 做一次 `parse_complete()` 拆分，使历史 session 与新对话保持一致的视觉呈现。`parse_complete()` 与流式的 `feed()` 是同一个状态机，只是输入是完整文本一次传入。

### 折叠状态持久化

不持久化折叠状态。session 恢复时默认全部展开。折叠是浏览操作而非数据操作，类似浏览器滚动位置，丢了无伤大雅。后续需要时再加。

### 前端 markdown 渲染库

使用 `react-markdown` + `remark-gfm`。`remark-gfm` 提供 GFM 扩展（表格、任务列表、删除线等），与 `react-markdown` 的 `allowedElements` 配合使用。fenced code block 已由 CodeBlockItem 独立渲染。

### 样式策略

所有组件使用 className + 独立 CSS 文件，不设内联 style。样式变量遵循 `--mutagent-*` 命名空间，通过 `var()` fallback 链支持全局和局部覆盖。`--mutgui-*` 核心 token（来自 mutgui 主题层）作为 fallback 默认值。

### MarkdownContent 空白处理

`white-space: pre-wrap` 在块级元素容器上会导致元素间换行空白被渲染为可见空行。改为默认 `normal`，让 react-markdown 产出的 HTML 元素自然排版。仅 `<pre>` fallback 保留 `pre-wrap`。

### `_flush_text` 空行优化

解析器 flush 时 `.strip()` 去头尾空行，跳过纯空白 MarkdownItem。中间段落间的 `\n\n` 保留，不影响多段落语义。头尾空行来自标题行与正文之间的排版空白，对内容无意义。

## 实施步骤清单

- [x] 1. 新增 `SectionHeadingItem`、`MarkdownItem`、`CodeBlockItem` 三个 dataclass 到 `_messages.py`，以及对应的三个 `ChatItemView` 子类和 View 实现（`SectionHeadingView`、`MarkdownView`、`CodeBlockView`）
- [x] 2. 新增 `MessageList.remove_items`、`begin_turn`、`end_turn`、`sync_from_parser`、`toggle_section_collapse` 方法
- [x] 3. 新建 `_markdown_parser.py`：实现 `IncrementalMarkdownParser` 类（状态机、Section 管理、可见性计算、`feed()` / `parse_complete()` / `finalize()` / `toggle_collapse()`），含 `.strip()` 空行优化
- [x] 4. 改动 `_conversation.py`：`response_start`/`text_delta`/`response_done` 改用解析器驱动；`turn_done` 调用 `end_turn`；`rebuild_items_from_messages` 改用 `parse_complete()`；移除 `_ensure_current_assistant` 和 `_extract_text`；移除折叠回调
- [x] 5. 安装 `react-markdown`、`remark-gfm` 到前端
- [x] 6. 新建 `SectionHeading.tsx` + `SectionHeading.css`：标题组件，`#` 前缀 CSS `::before` 生成，统一字号和缩进，per-level font-weight
- [x] 7. 新建 `MarkdownContent.tsx` + `MarkdownContent.css`：文本内容组件，`react-markdown` + `remark-gfm`，blockquote/table/list/p/hr/pre 全套 CSS
- [x] 8. 新建 `CodeBlock.tsx` + `CodeBlock.css`：纯文本风格，fence span 渲染 ` ``` ` 和语言名
- [x] 9. 在 `index.tsx` 注册三个新组件（`SectionHeading`、`MarkdownContent`、`CodeBlock`）
- [x] 10. 保留 `AssistantTextItem` / `BlockRenderer` / `AssistantMessage` 兼容旧 session，流式路径不再创建它们
