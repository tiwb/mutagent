# MCP Schema 字段在 help() 中完整展示

**状态**：✅ 已完成
**日期**：2026-05-13
**类型**：功能设计
**前置**：`bugfix-mcp-optional-param-binding.md`

## 需求

`help(playwright.browser_console_messages)` 和 `help(playwright)` 展示的信息**与
MCP server 返回的原始 JSON Schema 之间存在字段丢失，用户看不到完整的参数约束。

### 信息丢失清单

以 @playwright/mcp 的 21 个 tool 为样本，统计受影响字段：

| MCP Schema 字段 | 丢失 | 影响 tool 数 | 典型示例 |
|---|---|---|---|
| `enum` | ❌ | 5 | `level`: error/warning/info/debug — 用户不知道合法值 |
| `items` (array 元素类型) | ❌ | 5 | `paths: array` — 不知道数组里装什么 |
| `minimum` / `maximum` | ❌ | 1 | `index: number` — 不知道范围 |
| `additionalProperties` | ❌ | 1 | `data: object` — 不知道 object 结构 |
| `propertyNames` | ❌ | 1 | `data` 的 key 约束不可见 |

### 当前 help 输出 vs 原始数据对比

**`browser_console_messages`**：

```
# 原始 JSON Schema
level:    {type: "string", default: "info", enum: ["error","warning","info","debug"]}
all:      {type: "boolean"}
filename: {type: "string"}
required: ["level"]

# 当前 help
playwright.browser_console_messages(level: 'str' = 'info', all: 'bool' = ..., filename: 'str' = ...)

Args:
    level: string (required) — ...   ← 缺 enum 列表
    all: boolean — ...
    filename: string — ...
```

**`browser_tabs`**：

```
# 原始
action: {type: "string", enum: ["list","select","new","close"]}
required: ["action"]

# 当前 help
    action: string (required) — ...  ← 不知道只有 4 个合法值
```

**`browser_file_upload`**：

```
# 原始
paths: {type: "array", items: {type: "string"}}

# 当前 help
    paths: array — ...               ← 不知道数组元素是 string
```

### 范围声明

本 spec 只覆盖**外部 MCP server → 本地 Python wrapper** 的单向展示翻译
（场景 A）。反向需求（本地 Python 函数 → 对外 MCP schema，场景 B）属于独立
设计问题，不在此文档内。

B 场景的简要判断：mutbot `MutbotTools` 14 个函数 / 25 个参数的样本统计显示，
实际用到的非 signature 约束仅 `enum` 一项（`level`、`config_set.key`、`role`
三处），其余 JSON Schema 关键字（min/max、pattern、format、items、嵌套 object
等）全部未出现。因此 B 场景的设计大概率不需要扩展 docstring 格式，而应约定
「想让约束反映到 schema 就用 Python typing（`Literal`、`Annotated`）表达，
docstring 保持纯描述」。待第一个真实需求出现时另起 spec。

## 设计方案

### 定位

- **docstring 唯一消费者是 `help()` 的人类读者**。原始 JSON Schema 已经通过
  `tool_func._mcp_input_schema` attr 保留，docstring 不承担可逆序列化职责。
- **不扩展 signature**。Python signature 继续只表达 `name / type / default /
  required`，不塞 `Literal[...]` 或 `Annotated[...]`——避免自动生成的 wrapper
  signature 因 enum 过长而不可读。
- **docstring 扩展也不重复 signature 已表达的信息**。特别是 `default` 不再
  追加到 Args 描述里（signature 已经可见 `= "info"`），避免双写漂移。
- **`format_param_description_suffix` 是公共翻译器，不是 docstring 专用**。
  它被设计为纯函数（输入 property dict，输出后缀字符串），当前的
  `_make_tool_func` 拼 docstring 只是第一个消费者。未来
  `SandboxApp.describe_function`（见 `refactor-namespace-describe-api.md`）
  若要在参数表格里按参数独立展示约束，**应直接再次调用本函数**（输入
  `tool_func._mcp_input_schema['properties'][name]`），而不是反向解析已拼接
  好的 doc 字符串。

  > 注：`refactor-namespace-describe-api.md` Q2 的原措辞「可能在
  > `format_callable_signature` 层面就把信息编码进签名字符串」是历史推测，本
  > spec 最终决定走 docstring 后缀路线。Refactor 文档的 Q2 已同步订正。

### 职责划分

| 信息 | 承载位置 |
|---|---|
| 参数名 | signature |
| 基础类型（`string` → `str` 等） | signature |
| 默认值 | signature |
| required 与否 | signature（无 default = required） |
| 文字描述 | docstring Args 首句 |
| 其余 schema 约束（enum / range / pattern / ...） | docstring Args 描述后缀 |
| 罕见 / 无法翻译的字段 | docstring 兜底句 + `_mcp_input_schema` attr |

### JSON Schema 字段 → docstring 翻译规则

**翻译产出一段「描述后缀」**，追加在 Args 行原 description 之后。每个关键字
产出一句，句末带英文句号，使用英文固定词汇（避免中英散文混排时风格漂移）。

| JSON Schema 字段 | 翻译为 | 示例 |
|---|---|---|
| `description` | Args 行首句 | 原样 |
| `enum` | `Allowed: a \| b \| c.` | `Allowed: error \| warning \| info \| debug.` |
| `const` | `Must be: X.` | `Must be: "v1".` |
| `minimum` / `maximum` | `Range: a..b.`（闭区间） | `Range: 0..100.` |
| `exclusiveMinimum` | `Range: > a.` | `Range: > 0.` |
| `exclusiveMaximum` | `Range: < b.` | `Range: < 1.` |
| `multipleOf` | `Multiple of: N.` | `Multiple of: 0.5.` |
| `minLength` / `maxLength` | `Length: a..b.` | `Length: 1..256.` |
| `pattern` | `Pattern: ^xxx$.` | `Pattern: ^[a-z_]+$.` |
| `format` | `Format: X.` | `Format: email.` |
| `items.type`（scalar） | `Items: X.` | `Items: string.` |
| `items.type === "object"` | `Each item is an object; see raw schema.` | — |
| `minItems` / `maxItems` | `Items count: a..b.` | `Items count: 1..10.` |
| `uniqueItems: true` | `Items must be unique.` | — |
| `additionalProperties: false` | `No extra keys.` | — |
| `additionalProperties.type` | `Extra values: X.` | `Extra values: string.` |
| `propertyNames.pattern` | `Keys match: ^xxx$.` | — |
| `propertyNames.minLength` / `maxLength` | `Keys length: a..b.` | — |

**单边 range 的 fallback**：若只写了 `minimum=0` 无 `maximum`（或反之），
产出 `Range: >= 0.` / `Range: <= 100.`（不绕回 `Range: 0..∞.`）。

**后缀渲染顺序**（多个约束并存时固定排列，便于眼睛扫描）：

```
Allowed / Must be → Range / Multiple of → Length → Pattern → Format →
Items → Items count → Items must be unique →
Extra values / No extra keys → Keys match / Keys length
```

### 嵌套与兜底

**第一版不做嵌套展开**。

- `items` 是 object → 只给 `Each item is an object; see raw schema.`，不递归
  列举 object 的 properties
- 参数本身是 object 且有 `properties` → 描述照旧，properties 子字段不展开
- `oneOf` / `anyOf` / `allOf` / `not` / `$ref` → 翻译器遇到**不在上表里**的
  顶层约束关键字时，在描述末尾追加一句：

  ```
  Additional constraints apply; see raw schema via tool._mcp_input_schema.
  ```

  这条兜底句每个参数最多出现一次。

理由：嵌套在真实 MCP server 里非常少见（playwright 21 tool 里 0 个），spec
第一版不优化这类场景；真遇到时机器可读数据已经在 `_mcp_input_schema` attr
里完整保留，人读完整 schema 也能兜住。

### 示例对比

**`browser_console_messages`**（在 `bugfix-mcp-optional-param-binding.md`
完成后的基础上新增 enum 展示）：

```
playwright.browser_console_messages(level: 'str' = 'info', all: 'bool' = ..., filename: 'str' = ...)

Args:
    level: string — 日志级别。Allowed: error | warning | info | debug.
    all: boolean — 是否返回所有消息。
    filename: string — 限定来源文件名。
```

**`browser_tabs`**：

```
Args:
    action: string — 标签页操作。Allowed: list | select | new | close.
```

**`browser_file_upload`**：

```
playwright.browser_file_upload(paths: 'list')

Args:
    paths: array — 本地文件路径列表。Items: string.
```

**假想的综合样例**（同时含多种约束）：

```
Args:
    index: number — 目标元素索引。Range: 0..100.
    email: string — 联系邮箱。Length: 5..128. Format: email.
    tags: array — 标签集合。Items: string. Items count: 1..10. Items must be unique.
    metadata: object — 附加数据。Keys match: ^[a-z_]+$. Extra values: string.
```

### `help(playwright)`（namespace 级视图）

保持现状，不在 namespace 级列表里追加约束展示。namespace 视图是"清单页"，
每行一函数签名 + 一行摘要足够；需要细节时用户会 `help(playwright.foo)` 点入
详情。

### 消费侧约定（与 Refactor 协同）

本 spec 的约束信息**只烧录到 docstring 字符串里**，不在 `ParamDescr` /
`FunctionDescr` 上新增结构化字段。Refactor 完成后，消费侧按以下约定使用：

| 展示形式 | 数据来源 |
|---|---|
| 人读段落（如 `help()` 文本输出） | `FunctionDescr.doc`（原样） |
| 按参数表格显示约束（如 Settings Panel 的参数行展开） | 调用 `format_param_description_suffix(tool._mcp_input_schema['properties'][name])` 按参数单独产出 |
| 机器可读完整 schema | `tool._mcp_input_schema`（attr 原样） |

**明确不做的事**：

- 不在 `ParamDescr` 上新增 `constraints_text` / `constraints` 等字段。约束属于
  schema 域的语义，强塞进 signature 派生的 `ParamDescr` 会让该结构职责不清。
- 不让消费者反向解析 doc 字符串去抽取约束片段。
- 不在 `FunctionDescr.extensions` 里缓存约束翻译结果。原始 schema 本身即是最
  权威的结构化数据源，按需调翻译器即可，不需要中间缓存。

## 实施方案

### 改动点

1. **新增** `src/mutagent/sandbox/_signature.py::format_param_description_suffix`

   纯函数，输入 JSON Schema property dict，输出「描述后缀字符串」（可能为空）。
   不访问外部状态，便于单测穷举。

2. **修改** `src/mutagent/sandbox/_adapter_mcp.py::_make_tool_func` 的 docstring
   生成逻辑（现 811-827 行）

   - 保留现有 `f"    {pname}: {ptype}{req_mark} — {pdesc}"` 作为基线
   - 调用 `format_param_description_suffix(pinfo)` 得到后缀
   - 后缀非空则拼到 `pdesc` 之后
   - 去除 `req_mark` 里已有的 `(required)` 标记（可选，见下）

3. **不改** `mcp_schema_to_specs`。signature 构造与展示扩展解耦。

### `(required)` 标记去留

当前 docstring 里对 `name in required and not has_schema_default` 的参数追加
`(required)` 标记。这个信息 signature 已经表达（无 default），与「docstring
不重复 signature 已表达信息」原则冲突，**最终要移除**。

**但移除时机必须等到 Settings Panel `_fn_detail` 完成消费侧迁移
（`refactor-namespace-describe-api.md` R3）之后**。原因：
Settings Panel 的 `_fn_detail` 早期版本从 docstring 字符串里识别
`(required)` 字样做视觉强调。R3 已删除手拼 `Parameters:` 段，消费者
直接读 `inspect.signature(fn)` / `fn._mcp_input_schema['required']`，不再依赖
`(required)` 文案。此前置在本 Refactor R3 完成后已满足，Phase 2b 可启动。

因此拆成两步：

- **Phase 2a**（本 spec 主体，可立即落地）：接入 docstring 约束后缀，**保留
  `(required)` 标记**原样不动。
- **Phase 2b**（本 Refactor R3 完成后可做）：消费者已不再依赖 docstring 的
  `(required)` 文案，直接读
  `fn._mcp_input_schema['required']` / `inspect.signature(fn)` 即可；回头
  删掉 `(required)` 文案，跟 `default` 的处理逻辑对齐。

### 新函数 API

```python
def format_param_description_suffix(
    pinfo: Mapping[str, Any],
) -> str:
    """把 JSON Schema property 的约束字段翻译成 docstring 描述后缀。

    输入一个 property dict（MCP `input_schema.properties.<name>`），输出一段
    英文后缀字符串，形如 ``"Allowed: a | b. Range: 0..100."``。无约束时返回
    空串。

    翻译规则见 spec ``feature-mcp-schema-help-display.md``。本函数不处理
    `description`/`type`/`default`/`required`，这些由 signature 和 docstring
    基线格式承担。

    不认识的顶层关键字（``oneOf``/``anyOf``/``allOf``/``$ref``/``not`` 等）
    触发兜底句 ``"Additional constraints apply; see raw schema via
    tool._mcp_input_schema."``。
    """
```

返回字符串风格约定：

- 每子句独立完整，以英文句号结尾
- 子句间用单个空格分隔
- 整体前后不带空格
- 上层调用方负责在前面加一个空格（拼到 `pdesc` 之后）

### 实施步骤清单

**Phase 1：核心翻译器（纯函数 + 单测）**

- [x] 在 `_signature.py` 新增 schema 约束翻译纯函数（输入 property dict，输出英文后缀字符串）
- [x] 实现翻译规则表中全部关键字的 docstring 后缀翻译
- [x] 固定多约束并存时的子句排列顺序
- [x] 未知顶层关键字触发兜底提示
- [x] 单边 range 转为 `>=` / `<=` 形式
- [x] 新增翻译器单元测试（穷举各约束类型、组合、边界、兜底、顺序稳定性）
- [x] 样本回归：playwright 21 个 tool 翻译产物人工 review

**Phase 2a：接入 docstring 生成（保留 `(required)`）**

- [x] 在 `_adapter_mcp.py` 的 `_make_tool_func` 中接入翻译器，追加约束后缀到 Args 描述行
- [x] 保留 `(required)` 标记追加逻辑不动（留给 Phase 2b）
- [x] 新增集成测试（构造含约束的 input_schema → 生成 wrapper → 断言 docstring 含预期后缀）
- [x] 回归现有 MCP adapter 测试
- [x] 手工验收 3 个典型 tool 的 `help()` 输出
- [x] iter1 修正：约束后缀独立缩进 8 空格（四分支格式）
- [x] iter1 修正：`_MISSING` sentinel 跨 sharing 身份保留（`default_missing` 标记位）

**Phase 2b：`(required)` 清理（前置已满足，可启动）**

- [x] 前置确认：Settings Panel `_fn_detail` 已完成消费侧迁移（本文未继续，已由 refactor-namespace-describe-api R3 交付）
- [x] 移除 `_make_tool_func` 里 `(required)` 标记追加逻辑（由 iter2 实施一并完成）
- [x] 更新 Phase 2a 集成测试断言（iter2 重写了全套测试）
- [x] 手工 review Settings Panel 和 `help()` 输出（人工确认通过）

**依赖关系**

- Phase 1 无外部依赖，可立即开工。
- Phase 2a 在 Phase 1 合并后即可开工。
- Phase 2b 前置 “Settings Panel `_fn_detail` 已完成消费侧迁移” 已由 `refactor-namespace-describe-api.md` R3 交付，可启动。

## 关键参考

- `src/mutagent/sandbox/_adapter_mcp.py:799-856` — `_make_tool_func`，本次扩展
  docstring 生成逻辑的位置
- `src/mutagent/sandbox/_signature.py:82-115` — `mcp_schema_to_specs`，本次不动
- `src/mutagent/sandbox/_signature.py:63-75` — `json_type_to_annotation`，本次
  不动（不引入 `list[str]` / `Literal[...]` 等 signature 层扩展）
- `docs/specifications/bugfix-mcp-optional-param-binding.md` — 前置 bugfix
  （`_MissingSentinel` 机制）
- `docs/specifications/refactor-wrapper-faithful-signature.md` — 前置重构
  （真签名挂 `__signature__`）
- MCP 协议 schema：https://raw.githubusercontent.com/modelcontextprotocol/specification/main/schema/2025-11-25/schema.ts
  - `Tool.inputSchema` 定义为 `{type: "object", properties?: {...}, required?: string[]}`
  - properties 值遵循 JSON Schema 2020-12

## 迭代

- **iter1**：`feature-mcp-schema-help-display.iter1.md`（✅ 已完成）— 验收反馈与修正（方案 B 独立缩进行、`_MISSING` sentinel 丢失、上游 default 双写）
- **iter2**：`feature-mcp-schema-help-display.iter2.md`（✅ 已完成）— 方向性重写。与 mutio 协商后改为三段式渲染：signature 承担 `Literal[...]` enum、Args 段仅 `name: description.`、所有约束字段原词进 `Annotations:` 段以 JSON 透传。完全取代了本 spec 的英文约束翻译路线，同时完成了 Phase 2b 的 `(required)` 清理。本 spec 中关于 iter2/iter3 的原始描述（Google-style 参数头对齐、约束行规则化）不再适用。
