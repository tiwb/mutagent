# MCP 可选参数无默认值导致 TypeError

**状态**：✅ 已完成
**日期**：2026-05-13
**类型**：Bug修复
**前置**：`refactor-wrapper-faithful-signature.md`（已完成）

## 需求

### 问题现象

`playwright.browser_console_messages()` 调用直接抛 TypeError：

```
TypeError: missing a required argument: 'all'
```

但 docstring 明确说 `all` **不需要传**（`Defaults to false`），schema 也把它
放在 `required` 列表之外。

### 影响范围

playwright namespace 共 23 个 tool，**17 个**受此问题影响。任意一个 tool 有
至少一个「不在 required 列表中、也没有 default 字段」的参数，调用时必须把
这些参数全传齐——尽管 MCP 协议层完全允许省略它们。

受影响 tool 及参数数：

| tool | optional-no-default 参数数 |
|------|---------------------------|
| browser_click | 4 (element, doubleClick, button, modifiers) |
| browser_console_messages | 2 (all, filename) |
| browser_drag | 2 (startElement, endElement) |
| browser_drop | 3 (element, paths, data) |
| browser_evaluate | 3 (element, target, filename) |
| browser_file_upload | 1 (paths) |
| browser_handle_dialog | 1 (promptText) |
| browser_hover | 1 (element) |
| browser_network_request | 2 (part, filename) |
| browser_network_requests | 2 (filter, filename) |
| browser_run_code_unsafe | 2 (code, filename) |
| browser_select_option | 1 (element) |
| browser_snapshot | 4 (target, filename, depth, boxes) |
| browser_tabs | 2 (index, url) |
| browser_take_screenshot | 4 (element, target, filename, fullPage) |
| browser_type | 3 (element, submit, slowly) |
| browser_wait_for | 3 (time, text, textGone) |

只传 required 参数的调用验证：

```
browser_click(target="test")         → TypeError: missing 'element'
browser_snapshot()                   → TypeError: missing 'target'
browser_wait_for()                   → TypeError: missing 'time'
```

### 根因

传导链——三层语义，两层断裂：

```
JSON Schema 层:
  {all: {type: "boolean"}, required: []}
  → 语义: "all 是可选参数，不传时服务端兜底"

        ↓ mcp_schema_to_specs (不设 required: False, 不设 default)

ParamSpec 层:
  {name: "all", annotation: "bool"}   ← required 字段缺失, default 字段缺失
  → 信息已丢失: 不知道 all 是 optional

        ↓ build_signature (fallback: required = not has_default)

Python Signature 层:
  Parameter("all", KEYWORD_ONLY, default=Parameter.empty)
  → 语义: "all 是必填关键字参数"

        ↓ sig.bind()

TypeError: missing a required argument: 'all'
```

核心冲突：**`Parameter.empty` 在 Python 里表示「必填」，但 MCP 协议层
「不在 required」表示「可选、服务端兜底」。** 这两个语义在当前实现中
被直接等同。

### 为何此前未被发现

上一轮 `refactor-wrapper-faithful-signature` 改造时，`mcp_schema_to_specs`
把协议层的「不在 required」语义丢弃了（没写 `required: False`），
`build_signature` 的 fallback `required = not has_default` 又把「无 default」
等价为「required」。两层信息丢失恰好互相对冲——选错了维度但输出碰巧没塌，
直到 `Parameter.empty` 被 `sig.bind()` 严格解释。

## 关键参考

- `src/mutagent/sandbox/_signature.py:60` — `mcp_schema_to_specs`，MCP JSON Schema → ParamSpec 转换
- `src/mutagent/sandbox/_signature.py:95` — `build_signature`，参数分组策略 + 签名构造
- `src/mutagent/sandbox/_adapter_mcp.py:799` — `_make_tool_func`，MCP tool wrapper 构造入口，含 `sig.bind()` 调用
- `src/mutagent/sandbox/_adapter_mcp.py:850` — `call_with_retry`，实际 RPC 调用点，kwargs 在此处发给 MCP server
- `docs/specifications/refactor-wrapper-faithful-signature.md` — 上一轮真签名改造的设计方案
- Python stdlib `inspect.Parameter.empty` — 语义：「此参数没有默认值」=「必填」

## 设计方案

### 核心思路：sentinel 默认值 + RPC 前过滤

给所有 optional-no-default 参数注入一个 sentinel 默认值，让 Python signature
认为「有默认值所以不是必填」。在 RPC 调用前把 sentinel 值的 key 从 kwargs
中剔除，让 MCP server 收不到这个 key，自行应用服务端默认值。

两条路径分属不同模块：

```
mcp_schema_to_specs           → 注入 sentinel default
       ↓
build_signature               → 把 sentinel 写入 Parameter.default
       ↓
_make_tool_func / call_with_retry  → 把 sentinel 值从 kwargs 中剔除
```

### sentinel 设计

```python
class _MissingSentinel:
    """MCP optional-no-default 参数占位符。有默认值所以 sig.bind() 不报错；
       在 RPC 发送前被过滤，让 server 收到缺失 key 而非 null/空串。"""
    def __repr__(self):
        return "(omitted)"
    def __bool__(self):
        return False

_MISSING = _MissingSentinel()
```

- `__repr__` 返回 `(omitted)`，在 `help()` 签名里展示为 `all: 'bool' = (omitted)`，语义清晰
- `__bool__` 返回 False，避免 `if value:` 之类意外命中

### mcp_schema_to_specs 改动

1. 不在 `required` 列表中的参数显式设 `required: False`
2. 对于 `required=False` 且无 `default` 字段的参数，注入 `default: _MISSING`

```python
for pname, pinfo in properties.items():
    spec = {"name": pname}
    if pname in required:
        spec["required"] = True
    else:
        spec["required"] = False
    if "default" in info:
        spec["default"] = info["default"]
    elif pname not in required:
        spec["default"] = _MISSING   # ← 核心改动
    ...
```

### call_with_retry 改动

在 kwargs 传给 MCP server 之前，剔除值为 `_MISSING` 的 key：

```python
async def call_with_retry(kwargs: dict[str, Any]) -> Any:
    kwargs = {k: v for k, v in kwargs.items() if v is not _MISSING}
    await conn.ensure_connected()
    ...
    return await conn.client.call_tool(tool_name, kwargs)
```

### 为什么不推断真实默认值

`all` 描述说 "Defaults to false"，但 schema 不写 `default: false`。可选的
推断策略：

| 策略 | 行为 | 风险 |
|------|------|------|
| **按 type 推断零值** | boolean→False, string→"", number→0 | `filename: ""` 可能被 server 解释为"存到空路径"，而非"用户不想存文件" |
| **解析 "Defaults to" 文本** | 正则匹配 description | 脆弱的 NLP，不同 server 措辞不同 |
| **sentinel 不透传**（采纳） | 不传 key，server 自行兜底 | 无风险——MCP 协议原生语义就是省略 key |

**决策**：不推断默认值。MCP 协议规定 missing key = server default，这是唯一
可靠、无歧义的语义。客户端不应替 server 猜测默认值到底是什么。

### 与 downgrade 触发器的关系

sentinel 修复后，optional-no-default 参数有了 `has_default=True`，
`is_optional = has_default or not required = True or True = True`。
降级条件 `saw_optional and not is_optional` 不会误触发。这实际上绕过了
原来的维度错位问题——所以 Phase 2 的触发器清理是可选的语义改进，不影响
bugfix 正确性。但保留仍有价值：显式 `saw_default` 比间接 `saw_optional`
更不容易在未来改动中引入回归。

### 签名变化预期

修复前 `browser_console_messages` 签名：
```
(level: 'str' = 'info', *, all: 'bool', filename: 'str')
```
→ 三个参数都是 keyword-only，`all`/`filename` 无默认值 → call without them → TypeError

修复后：
```
(level: 'str' = 'info', all: 'bool' = (omitted), filename: 'str' = (omitted))
```
→ 三个参数都可以位置或关键字调用，都有默认值 → `browser_console_messages()` → 正常工作

## 兼容性

- **MCP 协议**：不传 key 和传 key=undefined 在 JSON 层等价，所有已知 MCP server
  都兼容
- **pysandbox 路径**：不受影响。`_describe_function` 在 server 端遇到无 default
  的参数时不输出 `default` 字段，客户端 `_make_namespace_func` 在 `params` 存在时
  走 `_build_signature`——同样受益于 sentinel 机制（在客户端那侧 `_build_signature`
  的 `required = not has_default` fallback 对 pysandbox 路径仍然生效）
- **help() 展示**：`= (omitted)` 替代了原来没有 `=` 的形态，观感变化但信息量更大
  （明确告知"这个参数可以不传"）
- **已有测试**：涉及 MCP tool wrapper 签名的测试需要更新期望字符串（`(omitted)`
  替换 `Parameter.empty` 导致的差异）

## 实施步骤清单

### Phase 1 — sentinel 机制

- [x] 新增 `_MissingSentinel` 类 + `_MISSING` 单例，放在 `_signature.py`
- [x] `mcp_schema_to_specs`：不在 required 的参数显式设 `required: False`；
  对 `required=False` 且无 `default` 的参数注入 `default: _MISSING`
- [x] `_make_tool_func` 的 `call_with_retry`：RPC 发送前剔除值为 `_MISSING` 的 key
- [x] 更新 `_signature.py` 模块 docstring，记录 Optional-no-default → sentinel 的
  设计理由

### Phase 2 — 降级触发器语义清理（可选）

- [x] `build_signature`：`saw_optional` → `saw_default`，触发条件改为
  `saw_default and not has_default`
- [x] 清理已无消费者的 `is_optional` 局部变量

### Phase 3 — 测试

- [x] 单元测试：7 种 schema 形态签名正确性（沿用已有矩阵），重点验证
  optional-no-default 参数有 `default=(omitted)` 而非 `Parameter.empty`
- [x] 集成测试：`_make_tool_func` 构造的 wrapper——位置调用、关键字调用、
  省略可选参数不报错、sentinel 值被正确过滤
- [x] playwright 端到端：`browser_console_messages()` 不报 TypeError；
  `browser_click(target="...")` 不报 TypeError；`browser_wait_for()` 不报 TypeError
- [x] 回归：已有 MCP wrapper 测试套件全通（更新因签名变化导致的期望差异）

### Phase 4 — 文档与 spec 同步

- [x] 更新 `refactor-wrapper-faithful-signature.md` 的关键参考章节，
  引用本 spec 作为已知局限的修正

## 测试验证

- `pytest tests/test_signature_build.py tests/test_adapter_mcp.py -q`
- `pytest tests/test_signature_build.py tests/test_adapter_mcp.py tests/test_mcp_settings_panel.py -q`
- `pytest -q`

## 关键决策记录

- **不做默认值推断**：`filename: ""` 与「不传 filename」在 server 端语义不同。
  MCP 协议层 missing key = server default 是唯一可靠语义
- **sentinel 用单例而非 None**：`None` 是合法的参数值（如 `browser_evaluate`
  的 `element: string | null`），不能复用为"未传"信号
- **Phase 2 可选**：sentinel 修复使 downgrade 触发器的维度错位不再产生可见影响，
  但清理它可降低未来改动的理解成本

### Phase 5 — 输出质量优化（2026-05-13）

Phase 1-4 完成了功能修复，但两处输出细节仍不符合预期：

**5a. `(omitted)` → `...`**

`_MissingSentinel.__repr__` 返回 `(omitted)`，在 `str(sig)` 中渲染为
`all: 'bool' = (omitted)`。这不像合法的 Python 默认值写法。

改为 `...`（Python `Ellipsis` 字面量），`def foo(x=...): pass` 是合法语法，
NumPy / PEP 484 stub / Protocol 等广泛使用 `...` 为占位符。

**5b. `required` + `default` 并存时压制 `(required)` 标签**

`_adapter_mcp.py:826` 的 docstring 生成逻辑：

```python
req_mark = " (required)" if pname in required else ""
```

这只看 `required` 数组，不管参数是否有已知 `default`。导致 @playwright/mcp
把 `level` 同时放在 `required` 和写了 `default: "info"` 时，help 输出显示
`level: string (required)` 但 signature 却有默认值 `= 'info'` —— 自相矛盾。

修复：有 schema 级 `default` 时压制 `(required)` 标签：

```python
has_schema_default = "default" in pinfo
req_mark = " (required)" if (pname in required and not has_schema_default) else ""
```

- [x] 5a. `_signature.py`：`_MissingSentinel.__repr__` 返回 `"..."`
- [x] 5b. `_adapter_mcp.py`：`req_mark` 逻辑加 `has_schema_default` 判断
- [x] 5c. 更新测试期望字符串（`(omitted)` → `...`）

实现时同步修复了 `webui/_settings_mcp.py::_fn_detail` 的同源 `(required)` 展示逻辑，
避免 CLI help 与 settings panel 在同一份 MCP schema 上出现不一致。
