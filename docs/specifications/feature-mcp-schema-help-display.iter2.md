# MCP Schema → Docstring：sandbox help() 渲染

**状态**：✅ 已完成
**日期**：2026-05-14
**类型**：功能设计

## 背景

mutagent 的 pysandbox 给远程 mcp tool 合成本地 wrapper 函数（`_adapter_mcp.py::_make_tool_func`）。agent 在 sandbox 里通过 `help(tool)` 了解 tool 的输入约束 — 这是 agent 唯一的入口。

本规范定义如何把 mcp `inputSchema` 渲染成本地 wrapper 函数的 signature + docstring，让 `help()` 输出的形态既贴近 python 原生、又不丢失 mcp 协议信息。

**适用范围**：仅 sandbox + `help()` 场景。如果 agent 直接走 mcp 协议（schema 直接给客户端），本规范不适用。

## 设计方案

### 信息源唯一：mcp schema 是源

外部 mcp tool 的 `inputSchema` 是**唯一信息源**，本地 wrapper 的 signature + docstring 是它的人读投影。

**不追求字符级双向对等**（往返 round-trip 不作设计锚点），但保证**软语义对等**：渲染输出的 docstring 应当能被 mutio 的解析器读回为语义等价的 schema。这条作为事后健全性测试，不作为 CI 强约束。

软语义对等的依据是 mutio 主导的 `Annotations:` 段语法（详见姊妹文档 `mutio/docs/specifications/feature-mcp-schema-docstring-source.md`）。

### 三处投影：把 mcp schema 字段分配位置

把 `inputSchema.properties[name]` 字段分到三个位置：

| schema 字段 | 投影位置 | 备注 |
|---|---|---|
| `type` | signature annotation | 通过 `json_type_to_annotation` 反推 |
| `default` | signature default | |
| 无 default | signature 必填 | |
| `enum`（同类型字面量） | signature `Literal[...]` | python first |
| `enum`（混合类型） | signature `Literal[...]`（python typing 支持） | 罕见 |
| `description` | docstring `Args:` 段散文 | `name: description.` |
| `items`（嵌套 object 等 typing 表达不了的复合类型） | annotation 降级为 `list[dict]` / `list` 等；完整 `items` 进 `Annotations:` 段 | 信息不丢失，与「typing 表达不了的全归 Annotations」原则一致 |
| 其他全部字段（`minimum` / `maximum` / `pattern` / `format` / `minLength` / `maxLength` / `multipleOf` / `uniqueItems` / `minItems` / `maxItems` / `additionalProperties` / `propertyNames` / `patternProperties` / `const` / 未知扩展） | docstring `Annotations:` 段 | mcp 原词 + JSON 字面量透传 |

**位置职责唯一**。已由 signature 表达的字段（`type` / `default` / `enum` / `items.type` 等）**不重复**写到 `Annotations:` 段，避免视觉冗余和潜在不一致。

### `Annotations:` 段渲染格式

格式定义在 mutio 主导规范，本规范的渲染输出必须符合。简述：

```
Annotations:
    {param_name}: {json_blob}
```

- 段头 `Annotations:` 顶格（与 `Args:` 同级）
- 每行 4 空格缩进 + 参数名 + `:` + JSON value
- value 严格 JSON（小写 `true` / `false` / `null`，字符串双引号）
- key 保持 mcp schema 原词（camelCase：`uniqueItems` / `additionalProperties` / `propertyNames`）
- value 一律 `json.dumps(..., ensure_ascii=False)`（docstring 给人读，含中文/Unicode 时直出原字符；mutio 解析侧 `json.loads` 接受非 ASCII）

### 长 value 自适应展开

每个 property 渲染时先 `json.dumps(value, ensure_ascii=False)` 单行：

- 单行长度（含 `    {name}: ` 前缀）≤ 阈值 → 单行
- 超阈值 → `json.dumps(value, indent=4, ensure_ascii=False)`，参数名 + `: {` 起首行，闭合 `}` 回到 4 空格缩进

仍然只有一种语法（`name: json`），多行只是 JSON pretty-print，**不引入第二种规则**。`json.loads` 原样接受多行 — 不破坏 mutio 解析器。

阈值：默认 100 列（常见 Python 行宽，作为渲染模块的内部参数集中定义，不散落写死）。

### 完整渲染示例

输入 mcp schema：

```json
{
    "name": "query",
    "description": "Run query against database.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "level":   {"type": "string", "enum": ["DEBUG", "INFO"], "default": "INFO", "description": "Log level."},
            "count":   {"type": "integer", "minimum": 0, "maximum": 100, "description": "Result limit."},
            "options": {"type": "object", "additionalProperties": false, "propertyNames": {"pattern": "^[a-z]+$"}}
        },
        "required": ["count"]
    }
}
```

渲染后 wrapper（agent 在 sandbox 里 `help(query)` 看到）：

```python
async def query(
    level: Literal["DEBUG", "INFO"] = "INFO",
    count: int = ...,
    options: dict = ...,
) -> Any:
    """Run query against database.

    Args:
        level: Log level.
        count: Result limit.
        options:

    Annotations:
        count: {"minimum": 0, "maximum": 100}
        options: {"additionalProperties": false, "propertyNames": {"pattern": "^[a-z]+$"}}
    """
```

观察：

- `level` 由 signature 完整表达（`Literal[...]` + default），无 Annotations 行
- `count` 描述在 Args，约束在 Annotations
- `options` 无 description，Args 行只有 `name:`（保留冒号，与有描述行视觉一致）
- `Annotations:` 段不出现 `level` 行（避免冗余 / 不一致）

### 与 namespace 聚合（pysandbox sharing）的关系

本规范**只动 MCP 入站口的渲染**，与 namespace 聚合路径正交。

**渲染只发生一次**：

- `_adapter_mcp.py::_make_tool_func` 在初次接入 MCP server 时把 `inputSchema` 翻译成真 `inspect.Signature` + 渲染好的 docstring，分别挂到 `__signature__` / `__doc__`。原始 `_mcp_input_schema` 仅作 Settings Panel 备用，不参与 sharing。

**聚合路径只搬运字符串和 spec**：

- `share.py::_describe_function` 透传 `signature: str` / `doc: str` / `params: list[dict]`。
- `pysandbox/namespaces.describe` 整条链路**没有任何代码读 `inputSchema`**——本机 `help(tool)` 看 `__doc__` + `__signature__`，远端通过 `params` 重组 `inspect.Signature` + 把 `doc` 字符串挂回 `__doc__`。
- 两端 `help()` 输出一致由「字符串穿透」保证，不是因为两端各跑一遍渲染。

**推论**：

1. 本次实施**不动 share.py**，sharing 链路对 schema 渲染规则的变更是 no-op。
2. `tests/test_pysandbox_sharing.py` 的一致性测试自动受益于 `_make_tool_func` 单点改动。
3. iter1 的 `_MISSING` sentinel 机制必须保留——它解决「必填参数怎么跨进程表达」，与渲染格式无关，是 sharing 路径自身需求。
4. 若未来出现「对端把已经聚合过的 MCP tool 再转发出来」，doc 是上一跳渲染结果的二次穿透，正符合「不重复渲染、不二次解析」的设计。

### 与 mutio 的契约关系

mutio 定义 `Annotations:` 段的**写入语法**（开发者怎么写），mutagent 的渲染规则**按同一语法生成**。两者由同一份语法约束，但实现独立。

**软语义对等验证**：

- mutagent 渲染产出 docstring → mutio 解析器读回 schema → 与原 mcp schema **结构等价**
- 等价定义：相同 properties 集合 / 相同 type / 相同 enum / 相同其他约束字段
- 允许差异：description 文本（渲染时可能格式化）、property 顺序、空段省略
- 验证作为**事后健全性测试**保留在 mutagent 测试套件，**不作为 CI 阻塞**

### 非目标

- **不做嵌套 schema 的展开渲染**（`items` 嵌套 object 时整体作为 JSON value 留在 Annotations，不递归列举字段）
- **不引入快捷格式**（`enum: a | b | c` / `range: 0..100` / `Pattern: ...` 等翻译句一律不做）
- **不强制字符级双向 round-trip**
- **不做未知字段的兜底翻译**（mcp 未来扩展字段原词透传到 Annotations，不做语言化翻译）

### 对 iter1 成果的处理

`feature-mcp-schema-help-display.iter1.md` 已落地：独立缩进行 + `_MISSING` sentinel 跨 sharing 一致性。本规范替换 iter1 的「英文约束行」格式（`Allowed: a | b | c.` / `Range: 0..100.` 等），但保留 iter1 的 sharing 一致性成果：

- **保留**：`_MISSING` sentinel 机制、`share.py::_describe_function` 透传链路（docstring 字符串穿透，无改动）
- **删除**：iter1 的英文翻译辅助函数（`format_param_description_suffix` / `_format_allowed_value` / `_format_range_clause`）及其调用点
- **改写**：`_make_tool_func` 的 docstring 拼装路径——`Args:` 只留 `name: description.`，约束统一进 `Annotations:` 段

## 消费者场景

| 消费者 | 场景 | 依赖的输出 | 验收标准 |
|---|---|---|---|
| `mutagent.sandbox._adapter_mcp._make_tool_func` | 远程 mcp tool 进 sandbox | 渲染后的 wrapper docstring | agent 在 sandbox 里 `help(tool)` 看到完整三段（signature + Args + Annotations），约束信息无丢失 |
| `mutagent.webui._settings_mcp._fn_detail` | Settings Panel 展示 mcp tool 详情 | 同一渲染输出 | 与 sandbox `help()` 视觉一致 |
| `mutagent.sandbox.share` | 跨进程 sharing 链路 | docstring 透传 | 8765 直连和 8700 sharing 两端 `help()` 输出严格一致 |
| 软语义对等测试（事后健全性） | mutagent 测试套件 | mutio 解析器 + mutagent 渲染器 | 一组 mcp schema 样本经渲染 + 解析后结构等价 |

## 关键参考

### 当前代码

- `mutagent/src/mutagent/sandbox/_adapter_mcp.py::_make_tool_func` — 渲染主路径
- `mutagent/src/mutagent/sandbox/_signature.py::json_type_to_annotation` — 已有，复用
- `mutagent/src/mutagent/sandbox/share.py::_describe_function` — sharing 透传，无改动
- `mutagent/src/mutagent/webui/_settings_mcp.py::_fn_detail` — Settings Panel 消费侧
- `mutagent/tests/test_pysandbox_sharing.py` — sharing 一致性测试

### 姊妹文档

- `mutio/docs/specifications/feature-mcp-schema-docstring-source.md` — `Annotations:` 段写入语法定义（mutio 主导，本规范遵守）

### 已废弃 / 取代的探索方向（保留不删，作历史参考）

- `feature-mcp-schema-help-display.md` 主 spec + iter1 — 早期英文翻译路线，iter1 部分成果（独立缩进行 / `_MISSING` sentinel）继续保留
- `feature-mcp-schema-help-display.iter1.md` — ✅ 已完成，sentinel 修复继续受益

### 外部协议

- MCP schema: https://raw.githubusercontent.com/modelcontextprotocol/specification/main/schema/2025-11-25/schema.ts
- JSON Schema: https://json-schema.org/

### 设计讨论脉络

本规范从 2026-05-14 与用户的协同设计讨论中推导：

1. 否决双向硬对等 + CI 闭环约束（设计税无收益，软语义对等已足够）
2. 收敛到 "mcp schema 是单一信息源 + 单向投影到三个位置"
3. 三处投影（signature / Args / Annotations）职责唯一，已表达字段不重复
4. enum 升级到 signature `Literal[...]`（python first + IDE 补全）
5. `Annotations:` 段格式遵循 mutio 主导定义（参数名 docstring 层级 + JSON value）
6. 长 value 自适应展开成多行 JSON（不引入第二种语法）
7. 软语义对等验证保留为事后健全性测试，不强制 CI
8. 澄清 namespace 聚合路径完全字符串穿透，本次实施对 sharing 是 no-op

## 实施步骤清单

- [x] `_signature.py` 增加 enum → `Literal[...]` 的 annotation 推导（同类型 / 混合类型字面量统一走 typing.Literal 字符串形式，与现有 `_RawAnnotation` 渲染兼容）
- [x] `_signature.py` 新增 `format_annotations_section(properties, required)` 渲染入口：
  - 按「已被 signature 表达的字段」白名单剔除（`type` / `default` / `description` / `enum` / 已用的 `items.type` 等）
  - 剩余字段（含未知扩展）原词透传为 JSON value
  - 单行长度 > 100 列时切换 `json.dumps(..., indent=4)`
  - 全程 `ensure_ascii=False`
- [x] `mcp_schema_to_specs` 检测 enum 并升级 annotation 为 `Literal[...]`
- [x] 改写 `_adapter_mcp.py::_make_tool_func` 的 docstring 拼装：
  - 头部 description
  - `Args:` 段：每行仅 `name: description.`，无描述时只 `name:`
  - `Annotations:` 段：调用上一步的渲染入口，无字段则整段省略
  - 删除对 `format_param_description_suffix` 的引用
- [x] 删除 `_signature.py` 中 iter1 残留的翻译辅助：`format_param_description_suffix` / `_format_allowed_value` / `_format_range_clause`（保留 `_MISSING` / `_format_json_literal`）
- [x] 新增 `tests/test_mcp_schema_render.py`（或并入既有 wrapper 测试）：覆盖 enum / range / pattern / nested object / 长 value 多行 / 中文 description / 无 description / required 与可选混合
  - 实际并入 `tests/test_signature_build.py`（新增 `TestFormatAnnotationsSection` + `TestFormatLiteralAnnotation`）和 `tests/test_adapter_mcp.py`（新增三段式布局 + Annotations 透传测试）
- [x] 跑 `tests/test_pysandbox_sharing.py`：确认 8765 直连 / 8700 聚合两端 `help()` 输出严格一致（应自动通过，不改 share.py）
  - 35/35 通过，未动 share.py
- [x] 视觉验证 `webui/_settings_mcp.py::_fn_detail` 渲染结果与 sandbox `help()` 一致（无需改动该文件，仅人工确认）
- [x] 自检：搜索 `format_param_description_suffix` / `_format_allowed_value` / `_format_range_clause` 全工程无残留 import
  - `grep -rn` 混 src/tests 返回 0 命中

## 实施验证

- 全量测试：`pytest mutagent/tests` → 950 passed, 4 skipped
- 关键三个测试集：
  - `test_signature_build.py` → 45 passed（含新增的 `TestFormatAnnotationsSection` 8 + `TestFormatLiteralAnnotation` 5）
  - `test_adapter_mcp.py` → 75 passed（含新增的三段式布局 + Annotations 透传 3 个）
  - `test_pysandbox_sharing.py` → 35 passed（未改 share.py，透传链路验证 sharing 一致性自动受益）
- 渲染示例验证（与 spec 示例一致）：
  ```
  Signature: (level: Literal["DEBUG", "INFO"] = "INFO", *, count: int, options: dict = ...)
  Doc:
    Run query against database.

    Args:
        level: Log level.
        count: Result limit.
        options:

    Annotations:
        count: {"minimum": 0, "maximum": 100}
        options: {"additionalProperties": false, "propertyNames": {"pattern": "^[a-z]+$"}}
  ```
- iter1 辅助函数（`format_param_description_suffix` / `_format_allowed_value` / `_format_range_clause`）全工程 0 残留
