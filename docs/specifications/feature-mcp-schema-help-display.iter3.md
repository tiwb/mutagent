# MCP Schema → Docstring:折行优化(iter3 修订版)

**状态**:✅ 已完成
**日期**:2026-05-14
**类型**:功能设计
**前置**:`feature-mcp-schema-help-display.iter2.md`(✅ 已完成)

## 需求

iter2 的"三段式"渲染(signature + Args + `Annotations:`)信息完整性已解决。
当前问题**纯粹是排版**--折行策略对 agent 不友好:

1. **Signature 永远单行**。`browser_click` 5 个参数带 `Literal["left","right","middle"]`,
   整行约 150 字符,agent 逐 token 解析长单行比逐行解析吃力。
2. **Annotations JSON 展开过深**。`indent=4` 在 3 层嵌套时最深 16 空格缩进,
   大量 blank token 对 agent 毫无信息量。
3. **`...` sentinel 语义模糊**。`...` 表示"可选、无默认值、省略即用服务端默认",
   但 Python 里 `...` 也是 `Ellipsis`,Pydantic 生态里 `Field(...)` 反而表示"必填",
   agent 容易误解。

iter3 同时顺手解决一个**结构性冗余**:

4. **`format_callable_signature` 三条 fallback 路径冗余**。本地迭代未发布,
   `_pysandbox_signature_str` / `_format_schema_signature` 没有兼容包袱;wrapper
   构造失败时签名自然降级为 `(**kwargs)`,参数信息全在 Annotations 段,
   "假签名拼接"反而提供低质量误导信息。

四件事都在 **help 的文本输出端**解决,不涉及数据流或 sharing 链路。

## 设计方案

### 阈值常量统一

新增一个统一阈值常量,覆盖 signature 折行与 JSON 折行决策:

```python
# _signature.py
_RENDER_LINE_WIDTH = 80  # iter2 原值 100 → iter3 改为 80
```

理由:
- 80 列是 PEP 8 / Black 的肌肉记忆数字,agent 训练数据中高频出现
- iter2 的 100 仅 Annotations 一处使用,没有兼容负担
- 一个常量管两处折行,避免后续散落维护

**`max_width` 语义统一约定**:所有渲染函数收到的 `max_width` 都指**渲染后整行实际占用的列数(含外层缩进与 prefix)**。递归调用内层时由调用方负责扣减外层缩进。

### Signature:`format_signature` 加 `max_width` 参数

> 不新增 `format_signature_multiline` 函数,给现有 `format_signature` 加可选参数,
> 内部按需折行。

```python
def format_signature(sig: inspect.Signature, *, max_width: int | None = None) -> str:
    ...
```

- `max_width=None`(默认)→ 当前行为,永远单行(向后兼容 `__repr__` 等老路径)
- `max_width=80` → 单行能放下保持单行,超宽切多行

**决策理由**(保留单一函数 vs 新增 multiline 同层函数):
- `format_callable_signature` 一次调用透传 `max_width` 即可,**不需要外层"先单行→判超宽→回头重渲染"两段式分发**
- 多行折行是 `format_signature` 的渲染细节,不是渲染层的策略决策
- 接口数量减少,调用方代码更简单

**Black 风格折行规则**:

- 单行总宽度(含 `qualified` 函数名前缀)≤ `max_width` → 保持单行(**单行不加 trailing comma**)
- 超过阈值 → 切换为每参数一行:

```
func_name(
    param1: type = default,
    param2: type,
    *,
    kwarg1: type = default,
) -> RetType
```

细节:
- 分隔符 `*` 和 `/` 独占一行(Black 规则:分隔符行末**不加** trailing comma)
- **trailing comma 仅多行模式追加**(含最后一个参数;diff 友好),单行模式不加
- 缩进 4 空格(与 Black 一致)
- 闭合 `)` 回到第一列缩进(0 空格),与函数名对齐
- **return annotation**(如有)跟在 `)` 同一行:`) -> RetType`;为空(`Signature.empty`)则只输出 `)`
- annotation 的字符串去引号保持 iter2 的 `_RawAnnotation` 机制
- `_MISSING` sentinel 渲染为 `<omit>`(详见下文)
- 空参数函数 `func()` 总是单行(无参数无折行需求)

**退化场景明确(接受现状不进一步处理)**:
- 单参数本身就超 `max_width`(例如 `option: Literal[100 项]`)→ 该行接受超宽,不再二次折行(Literal 内部折行另起 iter)
- `) -> VeryLongReturnType` 同行超 `max_width` → 接受超宽(return annotation 折行不在本 iter scope)

### `format_callable_signature` 简化:删除三条 fallback

**核心洞察**:所有入口路径(MCP 桥接 / pysandbox peer / 同进程 Namespace / CLI adapter)
在"正常情况"下 wrapper 都挂了 `__signature__`,全部收敛到 `format_signature(sig)` 主路径。
三条 fallback 只在 wrapper 构造失败时触发,且:

- `_format_schema_signature` 拼出的"伪签名"丢类型精度(`string` 不是 `str`)、丢 Literal、
  丢必填/可选区分(全是 `= ...`)、丢 description--**信息密度低于让 agent 直接看 Annotations 段**
- `_pysandbox_signature_str` 是老 server 兼容字段,**本地迭代尚未发布,无兼容包袱**
- `(**kwargs)` 是 wrapper 失败时的自然形态,本身就是 Python agent 的母语 hint

**简化后**:

```python
def format_callable_signature(func, *, max_width=None) -> str | None:
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return None
    return format_signature(sig, max_width=max_width)
```

**所有路径自然收敛**:

| 场景 | wrapper 形态 | 渲染结果 |
|---|---|---|
| MCP schema 合法 | 真 `__signature__` | 主路径多行折行 |
| MCP schema 失败(含 `oneOf`/`$ref`) | `def tool_func(**kwargs)` | `tool_func(**kwargs)`,参数细节看 Annotations 段 |
| pysandbox 新 server | 真 `__signature__` | 主路径 |
| pysandbox 老 server | `def ns_func(**kwargs)` | `ns_func(**kwargs)`,看 Annotations 段 |
| 同进程函数 | 真签名 | 主路径 |
| `inspect.signature` 报错 | - | 返回 `None`,由 `_render_function` iter2 兜底 |

**结构性收益**:
- 不需要"fallback 路径要不要参与折行"的特例规则--`(**kwargs)` 永远 18 字符,**物理上不可能超 80**
- `_mcp_input_schema` 属性**保留**(Annotations 段的数据源,不是签名 fallback)
- 连带清理:`_format_schema_signature` 函数、`_is_kwargs_only_fallback` 函数、
  `_adapter_pysandbox.py` 里 `ns_func._pysandbox_signature_str` 写入逻辑全部删除

### Annotations JSON:Black 风格递归紧凑

> 叶子紧凑、尽量一行、超出宽度才逐层展开。indent 从 4 降到 2。

**规则**(自定义递归 formatter,不依赖 `json.dumps(indent=)`):

1. **标量**(string / number / boolean / null)→ 始终 `json.dumps(value, ensure_ascii=False, separators=(",",":"))` 紧凑输出
2. **数组**:
   - 全标量元素 + 紧凑后 ≤ 可用宽度 → 紧凑单行(`["textbox","checkbox","radio"]`)
   - **全标量元素 + 紧凑后超宽 → 每元素一行展开**(覆盖长 enum 场景,例如百项语言代码列表)
   - 含嵌套结构 + 紧凑后 ≤ 可用宽度 → 紧凑单行
   - 含嵌套结构 + 紧凑后超宽 → 每元素一行展开
3. **对象**:
   - 紧凑后 ≤ 可用宽度 → 紧凑单行(`{"type":"string","enum":["left","right"]}`)
   - 紧凑后超宽 → 逐 key 展开,每行 `<indent>"key": <value>,`,结尾 `}` 回到上级缩进
4. **分隔符策略明确**:
   - **单行紧凑模式**:`separators=(",",":")`(无空格)
   - **多行展开模式**:元素间 `,\n`;对象 key/value 间用 `": "`(冒号+空格,可读性)
5. indent = 2(非 iter2 的 4)
6. 始终 `ensure_ascii=False`(与 iter2 一致)

**输入约束**:`_format_json_compact` 仅处理 JSON 兼容树(`Mapping` / `list` / `str` / `int` / `float` / `bool` / `None`)。MCP schema 来自协议层 JSON,已天然满足;非 JSON 兼容值(dataclass、自定义对象)行为未定义,调用方负责保证。

**`max_width` 与 `current_indent` 递归约定**:

```python
def _format_json_compact(obj, *, max_width, current_indent=0, indent_step=2) -> str:
    compact = _try_compact(obj)  # 标量直接返回;容器尝试紧凑序列化
    if current_indent + len(compact) <= max_width:
        return compact
    # 展开:子节点 current_indent = current_indent + indent_step,max_width 不变
    ...
```

外层调用传 `max_width=_RENDER_LINE_WIDTH`、`current_indent=外层已缩进列数`。
判断公式始终是「当前缩进 + 紧凑长度 ≤ max_width」。

**复杂度说明**:每节点先 try compact (`json.dumps`) 再决定是否展开,深嵌套有
O(深度 × 节点数) 重复序列化成本。MCP schema 实际深度 ≤ 4,可接受。

**效果对比**(`browser_fill_form` 的 Annotations 段):

```
# iter2(indent=4,~45 行)
    fields: {
        "items": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    ...

# iter3(Black 风格,~12 行)
    fields: {
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string", "description": "Human-readable field name"},
          "type": {"type": "string", "enum": ["textbox","checkbox","radio","combobox","slider"], "description": "Type of the field"},
          ...
        },
        "required": ["target","name","type","value"],
        "additionalProperties": false
      }
    }
```

**实现**:

- `_signature.py` 新增 `_format_json_compact(obj, *, max_width, current_indent=0, indent_step=2)`
- `format_annotations_section` 单行/多行决策保持「先试紧凑单行、超宽走 formatter」结构,但**多行分支替换为 `_format_json_compact`**,不再使用 `json.dumps(..., indent=4)`
- `_indent_continuation` helper 删除(旧多行用 `json.dumps(indent=4) + _indent_continuation` 拼接,新方案 formatter 直接产出最终缩进字符串)
- `_format_json_compact` 作为 `_signature.py` 内部私有 helper(前缀下划线表达私有意图)。
  当前唯一消费方是 Annotations 渲染,过早提取到公共库只增加维护面。函数 docstring
  留一句"未来若有第二个消费方(如 logs dump、debug pretty-print)可上提到 mutio"。

### `_MISSING` 渲染为 `<omit>`

iter2 的 `_MISSING.__repr__` 返回 `"..."`。本 iter 改为 `"<omit>"`。

**决策理由**(vs `...` + 脚注方案):

- `...` 与 Python `Ellipsis` / Pydantic `Field(...)` 反向语义碰撞,必须靠脚注 `(Parameters shown as "..." are MCP-optional ...)` 补救
- 脚注是补丁不是设计--既然要解释,说明 `...` 不自解释
- `<omit>` 尖括号是 CLI/template/OpenAPI 圈约定俗成的占位符标记(`<unset>`、`<auto>`、`<undefined>`、git 的 `<no commit>`),**一眼自解释,零歧义**
- 多 4 字符 token 成本 << 省下脚注一整行 + `has_missing` 检测代码的成本
- **把"标记 sentinel 的语义"封装在 sentinel 自己的 repr 里**,渲染层无需打补丁,符合"让正确的事简单"原则

**连带删除**:iter2 草案中的 `...` 脚注、`has_missing` 检测代码全部不再引入。

**视觉区分保证**:字符串字面量 `"..."` 仍渲染为 `'...'`(带引号 repr),与 sentinel `<omit>`(不带引号)天然区分。

## 测试验证

### Signature 折行

- [x] 单行 ≤80：保持原样，**不加 trailing comma**
- [x] 单行超 80（5 参数 + 长 Literal）：转多行，**每参数一行 + trailing comma**
- [x] 含 `*` / `/` 分隔符：分隔符独占一行，**行末不加 trailing comma**（中部位置随带语法 `,`，不额外补）
- [x] 含 return annotation：多行模式下 `) -> RetType` 同行
- [x] 空参数函数 `func()`：永远单行
- [x] 单参数本身就超 80：接受超宽，不二次折行
- [x] `format_signature(sig)` 不传 `max_width`：永远单行（向后兼容）
- [x] `__repr__` 等老调用路径：行为不变

### `format_callable_signature` 简化验证

- [x] MCP schema 合法 wrapper：渲染真签名，多行折行（实测 `browser_click`）
- [x] MCP schema 含 `oneOf` 失败 wrapper：`mcp_schema_to_specs` 返回 `[]` → 生成 `()` 签名，Annotations 段依靠结构完整 schema 补充（原调查中「生成 `(**kwargs)`」估计偏保守，实际 `()` 同样是自然降级结果）
- [x] pysandbox 新 server peer：渲染真签名（`test_with_params_builds_signature` / `test_peer_namespace_functions_carry_signature`）
- [x] pysandbox 老 server peer（`(**kwargs)` wrapper）：渲染 `ns_func(**kwargs)`（`test_without_params_falls_back_to_kwargs`）
- [x] 同进程 namespace 函数：渲染真签名（`test_normal_function_renders_single_line`）
- [x] `inspect.signature` 报错的奇葩 callable：返回 `None`（`test_unsignaturable_returns_none`）

### Annotations JSON 折行

- [x] 短标量数组（`["a","b","c"]`）：紧凑单行
- [x] 长 enum 标量数组（>80 列）：每元素一行展开
- [x] 短嵌套对象：紧凑单行
- [x] 深层嵌套（`browser_fill_form`）：行数从 ~45 降至 17 行（Annotations 段本身）
- [x] 中文 description：不被 `ensure_ascii` 转义
- [x] 单行紧凑分隔符：`(",",":")` 无空格
- [x] 多行展开 key/value 分隔：`": "` 含空格

### `<omit>` sentinel

- [x] `_MISSING` 渲染为 `<omit>`（不带引号）
- [x] 字符串字面量 `"..."` 渲染为 `'...'`（带引号，与 sentinel 视觉区分）
- [x] pysandbox 函数（无 `_MISSING`）：签名中不出现 `<omit>`
- [x] **不应**出现 iter2 草案的 `(Parameters shown as "..." are MCP-optional ...)` 脚注（从未引入）

### 测试断言策略

- 排版输出用**字符串精确对比**(snapshot 风格)
- 行数粗略数字(`~12 行`)仅作设计阶段直觉参考,断言用具体字符串

## 实施步骤清单

按依赖关系排序。每完成一项立即更新 checkbox。

### 第一波:核心渲染改动(`_signature.py`)

- [x] 修改 `_RENDER_LINE_WIDTH` 常量值 100 → 80
- [x] 修改 `_MISSING.__repr__` 返回 `"<omit>"`(替换 `"..."`)
- [x] 给 `format_signature` 加 `max_width: int | None = None` 参数,实现 Black 风格多行折行(含 `*` / `/` 分隔符独占行、trailing comma 多行模式追加、return annotation 同行规则)
- [x] 新增 `_format_json_compact(obj, *, max_width, current_indent=0, indent_step=2)` 私有 helper,docstring 注明未来上提到 mutio 的条件
- [x] 修改 `format_annotations_section` 多行分支,改用 `_format_json_compact` 替换 `json.dumps(indent=4)`
- [x] 删除 `_indent_continuation` helper

### 第二波:fallback 简化(`_signature.py`)

- [x] 简化 `format_callable_signature` 为 5 行版本:`inspect.signature` → `format_signature`,删除全部 fallback 分发
- [x] 删除 `_format_schema_signature` 函数
- [x] 删除 `_is_kwargs_only_fallback` 函数

### 第三波:调用方接入与连带清理

- [x] `_namespace.py::_render_function`:一次调用 `format_callable_signature(func, max_width=_RENDER_LINE_WIDTH)`,**删除 iter2 草案中规划的 `has_missing` 检测和 `...` 脚注追加逻辑**(如已落地则删,未落地则跳过)。实际采用:按 qualified 名长度折减 effective_width(下限 20 列)
- [x] `_adapter_pysandbox.py`:删除 `ns_func._pysandbox_signature_str = signature_str` 写入;评估上游 `signature_str` 形参的传递链是否可一并清理(若仅服务于此则删,若有其他消费方则保留)。实际采用:仅服务于 _pysandbox_signature_str 写入,**形参一并删除**,`build_peer_namespaces` 内不再提取 `signature` 字段
- [x] `_adapter_mcp.py`:确认 `_mcp_input_schema` 属性写入保留(Annotations 段数据源)
- [x] `_settings_mcp.py::_fn_signature` 与 `_fn_detail` 注释更新:移除"回落到 `_mcp_input_schema` 合成路径"等指向已删除 fallback 的措辞

### 第四波:测试用例

- [x] Signature 折行测试用例(覆盖单行/多行/分隔符/return annotation/空参/退化场景):`TestSignatureMultilineFolding` 12 个用例
- [x] `format_callable_signature` 简化路径测试(含 kwargs-only 实际渲染 + max_width 切换多行):`TestFormatCallableSignatureIter3` 4 个用例
- [x] Annotations JSON 折行测试(紧凑/长 enum 展开/深嵌套行数对比/中文/分隔符):`TestFormatJsonCompact` 8 个用例 + `TestAnnotationsBlackStyleLineCount`
- [x] `<omit>` sentinel 测试(含字符串 `"..."` 视觉区分用例):`TestOmitSentinel` 4 个用例
- [x] 删除 iter2 草案测试中关于 `...` 脚注的 4 条用例(未落地)。同时更新:原有 `format_annotations_section` 紧凑分隔符测试 / `_pysandbox_signature_str` fallback 测试 / `_make_namespace_func` 测试。

### 第五波:验收

- [x] `pytest mutagent/tests/` 全绿(980 passed, 4 skipped)
- [x] 实测 `browser_fill_form` 的 help 输出,行数从 ~45 降至 22 行(含 Args 段);Annotations 段本身 17 行
- [x] 实测 `browser_click` 多参数 + Literal 签名转多行:signature 成功转 Black 风格多行,optional-no-default 参数渲染为 `<omit>`
- [x] 实测含 `oneOf` 的输入 schema:`mcp_schema_to_specs` 安全返回 `[]`,wrapper 生成 `()` 签名,Annotations 段依靠结构完整 schema 补充

## 改动范围清单

实施清单的对应代码位置概览(仅供检索定位):

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `_signature.py` | 修改 | `format_signature` 加 `max_width` 参数 + 多行折行实现 |
| `_signature.py` | 修改 | `format_callable_signature` 简化为 5 行 |
| `_signature.py` | 修改 | `_RENDER_LINE_WIDTH` 100 → 80 |
| `_signature.py` | 修改 | `_MISSING.__repr__` `...` → `<omit>` |
| `_signature.py` | 修改 | `format_annotations_section` 多行分支改用 `_format_json_compact` |
| `_signature.py` | 新增 | `_format_json_compact(obj, *, max_width, current_indent, indent_step)` |
| `_signature.py` | 删除 | `_format_schema_signature` 函数 |
| `_signature.py` | 删除 | `_is_kwargs_only_fallback` 函数 |
| `_signature.py` | 删除 | `_indent_continuation` helper |
| `_namespace.py` | 修改 | `_render_function` 接入 `max_width`;不引入 `has_missing` / 脚注逻辑 |
| `_adapter_pysandbox.py` | 删除 | `ns_func._pysandbox_signature_str` 写入;评估上游 `signature_str` 清理范围 |
| `_settings_mcp.py` | 修改 | `_fn_signature` / `_fn_detail` 注释更新(移除已删 fallback 提法) |

## 关键参考

### 当前代码

- `mutagent/src/mutagent/sandbox/_signature.py::format_signature` - 当前单行签名(本 iter 加 `max_width`)
- `mutagent/src/mutagent/sandbox/_signature.py::format_callable_signature` - 含三条 fallback 的统一入口(本 iter 简化为 5 行)
- `mutagent/src/mutagent/sandbox/_signature.py::_format_schema_signature` - MCP schema 伪签名拼接(本 iter 删除)
- `mutagent/src/mutagent/sandbox/_signature.py::_is_kwargs_only_fallback` - `(**kwargs)` 形态识别(本 iter 删除)
- `mutagent/src/mutagent/sandbox/_signature.py::format_annotations_section` - 当前 Annotations 渲染
- `mutagent/src/mutagent/sandbox/_signature.py::_indent_continuation` - 当前 JSON 多行缩进辅助(本 iter 删除)
- `mutagent/src/mutagent/sandbox/_signature.py::_MISSING` - MCP optional-no-default sentinel(本 iter 改 repr 为 `<omit>`)
- `mutagent/src/mutagent/sandbox/_namespace.py::_render_function` - help() Layer 3 渲染入口
- `mutagent/src/mutagent/sandbox/_adapter_mcp.py::tool_func` - MCP wrapper 构造(保留 `_mcp_input_schema` 属性)
- `mutagent/src/mutagent/sandbox/_adapter_pysandbox.py::ns_func` - pysandbox peer wrapper 构造(删除 `_pysandbox_signature_str` 写入)
- `mutagent/src/mutagent/webui/_settings_mcp.py::_fn_signature` - settings 面板签名展示消费方

### 前置规范

- `feature-mcp-schema-help-display.iter2.md` - 三段式渲染基础,本 iter 在排版层改进

### 设计讨论脉络

1. **Signature 多行方案选型**:评估了 Black 风格 / 贪心填充 / 按类型长度分组 / Literal 内部折行四种方案,
   选定 Black 风格--可预测、符合 Python 社区肌肉记忆、agent 训练数据中高频出现。
2. **Signature 多行接口选型**:评估了"新增 `format_signature_multiline`"vs"`format_signature` 加 `max_width` 参数",
   选定后者--避免外层"先单行→判超宽→回头重渲染"两段式分发,接口更收敛,老调用路径默认行为不变。
3. **JSON 折行方案**:评估了 indent=2 微调 / 降阈值 / 自定义 Black 风格递归三种方案,
   选定 Black 风格--`browser_fill_form` 场景下行数从 45 → 12(减少 ~70%),token 大幅缩减。
4. **`_MISSING` repr 选型**:评估了 `...` + 脚注 / `OMIT` / `<omit>` / 完全省略 default / TS 风格 `?` 五种方案,
   选定 `<omit>`--尖括号占位符在 CLI/template/OpenAPI 圈约定俗成,自解释、零歧义、
   省下脚注 + `has_missing` 检测代码。Pydantic `Field(...)` 表"必填"的反向语义碰撞由此彻底消解。
5. **fallback 简化**:评估了"保留三条 fallback" / "删 `_format_schema_signature` 改 `[unavailable]` 标记" /
   "删全部 fallback 自然降级 `(**kwargs)`"三种方案。本地迭代未发布、`_pysandbox_signature_str` 无兼容包袱,
   选定第三种--`(**kwargs)` 是 Python agent 母语 hint,Annotations 段含完整 schema,
   "假签名"反而提供低质量误导信息。规范因此**整体删除**"fallback 路径要不要折行"这一类边界规则。
6. **阈值统一**:iter2 的 100 仅 Annotations 一处使用,无兼容负担;本 iter 统一为 80(PEP 8 + Black 默认)。
