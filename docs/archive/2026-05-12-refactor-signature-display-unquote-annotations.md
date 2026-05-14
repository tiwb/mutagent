# 签名展示去引号 — annotation 字符串不加引号显示

**状态**：✅ 已完成
**日期**：2026-05-12（2026-05-13 重写：方案收敛为 `_RawAnnotation` + 展示层 formatter；同日实施完成，112 个相关单测通过）
**类型**：改进
**前置**：`refactor-wrapper-faithful-signature.md`（已完成）

## 需求

`help(mutbot.logs)` 当前显示：

```
mutbot.logs(level: 'str' = 'INFO', logger: 'str' = '', ...) -> 'str'
```

类型标注带引号（`'str'`）是因为 `from __future__ import annotations`（PEP 563）
将 annotation 存为字符串，`inspect.formatannotation` 对字符串 annotation 走
`repr()` 分支，输出自带引号。这对 `help()` 和 MCP 设置 panel 的读者没有价值，
应显示为：

```
mutbot.logs(level: str = 'INFO', logger: str = '', ...) -> str
```

注意默认值 `'INFO'` 的引号是期望保留的（区分字符串值与其他值），只去掉
annotation 的引号。

## 根因

见 `refactor-wrapper-faithful-signature.md` 完整链路分析。两条路径殊途同归：

```
from __future__ import annotations
       │
       ▼
    ┌─────────────────────────────────────────┐
    │ 本地 sandbox: _wrap_async               │
    │   inspect.signature(coro_fn)            │
    │   → p.annotation = "str" (字符串)       │
    │                                          │
    │ peer describe: _describe_function       │
    │   _annotation_to_str → "str"            │
    │   build_signature → annotation = "str"  │
    └─────────────────────────────────────────┘
       │
       ▼
  Signature.__str__() → "level: 'str' = 'INFO'"
```

`Parameter.annotation` 无论来源如何，最终都是字符串 `"str"`，进入
`inspect.formatannotation` 后命中 `return repr(annotation)` 分支，必然带引号。

## 选定方案 — `_RawAnnotation` 包装类 + 展示层 formatter

**核心观察**：`Signature.__str__` → `Parameter.__str__` → `inspect.formatannotation`
的分派逻辑是：

```python
def formatannotation(annotation, base_module=None):
    if getattr(annotation, '__module__', None) == 'typing': ...   # ① typing 构造物
    if isinstance(annotation, types.GenericAlias): ...            # ② list[int] 这种
    if isinstance(annotation, type): ...                          # ③ 类对象 → __qualname__
    return repr(annotation)                                        # ④ 其余走 repr
```

四个分支排他且完备。字符串走 ④，`repr("str")` 变成 `"'str'"`，这就是引号的来源。

**关键技巧**：构造一个非 type、非 typing、非 GenericAlias 的轻量包装类，其
`__repr__` 返回原字符串内容。`formatannotation` 必定走 ④，输出就不带引号。

```python
class _RawAnnotation:
    """包装字符串 annotation，让 inspect.formatannotation 的 repr() 分支输出无引号。"""
    __slots__ = ('_s',)
    def __init__(self, s: str) -> None:
        self._s = s
    def __repr__(self) -> str:
        return self._s


def format_signature(sig: inspect.Signature) -> str:
    """渲染签名字符串，字符串 annotation 去引号，其余格式与标准库一致。"""
    new_params = [
        p.replace(annotation=_RawAnnotation(p.annotation))
        if isinstance(p.annotation, str) else p
        for p in sig.parameters.values()
    ]
    ret = sig.return_annotation
    if isinstance(ret, str):
        ret = _RawAnnotation(ret)
    return str(sig.replace(parameters=new_params, return_annotation=ret))


def format_callable_signature(func: Any) -> str | None:
    """统一展示入口：inspect.signature + wrapper fallback。"""
    ...
```

`build_signature()` **不改语义**：它继续把结构化 spec 忠实映射为
`inspect.Signature`，annotation 内部表示仍保持原始字符串。去引号只发生在
`format_signature()` / `format_callable_signature()` 这类展示层 helper 中，避免把
“观感问题”扩散成“内部模型变化”。

### 为何这是"最优解"

| 维度 | 该方案 |
|------|-------|
| 代码量 | ~25 行，两个 helper（`format_signature` + `format_callable_signature`） |
| 格式正确性 | 全部复用 CPython `Signature.__str__`——`/`、`*`、`**kwargs`、`empty` return、POSITIONAL_ONLY、KEYWORD_ONLY 等完全免维护 |
| 默认值引号 | 标准库照常用 `{!r}` 格式化默认值，`'INFO'` 保留引号；**零误伤** |
| Python 版本兼容 | 自动跟随 `inspect` 行为 |
| 测试负担 | 只需覆盖三类 annotation（字符串 / 类对象 / typing 泛型）+ return annotation 是否存在，无需逐 kind 组合 |
| 影响面 | 纯展示层工具函数，协议、`__signature__`、`build_signature` 语义都不变 |

### 正确性论证

`_RawAnnotation` 实例走 `formatannotation` 的四个分支：

1. `__module__` 是 `mutagent.sandbox.xxx`，不等于 `'typing'` — 不命中
2. 不是 `types.GenericAlias` — 不命中
3. 是实例不是 `type` 子类 — 不命中
4. 走到 `return repr(annotation)`，返回 `self._s`，即原字符串 ✓

四分支排他完备，**不存在"看脸"或边界依赖**。

`Parameter.replace` / `Signature.replace` 对 annotation 不做类型校验（typeshed
标注 `Any`，cpython 实现是直接赋值），`_RawAnnotation` 作为 annotation 值合法。

临时 Signature 仅用于 `str()` 渲染，用完即丢，不回流协议、不写入
`__signature__`，不会影响下游 `isinstance(ann, str)` 这类判断。

### annotation 形态覆盖

| annotation 值 | 来源 | 走哪条分支 | 最终显示 |
|--------------|------|-----------|---------|
| `"str"` | PEP 563 / `_annotation_to_str` | 包装后走 ④ | `str` ✓ |
| `"list[str]"` | typing 泛型 repr | 包装后走 ④ | `list[str]` ✓ |
| `"int \| None"` | union repr | 包装后走 ④ | `int \| None` ✓ |
| `"Any"` | MCP schema fallback | 包装后走 ④ | `Any` ✓ |
| 类对象 `int` | 非 PEP 563 环境 | 不包装，走 ③ | `int` ✓ |
| `list[int]` 对象 | 真运行期泛型 | 不包装，走 ② | `list[int]` ✓ |
| `typing.Optional[int]` | typing 对象 | 不包装，走 ① | `Optional[int]` ✓ |

## 兼容性

- `help()` 输出格式变化——纯观感改进，信息量不变
- 不改变协议结构（pysandbox describe 的 `signature` / `params` 字段仍保留）
- 不改变 `__signature__` 的内部表示
- 不改变 `build_signature()` 返回值中的 annotation 内部表示
- 不改变 wrapper 的调用行为
- MCP schema fallback 分支（`_fn_signature` 里 `_mcp_input_schema` 合成那条路径）
  本来就手动拼 `f"{pname}: {ptype}"`，天然不带引号，**无需改动，自动合规**

## 实施步骤清单

### Phase 1 — 共享工具函数

- [x] 新增 `_RawAnnotation` 包装类 + `format_signature(sig) -> str`
- [x] 新增 `format_callable_signature(func) -> str | None`
  - 放置位置：`src/mutagent/sandbox/_signature.py`
  - `build_signature()` 保持不变；展示 helper 统一消费 `inspect.signature()`、
    `_pysandbox_signature_str`、`_mcp_input_schema`
- [x] 单元测试：
  - 字符串 annotation 去引号（`"str"` → `str`）
  - 类对象 annotation 不动（`int` → `int`）
  - typing 泛型字符串（`"list[str]"` → `list[str]`，`"int | None"` → `int | None`）
  - 字符串默认值引号保留（`level: str = 'INFO'` 中 `'INFO'` 不动）
  - return annotation 为字符串 / 为空 / 为类对象 三种情况
  - 各 `Parameter.kind` 混合（POSITIONAL_ONLY `/`、KEYWORD_ONLY `*`、
    VAR_POSITIONAL `*args`、VAR_KEYWORD `**kwargs`）——由标准库自动处理，
    单测只需确认不被打破

### Phase 2 — 展示层接入

- [x] `src/mutagent/sandbox/_namespace.py:_render_function`
  - 改为统一调用 `format_callable_signature(func)`
- [x] `src/mutagent/webui/_settings_mcp.py:_fn_signature`
  - 改为统一调用 `format_callable_signature(func)`
- [x] `src/mutagent/webui/_settings_mcp.py:_fn_detail` 间接受益（调用 `_fn_signature`）
- [x] `src/mutagent/sandbox/share.py:_describe_function`
  - `signature` 字段改为 `format_signature(sig)`，保证 peer / 旧客户端兜底展示也去引号

### Phase 3 — fallback 收口

- peer 新路径：优先使用 `params -> build_signature -> format_signature`
- peer 旧路径：`format_callable_signature()` 收口 `_pysandbox_signature_str`
- server describe：`signature` 字段同步使用 `format_signature(sig)`，保证老客户端 /
  纯字符串兜底路径也得到无引号展示

### Phase 4 — 回归

- [x] `help()` 输出：本地 namespace、peer namespace、MCP tool 三种形态签名均不带引号
- [x] MCP 设置 panel：函数列表、函数详情签名均不带引号
- [x] 字符串默认值（如 `level='INFO'`）引号保留
- [x] 已有测试全部通过

## 附录：曾讨论过的替代方向

以下两个方向被排除，保留记录供后续回顾：

### 方案 A — 手动拼接签名字符串

遍历 `sig.parameters`，逐参数手写格式化，字符串 annotation 时不加引号。

**为何不选**：
- 需要完整复刻 `Parameter.__str__` / `Signature.__str__` 的所有细节：
  POSITIONAL_ONLY 组尾的 `/`、KEYWORD_ONLY 组前的裸 `*`、VAR_POSITIONAL
  的 `*args`、VAR_KEYWORD 的 `**kwargs`、默认值的 `{!r}`、return annotation
  为 `empty` 时跳过 ` -> ...`
- 代码量 ~30 行，单测要覆盖所有 kind 组合
- Python 版本升级时需跟踪 `inspect` 格式变化
- 选定方案完全覆盖同样场景，且代码量 1/2、测试面小得多

### 方案 B — `Signature.__str__()` 后正则替换

`str(sig)` 得到结果后用正则剥 annotation 引号。

**为何不选**：字符串默认值如 `name: str = "don't"` 会被正则误伤。无法安全
区分"annotation 位置的引号"和"默认值位置的引号"。

### 关于"保底方案"

不保留 A 或 B 作为保底。理由：

- 选定方案的分支分派是确定性的（`formatannotation` 四分支排他完备），不存在
  "某些输入会降级"的场景——要么全对，要么全错
- 若选定方案真出问题，A / B 同样面对相同输入，且 A 要写更多新代码出错概率
  反而更高，B 会误伤默认值
- 保底的合理语义是"主方案在部分输入上降级时兜底"，这里没有这样的降级路径

## 关键参考

- `src/mutagent/sandbox/_namespace.py:_render_function` — help() 渲染入口
- `src/mutagent/webui/_settings_mcp.py:_fn_signature` / `_fn_detail` — MCP 设置 panel
- `src/mutagent/sandbox/share.py:_annotation_to_str` — 类对象 → 字符串的既有映射
- `src/mutagent/sandbox/_signature.py:build_signature` — annotation 写入 `Parameter` 的方式
- CPython `Lib/inspect.py:formatannotation` — 本方案的核心依赖点（四分支分派）
