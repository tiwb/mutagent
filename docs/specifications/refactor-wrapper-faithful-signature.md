# MCP / Pysandbox Wrapper 真签名重构

**状态**：✅ 已完成
**日期**：2026-05-12
**类型**：重构（原 `bugfix-pysandbox-duplicate-signatures.md` 升级改名而来；展示层 bug 作为架构改造的自然副产品被消除）

## 需求
1. `help(fn)` 和 `_fn_detail` 对 pysandbox 包装的函数展示两份签名
2. 一份来自 `inspect.signature`（wrapper `(**kwargs: Any) -> Any`），对用户无意义
3. 一份来自 `__doc__` 首行（被 `_make_namespace_func` 硬拼进去的远程真实签名）
4. 直连 mutbot sandbox 时 help 输出正常，经过 pysandbox 包装后引入该问题

## 关键参考
- `src/mutagent/sandbox/_adapter_pysandbox.py:132` — `_make_namespace_func` 把签名拼进 `__doc__` 首行
- `src/mutagent/sandbox/_namespace.py:692-697` — `_render_function`（help layer 3）同时展示 `inspect.signature` + `__doc__`
- `src/mutagent/sandbox/_adapter_mcp.py:799-856` — `_make_tool_func` 的正确模式：用 `_mcp_description` + `_mcp_input_schema` 而非拼 `__doc__`

## 根因分析

`_make_namespace_func` 的设计初衷（注释可见）：

> signature_str 已经是 `(arg1, *, kwarg=...)` 风格的字符串，直接拼到 doc 顶部，help() 渲染时 inspect.signature 取本地空 wrapper 的 sig，但用户能在 doc 顶看到远端真实签名。

副作用：`help()` 的 `_render_function` 和 settings panel 的 `_fn_detail` 都可以同时拿到：
- `inspect.signature(func)` → `(**kwargs: Any) -> Any`（wrapper 空壳，对用户无价值）
- `__doc__` 首行 → 远程真实签名（被拼进去的）

两端都没做互斥处理，导致两份签名并列展示。

## 修复方向（初版，被后续设计方案替代）

1. `_make_namespace_func`：`__doc__ = doc`（不拼签名），远程签名存为 `_mcp_description`
2. `help()` 的 `_render_function`：当 `inspect.signature` 是 `**kwargs` 模式且存在 `_mcp_description` 时，签名展示 `()` 即可
3. `_fn_signature`：同上判断，避免展示无意义的 wrapper 签名

> 上述初版仅修显示层、保留 kwargs-only 的 wrapper 形态。讨论中发现同一组问题牵出更大的改造机会（位置调用、默认值丢失、MCP tool 展示同样有 `(**kwargs)` 误导），因此升级为下方的 Pythonic 方案。初版在此保留作为决策演进记录。

---

## 设计方案

### 目标

把 MCP tool wrapper 和 pysandbox namespace wrapper 从「`(**kwargs: Any)` 空壳 + 文档里夹真签名」改造为「`__signature__` 反映远端真实签名 + 运行时 bind 映射」。

收益：
- `mutbot.logs("ERROR")` / `fs.read("file.txt")` 位置调用可用（当参数形态允许）
- `help(fn)` / settings panel 展示一份忠实签名，原 bug 自然消失
- 默认值准确展示（当前 MCP tool 路径把 schema 里的 `default` 整个丢了）
- 参数名 typo 在客户端 `sig.bind` 阶段就 `TypeError`，不飞到服务端

### 核心机制：`__signature__` 伪装 + 运行时 bind

Python 的 `inspect.signature()` 优先读 `__signature__` 属性，不存在才 fallback 到函数实际形参。wrapper 代码仍是 `def ns_func(*args, **kwargs)` 通用接收，挂一个用 `inspect.Signature` 构造的真签名，wrapper 内部用 `sig.bind().apply_defaults()` 把位置参数规范化为 kwargs 再走 RPC：

```python
def ns_func(*args, **kwargs):
    bound = sig.bind(*args, **kwargs)   # 位置→kwargs、未知参数校验
    bound.apply_defaults()
    return _rpc_call(ns_name, fn_name, dict(bound.arguments))
ns_func.__signature__ = sig
```

纯标准库，无 `exec`，无代码注入面。

### 参数分组策略：保序 + 智能降级

Python 语法要求必选参数排在可选参数前。schema/签名里的顺序未必满足这个约束，所以：

> 按 properties / `inspect.signature` 原顺序遍历参数。遇到第一个「非 required 或自带 default」的参数之后，把后续所有参数强制转为 `KEYWORD_ONLY`（等效于插入 `*`）。

理由：
- 保留 server 原始顺序（调用者展示语义不被重排搅乱）
- required-first 的规范 schema 自动落到 `POSITIONAL_OR_KEYWORD`，位置调用可用
- 畸形 schema（required 夹在 optional 之间）自动降级为 keyword-only，不报错
- 不做「重排必选到前面」——一旦重排，位置参数序和用户看到的 schema 顺序就脱节了

### 协议扩展：pysandbox describe 新增 `params` 字段

当前 `pysandbox/namespaces.describe` 只返回 `signature: str`，字符串不够用来重建 `Signature` 对象（parse 字符串太脆弱）。扩展响应结构：

```python
functions: {
    fn_name: {
        "signature": "...",       # 保留，老字段、展示兜底
        "doc": "...",
        "params": [               # 新增，结构化参数列表
            {
                "name": "level",
                "kind": "POSITIONAL_OR_KEYWORD",   # Parameter.kind.name
                "default": "INFO",                # 省略 = 无默认值
                "annotation": "str",              # 字符串形式，仅展示
            },
            ...
        ],
        "kwargs_schema": {},      # 保留
    }
}
```

- `params` 是 additive optional 字段。老 server 不返回时客户端自动回落到 `(**kwargs)` wrapper（= 当前行为）
- `VAR_POSITIONAL`（`*args`）/ `VAR_KEYWORD`（`**kwargs`）在服务端生成 `params` 时跳过（RPC 无法承载位置可变参数）
- 不需要 capability 协商（新字段缺失即回退）

### MCP tool 路径：从 `input_schema` 构造

`_make_tool_func` 已经把 `input_schema` 挂在 wrapper 上。直接从 JSON Schema 构造同样的 params 列表后接入相同的签名构造逻辑。两条路径汇合到同一个 `_build_signature(params_spec) -> inspect.Signature` 工具函数。

### JSON Schema 类型映射（annotation 展示用）

| JSON Schema type | Python 展示 |
|---|---|
| `integer` | `int` |
| `number` | `float` |
| `string` | `str` |
| `boolean` | `bool` |
| `array` | `list` |
| `object` | `dict` |
| `null` | `None` |
| 缺失 / 复合 | `Any` |

### annotation 以字符串形式写入 `Parameter.annotation`

- JSON Schema → Python 类型是最佳努力，信息必然丢失（`integer` 不区分 `int` 与 `Literal[1,2,3]`）
- pysandbox server 端拿到的 `p.annotation` 是真对象，通过 RPC 只能传 `repr` 字符串
- 统一把字符串直接写进 `Parameter.annotation`（如 `"str"`、`"int | None"`），不单独维护 `_display_annotations` 字典（选字符串入 annotation：pysandbox/MCP wrapper 本就不是真 Python 类型正确的函数，`get_type_hints` 场景边缘；`help()` 展示价值明显更高；标准库 `inspect.signature` 字符串化时天然支持字符串 annotation）
- 代价：`typing.get_type_hints(fn)` 对这些 wrapper 会返回字符串而非类型对象，调用方需自行感知

### 默认值处理

- schema 的 `default` / signature 的 `p.default` 优先使用
- 非 JSON 可序列化的默认值（server 端：如 `datetime.now()` 这种）走 `_json_safe`，客户端以字符串形式占位，**仅用作展示**，`apply_defaults()` 时当作真值传回服务端是危险的（对象身份丢失）
- 策略：当 server 端 `_describe_function` 检测到默认值非 JSON 原生时，`default` 字段省略 + 挂 `default_repr: "<datetime.datetime(...)>"`；客户端构造 `Parameter` 时用 sentinel `inspect.Parameter.empty` 表示「没有默认值」，避免把 repr 字符串当默认值回传
- 等价于：这种参数在客户端变成必填，但这是安全的退化——宁可让用户显式传，也别悄悄传错

### 错误处理

- `sig.bind(*args, **kwargs)` 失败 → 直接抛原生 `TypeError`（与本地函数调用体验一致）
- 远端调用异常 → 保持现状（`MCPTransportError` / server 抛错透传）
- 签名构造失败（`params` 畸形等）→ 降级为 `(**kwargs)` wrapper + WARNING 日志，保护可用性

### 展示层简化

改造后：
- `_namespace.py::_render_function`：`inspect.signature(func)` 一把梭，真签名直接出，不需要任何特殊分支、不需要 `_display_signature` 属性
- `_settings_mcp.py::_fn_signature`：同样简化，`_mcp_input_schema` 分支可退役为仅在 wrapper 构造失败的 fallback 路径
- `_adapter_pysandbox.py::_make_namespace_func`：`__doc__ = doc`（停止往 doc 里拼签名）

### 兼容性

- **pysandbox 协议**：`params` 为 additive 可选字段，老 server 返回缺失 → 客户端回落 `(**kwargs)` wrapper，行为等同当前
- **wrapper 调用方**：原来只能 `fn(**kwargs)`，改造后仍然能 `fn(**kwargs)`；新增支持 `fn(positional)`——纯加法
- **`_async_original`、`_mcp_input_schema`、`_mcp_description`**：保留，`share.py::_handle_call` 等内部消费者不受影响
- **显示差异**：MCP tool `help()` 从 `(**kwargs: Any) -> Any` 变为真签名——观感变化明显但无功能破坏

### 实施顺序

先 MCP tool 路径（纯客户端改动 + 单测），再 pysandbox 路径（协议扩展 + server/client 双边）。这样 `_build_signature` 工具函数的接口在一期就被打磨定型，二期两条路径共享同一份实现（避免二期再回头改一期代码）。

## 关键参考（新增）

- `src/mutagent/sandbox/share.py:105` — `_describe_function`，服务端生成 describe 条目，需扩展 `params` 字段
- `src/mutagent/sandbox/_adapter_mcp.py:799` — `_make_tool_func`，MCP tool wrapper 构造入口
- Python stdlib `inspect.Signature` / `Parameter` / `BoundArguments` — 核心 API，详见 [inspect 文档](https://docs.python.org/3/library/inspect.html#inspect.Signature)
- `__signature__` 属性覆盖：`inspect.signature()` 查找顺序见 CPython `Lib/inspect.py::_signature_from_callable`

## 实施步骤清单

### Phase 1 — 共享工具函数 + MCP tool 路径

- [x] 新增 `_build_signature(params_spec) -> inspect.Signature` 工具函数，承载「保序 + 智能降级」参数分组策略、`annotation` 字符串化、默认值 sentinel 处理；放在 `src/mutagent/sandbox/` 下独立模块（具体文件名实施时定） → `sandbox/_signature.py`（`build_signature` + `try_build_signature` + `mcp_schema_to_specs` + `json_type_to_annotation`）
- [x] `_build_signature` 单元测试：required-first 正常、required 夹在 optional 中自动降级、全 optional、空参、annotation 缺失、default 缺失、非 JSON 原生 default（sentinel 路径） → `tests/test_signature_build.py` 26 例全通
- [x] `_adapter_mcp.py::_make_tool_func` 接入：从 `input_schema` 的 `properties` / `required` / `default` 构造 params_spec，调 `_build_signature` 生成 `__signature__`；wrapper 改用 `sig.bind().apply_defaults()` 路径；保留 `_async_original` / `_mcp_input_schema` / `_mcp_description` 属性不变
- [x] JSON Schema type → Python 展示字符串映射（按设计方案映射表），作为 `_build_signature` 的输入预处理或独立小函数 → `json_type_to_annotation`（含 union 类型合并）
- [x] 构造失败的降级路径：`_build_signature` 抛错时 wrapper 回落为 `(**kwargs)` 形态 + WARNING 日志 → `try_build_signature` 返回 `None` 时 `_make_tool_func` 选旧 wrapper
- [x] MCP tool 单元/集成测试：位置调用、关键字调用、默认值 `apply_defaults`、未知参数 `TypeError`、畸形 schema 降级、`help()` 输出单一真签名 → `TestMakeToolFuncSignature` 6 例全通

### Phase 2 — 展示层简化

- [x] `_namespace.py::_render_function`：移除为拼接 doc 签名而存在的特殊分支，简化为 `inspect.signature(func)` 一把梭 → 现有实现已经是“一把梭”风格，无需修改；bug 的根源（doc 拼接签名首行）在 Phase 4 停掉 pysandbox wrapper 的 `__doc__` 拼接后自然消失
- [x] `_settings_mcp.py::_fn_signature`：把 `_mcp_input_schema` 合成分支降级为 fallback（仅当 `__signature__` 缺失或降级为 空壳 `(**kwargs)` 时启用），主路径走 `inspect.signature` → 新增 `_is_kwargs_only_fallback` 检测 wrapper 降级形态
- [x] 回归测试：本地普通函数、MCP tool、未来的 pysandbox namespace 函数，三种形态 `help()` 输出均只有一份签名 → Phase 1 / 2 已测验前两种；pysandbox 路径 Phase 4 一并覆盖

### Phase 3 — pysandbox 协议扩展（server 端）

- [x] `share.py::_describe_function` 输出新增 `params` 字段：逐个参数生成 `{name, kind, default?, annotation}` 结构；跳过 `VAR_POSITIONAL` / `VAR_KEYWORD`
- [x] 非 JSON 原生默认值检测：省略 `default` 字段 + 另挂 `default_repr` 字符串（仅展示用），避免对象身份回传
- [x] annotation 以 `repr(p.annotation)` 或等价策略字符串化（空 annotation → 省略） → `_annotation_to_str`：类型对象用 `__name__`、其他走 `repr`
- [x] server 端单元测试：覆盖正常函数、无默认值、复杂默认值（dataclass/datetime）、含 `*args`/`**kwargs`、无类型注解、签名完全不可解析 → `TestDescribeFunctionParams` 8 例全通
- [x] **`_app_impl.py::_wrap_async — `__signature__``**：从 ``coro_fn`` 提取 ``inspect.Signature``，去掉 ``self`` 参数后挂到 ``wrapper.__signature__``。双路径受益：(a) 本地 sandbox ``help()`` 展示真签名；(b) ``_describe_function`` 透过 ``__signature__`` 获取去 ``self`` 的真签名 → peer describe 的 ``params`` 字段从此不再是空壳
- [x] **`_app_impl.py::_wrap_async` — 位置调用**：wrapper 形参从 ``(**kwargs)`` 改为 ``(*args, **kwargs)``，内部用 ``sig.bind(*args, **kwargs).apply_defaults()`` 规范化参数后调用 ``coro_fn``。签名不可解析时回落为仅 ``(**kwargs)``（旧行为）。至此本地 sandbox 位置调用 + ``help()`` 真签名全部兑现

### Phase 4 — pysandbox 协议扩展（客户端）

- [x] `_adapter_pysandbox.py::_make_namespace_func` 签名重构：接收 `params` 列表（可为 None），调 `_build_signature` 生成 `__signature__`；wrapper 用 `sig.bind().apply_defaults()` 转 RPC
- [x] 老 server 兼容：`params` 字段缺失或为 None 时，回落为 `(**kwargs)` wrapper（= 当前行为）；实现中用 `params is not None` 判断（空列表即无参函数，也走真签名路径）
- [x] 移除 `__doc__` 里拼接签名首行的代码，`__doc__ = doc`
- [x] 客户端测试：新 server（含 params）+ 老 server（无 params）两条路径各跑通；位置调用、默认值、未知参数错误、help() 输出 → `TestMakeNamespaceFuncSignature` 4 例 + `TestPeerBuildWithParams` 端到端 1 例全通

### Phase 5 — 端到端回归与清理

- [x] 端到端回归（`__signature__` 部分）：mutagent 连 mutbot 实例，``help(mutbot.logs)`` / ``help(mutbot.status)`` 签名正确
- [x] 端到端回归（位置调用部分）：``mutbot.exec_worker('1+1')`` 等位置调用可用
- [x] 原 bug 断言测试：``TestWrapAsyncSignature.test_render_function_single_signature`` — 断言 ``_render_function`` 输出中签名字符串仅出现一次（6 例）
- [x] 设计方案里被简化/退役的属性清理：``_display_signature`` 未实际引入（无需处理）；``_mcp_input_schema`` 在 ``_fn_signature`` / ``_fn_detail`` 中的使用是合理的 fallback 路径和参数表展示，保留
- [x] 相关文档/注释同步：``_make_namespace_func`` 顶部注释已更新（不再拼签名）；``_wrap_async`` 注释已补充 ``__signature__`` + 位置调用说明

