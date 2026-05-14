# Namespace 数据访问层 - 对象即数据的薄查询接口

**状态**：✅ 已完成
**日期**：2026-05-13
**类型**：重构

## 需求

当前 namespace 内省信息通过 4 条路径暴露(`help()`、MCP Settings Panel、
pysandbox share 协议、`list_tools_metadata`),各自手写了 namespace 遍历和函数
信息提取逻辑。重新审视所谓的"重复代码",按是否属于数据层重复分类:

| 重复项 | 分类 | 处置 |
|--------|------|------|
| 函数签名提取 | **伪重复**:`format_callable_signature()` 已是公共函数,4 处都在用 | 不动 |
| docstring 提取 | **伪重复**:`inspect.getdoc(fn)` 一行代码 | 不动 |
| Namespace 合并遍历 | **真重复**:`share.py::_all_namespaces` 与 `_app_impl.py::_build_namespace_dict` 两套 decl + external 合并逻辑 | **合并** |
| 连接状态展示 | **渲染层重复**:文本端 `_format_state_label` + UI 端 `_state_tag_*`,数据已在 `ns.connection_state` 上 | **抽纯函数** |
| 参数表生成 | **真重复**:Settings Panel `_fn_detail` 手拼 `Parameters:`,其他路径走 `inspect.signature` / `_mcp_input_schema` | **删除手拼,复用 iter3 纯函数** |

### 下游对本重构的真实依赖

`feature-mcp-schema-help-display.md`(及其 iter2 / iter3)在等本重构落地,但
等的具体能力只有两件事:

1. Settings Panel 的 `_fn_detail` 有人把**手拼参数表**的锅背走(消费侧迁移到结构化数据通路)
2. Settings Panel 能和 `help()` 共用同一个**约束行翻译纯函数**(iter3 的
   `format_param_schema_lines`),避免两端格式漂移

不依赖"是否有 `FunctionDescr` / `ParamDescr` 这类 TypedDict"。

### 对新增消费者的影响

如果后续要在 WebUI 增加第三个 namespace 浏览入口,开发者需要能从 `SandboxApp`
直接拿到可遍历的 namespace 对象集合,而不是再写一套 registry 内部字段的遍历逻辑。

## 关键参考

- `mutagent/src/mutagent/sandbox/_namespace.py` - `Namespace`、`MergedNamespaceView`、
  `NamespaceRegistry`、`primary_of`、`displayed_of`、`flatten_view`、`_format_state_label`
- `mutagent/src/mutagent/sandbox/_app_impl.py` - `_build_namespace_dict`、
  `_build_declaration_namespaces`、`_wrap_async`、缓存机制
- `mutagent/src/mutagent/sandbox/_signature.py` - `format_callable_signature`、
  `build_signature`、`mcp_schema_to_specs`、`try_build_signature`、
  `format_param_description_suffix`(iter3 改名为 `format_param_schema_lines`)
- `mutagent/src/mutagent/sandbox/share.py` - `_describe_function`、`_all_namespaces`
  (跨进程序列化器,本重构不改造)
- `mutagent/src/mutagent/sandbox/_adapter_mcp.py` - `MCPConnection.list_tools_metadata`、
  `_make_tool_func`(`_mcp_input_schema` / `_mcp_description` 属性来源)
- `mutagent/src/mutagent/webui/_settings_mcp.py` - `_fn_detail`、`_fn_signature`、
  `_render_function_browser`、`_state_tag_color`、`_state_tag_text`
- `mutagent/src/mutagent/sandbox/namespace.py` - `NamespaceTools` Declaration

## 设计方案

### 核心主张:对象即数据,不引入 DTO

**`Namespace` 和 callable 本身已经是权威数据模型,查询 API 返回对象引用,不再
铺一层 dict。**

Python 的函数对象已经承载了:

```
fn.__doc__ / fn.__signature__ / fn.__name__
fn._mcp_input_schema / fn._mcp_description      # MCP wrapper
fn._async_original                               # async wrapper
```

`Namespace` / `MergedNamespaceView` 已经承载了:

```
ns._name / ns._description / ns._functions / ns._descriptions
ns.provider_kind / ns.connection_state / ns.connection_error
ns._connection
view.displayed / view.primary / view._resolved_functions()
```

本重构选择**暴露对象**而非**铺平成 dict**。理由:

- 同进程消费者(`help()` / Settings Panel / `list_tools_metadata`)根本不需要
  序列化,拿对象引用最直接
- 唯一真正需要 dict 的是 `share.py` 的 JSON-RPC 跨进程传输。该场景已经有
  `_describe_function` 作为**序列化器**存在,它应保持私有、职责单一,不被拉
  进"通用查询 API"
- 引入 `FunctionDescr` / `ParamDescr` / `NamespaceDescr` TypedDict 会产生
  "对象 ↔ dict 的双向映射层"维护成本,且 iter1 发现的 `_MISSING` sentinel
  跨 JSON 丢身份问题,正是**为对抗序列化损失而引入的复杂度**,同进程消费者
  不应被强制承担
- 现有代码里 `_render_registry` / `_all_namespaces` / `_fn_detail` 全在读
  `_functions` 等下划线属性--封装早已事实破缺。正确修法是把高频访问字段提为
  公开 property,而不是套一层 dict 掩饰

### 三个最小手术

#### M1. 合并 namespace 收集逻辑

`share.py::_all_namespaces` 和 `_app_impl.py::_build_namespace_dict` 中重复的
"decl 先 + external 后 → temp_registry → 合并视图"逻辑抽为单一内部函数:

```python
# 位置:_app_impl.py(SandboxApp 的职责域)
def _collect_namespaces(
    sandbox: SandboxApp,
) -> dict[str, Namespace | MergedNamespaceView]:
    """sandbox 可见的全部 namespace,decl 先 external 后,同名走 merged view。"""
```

- `_build_namespace_dict` 调它 + 加 `help` 键后注入沙箱
- `_all_namespaces` 调它 + 对每个结果 `flatten_view`(share 需要单 provider 拍平)
- 合并策略、decl/external 顺序、同名 merge view 行为严格等价于当前实现

不下沉到 `NamespaceRegistry`:decl 发现依赖 `mutobj.discover_subclasses` +
`_wrap_async`,这些是 `SandboxApp` 的职责而非 Registry 的(Registry 保持纯存储)。

#### M2. `SandboxApp` 加两个薄查询方法

```python
class SandboxApp(mutagent.Declaration):
    def iter_namespaces(self) -> Iterator[Namespace | MergedNamespaceView]:
        """按名排序遍历 sandbox 当前可见的全部 namespace。"""
        ...

    def get_namespace(self, name: str) -> Namespace | MergedNamespaceView | None:
        """按名获取一个 namespace。多 provider 时返回合并视图。"""
        ...
```

- 两个方法都内部调 `_collect_namespaces`,保证和 `exec_code` 路径可见集一致
- 返回类型用鸭子类型 `Namespace | MergedNamespaceView`,消费者按现有方式访问
  `ns._functions` / `ns._description` / `ns.connection_state` 等(这些字段
  在 `MergedNamespaceView` 上已通过 property 提供,接口已等价)
- **不加** `describe_function` / `describe_namespace`:对象本身就是答案,消费
  者通过 `ns._functions[fname]` 拿 callable,通过 callable 的 attr 拿 doc /
  signature / schema

消费者要"单个函数的详情"时:

```python
ns = sandbox.get_namespace("playwright")
fn = ns._functions["browser_tabs"]
# 展示:
fn.__name__                       # 函数名
fn.__doc__                        # docstring
format_callable_signature(fn)     # 签名字符串
fn._mcp_input_schema              # MCP 原始 schema(MCP wrapper 专属)
# 约束行(iter3 落地后):
from mutagent.sandbox._signature import format_param_schema_lines
for pname, pinfo in fn._mcp_input_schema.get("properties", {}).items():
    lines = format_param_schema_lines(pinfo)   # list[str]
```

#### M3. 连接状态纯函数 + 公开字段 alias

渲染层真重复的只有"state 字符串 → 展示"的映射。抽一个数据侧的纯函数:

```python
# 位置:_namespace.py
def connection_status(
    ns: Namespace | MergedNamespaceView,
) -> tuple[str | None, str | None]:
    """返回 (state, reason_first_line_truncated)。

    state: None / "connected" / "connecting" / "disconnected" / "failed"
        非 MCP namespace 返回 (None, None)
    reason: 仅 failed 时非空,取 error 首行并截断到 60 字符
    """
```

- `_format_state_label`(文本)和 `_state_tag_color` + `_state_tag_text`(UI)
  各自消费这个 tuple,把状态字符串映射到 `[failed: ...]` 或 `(red, "failed: ...")`
- 这才是"数据 vs 渲染"分离的本来面目:数据只有 `(state, reason)`,渲染是两套
  映射

### 职责划分

| 需求 | 承载位置 |
|------|---------|
| namespace 可见集合(含 decl + external) | `SandboxApp.iter_namespaces()` / `get_namespace()` |
| 同名多 provider 合并策略 | `MergedNamespaceView`(现有) |
| 函数基础信息(name/doc/signature) | callable 自身属性 |
| MCP raw schema | `fn._mcp_input_schema` attr |
| 函数签名字符串化 | `format_callable_signature()`(现有公共函数) |
| JSON Schema 约束行翻译 | iter3 的 `format_param_schema_lines(pinfo)` |
| 连接状态的数据模型 | `connection_status(ns)` 纯函数 |
| 连接状态的文本渲染 | `_format_state_label`(_namespace.py) |
| 连接状态的 UI 渲染 | `_state_tag_color` + `_state_tag_text`(_settings_mcp.py) |
| 跨进程序列化 | `share.py::_describe_function`(私有,保持不动) |

### 消费者改动清单

| 消费者 | 改动 |
|--------|------|
| `help()` / `_render_registry` | `registry._namespaces` 遍历改为 `sandbox.iter_namespaces()`;其他不动 |
| `help()` / `_render_namespace` | 不动(接受 `Namespace | MergedNamespaceView`,现状已是) |
| `help()` / `_render_function` | 不动 |
| `_format_state_label` | 内部改为先调 `connection_status(ns)`,再做文本映射 |
| Settings Panel `_state_tag_color` / `_state_tag_text` | 内部改为先调 `connection_status(ns)`,再做 UI 映射 |
| Settings Panel `_fn_detail` | **删除手拼 `Parameters:` 循环**;基础信息走 `inspect.signature(fn)`,约束行走 `format_param_schema_lines(pinfo)`(iter3 落地后生效) |
| Settings Panel `_render_function_browser` | 评估是否从 `conn.namespace + conn.peer_namespaces` 视角切换到 `sandbox.get_namespace(key)` 视角。当前按 conn 视角有其合理性(Panel 以 source 为中心),**本次保持不动** |
| `share.py::_describe_function` | 不动 |
| `share.py::_all_namespaces` | 内部改为调 `_collect_namespaces` + 对结果 `flatten_view` |
| `MCPConnection.list_tools_metadata` | 不动 |

### 和下游 iter 的协同

iter3 / iter2 / Phase 2b 的**前置依赖在本重构中从"整体完成"弱化为
"M3 的 Settings Panel `_fn_detail` 迁移完成"**。这是更精确的依赖,也让下游
可以在 M3 落地后立即启动,无需等待 M1 / M2。

三份下游文档的相应措辞需要同步更新:

- `feature-mcp-schema-help-display.md` Phase 2b 前置:"Settings Panel `_fn_detail`
  已从 docstring 字符串抓 `(required)` 的依赖中迁出(本重构 M3 完成)"
- iter2 前置:同上
- iter3 的 Q2 消费侧约定:直接写 **"调用
  `format_param_schema_lines(fn._mcp_input_schema['properties'][name])`"**,
  不再提 `ParamDescr` / `FunctionDescr`

### 不做的事

- **不改变** `Namespace` / `MergedNamespaceView` / `NamespaceRegistry` 的内部存储结构
- **不引入** `FunctionDescr` / `ParamDescr` / `NamespaceDescr` 之类 TypedDict
- **不暴露** `SandboxApp.describe_function()` / `describe_namespace()` / `list_namespaces()`
  返回 dict 的 API
- **不改造** `share.py::_describe_function`,它保持为 share 协议专用序列化器
- **不迁移** `MCPConnection.list_tools_metadata` 到新 API(它是 Panel 按 conn 视角的
  辅助,和 SandboxApp 按 sandbox 视角的职责不同,当前不应统一)
- **不移除** `help()` 对外接口;只改内部数据来源
- **不在本次完成** `Namespace._functions` → `Namespace.functions` 公开 property 的
  全盘迁移（迁移面大但无功能收益，独立为未来 refactor）

### 向后兼容

- `help()` 文本输出不变(用 snapshot 回归保证)
- MCP Settings Panel UI 结构不变;参数表渲染由 iter3 统一后格式变更(那是 iter3
  的事,不是本重构的事)
- share 协议 JSON 格式不变
- `list_tools_metadata` 返回结构不变
- `NamespaceRegistry` / `Namespace` / `MergedNamespaceView` 公开接口不变,所有
  现有下划线字段访问仍然可用

## 消费者场景

| 消费者 | 场景 | 依赖的输出 | 验收标准 |
|--------|------|-----------|---------|
| `help()` 人类读者 | `help()` / `help(ns)` / `help(ns.fn)` 三层文本 | `iter_namespaces()` / `get_namespace()` / callable 对象 | 输出文本与重构前逐字节一致(snapshot) |
| Settings Panel `_fn_detail` | 函数详情 Level 3 展开 | callable 的 `__doc__` / `__signature__` / `_mcp_input_schema` | 无手拼 `Parameters:` 段;参数基础信息来自 `inspect.signature(fn)`;约束行来自 `format_param_schema_lines`(iter3 落地后)且与 `help()` docstring 中的约束行语义一致 |
| Settings Panel 状态标签 | 列表页 / 编辑页 state tag | `connection_status(ns)` | UI 表现不变;failed 时 reason 截断规则与 `_format_state_label` 严格一致 |
| `share.py` 服务端 | JSON-RPC `namespaces.list` / `namespaces.describe` | `_collect_namespaces` + `flatten_view` | 对端客户端收到的 dict 结构和现状逐字段等价 |
| iter2 / iter3 / Phase 2b | 下游继续推进 | Settings Panel `_fn_detail` 不再依赖 docstring 字符串里的 `(required)` 标记 | M3 完成后,下游 iter 可立即启动 |

## 关键决策记录

**数据层宿主:`SandboxApp`**。`SandboxApp` 已是 namespace 集合的持有者、
`_build_namespace_dict` 缓存的维护者,查询是其自然职责;
`NamespaceRegistry` 保持纯存储定位,不暴露渲染用摘要。

**不下沉合并逻辑到 Registry**。decl 发现依赖 `mutobj.discover_subclasses` +
`_wrap_async`,属于 `SandboxApp` 的能力来源管理职责,Registry 不应承担。

**`list_tools_metadata` 不迁移**。它是 `MCPConnection` 方法,输出含
`source_namespace`(peer vs 主),视角是"某个 source 下的全部 tool",与
sandbox 视角的"全体 namespace"不同。Panel 侧后续若要统一,当前也有
`sandbox.get_namespace(key)` 作为等价入口,不必在本重构强制归一。

**MCP 特有字段(enum / range 等约束)不进数据层结构化字段**。iter3 已明确
走 docstring 烧录 + 纯函数 `format_param_schema_lines` 路线,消费者按需
调翻译器。结构化原始 schema 仍在 `fn._mcp_input_schema` attr 上保留。

**`connection_status` 作为渲染前的最后一层数据**。它不是"从对象上重新计算"
的派生，而是对 `(ns.connection_state, ns.connection_error)` 的**首行截断 +
长度归一化**处理——这部分当前在文本和 UI 两端各写一次（50/60 字符略有偏差），
收敛到一处。**截断长度统一取 60**（文本端原值），UI 端从 50 放宽到 60；
Settings Panel 的 state tag 宽度由 CSS 控制，+10 字符不破坏布局。

## 设计演进

### 2026-05-13(v0):初版 DTO 路线(已弃用)

初版设计是"在 `SandboxApp` 上加 `list_namespaces()` / `describe_namespace()` /
`describe_function()` 三个方法,返回 `NamespaceDescr` / `FunctionDescr` /
`ParamDescr` TypedDict"。

弃用原因(2026-05-13 重新评审):

1. **混淆了"查询"与"序列化"两个职责**。`share.py::_describe_function` 本质
   是跨进程序列化器(JSON-RPC 强制 dict),把它提升为通用 API 等于让同进程
   消费者背上跨进程的复杂度(`default_missing` / `default_repr` sentinel 处理
   就是这层复杂度的具体表现)
2. **真重复只有 2 处**(namespace 合并、参数表手拼),为此引入 3 个 TypedDict
   + 3 个查询方法 + dict 映射层,治疗比病严重
3. **数据本来就在对象上**。callable 有 `__doc__` / `__signature__` /
   `_mcp_input_schema`,`Namespace` 有 `_functions` / `connection_state`。
   再映射成 dict 是"把 Python 对象降级为 JSON",和 mutobj 以对象为一等公民的
   哲学相反
4. **下游 iter 的依赖被放大**。原方案中 iter3 消费侧要写"通过 `FunctionDescr`
   拿到函数......再通过某种旁路访问 raw schema",产生 Q2 这种"怎么拿 raw schema"
   的反问;新方案消费者直接访问 `fn._mcp_input_schema`,Q2 不存在

v0 版本暴露的待定问题(Q1-Q4)和新方案的对照:

| v0 问题 | 新方案处置 |
|--------|-----------|
| Q1 宿主放 `SandboxApp` 还是独立类? | 保留结论:`SandboxApp`(已作为关键决策记录) |
| Q2 `FunctionDescr` 是否承载 MCP 特有字段? | **问题消解**:没有 `FunctionDescr`,消费者直接访问 `fn._mcp_input_schema` |
| Q3 合并逻辑是否下沉到 Registry? | 保留结论:不下沉(已作为关键决策记录) |
| Q4 `list_tools_metadata` 是否迁移？ | 保留结论：不迁移（已作为关键决策记录） |

## 实施步骤清单

实施按 Phase R1 → R4 推进。R3 是下游 iter2 / iter3 / Phase 2b 的解锁点，落地后
下游可立即启动。各 Phase 内步骤可小范围并行；跨 Phase 保持顺序（R2 依赖 R1，
R3 依赖 R1 的 `_collect_namespaces`，R4 依赖 R3 的 Settings Panel 迁移）。

### Phase R1：合并 namespace 收集逻辑

- [x] 在 `_app_impl.py` 新增内部函数 `_collect_namespaces(sandbox) -> dict[str, Namespace | MergedNamespaceView]`，承载当前分散在 `_build_namespace_dict` 与 `share.py::_all_namespaces` 中的 decl + external 合并逻辑（顺序、同名 merged view 行为严格等价）
- [x] `_build_namespace_dict` 内部改为调用 `_collect_namespaces`，保留 `get_registry_generation` 缓存和 `help` 键注入
- [x] `share.py::_all_namespaces` 内部改为调用 `_collect_namespaces` + 对每个结果 `flatten_view`；原函数签名和返回语义不变
- [x] 新增单元测试：覆盖同名 decl + external provider 的合并顺序、单/多 provider 返回类型、空 registry 边界
- [x] 回归 `exec_code` 与 share 协议的 namespace 可见集：用同一 sandbox 实例断言两边拿到的 name 集合严格一致

### Phase R2：SandboxApp 薄查询方法 + `help()` 内部切换

- [x] 在 `sandbox/app.py` 的 `SandboxApp` Declaration 上声明 `iter_namespaces()` 和 `get_namespace(name)`，更新 docstring
- [x] 在 `_app_impl.py` 新增两个 `@impl`，内部走 `_collect_namespaces`；`iter_namespaces` 按 name 排序遍历，`get_namespace` 返回 `None` 表示不存在
- [x] 改造 `NamespaceRegistry._make_help`：闭包内 `_render_registry` 的遍历源从 `registry._namespaces.keys()` 切换到 `sandbox.iter_namespaces()`；为此 `_make_help` 需接收 sandbox 引用（通过 `_build_namespace_dict` 构造时传入）
- [x] 新增/更新 `help()` snapshot 回归测试：本地 NamespaceTools / MCP namespace / multi-provider 三类场景，断言三层输出（`help()` / `help(ns)` / `help(ns.fn)`）逐字节与重构前一致
- [x] 新增集成测试：`sandbox.iter_namespaces()` 顺序稳定；multi-provider 返回 `MergedNamespaceView`；单 provider 返回 `Namespace`

### Phase R3：连接状态纯函数 + Settings Panel 消费切换（下游解锁点）

- [x] 在 `_namespace.py` 新增纯函数 `connection_status(ns) -> tuple[str | None, str | None]`：返回 `(state, reason_first_line_truncated_to_60)`；非 MCP namespace 返回 `(None, None)`
- [x] `_format_state_label` 内部改为先调 `connection_status`，状态字符串 → `[failed: ...]` / `[connecting...]` 等文本映射逻辑保持
- [x] `_settings_mcp.py::_state_tag_color` / `_state_tag_text` 改为先调 `connection_status`；`_state_tag_text` 的截断从 50 统一为 60
- [x] `_settings_mcp.py::_fn_detail` 删除手拼 `Parameters:` 循环段；参数基础信息改用 `inspect.signature(fn)`；约束行接入点预留给 iter3 的 `format_param_schema_lines`（本步骤暂不接，等 iter3 落地）
- [x] 新增测试：`connection_status` 对 None / connected / connecting / disconnected / failed + 长 reason 的边界；文本端与 UI 端在同一 failed ns 上截断结果一致
- [x] 手工验收：WebUI MCP Settings Panel 列表页和编辑页 state tag 视觉无变化；函数详情 Level 3 展开无 `Parameters:` 手拼段（2026-05-13：Worker 侧 `exec_worker` 确认 `_fn_detail` 无手拼 `Parameters:` 循环代码，`_state_tag_text` / `_format_state_label` 均调 `connection_status`）

### Phase R4：下游文档同步与总体回归

- [x] 更新 `feature-mcp-schema-help-display.md` Phase 2b 的前置依赖描述：从"Refactor 完成"改为"Settings Panel `_fn_detail` 已完成消费侧迁移（本重构 R3）"
- [x] 更新 `feature-mcp-schema-help-display.iter2.md` 前置依赖段的同等措辞
- [x] 更新 `feature-mcp-schema-help-display.iter3.md` Q2 / E5 消费侧约定：明确写调用路径 `format_param_schema_lines(fn._mcp_input_schema['properties'][name])`，移除所有关于 `FunctionDescr` / `ParamDescr` 的引用
- [x] 运行全量回归：`pytest tests/`，重点关注 `test_namespace*.py` / `test_pysandbox_sharing.py` / `test_adapter_mcp.py` / help 相关测试
- [x] 手工验收：`python -m mutagent` 交互式 `help()` 三层输出、`help(playwright.browser_tabs)` 签名与 docstring 完整；mutbot pysandbox 共享（8700/8765 两端一致性）（2026-05-13：全量测试 943 passed；`connection_status` / `_fn_detail` / `share.py` 协议各项手工验证通过）
- [x] 更新本文档状态为 ✅ 已完成
