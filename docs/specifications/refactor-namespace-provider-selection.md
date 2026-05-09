# Namespace Provider 选择逻辑收敛 设计规范

**状态**：🔄 实施中
**日期**：2026-05-09
**类型**：重构

## 需求

1. 当前 namespace 系统有多条独立路径各自决定「哪个 provider 是主角」，逻辑不一致，每次微调要改多处。
2. `_build_namespace_dict`（sandbox exec_code 用）和 `_all_namespaces`（pysandbox share export 用）各有一套合并逻辑，且 share export 直接 `dict.update` 替换整个 namespace，会丢 external provider 的非冲突函数。
3. `_description`、`_displayed_providers`、`_all_namespaces` 选代表各自对「主 provider」有独立定义，不在同一概念下。

## 关键参考

- `mutagent/src/mutagent/sandbox/_namespace.py` — `MergedNamespaceView`、`_displayed_providers`、`_render_*`、`NamespaceRegistry`
- `mutagent/src/mutagent/sandbox/_app_impl.py` — `_build_namespace_dict`、`_build_declaration_namespaces`
- `mutagent/src/mutagent/sandbox/share.py` — `_all_namespaces`（pysandbox export 侧选代表 provider）
- `mutagent/src/mutagent/sandbox/_adapter_mcp.py` — `_refresh_namespace`（tool namespace 描述来源 + 直接改 `ns._functions` 的 cache 失效问题）
- `mutagent/src/mutagent/sandbox/_adapter_pysandbox.py` — `build_peer_namespaces`（peer namespace 描述来源 + `(shared from ...)` 后缀）

## 当前架构（四条独立路径）

```
sandbox exec_code 路径：
  _build_namespace_dict
    ├── _build_declaration_namespaces (本地发现)
    ├── 合入 main registry 的外部 providers
    ├── temp_registry (decl 先 → 外部后，先注册先赢)
    └── MergedNamespaceView._description → 第一个有 active fn 的 provider

sandbox help() / sandbox 调用路径：
  MergedNamespaceView._displayed_providers → 有 active fn 的 provider
  _render_registry / _render_namespace → displayed 驱动多/单 provider 模式

pysandbox share export 路径：
  _all_namespaces
    ├── 遍历 main registry → 选 "connected 或首个" provider（独立算法）
    └── decl_namespaces.update() → decl 整个替换外部（非函数级合并）

MCP 连接注册路径：
  _adapter_mcp: tool namespace → _description 来自 MCP instructions
  _adapter_pysandbox: peer namespace → _description 来自 describe + "(shared from ...)"
```

四套「选主」算法接近但不一致。`_description` 走「首个有 active 函数」、`_displayed_providers` 走「所有有 active 函数」、`_all_namespaces` 走「首个 connected 否则首个」、`_connection` 走「OR 聚合」。前三者语义高度重叠，应当收敛到同一概念。

## 设计方案

### 核心概念：MergedNamespaceView 上的 displayed + primary

给 `MergedNamespaceView` 添加两个属性，统一表达「这个合并视图的有效 provider 是谁、权威 provider 是谁」：

```python
class MergedNamespaceView:
    @property
    def displayed(self) -> list[Namespace]:
        """参与渲染的 provider 列表：有 active 函数且 _functions 非空。

        语义：贡献了至少一个 active（非 shadowed）函数的 provider。
        过滤掉空壳 tool ns（连接前 _functions 还没填）和全被 shadow 的 provider。
        这是「权威」的根本属性。
        """
        resolved = self._resolved_functions()
        active_ids = {id(rf.active) for rf in resolved.values()}
        return [p for p in self._providers
                if p._functions and id(p) in active_ids]

    @property
    def primary(self) -> Namespace | None:
        """合并视图的主 provider。displayed[0] 的派生。

        全空壳时退化为 _providers[0]；无 provider 返回 None。
        """
        d = self.displayed
        if d:
            return d[0]
        return self._providers[0] if self._providers else None
```

`primary` 不是独立算法，是 `displayed` 的派生 —— 一份算法、一份缓存命中。

### 模块级 helper：统一 view / Namespace 的访问入口

`registry.get(name)` 既可能返回 `Namespace` 也可能返回 `MergedNamespaceView`。普通 `Namespace` 没有 `primary / displayed`，消费者本不该到处写 `isinstance`。

```python
def primary_of(ns: NamespaceLike) -> Namespace:
    """统一访问主 provider —— Namespace / view 通吃。

    - Namespace：返回自身
    - MergedNamespaceView：返回 view.primary（无 displayed 时退化首个 provider）
    """
    if isinstance(ns, MergedNamespaceView):
        return ns.primary or ns._providers[0]
    return ns


def displayed_of(ns: NamespaceLike) -> list[Namespace]:
    """统一访问 displayed providers 列表。Namespace 单一，返回空列表
    （表示「单 provider 路径，无多 provider 渲染」）。
    """
    if isinstance(ns, MergedNamespaceView):
        return ns.displayed
    return []
```

所有消费者一律调这两个 helper，原 `_displayed_providers(ns)` 模块函数变成对 `displayed_of` 的薄 alias（保留外部 import 兼容）。

### 策略统一表

| 属性 / 函数 | 当前算法 | 改为 |
|---|---|---|
| `MergedNamespaceView._description` | 内联：第一个有 active fn 的 provider | `self.primary._description` |
| `_displayed_providers(ns)` 模块函数 | 自己重算一遍 | `displayed_of(ns)`（→ `view.displayed`） |
| `_all_namespaces` 选代表 | 「connected 或首个」独立算法 | 走 view + 拍平（详见下节） |
| `_resolve_functions` 「先注册先赢」 | 不动 | 不动（调用语义） |
| `_render_registry / _render_namespace` | 调 `_displayed_providers(ns)` | 调 `displayed_of(ns)`，渲染分支不动 |

### 不收敛的部分（重要）

**`_connection / connection_state / connection_error` 保留 OR 聚合语义，不走 `primary`**。

这三个表达的是「这个合并视图的连接状态」，是任意一个 provider 还活着就算可用，不是「哪个 provider 是主角」。例如 decl provider 没有 connection（state=None），peer provider connected —— 不能因为 decl 是 primary 就报告 state=None。

```python
# 保留现状
@property
def _connection(self) -> "MCPConnection | None":
    # 优先 connected provider 的 connection；否则首个非 None
    ...

@property
def connection_state(self) -> str | None:
    # any connected → connected; any connecting → connecting; else 首个非 None
    ...

@property
def connection_error(self) -> str | None:
    # 取第一个 failed provider 的 error
    ...
```

「权威 provider」与「连接聚合状态」是两个独立概念，不要强行合并。

### `_all_namespaces` 改为走 view + 拍平（关键决策）

当前 `_all_namespaces` 用 `decl_namespaces.update(result)` 让 decl 整个 namespace 替换 external provider。这与 exec_code 路径的「函数级 shadow，external 非冲突函数仍可见」不一致 —— **export 出去的函数集比 exec_code 看到的少**。

新方案：用同一个 `temp_registry` 合并 decl + external，对每个 view 拍平成一个临时 `Namespace`，函数集取 view 合并后的全集（与 exec_code 路径完全一致）。

```python
def _all_namespaces(sandbox: SandboxApp) -> dict[str, Namespace]:
    """收集 sandbox 当前可见的全部 namespace（拍平成单 provider）。

    流程：
    1. 用 temp_registry 合并 decl + external，与 _build_namespace_dict 同算法
    2. 对每个 view / Namespace 拍平成单个 export Namespace：
       - description = primary_of(ns)._description
       - functions / descriptions = view 合并后的 active 集（exec_code 同集）
       - provider_kind = primary_of(ns).provider_kind
    3. 单 provider 名直接返回原 Namespace（无需拍平）
    """
    from mutagent.sandbox._app_impl import _build_declaration_namespaces

    registry = getattr(sandbox, "_registry", None)
    decl_namespaces = _build_declaration_namespaces(sandbox)

    temp_registry = NamespaceRegistry()
    # decl 先注册（与 exec_code 路径同序：decl 优先于 external）
    for ns in decl_namespaces.values():
        temp_registry.add(ns)
    if registry is not None:
        for providers in registry._namespaces.values():
            for p in providers:
                temp_registry.add(p)

    result: dict[str, Namespace] = {}
    for name in temp_registry._namespaces:
        ns = temp_registry.get(name)
        if isinstance(ns, MergedNamespaceView):
            result[name] = _flatten_view(ns)
        else:
            result[name] = ns
    return result


def _flatten_view(view: MergedNamespaceView) -> Namespace:
    """把 multi-provider view 拍平成对端可见的单 Namespace。

    description / provider_kind 走 primary；functions 走合并 active 集。
    与 exec_code 路径函数可见性完全一致，避免 export 丢函数。
    """
    p = view.primary or view._providers[0]
    flat = Namespace(view.name, description=p._description,
                     provider_kind=p.provider_kind)
    for fn_name, fn in view._functions.items():
        flat.register(fn_name, fn, view._descriptions.get(fn_name, ""))
    return flat
```

这样 `_all_namespaces` 与 `_build_namespace_dict` 共享同一份合并算法，且 export 函数集与本地可见函数集一致。

> **注**：`_flatten_view` 拍平后的临时 Namespace 不挂 `_connection`，对端拿到的是无连接状态的「快照」—— 对端 sandbox 只关心函数描述与调用，不应感知本端的 MCP 连接细节。

### 顺手修：`_refresh_namespace` 触发 view 缓存失效

`MergedNamespaceView._resolved_cache_key = tuple(id(p) for p in providers)`，只在 providers 列表变化时失效。但 `_adapter_mcp._refresh_namespace` **直接改 `ns._functions`**（id 不变），cache 不会失效，导致 `displayed / primary / _description` 拿到旧结果。

当前消费者基本不依赖这几个属性，影响小；refactor 后所有「选主」路径都依赖 view cache，必须修。

```python
# _adapter_mcp.py:_refresh_namespace 末尾
def _refresh_namespace(self, init_result, tools):
    ns = self.namespace
    ...
    # 函数表已变更，通知所属 view 失效缓存
    registry = getattr(self.app, "_registry", None)
    if registry is not None:
        view = registry._views.get(ns.name)
        if view is not None:
            view.invalidate()
```

`MCPConnection` 当前是否能拿到 `app` 引用待确认；如果不行，作为本次 refactor 的前置改动加上一个 `app` 引用，或改由 registry 监听 namespace 函数变化（后者复杂度高，不推荐）。

### MCP / peer 自身的 description 来源不动

- `_adapter_mcp._refresh_namespace`：`ns._description = instructions or serverInfo.title`
- `_adapter_pysandbox.build_peer_namespaces`：`ns._description = base_desc + "(shared from <source>)"`

这两条是 **provider 自身**怎么生成 description，view 层只负责「选哪个 provider 的 description」（即 `primary._description`）。两个层次不混。

### `primary` 的动态跳变

`primary` 派生自 `displayed`，`displayed` 派生自 `_resolved_functions()`，函数解析依赖注册顺序。

如果当前 primary provider 的 active 函数全部消失（连接重建清空函数表），`displayed` 自动跳到下一个有 active 函数的 provider，`primary._description` 随之变更。这是合理的信息更新 —— 失效 provider 不应继续占据主位置。如果未来发现 description 抖动困扰用户再考虑 sticky primary。

## 消费者场景

| 消费者 | 场景 | 依赖 | 验收标准 |
|---|---|---|---|
| sandbox `help()` 列表 | agent 查询能力清单 | `displayed_of(ns)` + `ns._description` | 描述来自 primary，多 provider 才显示 `[N providers]` 标记 |
| sandbox `help(ns)` 详情 | agent 看 ns 详情 | `displayed_of(ns)` 驱动多/单分支 | 单 displayed 走单 provider 模板，多 displayed 走 multi-provider 模板 |
| sandbox 函数调用 | agent 调用能力 | `_resolved_functions` | 调用不受 refactor 影响，active 解析与今天一致 |
| pysandbox share export | 对端融合 namespace | `_all_namespaces` → `_flatten_view` | export 描述来自 primary，函数集 = exec_code 可见集（不丢函数） |
| mutbot 自身 | 本地 MutbotTools + MCP peer | `displayed_of(ns)` 长度 = 1 | 单 provider 渲染，无 `[from #N]` 归属标签 |

## 变更兼容性

**接口层兼容**：

- `MergedNamespaceView._description / _functions / _descriptions / _connection / connection_state / connection_error` 签名与返回类型不变，仅 `_description` 内部实现改走 `primary`
- `_displayed_providers(ns)` 模块函数保留为 `displayed_of(ns)` 的 alias，下游 import 不受影响
- `_all_namespaces` 返回类型仍是 `dict[str, Namespace]`，但同名 multi-provider 时返回的是 `_flatten_view` 拍平的临时 Namespace（对端无感）

**行为层变化**：

- pysandbox share export：同名 multi-provider 时，对端可见函数集**变多**（从「decl 整体替换」→「函数级合并」）。这是修 bug，不是破坏
- pysandbox share export：description 不再随 `connection_state` 跳变（不再优先 connected），改为稳定走 primary（即注册顺序首个 displayed）
- `_refresh_namespace` 后 view cache 立即失效（修隐藏 bug）

## 实施步骤清单

### Step 1：`_namespace.py` 加 `displayed` / `primary` 属性 + 模块 helper

- [x] `MergedNamespaceView` 加 `displayed` property（移用现 `_displayed_providers` 算法）
- [x] `MergedNamespaceView` 加 `primary` property（派生自 `displayed`）
- [x] `_description` 改为 `return self.primary._description if self.primary else ""`
- [x] 新增模块级 `primary_of(ns)` / `displayed_of(ns)` helper
- [x] `_displayed_providers(ns)` 改为 `displayed_of(ns)` 的 alias（一行函数，保留以避免外部 import 断裂；后续 deprecation 单独 issue）
- [x] `_render_registry` / `_render_namespace` 把 `_displayed_providers(ns)` 替换为 `displayed_of(ns)`

### Step 2：`share.py` 重写 `_all_namespaces` 走 view + 拍平

- [x] 在 `share.py` 或 `_namespace.py` 加 `_flatten_view(view) -> Namespace` helper（位置看 import 方向，建议 `_namespace.py` 同文件）
      > 实际取名 `flatten_view`（导出供 share.py / 未来其他拍平场景复用，不加下划线前缀），放在 `_namespace.py`。
- [x] `_all_namespaces` 改为：建 temp_registry（decl 先 + external 后）→ 遍历 → view 拍平 / Namespace 直接返回
- [x] 验证 `share.py` 三个 handler（list / describe / call）调用点不需要改 —— 它们拿到的依旧是 `dict[str, Namespace]`

### Step 3：`_adapter_mcp.py` `_refresh_namespace` 后失效 view cache

- [x] 确认 `MCPConnection` 能否拿到 `app` 引用；不行则在 `MCPConnection.__init__` 加一个 `app` 字段（传入位置在 `_app_impl.py` 创建 connection 处）
      > `MCPConnection._sandbox` 已由调用方（`builtins/main_impl.py` / `cli/pysandbox.py` / `mutbot/web/server.py`）在 `add_namespace` 后立即赋值，直接复用，无需新字段。
- [x] `_refresh_namespace` 末尾：`registry._views.get(ns.name)` 取得 view 调 `view.invalidate()`
- [x] 评估是否同时需要在 `Namespace.register` 上加回调钩子（暂不做，靠调用方显式 invalidate；如后续发现 `_descriptions` 改动也有类似问题再补）

### Step 4：测试

- [x] 已有 `_namespace.py` 的 view / multi-provider 测试全部通过（不破坏）
- [x] 新增：`_all_namespaces` 在 multi-provider（decl + external 同名，函数部分重叠部分不重叠）下，返回的拍平 Namespace 函数集 = exec_code 路径函数集
- [x] 新增：`primary` / `displayed` 在「全空壳 view」「全 shadowed view」「正常 view」三种状态下的返回符合预期
- [x] 新增：`_refresh_namespace` 后 `view.primary` / `view._description` 立即看到新结果（cache 已失效）
- [x] 回归：sandbox `help()` 单 provider / multi-provider 渲染 snapshot 与 refactor 前一致（除了已知行为变化点：connection 状态不再影响 description 选择）
      > mutagent 788 + mutbot 508 全绿。

### Step 5：文档

- [x] `feature-namespace-multi-provider.md` / `feature-pysandbox-namespace-sharing.md` 同步更新「主 provider 选择策略」描述
      > `feature-namespace-multi-provider.md` 加了补正脚注；sharing 文档未提及选代表者算法细节，无需同步。

## 不做的事

- 不动 `NamespaceRegistry.add / remove_provider` / `_resolve_functions` 「先注册先赢」核心算法
- 不引入「sticky primary」（primary 一旦确定不跳变）—— 当前动态跳变是合理行为
- 不把 `Namespace` 改成必须经过 `MergedNamespaceView` 访问 —— 单 provider 直返保留
- 不动 MCP / peer 各自生成自身 `_description` 的来源逻辑
