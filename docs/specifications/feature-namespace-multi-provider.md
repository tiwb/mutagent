# Namespace Multi-Provider 模型 设计规范

**状态**：✅ 已完成
**日期**：2026-05-09
**类型**：功能设计

## 需求

1. mutbot 在打开 pysandbox-share（feature-pysandbox-namespace-sharing）后，作为 mutagent 的 MCP source 接入时永远卡 `connecting`，错误信息 `Pysandbox namespace conflict on source 'mutbot': peer namespace 'mutbot' clashes with tools of 'mutbot'`。
2. 直接原因：mutagent 配置中 source key = `mutbot`，因此该 source 的 tool namespace 名 = `mutbot`；而 mutbot server 自我暴露的 peer namespace 也恰好叫 `mutbot`（mutbot 自己的 debug_tools）。`MCPConnection._check_peer_name_conflicts` 视为 D1 冲突，抛 RuntimeError。
3. 这种「source 名 = peer ns 名」的撞名是**架构必然**：peer 服务器自我 export 的 namespace 命名几乎一定就是它自己的项目名，而用户配置 mcp_sources key 时最自然的命名也是项目名。零配置失误也会撞。
4. 次生 bug：`_do_rebuild` 抛 RuntimeError 后 state 永久卡在 `"connecting"`——因为只有 `connect()/list_tools()` 段做了 except → `_set_state("failed", ...)`，peer 构建/冲突检测段没兜底。autostart 上层 catch 了异常但不再回填 state，conn 被泄漏。
5. 即便修掉同名 namespace 的合并，**跨 source 同名**和**真正的同名函数冲突**仍未解决。希望把整个抽象抬正：namespace 应支持多 provider，冲突推到调用级。

## 关键参考

- `mutagent/src/mutagent/sandbox/_adapter_mcp.py` — `MCPConnection`（state 机、`_do_rebuild`、`_check_peer_name_conflicts` 在 ~604 行）
- `mutagent/src/mutagent/sandbox/_adapter_pysandbox.py` — `build_peer_namespaces`（peer 构建入口）
- `mutagent/src/mutagent/sandbox/app.py` — `SandboxApp._registry`（namespace 注册表，需要从 `dict[str, Namespace]` 改造）
- `mutagent/src/mutagent/sandbox/namespace.py` — `Namespace` 类（保留不动，新增 provider_kind 元数据字段）
- `mutbot/src/mutbot/web/server.py:178-205` — mutbot 把 mcp_sources 接入 sandbox 的现场，含一份本地 namespace-级冲突检测（也要按新模型放宽）
- `mutagent/docs/specifications/feature-pysandbox-namespace-sharing.md` — 共享机制基线，本设计沿用其 D2/D3/D4/D5/D6 决策，**改写 D1**

## 设计方案

### 模型转变：单 owner → multi-provider

`SandboxApp._registry` 从 `dict[str, Namespace]` 改为 `dict[str, list[Namespace]]`。每个 `Namespace` 实例继续作为一个 **provider** 存在（不动 Namespace 类的内部结构，便于回归与回退）。同名 namespace 不再阻塞注册，多个 provider 并存，由 SandboxApp 在 help/调用时按策略合并。

```
_registry["mutbot"] = [
    Namespace("mutbot", provider_kind="tool", _connection=conn_mutbot),  # 空壳，pysandbox tool 已被 D2 过滤
    Namespace("mutbot", provider_kind="peer", _connection=conn_mutbot),  # 真正的 logs/status/...
    Namespace("mutbot", provider_kind="peer", _connection=conn_other),   # 跨 conn 同名 provider
]
```

> 决策：每个 provider 仍是独立 `Namespace` 对象。`Namespace` 类只新增一个可选属性 `provider_kind: Literal["builtin", "tool", "peer", "cli"]` 用于策略判断，其它内部结构（`_functions / _descriptions / _connection / connection_state / connection_error`）零变更。

### 冲突检测时机：从注册级 → 调用/help 级

| 环节 | 旧 | 新 |
|---|---|---|
| MCP source 初始化 | namespace 名撞 → RuntimeError | append 到 list，**不报错** |
| 同 conn 内 peer 互撞 | RuntimeError | 仍 RuntimeError（同一 server export 两个同名 ns 是 server bug） |
| 跨 conn namespace 同名 | RuntimeError（mutbot 那段也阻塞） | 透明合并，不报错 |
| 同 namespace 同函数 | 不会发生（namespace 撞已经先死了） | 解析 active 时按策略选 + WARNING |

### 覆盖策略（同名函数冲突时谁是 active）

**默认顺序：先注册先赢**。理由：

- 静默换语义最危险；新加 source 不生效 + WARNING 是用户能立刻看到并改的故障模式
- 后来者赢会让旧调用悄悄换实现，更难排查

**同 priority 内的细则**：

- 同 conn 的 tool ns + peer ns 同名时，按注册顺序 peer 在 tool 后追加；但 tool ns 在过滤掉 `pysandbox` 后通常是空壳，函数级合并时 peer 的函数自然占位，不触发冲突日志。如果 tool ns 真的还有别的 tool 函数与 peer 撞名，按「先注册先赢」= tool 优先；这与上一轮讨论倾向 peer 优先相反，但与「先注册先赢」的全局规则一致，避免出现两条相互打架的策略。server owner 觉得不合适可以在 server 端去掉重复的 tool 暴露。

**priority 不进第一版**（YAGNI）。source 配置中暂不接受 `priority` 字段；通过控制 `mcp_sources` 配置中的 key 顺序间接控制注册顺序。真有冲突案例再加。

### 冲突日志

- 时机：在「构建合并视图 / 解析 active 函数」时打一次，不在每次调用时打（避免刷屏）。
- 级别：WARNING。
- 内容：`namespace 'mutbot' function 'logs': active=source<X>(peer), shadowed=[source<Y>(tool), source<Z>(peer)]`
- 缓存键：`(ns_name, fn_name, providers 签名)`。providers 列表变化（add/remove）时 invalidate，重新触发一次 WARNING。

### MergedNamespaceView

新增轻量 wrapper，给沙箱执行环境用。沙箱里 `mutbot` 这个名字解析到的不再是单个 `Namespace`，而是一个 view：

```python
class MergedNamespaceView:
    name: str
    providers: list[Namespace]              # 引用 _registry 的同名 list

    def __getattr__(self, fn_name): ...     # 调用：解析 active provider，dispatch
    def _resolved_functions(self) -> dict[str, ResolvedFn]: ...  # 给 help() 用
    @property
    def connection_state(self): ...         # 取 active provider 的状态；多 provider 时按 "any connected → connected" 聚合
```

`ResolvedFn` 记录 `(active_provider, shadowed_providers, fn)`。help() 渲染时使用。

> view 不持有状态，每次 attr 访问都即时从 `providers` 列表算。providers 列表变化无需通知 view（list 是同一个引用）。性能上加一层薄缓存即可，缓存 key 是 providers 的 id 序列。

### help() 渲染

`help(mutbot)` 输出形如：

```
mutbot — namespace
Providers:
  [1] source 'mutbot' (peer + tool, connected)        ← active for logs/status
  [2] source 'mutbot-backup' (peer, connecting)        ← shadowed for logs

Functions:
  logs(level='INFO', logger='', pattern='', last_n=50)
      from source 'mutbot' (peer)
      [shadowed: source 'mutbot-backup']
  status()
      from source 'mutbot' (peer)
  errors(last_n=20)
      from source 'mutbot-backup' (peer)
  health_check()
      from source 'mutbot' (tool)
```

实现集中在 `Namespace.__doc__ / repr` 当前路径上扩展，或在 SandboxApp 提供 `render_help(ns_name)` 由 sandbox `help()` 入口调用——具体落点实施时再定。

### 调用解析

`mutbot.logs(...)` 解析路径：

1. sandbox 全局 `mutbot` → `MergedNamespaceView`
2. `__getattr__("logs")` → 查 `_resolved_functions()` 拿 `ResolvedFn.fn`
3. 调用 fn（fn 内部已绑定到 active provider，沿用原有 conn 状态机 + cooldown + reconnect 逻辑，无变化）

active provider 的连接处于非 connected 时：沿用现有行为（peer 函数 wrapper 内部会触发 `ensure_connected` → 抛错或等重连）。第一版**不**自动 fallback 到 shadowed provider——fallback 改语义，留给后续按真实痛点考虑。

### Provider 移除（disconnect / reconnect）

`SandboxApp` 新增按 provider 维度的移除入口：

```python
sandbox_app.add_namespace(ns, on_remove=...)        # 已有
sandbox_app.remove_namespace(ns)                    # 新增：从 _registry[ns.name] 中 pop 该实例
```

`MCPConnection.close` 时遍历 `self.namespace + self.peer_namespaces`，逐个 `remove_namespace`。`_do_rebuild` 时先 remove 旧 peer providers，新建后再 add——provider 身份按 `Namespace` 实例区分，事务式更新。

list 空了从 `_registry` 删 key（避免 `dir(sandbox)` 出现幽灵名字）。

### 同 conn 内 peer 互撞仍报错

`_check_peer_name_conflicts` 重写：只检查 peer namespaces 之间是否有同名（同一 server 自我 export 两个同名 ns 必然是 server bug），不再检查 peer vs source tool ns。

### 连接异常路径完整性 — 修复 state 卡 connecting

**当前问题**：`_do_rebuild` 内只有 `connect()` / `list_tools()` 这一段被 `try/except` 包住并在异常时 `_set_state("failed", reason)`。后续两段没有同样的兜底：

- `build_peer_namespaces(...)` —— `pysandbox/namespaces.list` / `describe` 任何一步抛错（例如对端实现 bug、网络瞬断时只在第二个 RPC 才暴露）
- `_check_peer_name_conflicts(...)` —— 本设计前的旧 D1 冲突检测路径（即使新模型放宽了 namespace 级冲突，peer 互撞这条仍会抛）

这两段抛 `RuntimeError` 时，state 永远停在 `_do_rebuild` 开头设的 `"connecting"`。autostart 上层把异常 catch 成 WARNING 就忽略，conn 被泄漏在 connecting，help() 看到的就是这个状态，且 cooldown 不生效（`last_error` 也没填），下一次访问还会立刻重试连接、再次失败、再次卡 connecting，无限循环。

**修复**：把整个 `_do_rebuild` 后半段（peer 构建 + 冲突检测 + 任何后续逻辑）也纳入 `try/except`，统一翻 `_set_state("failed", reason)` + `last_attempt_at = time.time()`，与现有 connect/list_tools 段语义对齐：

```python
async def _do_rebuild(self) -> None:
    self._set_state("connecting", None)
    self.last_attempt_at = time.time()
    try:
        # connect / list_tools / refresh_namespace / build_peer_namespaces /
        # _check_peer_name_conflicts / _set_state("connected", ...)
        ...
    except MCPTransportError:
        self._set_state("failed", ...)  # 已有逻辑保留
        raise
    except Exception as exc:
        # 兜底：任何 RuntimeError / ValueError / 编程错都进 failed，不卡 connecting
        reason = str(exc) or exc.__class__.__name__
        self._set_state("failed", reason)
        self.last_attempt_at = time.time()
        logger.warning("MCP '%s' rebuild failed: %s", self.ns_name, reason)
        raise
```

**通用原则（D11）**：`_do_rebuild` 入口处只设 connecting 一次，出口要么 connected 要么 failed，**不允许任何路径让 state 留在 connecting**。新增任何后置逻辑都必须在该 try 范围内。

**配置错路径**：`make_client` 抛 `ValueError` 当前已有独立 except 翻 failed，保留不动（语义不同：配置错不进入 cooldown 的「网络问题」分类，但 state 仍是 failed，help 上能看见）。

### mutbot server 侧的对应改动

`mutbot/src/mutbot/web/server.py` 当前在 `add_namespace(peer)` 之前手动做了一次 namespace-级冲突检测：

```python
existing = sandbox_app._registry.get(peer.name)
if existing is not None:
    raise RuntimeError(...)
```

新模型下这段强约束应去掉，改为信任 SandboxApp 自身的 multi-provider 管理。

### 决策记录

#### D1（重写）：重名 namespace 不再阻塞注册

`feature-pysandbox-namespace-sharing.md` 的 D1「重名冲突 → 启动期直接报错」作废。新规则：

- 同 namespace 名允许多 provider 共存
- peer 互撞（同一 conn 内）仍报错
- 函数级冲突在 help/调用解析时按「先注册先赢」+ WARNING

#### D7：覆盖策略默认「先注册先赢」

理由：静默换语义不可接受；新增 source 不生效 + WARNING 是用户可立即定位的故障形态。priority 显式优先级不进第一版。

#### D8：冲突日志构建时一次

避免每次调用打日志刷屏。providers 列表变化时缓存 invalidate，重打一次。

#### D9：第一版不做调用 fallback

active provider 不可用（disconnected/failed）时按现有行为抛错，不自动 fallback 到 shadowed provider。fallback 改变调用语义，等真实痛点出现再加。

#### D10：第一版不开手动选 provider 的 API

不提供 `mutbot.from_source('xxx').logs(...)` 之类的逃生口。临时需求可通过 `sandbox_app._registry["mutbot"][i]._functions["logs"](...)` 直接走底层。

#### D11：`_do_rebuild` 异常路径完整性

`_do_rebuild` 入口设 connecting，出口必须是 connected 或 failed 二选一。任何后置逻辑（peer 构建、冲突检测、未来新增的握手步骤）都必须在统一 try 范围内，否则 state 卡 connecting + cooldown 失效 + autostart 静默吞错的复合故障会重现。

## 渲染改进（追加迭代）

**背景**：multi-provider 模型 v1 落地后，mutbot 实际使用中 help 输出有三处噪音：

1. mutbot 作为 source 接入时，自身 tool ns（已被 D2 过滤掉 `pysandbox`）经常是空壳 `[1] kind=tool, state=connected, functions=0`，与同名 peer provider 并列出现；信息为零却挤占视觉
2. 函数行无差别带 `[from kind#hex]` 归属标签——大多数实际场景过滤完空壳后只剩 1 个真有函数的 provider，标签反而干扰阅读
3. `peer#7f8a3b...` 的 hex id 没有定位价值，用户已经能从 Providers 段看到顺序编号

### 设计原则

**"真正贡献函数的 provider"作为渲染基准。** 由于 `MergedNamespaceView._resolved_functions` 中 `rf.active` / `rf.shadowed` 必然各有 ≥1 个函数（否则不会进 resolved），过滤 `len(_functions) == 0` 的 provider **绝不会丢失任何已 resolve 的函数归属**。

```python
displayed = [p for p in ns.providers if p._functions]
```

### 三点改动

#### R1：Providers 段过滤空壳

`_render_namespace` 的 `Providers (N):` 段只列 `displayed`；`_render_registry` 的 `[N providers]` 徽标也按 `len(displayed)` 算。

不分 connection_state 一刀切过滤——`connecting/failed` 时单 provider 路径已有 `⚠ Connection failed` 提示兜底，多 provider 视图不需要重复展示连接诊断信息。

#### R2：单 displayed provider 时退化为单 provider 渲染

multi-provider 分支判定从 `len(ns.providers) > 1` 改为 `len(displayed) > 1`：

- `len(displayed) <= 1`：走单 provider 路径（无 Providers 段、函数行无 `[from ...]` 标签），与 multi-provider 改造前完全一致
- `len(displayed) > 1`：进 multi-provider 分支

mutbot 场景过滤后只剩 1 个真有函数的 peer provider，自然回退为干净的单 provider 渲染。

#### R3：函数归属用 `#N` 替代 `kind#hex`

`displayed` 列表 1-based index 作为编号基准：

```python
idx_map = {id(p): i for i, p in enumerate(displayed, 1)}
```

函数行：`[from #2]`（不带 kind——Providers 段已经列了 kind/state，函数行越短越好扫；带 kind 会让多函数列表撑得太宽）。

shadowed 列表：`(shadowed: #1, #3)`。

#### 不动的部分

- `_resolved_functions` 内部 WARNING 日志保留 `kind#<hex>` 表示——后台 log 给工程师看，hex 是 process-stable 的引用，不依赖 displayed 顺序
- `Namespace` / `MergedNamespaceView` 接口不变；改动全部局限在 `_render_*` 渲染函数

## 待定问题

（暂无）

## 实施步骤清单

- [x] `_namespace.py` — 新增 `provider_kind` 字段；`NamespaceRegistry` 改造为 `dict[str, list[Namespace]]` multi-provider 存储；新增 `MergedNamespaceView`；扩展 `_render_registry`/`_render_namespace` 支持多 provider；冲突 WARNING 缓存
- [x] `_app_impl.py` — `_cleanups` 按实例 id 存储；`add_namespace`/`remove_namespace` 语义对齐 multi-provider（remove 接受 name 仍可，按全部 provider 处理；新增按实例 remove）；`_build_namespace_dict` 暴露 `MergedNamespaceView`
- [x] `_adapter_mcp.py` — `_do_rebuild` 后段 peer 构建 + 冲突检测纳入统一 try（D11）；`_check_peer_name_conflicts` 重写为只检 peer 互撞；`close` 走 instance-级移除入口
- [x] `_adapter_pysandbox.py` — peer namespace 标 `provider_kind="peer"`
- [x] `share.py` — `_all_namespaces` 在 multi-provider registry 下选「active provider」作单代表 export
      > **2026-05-09 补正**（`refactor-namespace-provider-selection.md`）：改为走 `flatten_view`，同名多 provider 拍平为单 Namespace，描述走 `primary_of(view)` (= displayed[0])，函数集 = view 合并后的 active 集。修复原“decl 整个覆盖 external”丢函数的 bug。
- [x] `mutbot/src/mutbot/web/server.py` — 去掉本地 namespace-级冲突检测；conn.namespace 标 `provider_kind="tool"`，peer 接入路径走新 API
- [x] **BUGFIX**：`_adapter_mcp.py` — `MCPConnection._sandbox` 回引；`_do_rebuild` 事务式同步 peer providers 到 sandbox registry（`_sync_peer_providers`）；D11 异常路径中摘除残留 peers；`close` 同步清理；`builtins/main_impl.py` + `cli/pysandbox.py` + `mutbot/web/server.py` 设置 `conn._sandbox`
- [x] **R1+R2+R3 渲染改进**：`_namespace.py` `_render_registry` / `_render_namespace` 按 `displayed = [p for p in providers if p._functions]` 过滤；徽标 / 分支判定 / 函数归属编号统一以 displayed 为基准；归属标签改为 `[from #N]` / `(shadowed: #i, #j)`
- [x] 单元测试：MergedNamespaceView 调用解析、冲突 WARNING 一次性、`_do_rebuild` 异常路径不卡 connecting、peer providers 同步注册/移除
- [x] 端到端验证：mutbot 作为 mutagent MCP source 的 4 个消费者场景

## 消费者场景

| 消费者 | 场景 | 依赖的输出 | 验收标准 |
|--------|------|-----------|---------|
| mutbot 作为 mutagent 的 MCP source | mutagent autostart mcp_sources['mutbot'] | mutbot peer namespace 成功融合，help(mutbot) 显示 logs/status/errors 等 | 不再报 namespace conflict；conn 状态在 connected；`mutbot.status()` 可调用 |
| 多 mutbot 实例同时挂 | 用户配置两个 source 都叫 `mutbot` 类（如 `mutbot` + `mutbot-backup`） | 两个 source 都能初始化，help() 看到 multi-provider | 注册不报错；help 显示 providers 列表；冲突函数 WARNING 一次 |
| mutagent 现有 sandbox 用户 | help(<其他 namespace>) 行为不变 | 没有 multi-provider 的 namespace 渲染应保持现状 | 单 provider 路径与改造前 diff 最小（输出可接受微调） |
| 用户排查 stuck connecting | source 配置 / 服务器异常时能立刻看到 failed 状态 | conn state 不再卡 connecting | 任意 RuntimeError 路径都进入 failed 分支；help(mutbot) 看到 `[failed: <reason>]`；cooldown 生效 |
