# Pysandbox Namespace Sharing 设计规范

**状态**：✅ 已完成
**日期**：2026-05-08（更新 2026-05-09）
**类型**：功能设计

## 需求

1. **mutagent 需要直接调用 mutbot 的运行时能力**：`mutbot.status()`、`mutbot.logs()`、`mutbot.exec_frontend()` 等，以 namespace 函数调用形态在 pysandbox 里直接使用，而非通过 MCP tool 的字符串拼接方式包一层。
2. **受众明确、互不重叠**：
   - **mutagent 面板**（mutbot 未来的主要 agent 形态）→ 通过本协议融合 mutbot 的 namespaces
   - **Claude Code / 其他 MCP agent** → 走标准 MCP，只看到 `pysandbox` 一个 tool（不变）
   - **pi / 无 MCP 的 agent** → 走 CLI `python -m mutbot pysandbox -c ...`（不变）
3. **对 LLM/sandbox 用户透明**：融合进来的 namespace 与本地 `NamespaceTools` 发现的 namespace 无感知差异 —— `help()`、`<ns>.<fn>()`、`help(<ns>.<fn>)` 完全一致。
4. **不堵死 Push 模式**（server 主动通知 namespace 变更），但本次不实现。

## 设计方案

### 协议定位：pysandbox 层的方言，借道 MCP 信道

这不是一个通用的 "MCP namespace 挂载协议"。**它是两个 pysandbox 之间的 namespace 融合协议**：两端都有 `Namespace` / `NamespaceRegistry` / `help` 这些 first-class 概念，一端把自己的 namespace registry 摊平暴露，另一端把它合并进本地 registry。一个不是 pysandbox 的普通 MCP server 没有能力提供这种端点。

层级归属：

```
mutio.mcp                   ← MCP wire 协议（JSON-RPC 封包、握手、扩展方法注册）
                              不认识"pysandbox"这个词，只提供扩展入口
mutagent.sandbox            ← pysandbox / Namespace / NamespaceRegistry 定义
mutagent.sandbox._adapter_* ← client 端：MCP 桥接（已有）+ pysandbox 融合（新增）
mutagent.sandbox.share      ← server 端：给 MCPView 注入 pysandbox 共享能力的入口
```

**mutio 的改动最小**：只需确认 `MCPView` 可以挂载自定义 JSON-RPC 扩展方法并在 `initialize` 响应里通告 capability。协议内容对 mutio 不可见。

### 动词与概念：share / merge，不叫 mount

- **Server 侧动作**：`share` —— "我把我的 pysandbox namespaces 分享出去"
- **Client 侧动作**：`merge` —— "我把远端 namespaces 平铺合并进本地 registry"
- **协议名**：pysandbox namespace sharing

不使用 `mount` 一词，因为 mount 隐含挂载点/层级，而本协议**明确是扁平融合**。

**扁平语义约束**：

- 远端返回的 namespace 结构是扁平的 `{name, description, functions}`，**不允许递归嵌套**（namespace 里不能套 namespace）
- Client 侧不引入远端前缀 —— 远端 `mutbot` namespace 以 `mutbot` 名字进入本地，不是 `mutbot_server.mutbot`
- 重名按冲突处理（策略见待定问题 Q1），不做层级回避

### 复用 MCP 端点 + Capability 通告

**不新增 endpoint**。在同一个 `/mcp` 上通过 capability 协商区分受众：

```jsonc
// initialize 响应
{
  "serverInfo": { ... },
  "capabilities": {
    "tools": { ... },
    "pysandbox": { "version": "1" }
  }
}
```

- 标准 MCP client（Claude Code）不认识 `pysandbox` capability，继续走 `tools/list`，只看到 `pysandbox` tool
- pysandbox peer client（mutagent）看到该 capability，触发扩展方法流程

**命名决策**：直接放一级字段 `capabilities.pysandbox`，不走 `experimental.pysandbox` 过渡。理由：受众明确是同生态的 pysandbox 对端，不预期被通用 MCP 客户端误解；MCP 规范允许 vendor 扩展直接放一级，不强制 experimental。

### 三个 JSON-RPC 扩展方法

方法名采用 **`pysandbox/namespaces.<verb>`** 风格，与 MCP 标准方法（`tools/list`、`tools/call`、`resources/read`）视觉对齐：

| 方法 | 入参 | 返回 |
|------|------|------|
| `pysandbox/namespaces.list` | `{}` | `{"namespaces": [{"name": str, "description": str, "function_count": int}]}` |
| `pysandbox/namespaces.describe` | `{"namespace": str}` | `{"name": str, "description": str, "functions": {fn_name: {"signature": str, "doc": str, "kwargs_schema": JSONSchema}}}` |
| `pysandbox/namespaces.call` | `{"namespace": str, "name": str, "arguments": {...}}` | `result` 或 JSON-RPC error |

**错误处理**：
- 传输错 → 走 JSON-RPC transport 层，由 `MCPConnection` 的 `_is_transport_error` 识别、触发重连
- 业务错（namespace 不存在 / 函数不存在 / 参数校验失败 / 业务异常）→ JSON-RPC `error` 字段返回，不触发重连，client 转为 Python 异常抛给 sandbox 用户
- 对等于现有 MCP tool 的 `MCPTransportError` vs `MCPToolError` 划分

### 调用形态：kwargs-only，与 MCP tool 对齐

协议层**只支持 kwargs 调用**，不支持位置参数。理由：

- MCP `tools/call` 的 `arguments` 本来就是 object（kwargs 语义）
- `inspect.signature` 可以序列化成字符串完整还原签名（默认值、类型注解、可变关键字参数），足够支撑 `help(mutbot.logs)` 展示
- Sandbox 用户不会对 namespace 函数做 `inspect.signature` 取 live `Signature` 对象 —— 他们用 `help()` 看字符串
- 客户端也只接 `**kwargs` 调用，本地 Namespace 函数本身就是 kwargs-friendly

`describe` 同时返回：
- `signature`：`inspect.signature(fn).__str__()` 产出的字符串，供 `help()` 原样展示
- `kwargs_schema`：JSON Schema（对齐 MCP `tools/list` 的 `inputSchema`），供协议层校验/文档生成使用

### Agent 端实现：复用 MCPConnection，最小扩展

关键不变量 —— **现有 `MCPConnection` 的状态机、cooldown、锁、重连重试、`_is_transport_error` 不动**。变化仅有：

1. **一个 source 可产出 0..N 个 namespace**（当前是 1 个）。`MCPConnection` 在 `_do_rebuild()` 成功后：
   - 标准路径：`tools/list` → 注册 tool 函数到一个 namespace（现状）
   - pysandbox 路径：检测到 `pysandbox` capability → 额外调 `pysandbox/namespaces.list` + 逐个 `describe` → 为每个远端 namespace 创建独立 `Namespace` 对象，追加到 `self.peer_namespaces: list[Namespace]`，随后由 startup 调用方批量进 registry
2. **函数 wrapper**：MCP tool 的 wrapper（`_make_tool_func`）和 pysandbox namespace 函数 wrapper 结构几乎一致，都走 `MCPConnection` 的 `ensure_connected` + 调用 + 传输错重试一次。差别只在最终调用的方法名（`tools/call` vs `pysandbox/namespaces.call`）和参数结构。
3. **不引入 `RemoteNamespace` 类**。融合进来的 namespace 就是普通 `Namespace` 对象，函数体是 RPC wrapper 闭包。与本地 `NamespaceTools` 产出的 namespace 对 registry 而言完全同构。

代码组织：

```
mutagent/sandbox/
  _adapter_mcp.py          # 已有：MCP tool → Namespace（tool 级桥接）
  _adapter_pysandbox.py    # 新增：pysandbox 融合 client（复用 HTTPMCPClient，加 3 个扩展方法调用）
```

`_adapter_pysandbox.py` 的核心是：
- `PysandboxPeerClient`：薄封装，在 `HTTPMCPClient` 已有的 `MCPClient` 上加 `list_namespaces` / `describe_namespace` / `call_namespace` 三个方法
- `build_peer_namespaces(conn, init_result, client) → list[Namespace]`：在 `MCPConnection._do_rebuild` 末尾、标准 tools 路径之后被调用，产出额外 namespace

`MCPConnection` 本身的改动：`_do_rebuild` 末段增加对 capability 的检测 + dispatch 到 `build_peer_namespaces`；状态字段同步 helper 拓展到 `peer_namespaces` 列表。不抽基类、不拷贝状态机。

### Server 端实现：MCPView 钩子 + class 属性注入

**前置：mutio 必须为 `MCPView` 子类暴露两个 hook**。当前 `_view_impl._setup_handlers` 是私有 + 写死的，没有给子类注入 extra capabilities / extra methods 的入口；又因为 view 实例由 ASGI 路由按请求实例化、ext 按实例缓存（`_MCPViewExt.get_or_create(view)`），不能用「对单个 view 实例 attach 一次」的旧思路。

mutio 的最小破口（落在 `mutio.mcp.view.MCPView` + `_view_impl`）：

```python
class MCPView(View):
    # 子类可覆盖：返回追加进 initialize capabilities 的字段
    def extra_capabilities(self) -> dict[str, Any]:
        return {}

    # 子类可覆盖：在 view 的 JsonRpcDispatcher 上注册扩展方法
    def register_extra_methods(self, dispatch: "JsonRpcDispatcher") -> None:
        return None
```

`_setup_handlers` 末尾调用 `view.register_extra_methods(ext._dispatch)`；`_handle_initialize` 把 `view.extra_capabilities()` 合并进 capabilities 字典。mutio 不感知 pysandbox 概念，仍是通用扩展点。

mutagent 提供两个工具：

```python
# mutagent/sandbox/share.py
PYSANDBOX_CAPABILITY: dict = {"pysandbox": {"version": "1"}}

def register_pysandbox_methods(
    dispatch: "JsonRpcDispatcher",
    sandbox: "SandboxApp",
) -> None:
    """在 view 的 dispatcher 上注册 3 个扩展方法。

    实现路径：直接查 sandbox._registry._namespaces[ns]._functions[fn](**arguments),
    不过 pysandbox 的 Python 代码解析,比 pysandbox tool 路径更直接,也更快。
    """
```

mutbot 的 `MutBotMCP` 子类覆盖钩子；`SandboxApp` 通过 **ClassVar 类级单例**注入
（利用 ``mutobj`` 的 ``ClassVar`` 识别机制，避免被包成 per-instance ``AttributeDescriptor``）：

```python
from typing import ClassVar

# mutbot/web/mcp.py
class MutBotMCP(MCPView):
    _sandbox_app: ClassVar["SandboxApp | None"] = None  # _on_startup 时赋值

    def extra_capabilities(self) -> dict:
        if self._sandbox_app is None:
            return {}
        return dict(PYSANDBOX_CAPABILITY)

    def register_extra_methods(self, dispatch) -> None:
        app = self._sandbox_app
        if app is not None:
            register_pysandbox_methods(dispatch, app)

# mutbot/web/server.py 的 _on_startup 末尾
MutBotMCP._sandbox_app = sandbox_app
```

同位置 ``PySandboxTools._app`` 也已同步修正为 ``ClassVar[SandboxApp | None] = None``。

**不走 Mixin**：`MutBotMCP` 已经是 `MCPView` 的直接子类，覆盖两个钩子比再叠 Mixin 直观，也避免 MRO 推理负担。

**`namespaces.call` 的实现路径**：直接查 `sandbox._registry._namespaces[ns]._functions[fn](**arguments)`，**不过 pysandbox 的 Python 代码解析**。比 `pysandbox` tool 路径更直接，也更快。

### 配置：收敛到 `mcp_sources`，零开关

**不引入 `rpc_sources`，也不加 per-source 策略开关**。mutagent 的 `mcp_sources` 配置保持不变：

```jsonc
{
  "mcp_sources": {
    "mutbot_local": { "url": "http://127.0.0.1:8741/mcp" }
  }
}
```

连接 handshake 后自动检测 capability：
- 只有标准 MCP → 注册 tools 到一个 namespace（现状）
- 有 `pysandbox` capability → **既融合 namespaces，也保留其他 tools**（仅自动过滤掉对端的 `pysandbox` tool 自身，避免递归，见决策 D2）

**行为可见性约束**：用户没动配置但启动后 help() 多出 N 个 namespace 会突兀，因此 `MCPConnection._do_rebuild` 在融合成功后必须 INFO 日志 `[merged N namespaces from <source_name>]`。

### 演进空间：Push 通知

MCP 天然支持 `notifications/` 机制。本协议复用：

- Server 在 namespace 变更时发 `notifications/pysandbox/namespaces.changed`（可选携带变更的 namespace 列表）
- Client 收到后重新调 `list` + `describe` 刷新

本次不实现，但方法名空间已预留。

### 接入路径对照

| 受众 | 协议 | 看到的能力 |
|------|------|-----------|
| mutagent 面板 | MCP + `pysandbox/namespaces.*` 扩展 | 融合后的远端 namespaces（`mutbot`、`web`...）直接可用 |
| Claude Code | 标准 MCP | `pysandbox` tool 一个 |
| pi / 其他 | CLI `python -m mutbot pysandbox -c ...` | 透传进 sandbox，看到完整 namespace dict |

## 决策记录

以下决策已敲定（日期 2026-05-09），不再做开关化处理；遇到实际问题再回炉。

### D1: 重名冲突 → 启动期直接报错

mutagent 本地已有 `mutbot` namespace（通过 `NamespaceTools` 发现）、远端又 share 了一个 `mutbot` 时：

- **行为**：`MCPConnection._do_rebuild` 在融合阶段一旦发现远端 namespace 名字与 registry 中已存在的冲突，**直接抛 RuntimeError**，错误信息包含两端来源（本地类路径 / 远端 source 名）。
- **不引入** `conflict` / `rename` 等配置项；不做 remote_wins / local_wins 策略；不做静默覆盖。
- **演进**：等真的撞上多源同名场景再加配置，避免过度设计。

### D2: Peer client 不可见对端的 `pysandbox` tool

mutagent 作为 peer client 连接到一个支持 `pysandbox` capability 的对端时：

- **行为**：`tools/list` 返回的 tool 列表中，**自动过滤掉名为 `pysandbox` 的 tool**；其他 tool 照常融合。
- **不引入** `include_peer_pysandbox_tool` 之类 opt-in 开关。
- **理由**：避免递归调用语义混乱、错误栈不可读；远端能力已通过 namespace 形态完整暴露，无需再保留 tool 入口。

### D3: Capability 命名 → 直接 `pysandbox`

- **行为**：server 在 capabilities 里通告 `"pysandbox": {"version": "1"}`，**不走 `experimental.pysandbox`**。
- **client 检测**：只检查一个位置 `capabilities.pysandbox`；不为前向兼容预留双位置查找。
- **理由**：受众明确是同生态对端，不预期通用 MCP 客户端误解；少一层包装，少一次升级动作。

### D4: 描述数据 Eager 拉取

- **行为**：`MCPConnection._do_rebuild` 在 `pysandbox/namespaces.list` 后，对每个 namespace 立即 `describe`，全部缓存进本地 `Namespace._functions`。
- **失败容错**：单个 namespace `describe` 失败不致命 —— skip + WARNING 日志，其他 namespace 正常融合。整条 source 的 `list` 失败才回到 failed 冷却态。
- **理由**：namespace 数量通常一位数，一次性拉取成本可忽略；`help(mutbot)` 首次调用不阻塞 RPC。

### D5: kwargs 必须 JSON-serializable

- **行为**：`pysandbox/namespaces.call` 的 `arguments` 走 JSON-RPC，必须是 JSON-native 类型（与 MCP `tools/call` 的 `arguments` 约束一致）。
- **不支持** `datetime` / `bytes` / 自定义对象的透传。
- **逃生口**：复杂对象用 `pysandbox(code=...)` tool 走代码字符串路径（已有能力）。

### D6: 半透明归属

- **`help()` 列表**：不标记来源，对 LLM 透明（与本地 namespace 一致）。
- **`help(<namespace>)`**：在描述区追加一行 `(shared from <source_name>)`。
- **连接状态**：远端 namespace 复用现有 `_format_state_label`，源连接断开时所有共享 namespace 一起显示 `[disconnected]` / `[failed: ...]`（共享同一 `MCPConnection` 状态）。

## 实施步骤清单

按依赖顺序，4 个阶段共 13 步。每步独立可编译/运行，便于分批 review 与回滚。

### 阶段 1：mutio 暴露 MCPView 扩展钩子（最小破口）

- [x] **步骤 1**：`mutio/src/mutio/mcp/view.py` 在 `MCPView` 上声明 `extra_capabilities` / `register_extra_methods` 两个可覆盖钩子
- [x] **步骤 2**：`mutio/src/mutio/mcp/_view_impl.py` 在 `_setup_handlers` 末尾调用 `view.register_extra_methods(ext._dispatch)`；`_handle_initialize` 顶层合并 `view.extra_capabilities()` 进 capabilities 字典
- [x] **步骤 3**：mutio 仓库补单测——最小 `MCPView` 子类覆盖两个钩子，断言 initialize 响应含自定义 capability 字段、自定义方法可被 dispatch 路由（`tests/test_mcp_view_extra_hooks.py`，5/5 通过；全量 175/175 回归绿）

*验收*：mutio 单测全绿；现有 mutbot 不接入也不受影响。

### 阶段 2：mutagent server 侧 share 能力

- [x] **步骤 4**：新建 `mutagent/src/mutagent/sandbox/share.py`，导出 `PYSANDBOX_CAPABILITY` 常量与 `register_pysandbox_methods(dispatch, sandbox)` 函数（实现 D3/D5：直接 `pysandbox` 顶级 capability、kwargs 走 JSON-RPC arguments）
- [x] **步骤 5**：`register_pysandbox_methods` 注册 3 个扩展方法
  - [x] `pysandbox/namespaces.list` → 返回 `[{name, description, function_count}]`
  - [x] `pysandbox/namespaces.describe` → 返回 `{name, description, functions: {fn: {signature, doc, kwargs_schema}}}`，`signature` 来自 `inspect.signature(fn).__str__()`，`kwargs_schema` v1 留空 dict
  - [x] `pysandbox/namespaces.call` → 直查 `sandbox._registry._namespaces[ns]._functions[fn](**arguments)`，业务异常包成 JSON-RPC error
- [x] **步骤 6**：`mutbot/src/mutbot/web/mcp.py` 的 `MutBotMCP` 增加 `_sandbox_app` ClassVar + 覆盖两个钩子
  - 最初用 `_sandbox_app: "SandboxApp | None" = None` 被 mutobj 包成 per-instance ``AttributeDescriptor``，仅因 ASGI per-request 时序巧合可工作
  - mutobj 落地 ``ClassVar`` 支持（2026-05-09）后修正为 ``ClassVar["SandboxApp | None"] = None``，成为真正的类级单例属性
  - 同步修正 ``mutagent/sandbox/entry_mcp.py`` 的 ``PySandboxTools._app``（同一错误范例）
- [x] **步骤 7**：`mutbot/src/mutbot/web/server.py` 的 `_on_startup` 在 `PySandboxTools._app = sandbox_app` 旁边追加 `MutBotMCP._sandbox_app = sandbox_app`

*验收*：`curl` 走完整 initialize → list → describe → call 流程通；标准 MCP 客户端连上仍只看到 `pysandbox` tool。

### 阶段 3：mutagent client 侧融合

- [x] **步骤 8**：新建 `mutagent/src/mutagent/sandbox/_adapter_pysandbox.py`
  - [x] `PysandboxPeerClient`：在 `MCPClient` 的 JSON-RPC `request` 入口上加 `list_namespaces` / `describe_namespace` / `call_namespace` 三个方法（mutio 侧同步新增 `MCPClient.request` 公开方法作为通用 JSON-RPC 入口）
  - [x] `_make_namespace_func(conn, ns_name, fn_name, signature_str, doc)`：结构对齐 `_make_tool_func`，走 `ensure_connected` + 传输错重试一次，最终调 `client.call_namespace`
  - [x] `build_peer_namespaces(conn, init_result, client) → list[Namespace]`：检测 `init_result['capabilities'].get('pysandbox')`；list + 逐个 describe（D4 Eager，单 describe 失败 skip+WARNING）；为每个远端 ns 创建 `Namespace`，描述追加 `(shared from <source_name>)` 行（D6）
- [x] **步骤 9**：改造 `mutagent/src/mutagent/sandbox/_adapter_mcp.py` 的 `MCPConnection`
  - [x] `_do_rebuild` 末尾：检测 capability，若有则调 `build_peer_namespaces` 挂到新增的 `self.peer_namespaces: list[Namespace]`
  - [x] tools 路径：当对端有 `pysandbox` capability 时，过滤掉名为 `pysandbox` 的 tool（D2）
  - [x] 重名冲突检查（D1）：peer namespace 名若与 `self.namespace.name` 或调用方 registry 中已存在的名冲突，抛 `RuntimeError`，错误信息含两端来源
  - [x] INFO 日志 `[merged %d namespaces from %s]`
  - [x] `_set_state` 同步状态到 `self.peer_namespaces` 中所有 namespace 的 `connection_state` / `connection_error` 字段
- [x] **步骤 10**：调整 `mutbot/src/mutbot/web/server.py` 中 `bridge_mcp_server` 调用循环——直接改为构造 `MCPConnection` + `reconnect()`，把 `conn.namespace` + `conn.peer_namespaces` 逐个 `sandbox_app.add_namespace`；registry 重名检测也在此处完成（D1）。mutagent 侧 legacy `bridge_mcp_server` 入口保留原签名不动，不引入 v2（唯一调用方已全部迁到新路径）
- [x] **步骤 11**：双实例联调 — 本轮以 `tests/test_pysandbox_sharing.py` 走完整协议链路的单测覆盖（13/13 过）代替 HTTP 双进程手测；HTTP transport 本身未改动，真实双 mutbot 联调留用户环境验收

*验收*：阶段 3 全部完成；mutagent + mutio 全量回归全绿（745+175 passed）；mutbot 预先存在的 20 个 fail 与本改动无关（agent 剥离重构遗留，已 stash 验证）。

### 阶段 4：文档与日志

- [x] **步骤 12**：`mutagent/README.md` 新增「Pysandbox namespace sharing」示例段；`mutbot/README.md` 补一说明与协议文档跳转
- [x] **步骤 13**：本文件状态翻牌为 ✅ 已完成

### 不做（明确排除）

- 不实现 `notifications/pysandbox/namespaces.changed` 推送（仅占位方法名，client 端不订阅）
- 不做 `kwargs_schema` 的 type-hint → JSON Schema 推断（v1 留空字典，等真有 LLM 校验需求再补）
- 不做 per-source `"include": ["namespaces" | "tools" | "both"]` 配置开关（默认行为：tools + namespaces 全融合，仅过滤对端 pysandbox tool）
- 不做重名 rename / 优先级配置（D1：直接报错）
- 不做对端 pysandbox tool opt-in 开关（D2：硬过滤）

## 消费者场景

| 消费者 | 场景 | 依赖的输出 | 验收标准 |
|--------|------|-----------|---------|
| mutbot 的 agent 面板 | 用户在 mutbot 里开 agent 面板，内部跑 mutagent；agent 的 pysandbox 里直接调 `mutbot.status()`、`mutbot.logs()` | 本协议 client 侧 + mutbot server 侧接入 | `help()` 显示本地 + 远端合并后的 namespace 列表；`mutbot.logs(level="ERROR")` 行为与 MCP pysandbox tool 路径一致 |
| mutagent CLI 开发者 | 在 `python -m mutagent` REPL 里连一个本地 mutbot 调试 | `mcp_sources` 配置 + 协议 | 启动时看到 `[merged N namespaces from mutbot_local]` 日志；sandbox 内无感知差异 |
| mutbot 本身（dogfooding） | mutbot server 通过现有 MCP 端点暴露 pysandbox 共享能力 | `attach_pysandbox_to_mcp_view` 一行接入 | `initialize` 返回包含 `experimental.pysandbox`；`pysandbox/namespaces.list` 返回当前 SandboxApp registry 的快照 |

## 关键参考

| 路径 | 说明 |
|------|------|
| `mutagent/src/mutagent/sandbox/_namespace.py` | `Namespace` / `NamespaceRegistry` 定义 + `help` 渲染逻辑 |
| `mutagent/src/mutagent/sandbox/namespace.py` | `NamespaceTools` Declaration，本地 namespace 自动发现 |
| `mutagent/src/mutagent/sandbox/_adapter_mcp.py` | 现有 MCP 桥接 —— `MCPConnection` 状态机、`HTTPMCPClient`、`_is_transport_error`，本协议直接复用 |
| `mutagent/src/mutagent/sandbox/_app_impl.py` | `SandboxApp` 实现 —— `add_namespace` / registry / namespace dict 构建 |
| `mutagent/src/mutagent/sandbox/_engine.py` | sandbox 执行引擎 —— namespace dict 如何注入 globals |
| `mutio/src/mutio/mcp/protocol.py` | `JsonRpcDispatcher` —— 扩展方法注册入口（协议层已支持，本协议不扩展 mutio） |
| `mutio/src/mutio/mcp/view.py` | `MCPView` —— server 侧视图基类，capability 通告位置 |
| `mutio/src/mutio/mcp/client.py` | `MCPClient` —— peer client 复用 |
| `mutbot/src/mutbot/builtins/debug_tools.py` | `MutbotTools`（`mutbot.*` namespace）—— 本次 share 的核心能力源 |
| `mutbot/src/mutbot/web/mcp.py` | `MutBotMCP` —— server 侧接入点，`attach_pysandbox_to_mcp_view` 在此调用 |
| `mutbot/src/mutbot/web/server.py` | `_on_startup` —— SandboxApp 初始化位置 |
| `mutbot/docs/specifications/refactor-agent-strip.md` | mutbot/mutagent 剥离重构方向 —— 本协议是该方向下的具体协议层 |
