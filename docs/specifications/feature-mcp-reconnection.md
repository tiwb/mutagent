# MCP 连接重连机制

**状态**：✅ 已完成
**日期**：2026-05-08
**类型**：功能设计

## 需求

1. **启动容错**：MCP 初始化时连接失败（server 未就绪、网络不通），namespace 不应永久丢失——后续可通过重连恢复
2. **运行时断连恢复**：已连接的 MCP 在使用过程中断开（子进程退出、HTTP 连接超时），调用时自动重连后重试
3. **help 显示连接状态**：`help()` 列出 namespace 时标注每个 MCP 的连接状态（connected / connecting / failed），失败的显示上次错误原因
4. **调用时自动重连**：直接调用 namespace 函数（如 `playwright.navigate(url="...")`）时，若连接处于失败状态，先尝试重连再执行；重连失败给出明确错误（含失败原因）
5. **区分 transport 语义**：stdio（子进程）和 streamable HTTP（网络）的重连策略应分别考虑——进程生命周期管理 vs 网络重试
6. **保留现有语义**：启动时不连（lazy connect），`connect_sources` 改为只创建代理不实际连接；`help()` 不触发重连（只读状态），只在真正调用 tool 时才尝试连接

## 调研结论

### 当前架构问题

`bridge_mcp_server()`（`_adapter_mcp.py:208`）是一次性的：

```
connect_sources()           # main_impl.py:289
  └─ for ns_name, cfg in mcp_sources:
       try:
           ns, client = await bridge_mcp_server(ns_name, cfg)   # ← 一次调用
           sandbox.add_namespace(ns, on_remove=client.close)
       except Exception as e:
           logger.warning("MCP source '%s' failed: %s", ns_name, e)  # ← 永久丢失
```

- `connect` 失败 → namespace 不创建 → help 看不到 → 永远无法调用
- 调用时（`_make_tool_func` 生成的 wrapper）如果 `client.call_tool` 失败 → 上抛异常到用户代码，无重试
- 没有连接状态记录，无法区分"从未连过"和"连过但断了"

### 调用链与 loop 绑定

`_make_tool_func`（`_adapter_mcp.py:237`）通过闭包持有 `client` 和 `main_loop`，所有调用通过 `run_coroutine_threadsafe` 回到主 loop（修复自 feature-mcp-http-adapter.md 迭代 2）。任何重连逻辑必须在这个主 loop 上执行（httpx.AsyncClient 的资源绑定约束）。

### stdio vs HTTP 断连特征

| 维度 | stdio | streamable HTTP |
|------|-------|-----------------|
| 断连原因 | 子进程退出、stdin pipe broken、stdout EOF | 网络超时、连接拒绝、HTTP 503 |
| 重连前提 | 终止旧进程 → 启动新进程 → 重做 initialize | 关闭旧连接 → 重建 client → 重做 initialize |
| 重连成本 | 进程启动开销（冷启动可能数秒） | 网络往返（毫秒级） |
| 幂等性 | 调用失败必定断连（进程死了一切调用都挂） | 可能只是单次请求失败（503 重试后恢复） |
| `list_tools` | 重连后必须重拉（新进程工具列表可能不同） | 重连后必须重拉（server 可能已更新） |

### 命名空间缓存问题

`_build_namespace_dict`（`_app_impl.py:117`）缓存 namespace dict。如果 MCP 重连后 tool 列表变化（新增/删除 tool），namespace 对象需要更新注册函数，同时 invalidate 缓存。当前 `add_namespace` 会自动 invalidate 缓存，重连时如果 tool 列表变了，需要重新 `add_namespace`（覆盖旧的）。

## 关键参考

### 源码
- `src/mutagent/sandbox/_adapter_mcp.py` — `StdioMCPClient`、`HTTPMCPClient`、`bridge_mcp_server`、`_make_tool_func`
- `src/mutagent/builtins/main_impl.py:289-311` — `connect_sources` 实现（遍历 mcp_sources、调 bridge_mcp_server、异常吞掉）
- `src/mutagent/main.py:46-70` — `App.connect_sources` Declaration + `setup_agent` 注释（声明"不连 MCP"）
- `src/mutagent/sandbox/app.py:54-78` — `SandboxApp.add_namespace(ns, on_remove=...)`
- `src/mutagent/sandbox/_app_impl.py:38` — `_add_namespace` 实现（同名替换时调旧 cleanup + invalidate cache）
- `src/mutagent/sandbox/_namespace.py` — `Namespace`、`NamespaceRegistry`、`_make_help`、渲染函数 `_render_registry`
- `mutio/src/mutio/mcp/client.py` — `MCPClient`（HTTP MCP 客户端）

### 相关规范
- `docs/specifications/feature-mcp-http-adapter.md` — HTTP adapter 设计与 loop 绑定修复
- `docs/specifications/feature-help-namespace-discovery.md` — help() 分层显示、connection state 展示的基础

### 外部参考
- MCP 协议 `initialize` 握手规范（`protocolVersion: "2024-11-05"`）
- `httpx` 异常体系：`ConnectError` / `ReadError` / `ReadTimeout` / `RemoteProtocolError` / `WriteError`

## 设计方案

### 核心抽象：MCPConnection 代理对象

把当前一次性的 `bridge_mcp_server` 升格为长生命周期代理对象，统一管理一个 MCP source 的连接状态和重连逻辑。

```python
class MCPConnection:
    ns_name: str
    config: dict[str, Any]
    main_loop: asyncio.AbstractEventLoop
    namespace: Namespace                  # 始终存在，按状态填充函数
    client: AnyMCPClient | None = None    # 当前 client，断开时 None
    state: Literal["disconnected", "connecting", "connected", "failed"]
    last_error: str | None = None
    last_attempt_at: float | None = None  # 用于冷却期判定
    lock: asyncio.Lock                    # 防并发重连

    async def ensure_connected(self) -> None: ...   # 幂等，含冷却判定
    async def reconnect(self) -> None: ...           # 完整重建：close → new client → connect → list_tools → 更新 ns
    async def close(self) -> None: ...               # sandbox cleanup 入口
```

`connect_sources` 不再直接 `bridge_mcp_server`，而是为每个 mcp source 创建一个 `MCPConnection`，把 `connection.namespace` 加入 sandbox（始终注册，即使未连），cleanup 绑定 `connection.close`（不再绑死单个 client，规避重连后 cleanup 调到旧 client 的问题）。

### 启动模式：autostart 配置

mcp source 配置新增 `autostart` 字段（默认 `true`）：

```jsonc
"playwright": {
    "transport": "stdio",
    "command": "npx",
    "args": ["@playwright/mcp"],
    "autostart": true   // 启动时异步预连，不阻塞 setup
},
"experimental": {
    "transport": "http",
    "url": "http://localhost:9999",
    "autostart": false  // 完全 lazy，首次调用才连
}
```

- `autostart: true`：`connect_sources` 内 `asyncio.create_task(conn.reconnect())`，立即返回；状态 `disconnected → connecting → connected/failed`。**autostart 不阻塞 setup**，启动失败只 log warning，不影响其他 namespace。
- `autostart: false`：`MCPConnection` 创建后留在 `disconnected`，等首次 `__getattr__` 触发。
- 运行时断连后的重连**永远是 lazy**（下次调用触发），不受 `autostart` 影响。

### 调用触发：Namespace `__getattr__` 拦截

纯 lazy 场景下用户调用 `playwright.navigate(...)` 时函数还不存在。解决：`Namespace.__getattr__` 持有 `MCPConnection` 反向引用，未知属性时同步阻塞触发 connect，连接成功后从填充好的 `_functions` 取出函数返回。

```python
def __getattr__(self, name: str) -> Any:
    if name.startswith('_'):
        raise AttributeError(name)
    if name in self._functions:
        return self._functions[name]
    # 未连接 / 上次失败 → 触发重连
    if self._connection is not None and self._connection.state != "connected":
        future = asyncio.run_coroutine_threadsafe(
            self._connection.ensure_connected(), self._connection.main_loop)
        future.result(timeout=30)   # stdio 冷启动按 30s 留余量
        if name in self._functions:
            return self._functions[name]
    raise AttributeError(f"'{self._name}' has no function '{name}'")
```

关键点：
- `__getattr__` 仅在属性不在 `_functions` 时被调用，已连接状态下零开销
- `help()` 渲染只读 `_functions` 和状态字段，不走 `__getattr__`，**不会触发重连**（满足需求 6）
- 阻塞 timeout 用 30s（覆盖 stdio 冷启动 + npx 下载场景）

### 异常分类

新增两个异常类型（放 `_adapter_mcp.py`，与 client 相邻）：

```python
class MCPTransportError(Exception):
    """传输层错误 — 触发重连。"""
class MCPToolError(Exception):
    """业务层错误（isError=True）— 直接抛给用户，不重连。"""
```

改造点：
- `_extract_content` 的 `isError` 分支 → `MCPToolError`
- `StdioMCPClient._send_and_receive`：`BrokenPipeError` / EOF / `"closed unexpectedly"` → `MCPTransportError`
- `HTTPMCPClient.call_tool`：捕获 `httpx.ConnectError/ReadError/ReadTimeout/RemoteProtocolError/WriteError`、`ConnectionResetError`、以及 `HTTPStatusError(404|410)`（session 失效，见下方 HTTP 断连场景 3.4）→ wrap 成 `MCPTransportError`

判定函数：

```python
_TRANSPORT_EXCEPTIONS = (
    httpx.ConnectError, httpx.ReadError, httpx.ReadTimeout,
    httpx.RemoteProtocolError, httpx.WriteError,
    BrokenPipeError, ConnectionResetError,
)

def _is_transport_error(exc: Exception) -> bool:
    if isinstance(exc, MCPTransportError):
        return True
    if isinstance(exc, _TRANSPORT_EXCEPTIONS):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (404, 410):
        return True
    if isinstance(exc, RuntimeError) and "closed unexpectedly" in str(exc):
        return True
    return False
```

### 调用链：tool wrapper 自动重连重试

`_make_tool_func` 改造，闭包持有 `MCPConnection` 而非裸 `client`：

```python
def tool_func(**kwargs):
    async def call_with_retry():
        await conn.ensure_connected()
        try:
            return await conn.client.call_tool(tool_name, kwargs)
        except Exception as exc:
            if not _is_transport_error(exc):
                raise
            # 标记断开，重连一次后重试
            conn.mark_disconnected(str(exc))
            await conn.reconnect()        # 失败时抛 MCPTransportError 给用户
            return await conn.client.call_tool(tool_name, kwargs)
    future = asyncio.run_coroutine_threadsafe(call_with_retry(), conn.main_loop)
    return future.result(timeout=120)
```

**最多重试一次**，避免无限循环。第二次仍失败的异常直接上抛，用户看到 `MCPTransportError: <原因>`。

### 重连策略：stdio / HTTP 走同一流程

`MCPConnection.reconnect()` 不分 transport，统一：

```
1. async with self.lock:
2.   if state == "connected": return        # 别人已经重连完了
3.   if cooldown_active(): raise last_error  # 冷却期内直接抛上次的错
4.   state = "connecting"; last_attempt_at = now()
5.   try:
6.     if old_client: await old_client.close()  # 旧进程/连接释放
7.     new_client = make_client(config)         # stdio: 新 Popen / HTTP: 新 MCPClient
8.     init_result = await new_client.connect()
9.     tools = await new_client.list_tools()
10.    self.client = new_client
11.    _refresh_namespace_functions(tools)      # 增删 ns._functions, invalidate cache
12.    state = "connected"; last_error = None
13.  except Exception as exc:
14.    state = "failed"; last_error = str(exc)
15.    raise MCPTransportError(...) from exc
```

不做 HTTP 专属的「廉价重试」（不 close 复用 httpx 连接池）— 实现单一、行为一致；HTTP 重建毫秒级，成本可接受。

HTTP session 失效（场景 3.4）天然由「完整重建 client」覆盖：旧 `Mcp-Session-Id` 随旧 `MCPClient` 一起丢弃。

### 并发保护

`asyncio.Lock` 在主 loop 上，多个并发调用同时触发 `ensure_connected` → 第一个真重连，其余 await 同一个 lock，醒来后 step 2 短路返回。

### 冷却期

`failed` 状态下短时间内（默认 5s）再次调用直接抛上次的错，不重新发起 connect。避免 stdio stdout 污染等「伪断连」每次调用都 fork 子进程。

配置项 `retry_cooldown`（每个 source 可覆盖，默认 5s，0 表示禁用）。冷却时间过后下次调用会重新尝试；用户也可以等待后重试。

### 非目标

以下能力本期不实现，避免过度设计；如后续出现真实需求再加：

- **不做重连熔断**（连续 N 次失败后进入永久 failed）— 仅靠 `retry_cooldown` 节流；需要彻底禁用某 source 的用户可设 `autostart: false` 并不调用。
- **不暴露显式重连 API**（如 `playwright._reconnect()` / `reconnect("playwright")`）— `__getattr__` 触发 + 冷却过期已足够覆盖需求 1-6。
- **不做 HTTP 专属「廉价重试」**（不 close 复用 httpx 连接池）— stdio / HTTP 重连走同一流程，行为一致、实现单一。
- **不做全局默认配置**（如 `mcp_defaults`）— `autostart` / `retry_cooldown` 每个 source 独立配置。

### Namespace 状态字段与 help 渲染

`Namespace` 增加可选字段：

```python
connection_state: str | None = None       # None = 非 MCP，不显示状态
connection_error: str | None = None       # 失败原因（最近一次）
```

非 MCP 的 NamespaceTools / CLI namespace 这两个字段都是 None，渲染时跳过状态显示，行为与现状一致。

`_render_registry`（顶层 help()）渲染规则：
- `connected` 不显示标签（保持现状简洁）
- `connecting` / `disconnected` / `failed` 才显示标签
- `failed` 后跟简短 reason（截断 60 字符）
- 函数数：连过的显示真实数（断开后保留 last seen），从未连过的显示 `(? functions)`

```
Available namespaces:
  fs                            — 文件系统工具 (8 functions)
  playwright                    — 浏览器自动化 (12 functions)
  serena       [connecting...]
  weather      [failed: connection refused]
  experimental [disconnected]   — 懒连接，调用时启动 (? functions)
```

`_render_namespace`（help(weather)）在 `failed` 状态追加 hint：

```
Namespace: weather

⚠ Connection failed: connection refused
  Last attempt: 2026-05-08 10:23:15
  Calling any function will retry the connection.
```

### 断连场景参考

**stdio**（出现频率：低-中）：
- server 自身崩溃 / 依赖进程（Chrome 等）退出导致 server 自杀
- **server 误把 logger 写到 stdout，污染 JSON-RPC 流**（最常见的实现质量问题，重连大概率仍会再污染 → 冷却期能避免无意义重连）
- pipe broken：Popen.poll() 非 None / write 抛 BrokenPipeError / readline 返回空

**HTTP**（出现频率：高）：
| 场景 | 表现 |
|------|------|
| server 退出，端口关 | `httpx.ConnectError` |
| graceful shutdown | `httpx.ConnectError` |
| 处理中崩溃，连接已建 | `httpx.RemoteProtocolError` / `ReadError` / `ReadTimeout` |
| **server 重启，session-id 失效** | `404 Session not found` (规范) / `MCPError` (不规范) |
| SSE 长连接断 | `httpx.ReadError` / `RemoteProtocolError` |

## 实施步骤清单

### 一、适配层：异常体系 + MCPConnection

- [x] 在 `_adapter_mcp.py` 引入 `MCPTransportError` / `MCPToolError`，提供 `_is_transport_error(exc)` 判定函数（覆盖 httpx 连接类异常、`BrokenPipeError` / `ConnectionResetError`、`HTTPStatusError(404|410)` session 失效）
- [x] 改造 `_extract_content`：`isError=True` 分支抛 `MCPToolError` 而非 `RuntimeError`
- [x] 改造 `StdioMCPClient`：`_send_and_receive` 中 EOF / `BrokenPipeError` / "closed unexpectedly" 全部 wrap 为 `MCPTransportError`
- [x] 顺手统一 `StdioMCPClient.connect()` 中硬编码的协议版本：从 `mutio.mcp.protocol` 导入 `PROTOCOL_VERSION`，替换手写的 `"2024-11-05"`（当前与 mutio 的 `"2025-03-26"` 不一致）
- [x] 改造 `HTTPMCPClient.call_tool` / `connect`：捕获 httpx 传输类异常与 session 失效 wrap 为 `MCPTransportError`
- [x] 新增 `MCPConnection` 类（状态机 + `ensure_connected` + `reconnect` + `close` + `asyncio.Lock` + 冷却期判定），代替现有 `bridge_mcp_server` 的管理职能；保留 `make_client(config)` 工厂函数供重连复用
- [x] 重写 `_make_tool_func`：闭包从 `client` 改为 `MCPConnection`；调用路径「`ensure_connected` → `call_tool` → 传输错时 `mark_disconnected` + `reconnect` + 重试一次」

### 二、Namespace：状态字段 + 懒触发

- [x] `Namespace` 增加可选字段 `connection_state` / `connection_error` / `_connection`（反向引用）；对非 MCP namespace 保持 None 不影响现有表现
- [x] `Namespace.__getattr__` 未命中 `_functions` 且 `_connection 存在 且 状态非 connected` 时，同步阻塞触发 `ensure_connected`（`run_coroutine_threadsafe` + 30s timeout），连接后重查函数表
- [x] 改造 `_render_registry`：`connected` 不显示标签，`connecting` / `disconnected` / `failed` 显示状态标签，`failed` 附 reason（截断 60 字符）
- [x] 改造 `_render_namespace`：`failed` 状态追加 hint 段（connection failed: ... / last attempt: ... / calling any function will retry）

### 三、启动逻辑：connect_sources 重构

- [x] 改造 `main_impl.py:connect_sources`：为每个 mcp source 创建 `MCPConnection`，把 `conn.namespace` 加入 sandbox（`on_remove=conn.close`）；`autostart=true` 的 `asyncio.create_task(conn.ensure_connected())` 并立即 return，不阻塞 setup
- [x] 读配置新字段 `autostart`（默认 `true`）、`retry_cooldown`（默认 `5`秒，0 禁用）；未识别字段作为未来扩展位不报错
- [x] 保留 `bridge_mcp_server` 为 Legacy 入口（内部走 MCPConnection），pysandbox CLI 同步改走 `MCPConnection`

### 四、测试

- [x] **单元测试**：`MCPConnection` 状态机（disconnected→connecting→connected与→failed 路径）、冷却期（failed 期间调用不重试、过后重试）、Lock 并发（两个协程同时 ensure，只需一次真 connect）
- [x] **单元测试**：`_is_transport_error` 覆盖各类 httpx 异常、stdio EOF、`HTTPStatusError(404)` 与非传输错误（`MCPToolError`）的判定区分
- [x] **集成测试**（用 mock client / fake stdio 进程）：autostart=true 启动不阻塞、lazy 调用触发、运行时传输错重连重试一次后成功、重连后 tool 列表变化能反映到 ns._functions
- [x] **渲染测试**：`help()` / `help(ns)` 在 4 种状态下的文本输出符合设计（connected 无标签）

### 五、文档与配置说明

- [x] 更新 `_adapter_mcp.py` / `main.py:App.connect_sources` 的 docstring，说明 `autostart` / `retry_cooldown` 语义与默认值
- [x] 检查 `mutagent/README.md` 或使用文档中 `mcp_sources` 配置示例（README 未包含 MCP 示例，无需修改）
- [x] 检查 `feature-mcp-http-adapter.md` / `feature-help-namespace-discovery.md` 是否需补充互许链接（已在本文档「关键参考 → 相关规范」中列出，反向链接待后续需要时补）

## 实施总结

- **代码变化**
  - `src/mutagent/sandbox/_adapter_mcp.py`：重写。新增 `MCPTransportError` / `MCPToolError` / `_is_transport_error` / `make_client` / `MCPConnection`。原 `bridge_mcp_server` 退化为 Legacy 薄包装，内部走 `MCPConnection.reconnect`。协议版本从 `mutio.mcp.protocol.PROTOCOL_VERSION` 取，消除与 mutio 不一致。
  - `src/mutagent/sandbox/_namespace.py`：`Namespace` 增 `_connection` / `connection_state` / `connection_error`；`__getattr__` 未命中函数且 MCP 未连时阻塞触发 `ensure_connected`（30s timeout）。`_render_registry` / `_render_namespace` 增状态标签与 failed hint。
  - `src/mutagent/builtins/main_impl.py:connect_sources`：改为创建 `MCPConnection` 并 `add_namespace(conn.namespace, on_remove=conn.close)`；`autostart=true` 后台 `create_task(ensure_connected)`，不阻塞 setup。
  - `src/mutagent/cli/pysandbox.py`：同步走 `MCPConnection` 路径，保持与 App.connect_sources 一致。
  - `src/mutagent/main.py:App.connect_sources` docstring：补充 `autostart` / `retry_cooldown` 语义说明。

- **测试变化**
  - `tests/test_adapter_mcp.py` 从 23 个加到 56 个：新增 `TestIsTransportError` (12)、`TestMCPConnectionStateMachine` (8)、`TestToolFuncAutoReconnect` (4)、`TestNamespaceRender` (7)；原有 `isError` 测试从匹配 `RuntimeError` 改为 `MCPToolError`。
  - 全量测试：732 passed。4 skipped（均与本改动无关）。

- **设计与实现偏差**
  - `MCPConnection.reconnect` / `ensure_connected` 拆分为两个公开方法 + 内部 `_do_rebuild`。原设计「`reconnect` 内部看 state 短路」不能同时满足「显式重连刷新 tool 列表」；ensure_connected 负责幂等与并发保护，reconnect 始终全量重建。
  - `make_client` 抛 `ValueError`（配置错、不是传输错）时保留原始异常上抛，不 wrap 为 `MCPTransportError`，但仍记为 `failed` 状态供 help 展示。保证 `bridge_mcp_server` 的原有 ValueError 语义（config 错误直接上报）不被破坏。
