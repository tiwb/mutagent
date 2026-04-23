# Sandbox MCP Adapter 支持 Streamable HTTP

**状态**：✅ 已完成
**日期**：2026-04-23
**类型**：功能设计

## 背景

`mutagent.sandbox._adapter_mcp` 负责把外部 MCP server 的 tools 桥接成 pysandbox 命名空间。当前实现（`_adapter_mcp.py:21` `StdioMCPClient`）只支持 stdio transport，文件顶部注释遗留 TODO：

```python
# TODO: 支持 Streamable HTTP 模式（通过 mutagent.net.client.MCPClient）。
```

而 Streamable HTTP 的 MCP 客户端其实已经在 `mutio.mcp.client.MCPClient` 实现完毕（`mutio/src/mutio/mcp/_client_impl.py`，229 行），`mutagent.net.client` 已经 re-export：

```python
from mutio.mcp.client import MCPClient, MCPError
```

只是 sandbox 侧没有接入路径。

### 驱动场景

用户希望把 `mcp__serena__*`（30+ 工具的 schema 长期占用 Claude Code context）改成通过 `mcp__mutbot__exec_python` 间接调用。Serena 已经以 streamable-http 模式常驻在 `http://127.0.0.1:8800/mcp`，天然适合被 mutbot worker 作为外部 MCP 桥接。

如果 adapter 只支持 stdio，就要么放弃现有 HTTP 常驻（让 mutbot worker 独占 serena 子进程），要么走 supergateway 转接（多一层进程）。两种方案都绕开了已有的 `MCPClient` 实现。

### 相关约束

- mutbot worker 是长驻进程，`mcp_sources` 在 `_app_impl.py:_setup` 启动时一次性连接，后续通过 cache 复用
- `bridge_mcp_server` 返回 `(Namespace, client)`，client 生命周期绑定 app，`_get_mcp_clients` 维护
- 现有工具包装函数 `_make_tool_func` 在 sandbox 线程池里通过 `asyncio.run_coroutine_threadsafe` 调回主 loop

## 目标

1. `mcp_sources` 配置支持声明 transport 类型，值为 `"stdio"`（默认，保持兼容）或 `"http"`
2. `http` 类型复用 `mutio.mcp.client.MCPClient`，不重写 HTTP MCP 协议
3. `bridge_mcp_server` 对上层返回统一的抽象 client（或两种 client 有统一接口），上层 `_app_impl._setup` 无需关心 transport
4. 工具包装函数不变——仍是 `ns.tool_name(**kwargs)` → 返回文本/JSON

### 非目标

- 不实现 MCP resource / prompt 桥接（与本需求无关，见 feature-mcp-declarations.md）
- 不改动 CLI adapter（`_adapter_cli.py`）
- 不改 mutbot 自身的 config schema 之外的逻辑（mutbot 只是消费者）

## 设计要点

### 配置 schema

新增 `transport` 字段，默认 `"stdio"` 保持现有行为：

```jsonc
// ~/.mutbot/config.json
{
  "mcp_sources": {
    "playwright": {
      "transport": "stdio",        // 可省略
      "command": "npx",
      "args": ["-y", "@playwright/mcp"],
      "shell": true
    },
    "serena": {
      "transport": "http",
      "url": "http://127.0.0.1:8800/mcp",
      "timeout": 60                // 可选，透传给 MCPClient
    }
  }
}
```

字段与 transport 的关系：

| 字段 | stdio | http |
|------|-------|------|
| `command` | 必需 | 忽略 |
| `args` | 可选 | 忽略 |
| `shell` | 可选 | 忽略 |
| `url` | 忽略 | 必需 |
| `timeout` | 忽略 | 可选（默认 30） |

校验时机：`_setup` 循环里对每个 server 读 transport，字段缺失记 warning 并跳过（和现有失败处理一致，不中断整个 setup）。

### Adapter 结构

方案 A（推荐）：`_adapter_mcp.py` 里新增 `HTTPMCPClient` 适配器，薄包一层 `mutio.mcp.client.MCPClient`，把它的接口对齐 `StdioMCPClient`：

```python
class HTTPMCPClient:
    def __init__(self, url: str, timeout: float = 30.0) -> None:
        self._mcp = MCPClient(url=url, timeout=timeout)

    async def connect(self) -> dict[str, Any]:
        await self._mcp.connect()
        return {"serverInfo": self._mcp.server_info,
                "capabilities": self._mcp.server_capabilities}

    async def list_tools(self) -> list[dict[str, Any]]:
        return await self._mcp.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        result = await self._mcp.call_tool(name, **arguments)
        # 复用 StdioMCPClient 的 content 抽取逻辑（isError → raise，text → json.loads 尝试）
        return _extract_content(result)

    async def close(self) -> None:
        await self._mcp.close()
```

抽取一个公共 `_extract_content(result) -> Any` 函数，把现有 `StdioMCPClient.call_tool:78-93` 的逻辑复用。

`bridge_mcp_server` 改签名，加一个 server_config 入口（或保留现签名 + 新增 `bridge_mcp_server_http`）：

**子方案 A1**（推荐）：统一入口 + 分派

```python
async def bridge_mcp_server(
    ns_name: str,
    server_config: dict[str, Any],
) -> tuple[Namespace, AnyMCPClient]:
    transport = server_config.get("transport", "stdio")
    if transport == "stdio":
        client = StdioMCPClient(
            server_config["command"],
            server_config.get("args", []),
            shell=server_config.get("shell", False),
        )
    elif transport == "http":
        client = HTTPMCPClient(
            server_config["url"],
            timeout=server_config.get("timeout", 30.0),
        )
    else:
        raise ValueError(f"Unknown MCP transport: {transport!r}")
    await client.connect()
    ...
```

- `AnyMCPClient = StdioMCPClient | HTTPMCPClient`，`_get_mcp_clients` 的类型注解跟着改
- `_app_impl._setup:198` 只传 `ns_name, server_config`，解包逻辑下沉到 adapter

**子方案 A2**：保持 `bridge_mcp_server` 原签名，新增 `bridge_mcp_server_http`，分派放 `_app_impl._setup`

- 好处：改动小，不动调用方
- 坏处：上层被迫感知 transport，未来加 websocket 又要再改一次

推荐 A1，改动可控且扩展性好。

### `_make_tool_func` 无需改动

两个 client 的 `call_tool(name, arguments: dict) -> Any` 对齐后，函数体不变。stdio 版 `arguments` 是 `**kwargs` 封装成 dict，http 版通过 `**arguments` 展开回 `MCPClient.call_tool` 的签名——行为等价。

### 连接失败处理

HTTP 模式下，server 未启动/端口不通在 `await client.connect()` 抛异常（`httpx.ConnectError` 等）。沿用现有 `_setup` 里的 `try/except Exception` 模式，记 warning、跳过这个 ns、不影响其他 mcp_sources。

## 已决策项

- **不加 client 基类**：两个 client 用 duck typing + `Union` 类型注解表达约束，`abc.ABC` 不解决实际问题（未来 websocket / sse 出现时再评估）
- **不做 session 失效重连**：初版 HTTP session 失效直接报错让上层看 log，后续遇到再加重试层
- **不引入鉴权字段**：`MCPClient` 未支持，需要时在 mutio 层增强后 adapter 自然透传
- **不支持 SSE legacy transport**：serena 已走 Streamable HTTP，无驱动场景

## 实施步骤清单

- [x] `_adapter_mcp.py`：抽取 `_extract_content(result) -> Any` 公共函数，替换 `StdioMCPClient.call_tool` 里的 content 抽取逻辑
- [x] `_adapter_mcp.py`：新增 `HTTPMCPClient` 类（依赖 `mutio.mcp.client.MCPClient`），接口对齐 `StdioMCPClient`
- [x] `_adapter_mcp.py`：改 `bridge_mcp_server` 签名为 `(ns_name, server_config)`，按 `transport` 字段分派
- [x] `_adapter_mcp.py`：删除文件顶 TODO 注释，更新 docstring
- [x] `_app_impl.py:_setup`：调用点改为 `bridge_mcp_server(ns_name, server_config)`，`_get_mcp_clients` 类型注解扩为联合类型
- [x] 单测：`HTTPMCPClient.connect/list_tools/call_tool` 对 mock `MCPClient` 行为正确、`_extract_content` 处理 isError/text/JSON/多 text 的所有分支、`bridge_mcp_server` 分派 stdio/http/unknown
- [x] 集成测试：用户手动用 serena (streamable-http) 验收（见验收标准）——2026-04-23 两条验收标准均通过
- [x] 外部影响：`D:/ai_skills/serena/skill.md` 已在 "接入形式：直连 MCP vs mutbot 桥接（推荐）" 章节（line 50-89）详细说明，无需追加

## 验收标准

- `~/.mutbot/config.json` 配 `mcp_sources.serena` with `transport: "http"`, `url: "http://127.0.0.1:8800/mcp"`，重启 mutbot worker 后：
  - `exec_python code="help(serena)"` 列出 serena 的所有 tool
  - `exec_python code="serena.find_symbol(name_path_pattern='MCPClient', relative_path='mutio/src/mutio/mcp/client.py')"` 返回结果
- 原有 stdio 配置（如 playwright）行为不变
- HTTP server 未启动时，`setup` 记 warning 并跳过，不阻塞其他 mcp_sources 和 cli_sources 初始化

---

## 迭代 2：修复 HTTP 调用 "Event loop is closed"（2026-04-23）

### 现象

初版实施后，`help(serena)` 和 `help(serena.find_symbol)` 正常（元数据展示来自 setup 阶段缓存），但实际调用任何工具时炸：

```
serena.find_symbol(name_path_pattern="MCPClient", ...)
→ RuntimeError: Event loop is closed
  (traceback 深入 httpx.AsyncClient.aclose → anyio → BaseEventLoop._check_closed)
```

stdio transport 的 MCP（playwright 等）不受影响。

### 根因分析

串起执行链：

```
worker 主 loop (L0)
  └─ PySandboxToolkit.pysandbox  (mutbot/builtins/pysandbox_toolkit.py:48)
       └─ loop.run_in_executor(None, exec_code, ...)    ← 甩到线程池
            └─ [线程池] exec_code 执行用户代码
                 └─ serena.find_symbol(...)
                      └─ _make_tool_func 内 (_adapter_mcp.py:215-224):
                          try: loop = asyncio.get_running_loop()    # 线程池里无 loop
                          except RuntimeError:
                              asyncio.run(client.call_tool(...))     # ★ 开临时 loop L1
                                   └─ httpx.AsyncClient.post(...)    # client 绑在 L0 创建
                                                                      # → 跨 loop → 炸
```

**关键差异**：

- `httpx.AsyncClient` 内部走 anyio，**资源（transport、连接池、streams）绑定创建时的 event loop**。`MCPClient.connect()` 在 `_app_impl._setup` 里跑，绑到主 loop L0。后续 `asyncio.run()` 开的 L1 里调用它，anyio 找不到原 loop → `Event loop is closed`。
- `StdioMCPClient` 不炸的原因：`_send_and_receive` 是**纯同步 IO**（`subprocess.stdin/stdout.readline`），`async def _request` 只是 `loop.run_in_executor(None, self._send_and_receive, ...)` 的薄壳——没有 loop 绑定资源，换任何 loop 都能跑。

所以初版 `_make_tool_func` 里 `try: get_running_loop() except RuntimeError: asyncio.run()` 的 fallback **对 stdio 能用纯属巧合**，对 HTTP 直接暴露了语义错误。

### 修复方案选型

| 方案 | 改动位置 | 成本 | 副作用 |
|------|---------|------|--------|
| **A. 捕获主 loop + run_coroutine_threadsafe**（推荐）| `_adapter_mcp.py` | 最小 | 要求 setup 在主 loop 跑（已成立） |
| **B. 每次调用新建 MCPClient** | `_adapter_mcp.py` | 小 | 每次 call 多一次 initialize 握手；丢失 session_id（server 侧状态被重置） |
| **C. MCPClient 改用 `httpx.Client`（同步）** | `mutio/mcp/_client_impl.py` | 大 | 丢失 async 并发；影响 MCPClient 所有消费者；不改 adapter 也能跑但侵入性最大 |

**选 A**：精准打击根因，不改 mutio，语义清晰。

### 方案 A 设计

核心思路：**所有 MCP client 的 IO 都汇聚到 setup 时捕获的主 loop**，通过 `run_coroutine_threadsafe` 跨线程调度过去。

#### 修改 1：`bridge_mcp_server` 捕获主 loop

```python
async def bridge_mcp_server(
    ns_name: str,
    server_config: dict[str, Any],
) -> tuple[Namespace, AnyMCPClient]:
    client = _make_client(server_config)        # stdio/http 分派
    await client.connect()
    main_loop = asyncio.get_running_loop()      # ← 捕获 setup 时的主 loop

    tools = await client.list_tools()
    ns = Namespace(ns_name)
    for tool in tools:
        fn = _make_tool_func(
            client, tool["name"], tool.get("description", ""),
            tool.get("inputSchema", {}),
            main_loop,                          # ← 闭包到 tool_func
        )
        ns.register(tool["name"], fn, tool.get("description", ""))
    return ns, client
```

#### 修改 2：`_make_tool_func` 永远走主 loop

```python
def _make_tool_func(client, tool_name, description, input_schema, main_loop):
    ...
    def tool_func(**kwargs):
        # 不再区分"有无 running loop" —— pysandbox 本就在线程池里跑
        # MCP client 的 IO 必须在创建它的 loop 里做
        future = asyncio.run_coroutine_threadsafe(
            client.call_tool(tool_name, kwargs), main_loop)
        return future.result(timeout=120)
    return tool_func
```

删掉原 try/except。该 fallback 本来是想照顾"独立脚本模式"（无主 loop 直接跑 sandbox），但：
- stdio 模式下它能用是巧合（内部 `run_in_executor` 无 loop 绑定）
- http 模式下它不能用
- 真要支持独立脚本模式，应在 setup 时判断：`main_loop = asyncio.get_running_loop() if running else asyncio.new_event_loop()`，并把 loop 的生命周期管起来。本迭代不做。

### 为什么方案 A 是对的

1. **对齐实际约束**：httpx.AsyncClient 要求所有 IO 在创建它的 loop 里跑；方案 A 强制所有 call 汇聚到那个 loop
2. **stdio 也统一**：stdio 当前靠巧合能工作的路径被收敛，所有 MCP 调用统一在主 loop 排队，行为可预测
3. **并发友好**：worker 主 loop 单线程，多个并发 pysandbox 调用通过 `run_coroutine_threadsafe` 自然排队；`MCPClient._request_id += 1` 这种非原子操作天然安全
4. **改动隔离**：不动 mutio，不动 mutbot，只改 adapter

### 新增待定问题

#### QUEST Q5：close 路径是否也要回主 loop

`_app_impl._setup` 里的 try/except 没调 `client.close()`，但 app 销毁时 `_get_mcp_clients` 里的 client 需要被清理。HTTP client close 也走 `httpx.AsyncClient.aclose()`，同样需要在创建它的 loop 里跑。

**建议**：销毁路径如果本来就在主 loop 里（app 生命周期通常同主 loop），`await client.close()` 直接可用。如果存在跨线程销毁的路径，再走 `run_coroutine_threadsafe`。本迭代先假设销毁在主 loop，留作 follow-up 验证。

#### QUEST Q6：`future.result(timeout=120)` 的 120s 是否够

serena 大项目首次 `find_symbol` / `search_for_pattern` 可能触发 LSP 冷索引，超过 120s 存在概率。但这是**初版就有的**问题，不是本迭代引入的，不在本次修复范围。后续可考虑：
- 参数化 timeout（per-transport 或 per-tool）
- 让 `MCPClient.timeout` 也透传给 future 等待时间

### 实施步骤（增量）

- [x] `_adapter_mcp.py`:
   - [x] `bridge_mcp_server` 里捕获 `main_loop = asyncio.get_running_loop()`
   - [x] 把 `main_loop` 传进 `_make_tool_func`
   - [x] `_make_tool_func` 删 try/except，改为只走 `run_coroutine_threadsafe(main_loop)`
- [x] 本迭代不涉及 `_app_impl.py`
- [x] 回归测试：
   - [x] HTTP：`serena.find_symbol(...)` 连续调 3 次均返回，无 "Event loop is closed"（2026-04-24 单个 exec_code 块内连续 4 次均成功）
   - [x] Stdio：单测 `test_stdio_default_transport` 回归通过；`_make_tool_func` 与 stdio 无 transport 特异性逻辑，行为等价
   - [x] 新增单测 `TestToolFuncCrossThread`：`tool_func_invoked_from_worker_thread` / `tool_func_repeated_calls` 覆盖线程池场景

### 新增验收标准

- [x] `serena.find_symbol(name_path_pattern="MCPClient", relative_path="mutio/src/mutio/mcp/client.py")` 返回符号结果
- [x] 连续调用 3 次 serena 任意 tool，全部成功（单 exec_code 块内 4 次调用全返回）
- [x] stdio MCP（如 playwright）回归通过（由单测覆盖）
