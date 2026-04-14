# ASGI Access Log 与安全决策日志 设计规范

**状态**：✅ 已完成
**日期**：2026-03-31
**类型**：功能设计

## 需求

1. mutagent ASGI 服务器无 access log，HTTP 请求的 client_ip、method、path、status_code 完全不可见
2. mutbot 安全中间件无决策日志，拦截/放行的原因无从追查
3. 公网安全加固功能实施完毕、单元测试通过，但实际未生效——因为 Supervisor TCP 透传导致 Worker 看到的 client_ip 全是 `127.0.0.1`。如果有 access log，一眼就能发现

## 关键参考

- `mutagent/src/mutagent/net/_protocol.py:359-373` — `RequestResponseCycle.run()`，ASGI 应用调用点，目前无 access log
- `mutagent/src/mutagent/net/_protocol.py:408-410` — `_send_response_start()`，此处可获取 `status_code`
- `mutagent/src/mutagent/net/_protocol.py:234-269` — WebSocket upgrade 处理，无日志
- `mutbot/src/mutbot/auth/middleware.py:98-139` — `_mutbot_before_route()`，安全中间件，无决策日志
- `mutbot/src/mutbot/auth/network.py:69-108` — `resolve_client_ip()`，IP 解析逻辑

## 设计方案

### 层次划分

两层日志职责分明，不重复：

| 层 | 位置 | logger | 内容 | 级别 |
|----|------|--------|------|------|
| Access Log | mutagent `_protocol.py` | `mutagent.net.access` | `client_ip method path → status_code` | INFO |
| 安全决策 | mutbot `middleware.py` | `mutbot.auth.middleware` | 拦截/放行原因 | DEBUG（放行本地）/ INFO（拦截远程） |

### Access Log（mutagent 层）

采用 uvicorn 兼容格式 + response size + 响应耗时（CLF 标准字段 + 常用扩展）：

```
INFO  mutagent.net.access — 127.0.0.1:54321 - "GET /api/workspaces HTTP/1.1" 200 1234 45ms
INFO  mutagent.net.access — 10.219.26.186:12345 - "WebSocket /ws/workspace/abc123"
```

- HTTP：`{client_addr} - "{method} {path} HTTP/{version}" {status_code} {response_bytes} {elapsed_ms}ms`
  - `response_bytes`：优先取 Content-Length header；chunked 响应累计 body 字节数；无 body 时为 `0`
  - `elapsed_ms`：从 `RequestResponseCycle.__init__()` 到响应完成的耗时，`time.perf_counter()` 计时
  - 记录时机：`_send_response_body()` 中 `more_body=False` 时（响应完成，status_code/body size/耗时三者均已确定）；`_send_500()` 单独记录
- WebSocket：`{client_addr} - "WebSocket {path}"`，在 `_handle_ws_upgrade()` 末尾记录（无 size/耗时，WebSocket 是长连接）

### 安全决策日志（mutbot 层）

在 `_mutbot_before_route()` 的关键分支点添加日志：

| 场景 | 级别 | 消息 |
|------|------|------|
| 无 auth + 本地请求 → 放行 | DEBUG | `allow local (no auth): {client_ip} {path}` |
| 无 auth + 非本地 → 重定向 setup | **INFO** | `redirect to /auth/setup (no auth): {client_ip} {path}` |
| 无 auth + 非本地 WebSocket → 4401 | **INFO** | `reject ws (no auth): {client_ip} {path}` |
| /mcp 或 /internal/ 非本地 → 403 | **WARNING** | `deny local-only path: {client_ip} {path}` |
| 已认证 → 放行 | DEBUG | `allow authenticated: {user_sub} {path}` |
| 未认证 HTTP → 302 | INFO | `redirect to login: {client_ip} {path}` |
| 未认证 WebSocket → 4401 | INFO | `reject ws (unauthenticated): {client_ip} {path}` |

拦截/拒绝类用 INFO 或 WARNING（操作人员需要看到）；本地放行用 DEBUG（正常情况太频繁）。

### 实施要点

#### Access Log — `_protocol.py` 改动

**计时**：在 `RequestResponseCycle.__init__()` 中记录 `self._start_time = time.perf_counter()`。

**response size**：`_send_response_start()` 已解析 `content-length` 存入 `self._expected_content_length`（L417-419）。对于 chunked 响应（`self._chunked=True`），需在 `_send_response_body()` 中累计 `self._response_body_size += len(body)`。

**日志记录时机**：不能在 `_send_response_start()` 中记录——此时 chunked 响应的 body size 尚未确定。应在 **`_send_response_body()` 的 `more_body=False`** 分支（L451，即响应完成时）记录，此时 status_code、body size、耗时三者都已确定。

需要在 `_send_response_start()` 中暂存 `self._status_code = status_code`，供后续日志使用。

**500 异常**：`run()` 中 catch 异常调用 `_send_500()` 时（L362-366），`_send_500()` 直接写 h11 不经过 `_send_response_body()`，需单独记录 access log。

**WebSocket**：在 `_handle_ws_upgrade()` 末尾（L269 之后）记录，直接用 `self.client` 和 scope 信息。

#### 安全决策日志 — `middleware.py` 改动

纯日志添加，在 `_mutbot_before_route()` 各分支的 return 前加一行 `logger.xxx()`。文件顶部已有 `logger = logging.getLogger(__name__)`（即 `mutbot.auth.middleware`），无需新增 logger。

共 7 个分支点（对应设计表格的 7 行），每个加一行日志。

## 实施步骤清单

- [x] `_protocol.py`：`RequestResponseCycle` 加 access log（`_start_time`、`_status_code`、`_response_body_size` 字段 + 日志记录）
- [x] `_protocol.py`：`_handle_ws_upgrade()` 加 WebSocket access log
- [x] `middleware.py`：7 个分支加安全决策日志
- [x] 启动服务验证日志输出

## 反向代理场景下 access log 显示错误 IP（2026-03-31）

### 问题现象

mutbot Supervisor 注入 `X-Forwarded-For` 修复安全拦截后，中间件已能正确解析真实客户端 IP（拦截生效），但 access log 仍然显示 `127.0.0.1`——所有请求看起来都来自本地，无法区分内外部访问。

### 根因

access log 在 `_protocol.py` 中记录的是 TCP 层 `self.client`（即 `transport.get_extra_info("peername")`），这是直连的 TCP 对端地址。在反向代理（Supervisor、nginx）场景下，TCP 对端是代理而非真实客户端。

标准做法（nginx/Apache/uvicorn）：有反向代理时，access log 应从 `X-Forwarded-For` 解析真实 IP。

### 需要解决的问题

access log 当前在 mutagent 协议层记录，但 XFF 解析逻辑（`resolve_client_ip()` + `trusted_proxies` 配置）在 mutbot 层。需要一种方式让协议层拿到解析后的真实 IP。

### 方案：ASGI scope 注入

采用 scope 注入方案（方向 1）。上层中间件将解析后的 IP 写入 `scope["real_client_ip"]`，协议层日志优先取此字段，fallback 到 TCP client。

- mutagent 不依赖 mutbot，只读一个可选 scope 字段——不破坏通用性
- 符合 ASGI 生态惯例（uvicorn `--proxy-headers` 也是类似思路）
- 排除方向 2（协议层自行解析 XFF 需引入 trusted_proxies 配置，违反依赖方向）和方向 3（回调/钩子，过度设计）

### 关键参考

- `mutagent/src/mutagent/net/_protocol.py:501` — `_log_access()` 使用 `self.client` 记录地址
- `mutagent/src/mutagent/net/_protocol.py:275` — WebSocket access log 同样使用 `self.client`
- `mutbot/src/mutbot/auth/middleware.py:111-114` — `resolve_client_ip()` 解析后写入 `current_client_ip` ContextVar
- `mutbot/src/mutbot/auth/network.py:69-108` — `resolve_client_ip()` XFF 解析逻辑

## 反向代理 IP 修复 — 实施步骤

- [x] `middleware.py`：`resolve_client_ip()` 后将结果写入 `scope["real_client_ip"]`
- [x] `_protocol.py`：`_log_access()` 优先取 `scope["real_client_ip"]`，fallback TCP client
- [x] `_protocol.py`：WebSocket access log 同理，优先取 `scope["real_client_ip"]`
- [x] 启动服务验证反向代理场景下日志显示真实 IP
