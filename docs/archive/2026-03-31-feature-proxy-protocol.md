# HTTPProtocol PROXY Protocol v1 支持 设计规范

**状态**：✅ 已完成
**日期**：2026-03-31
**类型**：功能设计

## 需求

mutagent 的自研 ASGI 服务器（`HTTPProtocol`）直接从 `transport.get_extra_info("peername")` 获取客户端 IP，写入 `scope["client"]`。当前面有 TCP 代理（如 mutbot Supervisor）时，`peername` 是代理地址而非真实客户端。

mutbot Supervisor 因 Windows 无法传递 socket fd 给子进程，采用 TCP 代理模式转发流量。Worker 的 `scope["client"]` 永远是 `("127.0.0.1", port)`，导致所有依赖 client IP 的逻辑失效（安全拦截、access log、MCP 本地限制）。

XFF（X-Forwarded-For）是 HTTP 请求级 header，但 Supervisor 的 pipe 架构只能处理第一个请求的 header，HTTP/1.1 keep-alive 后续请求无法注入 XFF。需要一个**连接级**的解决方案。

## 设计方案

### PROXY Protocol v1

[PROXY protocol](https://www.haproxy.org/download/1.8/doc/proxy-protocol.txt) 由 HAProxy 设计，专门解决"TCP 代理后端无法得知真实客户端 IP"的问题。v1 是纯文本格式，TCP 连接建立后、应用数据之前发送一行：

```
PROXY TCP4 <src_ip> <dst_ip> <src_port> <dst_port>\r\n
```

示例：
```
PROXY TCP4 10.219.26.186 192.168.1.1 56789 8741\r\n
```

**为什么适合**：
- **连接级**：发一次，该连接上所有请求（keep-alive、WebSocket）都知道真实 IP
- **不侵入应用协议**：在 HTTP 数据之前，不影响 h11 解析
- **行业标准**：nginx、HAProxy、AWS NLB 均支持，未来在 Worker 前加 nginx 无需改 Worker
- **极简**：一行文本，解析代码 ~15 行

### 改动点

**`_protocol.py` — `HTTPProtocol`**：

1. 新增实例变量 `_proxy_header_parsed: bool = False`
2. `data_received()` 首次调用时，检查数据是否以 `b"PROXY "` 开头
3. 如果是：解析该行，提取 `src_ip` 和 `src_port`，覆盖 `self.client`，将剩余数据交给 h11
4. 如果不是：正常处理（兼容无 PROXY protocol 的直连场景）

**不影响**：
- 直连模式（无代理）：数据以 HTTP 方法开头（`GET `、`POST ` 等），不会匹配 `PROXY ` 前缀，行为不变
- WebSocket：PROXY protocol 在 HTTP 握手之前，`self.client` 已被正确设置

### 安全考虑

PROXY protocol 头可以被任何 TCP 客户端伪造。但在 mutagent 的使用场景中，Worker 监听 `127.0.0.1`，只有本机进程能连接，外部无法直接发送伪造的 PROXY protocol 头。

如果未来需要在非 loopback 场景使用，应增加可信来源 IP 白名单校验。当前不需要。

## 关键参考

### 源码
- `src/mutagent/net/_protocol.py:81` — `HTTPProtocol` 类定义
- `src/mutagent/net/_protocol.py:110-121` — `connection_made()` 提取 peername
- `src/mutagent/net/_protocol.py:142-145` — `data_received()` 入口

### 外部参考
- [HAProxy PROXY Protocol v1 规范](https://www.haproxy.org/download/1.8/doc/proxy-protocol.txt)

### 消费者
- mutbot Supervisor（`src/mutbot/web/supervisor.py`）— 连接 Worker 时发送 PROXY protocol 头

## 实施步骤清单

- [x] `HTTPProtocol.__init__` 新增 `_proxy_header_parsed = False` 标志
- [x] `data_received()` 首次调用时检测 `PROXY ` 前缀，解析并覆盖 `self.client`，剩余数据交给 h11
- [x] 单元测试：PROXY protocol 正常解析、无 PROXY 前缀兼容、畸形 PROXY 行处理
