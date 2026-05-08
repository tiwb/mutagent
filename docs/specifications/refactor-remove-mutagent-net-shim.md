# 移除 mutagent.net 兼容层

**状态**：✅ 已完成
**日期**：2026-05-08
**类型**：重构

## 需求

1. `mutagent.net` 已成为纯 re-export shim，所有实现已迁入 `mutio.net` + `mutio.mcp`，不再需要兼容层
2. mutbot 已完成迁移（src 中零引用 `mutagent.net`）
3. 将 mutagent 内部剩余的 `mutagent.net` import 全部替换为直接 `from mutio.*`，然后删除 `mutagent.net` 目录

## 关键参考

- `mutagent/src/mutagent/net/__init__.py` — 兼容层入口，仅 `import mutio.net; import mutio.mcp`
- `mutagent/src/mutagent/net/client.py` — `from mutio.net.client import *`
- `mutagent/src/mutagent/net/server.py` — `from mutio.net.server import *`
- `mutagent/src/mutagent/net/asgi.py` — `from mutio.net.asgi import *`
- `mutagent/src/mutagent/net/mcp.py` — `from mutio.mcp.toolset/promptset/view import ...`
- `mutagent/src/mutagent/net/_mcp_proto.py` — `from mutio.mcp.protocol import *`
- `mutagent/src/mutagent/net/_mcp_impl.py` — `from mutio.mcp._view_impl import ...`
- `mutagent/src/mutagent/net/_protocol.py` — `from mutio.net._protocol import *`
- `mutagent/src/mutagent/net/_client_impl.py` — `from mutio.mcp._client_impl import _ext`
- `mutagent/src/mutagent/net/_server_impl.py` — 空文件，只需删除

## 设计方案

### 映射表

| 旧 import | 新 import |
|-----------|-----------|
| `from mutagent.net.client import HttpClient` | `from mutio.net.client import HttpClient` |
| `from mutagent.net.server import Server, View, Request, Response` | `from mutio.net.server import Server, View, Request, Response` |
| `from mutagent.net.mcp import MCPToolSet` | `from mutio.mcp.toolset import MCPToolSet` |
| `from mutagent.net.mcp import MCPView` | `from mutio.mcp.view import MCPView` |
| `from mutagent.net._mcp_proto import ToolResult` | `from mutio.mcp.protocol import ToolResult` |
| `from mutagent.net._mcp_impl import MCPToolProvider` | `from mutio.mcp._view_impl import MCPToolProvider` |
| `from mutagent.net._mcp_impl import _get_declaration_doc` | `from mutio.mcp._view_impl import _get_declaration_doc` |
| `from mutagent.net._protocol import HTTPProtocol` | `from mutio.net._protocol import HTTPProtocol` |

### 影响范围

**mutagent 源码（6 个文件）**：

| 文件 | 行号 | 改动 |
|------|------|------|
| `builtins/anthropic_provider.py` | 10 | `mutagent.net.client` → `mutio.net.client` |
| `builtins/openai_provider.py` | 9 | 同上 |
| `builtins/web_jina.py` | 12 | 同上 |
| `builtins/web_local.py` | 17 | 同上 |
| `builtins/web_toolkit_impl.py` | 11 | 同上 |
| `sandbox/entry_mcp.py` | 8-9 | 两处 import 替换 |

**测试文件（4 个）**：移至 mutio（零 mutagent 依赖，mutio 自身未覆盖这些场景）

| 文件 | 改动 |
|------|------|
| `tests/test_mcp_multi_view.py` | import 替换后移至 `mutio/tests/` |
| `tests/test_mcp_tool_doc.py` | import 替换后移至 `mutio/tests/` |
| `tests/test_server_views.py` | import 替换后移至 `mutio/tests/` |
| `tests/test_proxy_protocol.py` | import 替换后移至 `mutio/tests/` |

**删除**：

- `mutagent/src/mutagent/net/` 目录（10 个文件）

**不改动**：

- mutbot src — 已迁移完，零引用
- `docs/archive/` — 历史记录，保持原样
- `build/` — 构建产物，下次构建自动更新
- 活跃 spec 文档中两处叙述性提及（`feature-mcp-declarations.md`、`feature-mcp-http-adapter.md`）— 不影响功能，暂不动

### 风险

- **无风险** — 纯 import 路径替换，语义完全等价。10 个文件的改动都是单行替换，每个目标模块已存在且导出对应符号。

## 实施步骤清单

- [x] 替换 mutagent 源码 6 个文件中的 `mutagent.net` import
- [x] 替换测试 4 个文件中的 `mutagent.net` import 并移至 `mutio/tests/`
- [x] 删除 `mutagent/src/mutagent/net/` 目录
- [x] 运行测试验证
