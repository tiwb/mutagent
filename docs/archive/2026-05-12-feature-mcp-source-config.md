# MCP 连接设置 UI

**状态**：✅ 已完成
**日期**：2026-05-12
**类型**：功能设计

## 需求

1. 在 mutagent WebUI 工具栏 ☰ 菜单中增加「MCP 连接设置」入口
2. 点击后打开设置 Drawer，展示已配置的 MCP 源列表，支持添加 / 编辑 / 删除
3. 提供显式 **Connect / Disconnect / Reconnect** 控制，覆盖 `autostart` 自动策略
4. 连接成功后展示该 source 提供的 tools 列表（含参数说明）
5. 配置持久化到 `config.mcp_sources`，与 LLM API 设置体验对齐

## 现状对照

### Settings 子系统已重构为统一容器

webui 的设置 UI 已不是「一个 Action 对应一个 Drawer」的旧模型，当前结构是：

```
☰ 菜单（_toolbar_impl.py）
  └─ drawer.list_panels() 自动遍历，每个 panel 生成一个 OpenSettingsAction
       ↓ execute(panel_id)
  SettingsDrawer（settings.py / _settings_drawer_impl.py）
  └─ active_panel_id 路由到对应 SettingsPanel 子类
       ├─ LLMSettingsPanel         _settings_llm.py   panel_id="llm"
       └─ MCPSettingsPanel（本文） _settings_mcp.py   panel_id="mcp"
```

`SettingsDrawer.__init__` 通过 `mutobj.discover_subclasses(SettingsPanel)` 自动收集
所有 `SettingsPanel` 子类，按 `panel_placement` 排序后注册路由。
菜单项 `OpenSettingsAction(panel_id, label, placement)` 是参数化通用 Action，
**新增面板不需要改 toolbar / conversation / 任何 Action 类**。

### MCP 运行时栈

| 关注点 | 位置 |
|--------|------|
| Config key | `config.mcp_sources` — `dict[str, ServerConfig]` |
| 连接代理 | `sandbox/_adapter_mcp.py::MCPConnection` |
| Client 工厂 | `sandbox/_adapter_mcp.py::make_client`（stdio / http） |
| 状态字段 | `MCPConnection.state` ∈ `disconnected / connecting / connected / failed` |
| 启动期注入 | `builtins/main_impl.py::connect_sources` |
| 已有 API | `ensure_connected()` 幂等懒连，`reconnect()` 全量重建（绕过 cooldown），`close()` 释放 client + `_set_state("disconnected")` + 摘除 peer providers |
| Sandbox 摘除钩子 | `sandbox.add_namespace(ns, on_remove=conn.close)` — 摘 namespace 自动触发 close |
| Peer namespaces | `MCPConnection.peer_namespaces` — pysandbox 源融合进来的额外命名空间 |

## 配置模型

### 字段表

| 字段 | 类型 | 必需 | 默认值 | 说明 |
|------|------|:---:|--------|------|
| `transport` | `"stdio" \| "http"` | — | `"stdio"` | 传输方式 |
| `autostart` | `bool` | — | `true` | 启动后自动连接；`false` 则完全 lazy |
| `retry_cooldown` | `float` | — | `5.0` | 自动重试冷却秒数（0 禁用）。**显式 Connect / Reconnect 总是绕过 cooldown** |
| **stdio 专用** | | | | |
| `command` | `str` | ✅ | — | 可执行文件 |
| `args` | `list[str]` | — | `[]` | 命令行参数 |
| `shell` | `bool` | — | `false` | 是否通过 shell 执行（注意命令注入风险） |
| `env` | `dict[str, str]` | — | `{}` | 环境变量。⚠️ **当前 `StdioMCPClient` 尚未透传，需要随本特性补实现** |
| **http 专用** | | | | |
| `url` | `str` | ✅ | — | MCP server URL |
| `timeout` | `float` | — | `30.0` | HTTP 请求超时秒数 |

> HTTP 自定义 headers / 鉴权（Bearer 等）当前不支持，留给后续特性扩展。

### 示例

```json
{
  "mcp_sources": {
    "my-filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-filesystem"],
      "env": {},
      "autostart": true,
      "retry_cooldown": 5.0
    },
    "playwright": {
      "transport": "http",
      "url": "http://127.0.0.1:8800/mcp",
      "timeout": 30.0,
      "autostart": false
    }
  }
}
```

### 名字归一化与冲突

`MCPConnection` 内部用 `_sanitize_ns_name(ns_name)` 把非 Python 标识符字符
（`-`、`.`、首位数字等）替换为下划线，作为 sandbox registry 中的 namespace 名。

- **dict key 保留用户原始输入**（用于配置展示与持久化）
- **sandbox 注册名是 sanitized 后的版本**
- **Save 时校验**：所有 source 的 sanitized 名两两不重复，否则报错
  （例如 `my-fs` 与 `my.fs` 会撞成 `my_fs`）
- 编辑页输入 Name 时实时显示 `运行时名: my_fs`，避免意外

## 连接状态机

`MCPConnection` 管理单个 MCP source 的完整生命周期：

```
                      Connect / 首次 lazy 调用
   ┌──────────────┐   ──────────────────────→   ┌──────────────┐
   │ disconnected │                              │  connecting  │
   └──────────────┘   ←──────────────────────    └──────┬───────┘
       ▲     ▲           Disconnect                     │
       │     │           （sandbox.remove_namespace）     │
       │     │                                          │成功    │失败
       │     │                                          ▼        ▼
       │     │                                  ┌──────────────┐ ┌──────────────┐
       │     └──Disconnect─────────────────────│  connected   │ │    failed    │
       │                                       └──────┬───────┘ └──────┬───────┘
       │              tool 调用传输错                    │                │
       │              (mark_disconnected)               │                │
       │           ────────────────────────────────────┘                │
       │                                                                │
       └─────────────Disconnect─────────────────────────────────────────┘
                                          ▲
                                          │ Reconnect / cooldown 期满后下次自动 ensure
                                          │
                                  （回到 connecting）
```

| 状态 | 含义 | namespace 是否在 registry |
|------|------|:---:|
| `disconnected` | 从未连过 / 用户主动 Disconnect / 启动 `autostart=false` | ❌ |
| `connecting` | `_do_rebuild` 进行中 | — |
| `connected` | 当前可用，tool 可调用 | ✅ |
| `failed` | 上次连接失败；冷却期内自动重试被抑制，**期满后下次 `ensure_connected` 会重建** | ✅（保留，等下次调用自动重连） |

`failed` 与 `disconnected` 的差别：

- `failed` 由系统判定（连不上 / tool call 传输错），namespace 留在 registry，下次访问自动尝试
- `disconnected` 由用户判定（主动 Disconnect），namespace 摘除，等同于「该 source 已停用」

### autostart 与显式控制

| autostart | 启动后状态 | 行为 |
|:---------:|-----------|------|
| `true`（默认） | 后台任务驱动 `connecting → connected/failed` | 不阻塞启动 |
| `false` | `disconnected`，namespace **不进 registry** | 用户点 Connect 时再注册 |

两种策略下 UI 都可以覆盖：失败的可以 Reconnect、已停用的可以 Connect、连着的可以 Disconnect 或 Reload tools。

## MCPSettingsPanel 设计

### 文件与接入

新增 `mutagent/src/mutagent/webui/_settings_mcp.py`，类 `MCPSettingsPanel(SettingsPanel)`：

```python
class MCPSettingsPanel(SettingsPanel):
    panel_id: ClassVar[str] = "mcp"
    panel_title: ClassVar[str] = "MCP 连接设置"
    panel_placement: ClassVar[str] = "settings:10/20"  # LLM=10/10 在前
```

在 `webui/settings.py` 末尾追加 `from . import _settings_mcp  # noqa`，
`SettingsDrawer` 自动发现并路由，**不动 toolbar / conversation / Action**。

### 面板状态字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `current_step` | `str` | `"list"` / `"edit"` |
| `editing_key` | `str` | 编辑中的 source dict key（"" = 新建） |
| `editing_is_new` | `bool` | 是否新建（决定 Remove 按钮可见性） |
| `form_*` | 多个 | 表单字段镜像（name / transport / command / args / env_text / url / timeout / autostart / retry_cooldown） |
| `env_text` | `str` | env 的文本输入态（`KEY=VALUE` 多行），保存时解析为 dict |
| `error` / `notice` | `str` | 表单级错误 / 提示 |
| `pending_button` | `str` | 进行中的异步操作（行 key + "connect" / "disconnect" / "reconnect"），用于按钮置灰 |

### 列表页

每行展示：source 名 / transport tag / 状态 tag / tool 数量 / 单按钮，按状态映射：

| 状态 | 列表行按钮 | 行为 |
|------|-----------|------|
| `disconnected` | **Connect** | `await conn.reconnect()` |
| `connecting` | **Connect** 置灰 + spinner | 见「取消语义」 |
| `connected` | **Disconnect** | `sandbox.remove_namespace(conn.namespace)`（触发 `on_remove=conn.close`） |
| `failed` | **Reconnect** | `await conn.reconnect()` |

```
┌─ MCP 连接设置 ───────────────────────────────────────────┐
│ [+ Add stdio]  [+ Add HTTP]                              │
│                                                          │
│  my-filesystem  [stdio]  [connected]  3 tools  [Disconnect] │
│  playwright     [http]   [failed: connection refused]    │
│                                          [Reconnect]     │
│  notion         [http]   [disconnected]   [Connect]      │
│                                                          │
│  Config: ~/.mutagent/config.json                         │
└──────────────────────────────────────────────────────────┘
```

> Connected 行只放 Disconnect。「不断连接刷新 tool 列表」的需求由编辑页的
> Reload tools 按钮承担（见编辑页）。

### 编辑页

```
┌─ MCP 连接设置 ───────────────────────────────────────────┐
│ [← Back]  my-filesystem   [stdio]  [connected]           │
│                                                          │
│  Name:       [my-filesystem                          ]   │
│              运行时名: my_filesystem                       │
│  Transport:  [stdio ▾]                                   │
│                                                          │
│  ── stdio ──                                             │
│  Command:    [npx                                    ]   │
│  Args:       [-y, @anthropic/mcp-server-filesystem   ]   │
│  Shell:      [ ]                                         │
│  Env:        [TEXTAREA, KEY=VALUE 一行一对，autoSize]    │
│                                                          │
│  ── http ──（transport=http 时显示）                      │
│  URL:        [                                       ]   │
│  Timeout:    [30                                     ] s │
│                                                          │
│  [✓] Autostart      Retry cooldown: [5.0] s              │
│                                                          │
│  ── Connection ──                                        │
│  Status: [connected]                                     │
│  Buttons: [Disconnect] [Reload tools]                    │
│                                                          │
│  ── Tools (3) ──                                         │
│  ▸ read_file(path)            Read contents of a file    │
│  ▸ write_file(path, content)  Write content to a file    │
│  ▸ list_directory(path)       List directory contents    │
│                                                          │
│  ── Peer namespaces ──（仅 pysandbox 类 source 显示）     │
│  • mutbot.web    (8 functions)                           │
│  • mutbot.logs   (5 functions)                           │
│                                                          │
│  [Remove]                                  [Save]        │
└──────────────────────────────────────────────────────────┘
```

编辑页按钮规则（与列表页有差异，因为多按钮可放）：

| 状态 | 编辑页按钮 |
|------|-----------|
| `disconnected` | `[Connect]` |
| `connecting` | `[Connect]`(disabled) + spinner |
| `connected` | `[Disconnect]  [Reload tools]` |
| `failed` | `[Reconnect]  [Disconnect]` |

`Reload tools` = `await conn.reconnect()`（已 connected 状态下的全量重建，
适合 server 端动态新增 tool 后刷新展示）。

### Tools 列表渲染

- 折叠态：一行一个 `tool_name(params)` + 描述首行
- 展开态：完整描述 + 参数表（name / type / required / description）
- 多个 tool 可同时展开（非手风琴）
- 未连接：显示 `(not connected — use Connect to discover tools)`
- Peer namespaces 区只展示数量，不下钻（要看具体 tool 走 sandbox `help()` 即可）

**数据来源**：建议在 `MCPConnection` 上加 public 方法 `list_tools_metadata() ->
list[dict]` 返回 `[{name, description, input_schema, source_namespace}, ...]`，
避免 panel 直接读 `namespace._functions` 这类私有字段。peer namespaces 同理走
`MCPConnection.peer_namespaces` 公开属性。

### 表单细节

#### Transport 切换

- `stdio` 显示 command / args / shell / env，隐藏 url / timeout
- `http` 反之
- 切换 transport 不清空对侧字段（用户切回去还能用），但 Save 时只持久化对应 transport 字段

#### env 文本格式（`.env` 风格）

- 一行一对 `KEY=VALUE`
- 第一个 `=` 为分隔符，value 中可含 `=`（如 `URL=https://x.com/?a=b`）
- value 首尾去空格；以 `"` 或 `'` 包裹时脱壳并保留内部空白
- `#` 开头或纯空白行跳过（支持注释）
- KEY 必须匹配 `[A-Za-z_][A-Za-z0-9_]*`，否则在字段下方红字显示错误行号
- antd 控件：`Input.TextArea autoSize={{ minRows: 1, maxRows: 8 }}`

#### 敏感字段

env 与未来可能的 HTTP 鉴权字段（API key、token）需要：

- 列表页 / 编辑页**默认遮蔽**（`••••`），点眼睛图标切换显式
- 日志侧确保不打印 env / token（`MCPConnection` 现有 log 已 OK，需在 panel 侧 review）

与 `LLMSettingsPanel.auth_token` 处理对齐。

> 文件权限收紧（chmod 0600）暂不在本特性范围，留待后续统一处理（含 LLM Settings 写盘路径）。

## 运行时同步策略

### conn 实例所有权

`MCPSettingsPanel` 持有：

```python
self._conns: dict[str, MCPConnection]   # dict key = source 原始名
```

**conn 来源统一**：sandbox 暴露 `mcp_connections() -> dict[str, MCPConnection]`，panel 首次 `on_open()` 直接拿这个 dict 填 `self._conns`，**不去钻 `_registry` 私有结构**。

**`autostart=false` 的 source 也创建 conn 实例**（决策 D3，避免 panel 反查时漏掉这类 source）：`main_impl.connect_sources` 对 `autostart=false` 仍走 `MCPConnection(...)` 实例化，但**不调用 `sandbox.add_namespace`、不发起后台连接**——实例只挂在 `sandbox.mcp_connections()` 返回的 dict 里待命。点 Connect 时 panel 调 `sandbox.add_namespace(conn.namespace, on_remove=conn.close)` + `await conn.reconnect()`。

新增 source 时 panel 用最新 config 创建 `MCPConnection` 并塞进 dict；删除 / rename 时摘 namespace（触发 close）+ 从 dict 中删除。

> 启动期 `main_impl.connect_sources` 仍负责 `autostart=true` 的批量初始化；panel 只是「接管已存在的 conn 实例 + 新增/删除」。两边不耦合。

### Save 后的运行时同步策略

明确选择 **C 方案：保存只持久化 config，不动运行时**：

- Save Provider 把表单写回 `config.mcp_sources` 并落盘
- 如果该 source 当前处于 `connected/failed/connecting`，且表单字段（command/args/url/env/timeout/transport/shell）实际有变更，编辑页头部显示
  `⚠ 配置已修改，需要 Disconnect → Connect 才能生效` 横幅
- `autostart` / `retry_cooldown` 修改不显示横幅，仅下次启动 / 下次失败生效（运行中的 conn 保留旧 cooldown 值，不影响）
- 用户主动 Disconnect → 旧 conn 释放；下次 Connect 时 panel 用最新 config 新建 conn

> 选 C 而不是「Save 时自动重连」，是因为后者会在用户编辑过程中产生不可预期的中断，且编辑期不一定想立即应用（可能还要继续改其他 source）。

### Rename / Delete 语义

- **Rename**（修改 Name 字段）：等同于「删旧 + 新增」。如果旧 source 处于
  `connected/failed/connecting`，Save 时弹确认框「重命名会断开当前连接，是否继续？」；确认后摘旧 namespace、删旧 dict key、新建 conn（按新名 + autostart 决定是否自动连）
- **Delete**（Remove 按钮）：摘 namespace（触发 close）、删 dict key、删 panel 内 conn 引用、落盘

### 取消「connecting 中」的语义

`_do_rebuild` 持有 `MCPConnection._lock`，本设计**不引入主动 cancel**：

- `connecting` 状态下，列表页的 Connect 按钮置灰 + spinner，编辑页的 Connect 也置灰
- Disconnect 在 connecting 时**也置灰**（避免用户点了之后串行等待，体验不可预期）
- 如果连接卡住（>30s 无响应），靠底层 client 自身超时落到 `failed` 状态后再让用户重试

> 真正的 cancel（`asyncio.Task.cancel()`）留作后续优化，需要先在 MCPConnection 层加协作式取消支持。

### 异步执行

Connect / Reconnect / Reload tools 通过 `asyncio.run_coroutine_threadsafe(...,
sandbox.main_loop)` 提交到 sandbox 所在 loop。点击瞬间设置 `pending_button` 字段
触发 mutgui 重渲染（按钮置灰 + spinner），future 完成后再次触发重渲染（清 pending、
刷新状态 tag 与 tools 列表）。

## 配置持久化

与 LLM Settings 一致：

```python
def _write_config(self, mcp_sources: dict) -> None:
    self._agent.config.set("mcp_sources", mcp_sources, source="webui")
    path = Path("~/.mutagent/config.json").expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged_config, indent=2, ensure_ascii=False))
```

保存配置 ≠ 触发连接（见上）。

## 按钮 → 已有 API 映射

不新增 `MCPConnection.connect()` / `disconnect()` 方法。所有按钮直接复用现有 API：

| 按钮 | 实现 | 状态变化 |
|------|------|---------|
| Connect | `await conn.reconnect()`（panel 在 conn 不存在时按 config 新建） | `disconnected` → `connecting` → `connected` / `failed` |
| Disconnect | `await sandbox.remove_namespace(conn.namespace)` 触发 `on_remove=conn.close` | 任意 → `disconnected`，namespace 出 registry，conn 实例释放 |
| Reconnect | `await conn.reconnect()` | `failed/connected` → `connecting` → `connected` / `failed` |
| Reload tools | `await conn.reconnect()` | `connected` → `connecting` → `connected`（典型情况） |

> `MCPConnection.reconnect()` 现状已经做「直接 `_do_rebuild`，不看当前状态、绕过 cooldown」，正好满足显式按钮语义。

## 需要补的代码改动清单

为了让本设计能落地，除 `_settings_mcp.py` 自身外还需：

1. **`MCPConnection.list_tools_metadata()`** — 公开方法，封装当前 `namespace._functions` 与 tool input_schema 的拼装
2. **`StdioMCPClient` 透传 `env`** — 当前 `Popen` 没接 env 参数，配置字段表里的 `env` 实际未生效
3. **`SandboxApp.remove_namespace()` 的 await 语义** — 确认它能在异步路径上调用并等待 `on_remove` 回调完成（如果当前是同步移除 + 后台 close，需要补 await）
4. **panel 的状态反查入口** — `sandbox` 上加 `mcp_connections() -> dict[str, MCPConnection]`，避免 panel 钻 `_registry` 私有结构

## 实施步骤清单

按依赖顺序：底层先（adapter / sandbox 公开 API）→ 启动期接入 → panel 实现 → 测试。每步在新 session 开工时按需读源码再决定具体改法。

### 底层基础设施

- [x] `_adapter_mcp.py`：`StdioMCPClient` 接收并透传 `env` 到 `Popen`（覆盖两个 shell/非 shell 分支）；`make_client` 把 `env` 从 server_config 取出转发
- [x] `_adapter_mcp.py`：`MCPConnection` 新增 `list_tools_metadata()` 公开方法，返回 `[{name, description, input_schema, source_namespace}, ...]`，含 peer namespaces 中的 tools
- [x] `sandbox/app.py`（及 `_app_impl.py`）：新增 `mcp_connections() -> dict[str, MCPConnection]` + `register_mcp_connection(name, conn)` + `unregister_mcp_connection(name)`。`remove_namespace` 的异步语义 — `_schedule_cleanup_sync` 已在现有 loop 上以 `create_task` 调度 cleanup；panel 按钮路径额外 `await conn.close()` 补上同步等待，无需改 SandboxApp 接口。

### 启动期接入

- [x] `main_impl.connect_sources`：为所有 source（含 `autostart=false`）创建 `MCPConnection` 实例并 `register_mcp_connection`；autostart=true 才 `add_namespace` + bg 连，autostart=false 仅挂 conn dict 供 panel 反查

### MCPSettingsPanel 实现

- [x] 新增 `webui/_settings_mcp.py`：`MCPSettingsPanel(SettingsPanel)`，`panel_id="mcp"` / `panel_title="MCP 连接设置"` / `panel_placement="settings:10/20"`；状态字段、`render()`、`on_open()` 反查 conn dict
- [x] 列表页：每行按状态渲染单按钮（Connect / Disconnect / Reconnect），异步执行 + `pending_button` 置灰 + 完成后刷新
- [x] 编辑页：transport 切换、env 文本解析、表单校验（KEY 命名 + sanitized 名冲突）、`Connection` 区按钮组、`Tools` 折叠列表、`Peer namespaces` 区
- [x] Save 流程：写 `config.mcp_sources` 落盘 + 「配置已修改需要重连」横幅判断 + Rename 处理（旧 conn 释放 + dict key 调整）
- [x] `webui/settings.py` 末尾追加 `from . import _settings_mcp  # noqa`（并在 `webui/__init__.py` 同步加载，与 LLM panel 一致）

### 测试

- [x] `tests/test_adapter_mcp.py` 补充：`make_client` 转发 env、`StdioMCPClient.connect` Popen 收到合并后 env（并保留父进程 PATH）、缺省时 env=None、`list_tools_metadata` 返回结构与空列表退化
- [x] `tests/test_mcp_settings_panel.py`（8 个分类、共43 个 case）：env 文本解析全分支、args 解析、draft↔config、加载跳转、sanitized 名冲突、按钮状态映射、运行期变更横幅、Save / Delete / Rename 流程、stdio/http 表单渲染、autostart=false conn 反查
- [x] 手动验证：mutagent webui 实跑，加 stdio + http 两类 source，验证 Connect/Disconnect/Reconnect/Reload tools 全链路

## 变更记录

### 2026-05-12: Functions 浏览器 + UI 细节打磨

与初始设计相比的变更：

**Functions 浏览器（替代 Tools 列表渲染）**
- 三级逐级展开：namespace 列表 → 函数列表 → 函数详情
- 同名 namespace（主 ns + peer namespaces）按 name 分组聚合，合并 `_functions` + `_descriptions`，过滤 0-function 组
- 展开详情含签名（从 `_mcp_input_schema` 构建）+ 描述 + 参数表
- 展开箭头在 name 左侧，详情 block 为点击行的平级 sibling（点击不冒泡触发折叠）
- 不再独立展示 Peer namespaces 区，全部融合到 Functions 浏览器
- Functions 标题行：左 "Functions (N)" 文本，右 Reload 按钮（仅 connected 状态显示）

**List 页**
- 函数计数聚合同名 namespace（主 ns + peer namespaces），显示 "N functions" 而非 "N tools"

**Disconnect 语义**
- 只 `await conn.close()`，不调 `sandbox.remove_provider(conn.namespace)`
- Namespace 留在 registry 中 state=disconnected，re-connect 时复用
- Delete / Rename 仍调 `remove_provider` 永久摘除

**UI 细节**
- 编辑页 Form 从 `layout="vertical"` 改为 `"horizontal"` + `labelAlign="left"`
- 删除 `antd.Divider`，全部替换为间距 + `borderTop` 细线
- Connection 分类移除，操作栏 flex space-between：左 Remove，右 Connect/Disconnect/Reconnect + Save
- 修复 banners 误报：`_config_changed_at_runtime` 加归一化管道

**其他修复**
- `_server_impl._on_startup`：`connect_sources` 后 `sandbox.bind_main_loop()` 注入 `_async_loop`（修复 Connect/Disconnect 按钮报 "Sandbox event loop not available"）
- `bugfix-pysandbox-duplicate-signatures.md`：记录 pysandbox 包装函数签名重复的根因与修复方向（该 bug 为现有问题，不在本次修复范围）

### 完成态

- [x] 所有 checkbox 完成后向用户确认 → 改 status 为 ✅ 已完成
