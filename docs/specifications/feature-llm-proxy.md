# LLM 代理端点迁入 mutagent 设计规范

**状态**：✅ 已完成
**日期**：2026-05-28
**类型**：功能设计

## 背景

mutbot 早期实现了一组 LLM 代理 HTTP 端点（`/llm/v1/messages`、`/llm/v1/chat/completions`、`/llm/v1/models`），
让外部客户端（IDE 插件、Claude Code、其他 agent 框架）通过 mutbot 统一接入已配置的多家 LLM provider，
并在 Anthropic / OpenAI 两种格式之间自动翻译。

`refactor-agent-strip.md`（mutbot）已明确将 `proxy/` 列为"未来归位 mutagent"。当前 mutbot 已不挂载这些路由，
proxy/ 在 mutbot 内是死代码。代理逻辑的本质——多 provider 路由、协议翻译、统一认证转发——与 mutbot 的
sandbox/terminal 定位完全无关，自然属于 mutagent 这一层。

mutagent 完成 LLMApiClient 重构、并接入 Copilot provider 之后，正是把 LLM 代理端点搬过来的时机。

## 需求

1. mutagent 提供一组 LLM 代理 HTTP 端点，挂载在 `/llm` 前缀下：
   - `GET /llm` 与 `GET /llm/`：人类可读的 API 说明页（HTML）
   - `GET /llm/v1/models`：列出当前配置中所有可用模型
   - `POST /llm/v1/messages`：Anthropic Messages 格式入口
   - `POST /llm/v1/chat/completions`：OpenAI Chat Completions 格式入口
2. 客户端格式与后端 provider 格式可不同时，代理需自动做 JSON 层翻译（双向）：
   - Anthropic 请求 → OpenAI 后端 → Anthropic 响应（含流式 SSE 转译）
   - 同格式直通时不引入额外解析开销
3. 代理基于 mutagent `Config.providers` 配置发现可用模型，配置变更（含 hot reload）后端点行为应跟随更新
4. 模型路由策略：
   - 支持 provider 配置中的 list / dict 两种 models 形式（dict 用于别名解决重名）
   - 支持模型名归一化（如剥离 Anthropic 日期后缀）
   - 找不到模型时返回 404 + 结构化错误信息
5. 流式响应使用 SSE，需正确转发 cache-control / connection headers，且保持端到端取消语义
6. 提供代理调用日志能力：
   - 记录每次代理调用的 model、format、duration、token 用量等元信息
   - 落到 JSONL 文件，便于离线分析外部客户端的 LLM 使用情况
   - 提供按日期读取与汇总的查询函数（供 CLI / 内省使用）
7. 代理端点是**可选挂载**而非默认开启：mutagent 提供 View 类与注册入口，由调用方（mutagent CLI server / mutbot / 其他宿主）按需 import 触发注册，不强加路由
8. 代理只复用 LLMApiClient 实例上的连接信息（base_url / 认证 headers），不强制走 LLMApiClient.send（代理工作在 JSON 层，避免双重解析与序列化开销）

## 关键参考

### mutbot 待迁出源文件
- `mutbot/src/mutbot/proxy/__init__.py` — 触发 View 子类注册的入口
- `mutbot/src/mutbot/proxy/routes.py` — View 子类 + provider 实例池 + 转发主流程（~488 行，依赖 mutio.net + mutagent provider + Copilot）
- `mutbot/src/mutbot/proxy/translation.py` — Anthropic ↔ OpenAI JSON 双向翻译（~390 行，纯函数，零依赖）
- `mutbot/src/mutbot/proxy/logging.py` — `ProxyLogger` + JSONL 读写 + 汇总（~125 行，零外部依赖）
- `mutbot/src/mutbot/cli/proxy_log.py` — 日志查询 CLI（mutbot 侧仅此一处仍依赖 proxy.logging，迁移后改 import 路径）

### mutagent 接入点
- `mutagent/src/mutagent/core/llm.py` — `LLMApiClient` 基类
- `mutagent/src/mutagent/core/_llm_impl_anthropic.py` / `_llm_impl_openai.py` — 后端 provider 实现，需要为代理暴露 base_url / headers
- `mutagent/src/mutagent/app/config.py` — `Config.list_models` / `Config.resolve_model` / `Config.on_change`，代理路由依赖这套配置 API
- `mutagent/src/mutagent/app/log_store.py` — mutagent 现有日志体系（评估代理日志是否复用此设施 vs 独立 JSONL）
- `mutio.net.server` — `View` / `Request` / `Response` / `StreamingResponse` 等基础类型（mutbot proxy 已基于此实现）

### 关联规范
- `mutagent/docs/specifications/feature-copilot-provider.md` — Copilot 必须先归位，代理才能完整覆盖三家 provider
- `mutbot/docs/specifications/refactor-agent-strip.md` — 明确写明 `proxy/` 回归 mutagent 的方向

## 消费者场景

| 消费者 | 场景 | 依赖的输出 | 验收标准 |
|--------|------|-----------|---------|
| mutbot | 删除 `mutbot/proxy/` 后，仍能以"挂载 mutagent 代理"方式继续提供 `/llm/*` 端点 | mutagent 暴露 import 即注册的 View 子类（或显式 `register_proxy(router)` 入口） | mutbot worker 启动后 `/llm/v1/models` 返回当前配置模型列表，已有外部 IDE 插件不感知后端搬迁 |
| Claude Code / 外部 IDE 客户端 | 通过 `/llm/v1/messages` 或 `/llm/v1/chat/completions` 调用任意 provider | 端点行为、SSE 字节序列与 mutbot 旧版一致 | 同一客户端配置在迁移前后表现等价（含流式取消、token 统计、错误透传） |
| mutagent 自身 server | 独立运行 mutagent server 时也能直接挂载代理 | 代理模块不绑定 mutbot 任何概念，仅依赖 Config + LLMApiClient 注册表 | 在仅安装 mutagent 的环境下能起 server 并通过 `/llm/*` 调到 Anthropic/OpenAI/Copilot |

## 设计方案

### 模块拆分

- 新增 `src/mutagent/llmproxy/` 包，拆成 `routes.py` / `translation.py` / `logging.py`
- `__init__.py` 仅负责导出配置入口并 import `routes`，保持"按需 import 才注册 View"的可选挂载语义
- `routes.py` 中维护一个轻量 `LLMProxyRuntime`，持有 `Config` 与 provider 实例缓存

### 配置与缓存策略

- 宿主在启动阶段显式调用 `configure_llm_proxy(config)` 绑定 `Config`
- 代理每次请求前读取 `Config.providers` 的序列化签名；签名变化时清空 provider cache，因此普通 `config.set()` 热更新即可生效，不需要宿主额外通知
- provider 实例仍按 provider 配置 + backend model 缓存，避免 Copilot 每次请求都重新创建设备流/JWT 状态

### 路由与后端适配

- 暴露四个端点：`/llm`、`/llm/`、`/llm/v1/models`、`/llm/v1/messages`、`/llm/v1/chat/completions`
- 模型解析支持 `models` 的 list / dict 两种形式，并额外允许用"去日期后缀"的 Anthropic 名称命中真实 backend model
- 后端格式通过 provider 类型推断：`AnthropicApiClient` 走 Anthropic 协议，`OpenAIApiClient` / `CopilotApiClient` 走 OpenAI 协议

### 翻译策略

- 复用 mutbot 旧实现思路，并补齐缺失的 `OpenAI -> Anthropic` 请求/响应/SSE 反向翻译，使两类客户端都能访问任意后端 provider
- 同格式请求保持 JSON 透传，只替换 backend `model`
- OpenAI 流式请求统一补 `stream_options.include_usage = true`，确保代理日志可拿到 token usage

### 日志策略

- 代理日志独立落到 `.mutagent/logs/proxy/YYYY-MM-DD.jsonl`
- 每条记录包含 `client_format` / `backend_format` / `model` / `backend_model` / `provider` / `translated` / `usage` / `duration_ms`
- 同时提供 `read_log_file()` / `get_summary()` 供后续 CLI 或内省复用

### 错误语义

- 未配置 runtime 返回 503
- 缺少 `model` 返回 400
- 模型不存在返回 404
- 错误响应按客户端格式分别包装成 Anthropic / OpenAI 风格结构，避免客户端因为协议不符而再次失败

## 实施步骤清单

- [x] 在 `src/mutagent/llmproxy/` 中新增可选挂载入口、代理 runtime、HTTP Views 与 JSONL 日志模块
- [x] 迁移并补齐 Anthropic / OpenAI 双向请求、响应与 SSE 翻译 helper
- [x] 基于 `Config.providers` 实现模型解析、模型名归一化命中与 provider cache 热更新
- [x] 保持 Copilot / OpenAI / Anthropic 三类 provider 都能暴露 base URL 与认证头给代理层复用
- [x] 补充 llmproxy 测试，覆盖翻译、runtime 缓存失效与日志汇总
- [x] 运行 `pytest`、`pyright`、`mutobj-lint` 验证 mutagent 改动

## 测试验证

- `pytest`
- `pyright`
- `mutobj-lint`
