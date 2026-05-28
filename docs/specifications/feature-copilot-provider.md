# CopilotProvider 迁入 mutagent 设计规范

**状态**：✅ 已完成
**日期**：2026-05-28
**类型**：功能设计

## 背景

`mutbot/copilot/` 目录原本承载 GitHub Copilot LLM provider 实现。`refactor-agent-strip.md`（mutbot）已明确：
agent 相关功能（包括 copilot/）"未来归位 mutagent"。当前 mutbot 已切断 import 链，
copilot/ 模块在 mutbot 里实际是死代码。

mutagent 这边完成 LLMApiClient 重构后，正是把 Copilot 整体接入 mutagent 内置 provider 体系的时机。

## 需求

1. 在 mutagent 中新增 GitHub Copilot LLM provider，作为内置 LLMApiClient 子类
2. provider 通过短名 `"Copilot"` 在配置中引用（与 `"Anthropic"`、`"OpenAI"` 平级），自动 discover 注册
3. Copilot 协议本质是 OpenAI Chat Completions 格式 + 专用认证头，实现上应复用 mutagent 现有 OpenAI helper，避免代码重复
4. 提供 GitHub OAuth 设备流认证能力：
   - 用户首次未持有 token 时引导完成设备码授权，拿到 GitHub access token
   - GitHub access token 换取 Copilot JWT，JWT 过期时自动刷新
   - GitHub access token 通过 mutagent 配置（如 `github_token`）持久化；Copilot JWT 仅驻内存
5. 支持账户类型切换（individual / business / enterprise），对应不同 API base URL
6. 与 mutagent 现有 provider 在配置 schema、from_spec 行为、send 流式协议上保持一致

## 关键参考

### mutbot 待迁出源文件
- `mutbot/src/mutbot/copilot/auth.py` — `CopilotAuth` 单例，OAuth 设备流 + JWT 刷新（~205 行，仅依赖 httpx 和标准库）
- `mutbot/src/mutbot/copilot/provider.py` — `CopilotProvider`（旧 LLMProvider 子类，~93 行，复用 OpenAI helper）

### mutagent 接入点
- `src/mutagent/core/llm.py` — `LLMApiClient` 基类（新版 API：实例绑定 model_id，send 不再传 model）
- `src/mutagent/core/_llm_impl_openai.py` — OpenAI 实现，含 `_messages_to_openai` / `_tools_to_openai` / `_send_stream` / `_send_no_stream` 等可复用 helper
- `src/mutagent/core/_llm_impl.py::_get_provider_aliases` — 短名 alias 表，Copilot 需要在此被发现
- `src/mutagent/app/config.py` — Config Declaration（github_token 等持久化字段的载体）
- `src/mutagent/core/_llm_impl_copilot.py` — 新增 Copilot provider + auth backend（设备流、JWT 刷新、专用 headers、账户类型 base URL）
- `tests/core/test_copilot_provider.py` — provider alias / auth / send 行为测试

### mutbot 侧规划文档
- `mutbot/docs/specifications/refactor-agent-strip.md` — 明确写明 `copilot/` 回归 mutagent 的方向

### 外部协议参考
- VS Code Copilot Chat 客户端使用的 Client ID、headers 格式见 `mutbot/copilot/auth.py` 中 `GITHUB_CLIENT_ID` / `get_headers` 实现

## 消费者场景

| 消费者 | 场景 | 依赖的输出 | 验收标准 |
|--------|------|-----------|---------|
| mutbot | 删除自身 `copilot/` 目录后，已有用户配置（含 github_token）继续可用 | mutagent 提供 `"type": "Copilot"` provider 类型 + github_token 持久化字段语义 | 老用户的 `~/.mutbot/config.json` 中 Copilot 模型仍可成功对话，无需重新走设备流 |
| mutagent CLI / agent 自身 | 直接通过 mutagent 调用 Copilot 模型 | `Config.resolve_model` 解析 Copilot spec 后能创建 Provider 实例并 send | mutagent 单元测试覆盖 from_spec / 流式 send / JWT 刷新路径 |
| 后续 LLM proxy（独立 spec） | 在代理后端中支持 Copilot | provider 实例可向 proxy 暴露 base_url + 认证 headers | proxy 透传 Copilot 流式响应正常 |

## 设计方案

### provider 形态

- 在 `src/mutagent/core/_llm_impl_copilot.py` 新增 `CopilotApiClient(LLMApiClient)`，`api_type = "Copilot"`，与 `Anthropic` / `OpenAI` 同级 discover
- `send()` 保持 mutagent 新版签名（实例自带 `model_id`），请求体与流式解析直接复用 `src/mutagent/core/_llm_impl_openai.py` 的 helper
- `base_url` 按 `account_type` 解析：`individual` / `business` / `enterprise`；如显式提供 `base_url`，以配置覆盖默认值

### 认证生命周期

- GitHub access token 使用配置字段 `github_token`
- Copilot JWT 不落盘，只在 `CopilotAuth` 实例内缓存，并在过期前 5 分钟自动刷新
- 当 `github_token` 缺失时，backend 可在当前进程内执行 GitHub OAuth 设备流获取 token；framework 本身不回写 config 文件
- `github_token` 的持久化由外部流程负责（如未来 webui / setup wizard），不由 `mutagent.core` 或 `App.setup_agent()` 自动执行

### 兼容性策略

- 新配置推荐 `type: "Copilot"`
- provider 解析层兼容 `"CopilotProvider"` 与旧全路径 `mutbot.copilot.provider.CopilotProvider`（避免 mutbot 旧配置失效）
- provider 实例只消费配置，不承担配置持久化职责

### 非范围

- 本次不做 webui/settings 面板适配
- 本次不实现 Copilot 模型列表拉取或 setup wizard，仅保证 backend provider/auth 代码和 CLI/agent 路径可用

## 实施步骤清单

- [x] 在 `src/mutagent/core/_llm_impl_copilot.py` 新增 Copilot provider，实现设备流认证、JWT 刷新、账户类型 base URL 与 OpenAI helper 复用
- [x] 扩展 provider alias 解析，支持 `Copilot` 短名并兼容旧 `CopilotProvider` / 全路径配置
- [x] 保持 framework 仅读取 `github_token`，不自动回写配置文件
- [x] 补充 core 测试，覆盖 alias、设备流行为与 send 行为
- [x] 运行 mutagent 现有 pytest、pyright、mutobj-lint 验证改动

## 测试验证

- `pytest`
- `pyright`
- `mutobj-lint`
