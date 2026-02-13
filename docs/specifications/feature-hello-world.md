# mutagent 首次真实启动 设计规范

**状态**：✅ 已完成
**日期**：2026-02-13
**类型**：功能设计

## 1. 背景

让 mutagent 可以通过 `python -m mutagent` 启动交互式会话，读取配置，使用合理的系统提示词让 Agent 理解自己的定位与使命，为自我迭代进化做准备。

## 2. 设计方案

### 2.1 `python -m mutagent` 入口

创建 `src/mutagent/__main__.py`，实现：
- 读取配置（见 2.2）
- 创建 Agent
- 进入交互循环：提示用户输入 → agent.run() → 打印结果 → 循环
- Ctrl+C / 空输入退出

### 2.2 配置读取

优先级：**环境变量 > mutagent.json**（环境变量覆盖文件配置）

**环境变量**：
| 变量名 | 对应参数 | 默认值 |
|---|---|---|
| `ANTHROPIC_AUTH_TOKEN` | `api_key` | 无（必须） |
| `ANTHROPIC_BASE_URL` | `base_url` | `https://api.anthropic.com` |
| `ANTHROPIC_MODEL` | `model` | `claude-sonnet-4-20250514` |

**mutagent.json**（同 Claude Code 格式）：
```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://...",
    "ANTHROPIC_AUTH_TOKEN": "...",
    "ANTHROPIC_MODEL": "..."
  }
}
```

`mutagent.json` 的 `env` 字段中的值作为环境变量的 fallback，不覆盖已有环境变量。

在 `main.py` 中新增 `load_config()` 函数，返回 `(api_key, model, base_url)`。

### 2.3 system prompt 传递修复

**当前问题**：`Agent.system_prompt` 在 `step()` 中没有传给 `send_message`，LLM 收不到系统提示。

**修复方案**：
- `LLMClient.send_message` 签名新增 `system_prompt: str = ""` 参数
- `claude.impl.py` 中将 `system_prompt` 放入 payload 的 `system` 字段
- `agent.impl.py` 的 `step()` 将 `self.system_prompt` 传给 `send_message`

### 2.4 系统提示词

系统提示词应让 Agent 理解：
- **身份**：mutagent — 可运行时自我迭代的 Python AI Agent
- **能力**：通过 5 个核心工具（inspect_module, view_source, patch_module, save_module, run_code）操作 Python 运行时
- **核心理念**：运行时优先、声明-实现分离、自举能力
- **工作流**：inspect → view_source → patch → run_code 验证 → save 固化

## 3. 已解决问题

无待定问题，需求明确。

## 4. 实施步骤清单

### 阶段一：实现 [✅ 已完成]

- [x] **Task 1.1**: 修复 system prompt 传递链
  - [x] `LLMClient.send_message` 新增 `system_prompt` 参数
  - [x] `claude.impl.py` 将 system_prompt 放入 payload `system` 字段
  - [x] `agent.impl.py` 的 `step()` 传递 `self.system_prompt`
  - 状态：✅ 已完成

- [x] **Task 1.2**: 配置读取
  - [x] `main.py` 新增 `load_config()` 函数
  - [x] 读取 mutagent.json 的 env 字段作为 fallback
  - [x] 读取环境变量覆盖
  - 状态：✅ 已完成

- [x] **Task 1.3**: 交互式入口
  - [x] 创建 `__main__.py`
  - [x] 实现输入循环
  - [x] 编写系统提示词
  - 状态：✅ 已完成

- [x] **Task 1.4**: 测试验证
  - [x] 已有单元测试通过 (161 passed, 2 skipped)
  - [x] 真实 API 验证：system prompt 生效 + 工具调用正常
  - 状态：✅ 已完成
