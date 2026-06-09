"""mutagent message models for LLM communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from mutio.codec.json import JsonObject


# ---------------------------------------------------------------------------
# ContentBlock 类型体系
# ---------------------------------------------------------------------------

@dataclass
class ContentBlock:
    """内容块基类。"""

    type: str


@dataclass
class TextBlock(ContentBlock):
    """文本内容块。"""

    type: str = "text"
    text: str = ""


@dataclass
class ImageBlock(ContentBlock):
    """图像内容块。data 与 url 二选一。"""

    type: str = "image"
    data: str = ""              # base64
    media_type: str = ""
    url: str = ""


@dataclass
class DocumentBlock(ContentBlock):
    """文档内容块（PDF 等）。"""

    type: str = "document"
    data: str = ""              # base64
    media_type: str = ""        # "application/pdf"


@dataclass
class ThinkingBlock(ContentBlock):
    """Thinking 内容块。

    thinking 非空 = 可见推理过程。
    data 非空 = 被 Anthropic 安全系统屏蔽的加密数据（内容不可读，后续轮次原样回传）。
    signature = Anthropic 加密签名，验证 thinking 未被篡改。
    Provider 负责映射回 API 的 thinking / redacted_thinking type。
    """

    type: str = "thinking"
    thinking: str = ""
    signature: str = ""
    data: str = ""


@dataclass
class ToolUseBlock(ContentBlock):
    """工具调用请求块。"""

    type: str = "tool_use"
    id: str = ""                # 工具调用标识（LLM 生成）
    name: str = ""
    input: JsonObject = field(default_factory=JsonObject)


@dataclass
class ToolResultBlock(ContentBlock):
    """工具执行结果块。"""

    type: str = "tool_result"
    tool_use_id: str = ""
    tool_name: str = ""
    content: str = ""
    is_error: bool = False
    duration: float = 0.0


@dataclass
class TurnStartBlock(ContentBlock):
    """Turn 开始标记。输入 Message 含此 block 时触发 agent 处理。"""

    type: str = "turn_start"
    turn_id: str = ""


@dataclass
class TurnEndBlock(ContentBlock):
    """Turn 结束标记。agent 在最后一条 assistant Message 末尾追加。"""

    type: str = "turn_end"
    turn_id: str = ""
    duration: float = 0.0       # 整轮耗时（秒）


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """对话消息。

    role: "user" | "assistant" | "system"
    blocks: 内容块列表，唯一内容容器。
    """

    role: str
    blocks: list[ContentBlock] = field(default_factory=list[ContentBlock])

    # --- 标识 ---
    id: str = ""                # 消息标识（空 = 未分配，应用层生成）
    label: str = ""             # 段标识（prompt: "base"/"memory"，对话消息通常为空）
    sender: str = ""            # 创建者身份
    model: str = ""             # AI 模型标识

    # --- 事实性元数据 ---
    timestamp: float = 0.0
    duration: float = 0.0       # 生成耗时（秒）
    input_tokens: int = 0
    output_tokens: int = 0

    # --- Provider 提示 ---
    cacheable: bool = True
    priority: int = 0           # Prompt 排序优先级（值越大越靠前）


# ---------------------------------------------------------------------------
# ToolSchema / Response / StreamEvent / Content
# ---------------------------------------------------------------------------

@dataclass
class ToolSchema:
    """JSON Schema description of a tool for the LLM."""

    name: str
    description: str
    input_schema: JsonObject = field(default_factory=JsonObject)


@dataclass
class Usage:
    """Normalized LLM token usage."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class Response:
    """LLM response wrapper."""

    message: Message
    stop_reason: str = ""
    usage: Usage = field(default_factory=Usage)


@dataclass
class StreamEvent:
    """A single event in a streaming LLM response.

    Event types:
        "response_start"  - LLM 调用开始，携带预生成的 Message 元数据(id/model/timestamp)
        "text_delta"      - 增量文本
        "tool_use_start"  - LLM 开始构造工具调用
        "tool_use_delta"  - 增量 JSON（工具参数）
        "tool_use_end"    - LLM 完成工具调用块
        "tool_exec_start" - Agent 开始执行工具（tool_call = ToolUseBlock）
        "tool_exec_end"   - Agent 完成执行工具（tool_call = ToolResultBlock）
        "response_done"   - 一次 LLM 调用完成，携带 Response
        "turn_done"       - Agent 完成处理一条用户消息
        "error"           - 错误
    """

    type: str
    text: str = ""
    tool_call: Optional[ToolUseBlock | ToolResultBlock] = None
    tool_json_delta: str = ""
    response: Optional[Response] = None
    turn_id: str = ""
    error: str = ""
    timestamp: float = 0.0

