"""mutagent.core._provider_impl_anthropic -- Anthropic Claude API provider."""

import logging
from typing import AsyncGenerator, AsyncIterator, ClassVar, cast

import httpx

from mutio.codec.json import (
    JSONDecodeError,
    JsonObject,
    JsonValue,
    dumps,
    get_field,
    loads,
    narrow_value,
)
from mutio.net.client import HttpClient
from .messages import (
    ContentBlock,
    DocumentBlock,
    ImageBlock,
    Message,
    Response,
    StreamEvent,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolSchema,
    ToolUseBlock,
    Usage,
)
from .llm import LLMApiClient
from ._llm_impl import get_default_context_window

logger = logging.getLogger(__name__)


class AnthropicApiClient(LLMApiClient):
    """Anthropic Claude API provider。

    Attributes:
        base_url: API 基础 URL（如 "https://api.anthropic.com"）。
        api_key: Anthropic API key。
    """

    api_type: ClassVar[str] = "Anthropic"

    base_url: str
    api_key: str

    def __init__(self, spec: JsonObject):
        auth_token = get_field(spec, "auth_token", str, default="")
        if not auth_token:
            raise ValueError("AnthropicProvider requires 'auth_token' in model spec.")
        model_id = get_field(spec, "model_id", str, default="")
        super().__init__(
            model_id=model_id,
            context_window=get_default_context_window(model_id),
            base_url=get_field(spec, "base_url", str, default="https://api.anthropic.com"),
            api_key=auth_token,
        )

    async def send(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        prompts: list[Message] | None = None,
        stream: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Send messages to Claude API and yield streaming events."""
        claude_messages = _messages_to_claude(messages)
        payload: JsonObject = {
            "model": self.model_id,
            "messages": claude_messages,
            "max_tokens": 4096,
        }
        if prompts:
            payload["system"] = _prompts_to_claude(prompts)
        if tools:
            payload["tools"] = _tools_to_claude(tools)

        headers = {
            "authorization": f"Bearer {self.api_key}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        if stream:
            async for event in _send_stream(self.base_url, payload, headers):
                yield event
        else:
            async for event in _send_no_stream(self.base_url, payload, headers):
                yield event


# ---------------------------------------------------------------------------
# Message → Claude API 转换
# ---------------------------------------------------------------------------

def _block_to_claude(block: ContentBlock) -> JsonObject | None:
    """将单个 ContentBlock 转换为 Claude API content block。未知类型返回 None。"""
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text} if block.text else None
    if isinstance(block, ImageBlock):
        if block.data:
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": block.media_type, "data": block.data},
            }
        if block.url:
            return {
                "type": "image",
                "source": {"type": "url", "url": block.url},
            }
        return None
    if isinstance(block, DocumentBlock):
        return {
            "type": "document",
            "source": {"type": "base64", "media_type": block.media_type, "data": block.data},
        }
    if isinstance(block, ThinkingBlock):
        if block.data:
            # redacted thinking — 原样回传
            return {"type": "redacted_thinking", "data": block.data}
        if block.thinking:
            return {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
        return None
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if isinstance(block, ToolResultBlock):
        result: JsonObject = {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
        }
        if block.is_error:
            result["is_error"] = True
        return result
    # 未知类型跳过
    return None


def _messages_to_claude(messages: list[Message]) -> list[JsonObject]:
    """Convert internal Message list to Claude API messages format.

    处理 blocks 模型：
    - assistant 消息中的 ToolUseBlock → tool_use content
    - user 消息中的 ToolResultBlock → tool_result content
    - 保证 user/assistant 严格交替（通过 merge）
    """
    result: list[JsonObject] = []
    for msg in messages:
        if msg.role == "assistant":
            assistant_content: list[JsonObject] = []
            for block in msg.blocks:
                api_block = _block_to_claude(block)
                if api_block:
                    assistant_content.append(api_block)

            if assistant_content:
                result.append({"role": "assistant", "content": assistant_content})
        else:
            # user / system 消息
            content: list[JsonObject] = []
            for block in msg.blocks:
                api_block = _block_to_claude(block)
                if api_block:
                    content.append(api_block)
            if content:
                if len(content) == 1 and content[0].get("type") == "text":
                    result.append({"role": msg.role, "content": content[0]["text"]})
                else:
                    result.append({"role": msg.role, "content": content})

    return _merge_consecutive_roles(result)


def _prompts_to_claude(prompts: list[Message]) -> list[JsonObject]:
    """将 prompt Messages 转换为 Claude API system 字段的 content block 数组。"""
    system_blocks: list[JsonObject] = []
    for msg in prompts:
        for block in msg.blocks:
            if isinstance(block, TextBlock) and block.text:
                entry: JsonObject = {"type": "text", "text": block.text}
                if msg.cacheable:
                    entry["cache_control"] = {"type": "ephemeral"}
                system_blocks.append(entry)
    return system_blocks


def _merge_consecutive_roles(messages: list[JsonObject]) -> list[JsonObject]:
    """Merge consecutive messages with the same role into one."""
    if not messages:
        return messages
    merged: list[JsonObject] = [messages[0]]
    for msg in messages[1:]:
        if msg["role"] == merged[-1]["role"]:
            prev = merged[-1]
            prev_content = _to_content_blocks(prev["content"])
            cur_content = _to_content_blocks(msg["content"])
            prev["content"] = prev_content + cur_content
        else:
            merged.append(msg)
    return merged


def _to_content_blocks(content: JsonValue) -> list[JsonObject]:
    """Normalize message content to a list of content blocks."""
    if isinstance(content, list):
        # list[JsonValue] → list[JsonObject]：已知 content block 都是 dict
        return cast(list[JsonObject], content)
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return []


def _tools_to_claude(tools: list[ToolSchema]) -> list[JsonObject]:
    """Convert internal ToolSchema list to Claude API tools format."""
    result: list[JsonObject] = []
    for tool in tools:
        entry: JsonObject = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema or {"type": "object", "properties": {}},
        }
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Claude API → 内部模型转换
# ---------------------------------------------------------------------------

def _normalize_cache_creation(value: JsonValue) -> int:
    """Normalize cache_creation_input_tokens from Anthropic (may be plain int or nested dict)."""
    if isinstance(value, dict):
        # isinstance 窄化后 value 为 dict[str, JsonValue] = JsonObject
        return get_field(value, "input_tokens", int, default=0)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _response_from_claude(data: JsonObject) -> Response:
    """Convert Claude API response to internal Response."""
    stop_reason = get_field(data, "stop_reason", str, default="")
    raw_usage = get_field(data, "usage", JsonObject, default={})
    usage = Usage(
        input_tokens=get_field(raw_usage, "input_tokens", int, default=0),
        output_tokens=get_field(raw_usage, "output_tokens", int, default=0),
        cache_read_input_tokens=get_field(raw_usage, "cache_read_input_tokens", int, default=0),
        cache_creation_input_tokens=_normalize_cache_creation(raw_usage.get("cache_creation_input_tokens", 0)),
    )

    blocks: list[ContentBlock] = []
    content = get_field(data, "content", list[JsonObject], default=[])
    for block_data in content:
        block_type = get_field(block_data, "type", str, default="")
        if block_type == "text":
            blocks.append(TextBlock(text=get_field(block_data, "text", str, default="")))
        elif block_type == "tool_use":
            blocks.append(ToolUseBlock(
                id=get_field(block_data, "id", str, default=""),
                name=get_field(block_data, "name", str, default=""),
                input=get_field(block_data, "input", JsonObject, default={}),
            ))
        elif block_type == "thinking":
            blocks.append(ThinkingBlock(
                thinking=get_field(block_data, "thinking", str, default=""),
                signature=get_field(block_data, "signature", str, default=""),
            ))
        elif block_type == "redacted_thinking":
            blocks.append(ThinkingBlock(
                data=get_field(block_data, "data", str, default=""),
            ))

    message = Message(role="assistant", blocks=blocks)
    return Response(message=message, stop_reason=stop_reason, usage=usage)


# ---------------------------------------------------------------------------
# HTTP 发送
# ---------------------------------------------------------------------------

def _extract_error_msg(data: JsonValue, status_code: int, prefix: str) -> str:
    """从 API 错误响应 JSON 中提取错误消息。"""
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            msg = error.get("message")
            if isinstance(msg, str):
                return f"{prefix} ({status_code}): {msg}"
    return f"{prefix} ({status_code}): {dumps(data)}"


async def _send_no_stream(
    base_url: str,
    payload: JsonObject,
    headers: dict[str, str],
) -> AsyncIterator[StreamEvent]:
    """Non-streaming path: make a regular HTTP request and wrap as StreamEvents."""
    async with HttpClient.create(timeout=httpx.Timeout(None, connect=10)) as client:
        resp = await client.post(
            f"{base_url}/v1/messages",
            headers=headers,
            json=payload,
        )
    data_raw: JsonValue = resp.json()
    if resp.status_code != 200:
        error_msg = _extract_error_msg(data_raw, resp.status_code, "Claude API error")
        logger.warning("API error (%d): %s", resp.status_code, error_msg)
        yield StreamEvent(
            type="error",
            error=error_msg,
        )
        return

    if not isinstance(data_raw, dict):
        yield StreamEvent(type="error", error="Claude API response is not a JSON object")
        return
    response = _response_from_claude(data_raw)

    # Emit text deltas
    for block in response.message.blocks:
        if isinstance(block, TextBlock) and block.text:
            yield StreamEvent(type="text_delta", text=block.text)
        elif isinstance(block, ToolUseBlock):
            yield StreamEvent(type="tool_use_start", tool_call=block)
            yield StreamEvent(type="tool_use_end")

    yield StreamEvent(type="response_done", response=response)


async def _send_stream(
    base_url: str,
    payload: JsonObject,
    headers: dict[str, str],
) -> AsyncIterator[StreamEvent]:
    """Streaming path: parse SSE events from Claude API and yield StreamEvents."""
    payload["stream"] = True

    async with HttpClient.create(timeout=httpx.Timeout(None, connect=10)) as client:
        async with client.stream(
            "POST",
            f"{base_url}/v1/messages",
            headers=headers,
            json=payload,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                try:
                    error_data = loads(body)
                    error_msg = _extract_error_msg(error_data, resp.status_code, "Claude API error")
                except Exception:
                    error_msg = f"Claude API error ({resp.status_code}): HTTP {resp.status_code}"
                logger.warning("API stream error (%d): %s", resp.status_code, error_msg)
                yield StreamEvent(
                    type="error",
                    error=error_msg,
                )
                return

            # Accumulate blocks for final Response
            blocks: list[ContentBlock] = []
            stop_reason = ""
            usage = Usage()

            current_block_type: str = ""
            current_tool_id: str = ""
            current_tool_name: str = ""
            current_tool_json_parts: list[str] = []
            current_text_parts: list[str] = []
            current_thinking_parts: list[str] = []
            current_thinking_signature: str = ""
            current_redacted_data: str = ""

            event_type = ""
            async for raw_line in resp.aiter_lines():
                line = raw_line

                if line.startswith("event: "):
                    event_type = line[7:]
                    continue

                if line.startswith("data: "):
                    data_str = line[6:]
                    if not event_type:
                        continue

                    try:
                        data_raw = loads(data_str)
                    except JSONDecodeError:
                        continue

                    # SSE 事件体始终是 JSON object
                    if not isinstance(data_raw, dict):
                        continue
                    data: JsonObject = data_raw

                    try:
                        if event_type == "message_start":
                            msg_data = get_field(data, "message", JsonObject, default={})
                            raw_usage = get_field(msg_data, "usage", JsonObject, default={})
                            usage.input_tokens = max(usage.input_tokens, get_field(raw_usage, "input_tokens", int, default=0))
                            usage.output_tokens = max(usage.output_tokens, get_field(raw_usage, "output_tokens", int, default=0))
                            usage.cache_read_input_tokens = max(usage.cache_read_input_tokens, get_field(raw_usage, "cache_read_input_tokens", int, default=0))
                            usage.cache_creation_input_tokens = max(
                                usage.cache_creation_input_tokens,
                                _normalize_cache_creation(raw_usage.get("cache_creation_input_tokens", 0)),
                            )

                        elif event_type == "content_block_start":
                            block = get_field(data, "content_block", JsonObject, default={})
                            current_block_type = get_field(block, "type", str, default="")
                            if current_block_type == "tool_use":
                                current_tool_id = get_field(block, "id", str, default="")
                                current_tool_name = get_field(block, "name", str, default="")
                                current_tool_json_parts = []
                                tc = ToolUseBlock(
                                    id=current_tool_id,
                                    name=current_tool_name,
                                )
                                yield StreamEvent(
                                    type="tool_use_start", tool_call=tc
                                )
                            elif current_block_type == "thinking":
                                current_thinking_parts = []
                                current_thinking_signature = ""
                            elif current_block_type == "redacted_thinking":
                                current_redacted_data = ""
                            elif current_block_type == "text":
                                current_text_parts = []

                        elif event_type == "content_block_delta":
                            delta = get_field(data, "delta", JsonObject, default={})
                            delta_type = get_field(delta, "type", str, default="")
                            if delta_type == "text_delta":
                                text = get_field(delta, "text", str, default="")
                                if text:
                                    current_text_parts.append(text)
                                    yield StreamEvent(
                                        type="text_delta", text=text
                                    )
                            elif delta_type == "input_json_delta":
                                json_chunk = get_field(delta, "partial_json", str, default="")
                                if json_chunk:
                                    current_tool_json_parts.append(json_chunk)
                                    yield StreamEvent(
                                        type="tool_use_delta",
                                        tool_json_delta=json_chunk,
                                    )
                            elif delta_type == "thinking_delta":
                                thinking_text = get_field(delta, "thinking", str, default="")
                                if thinking_text:
                                    current_thinking_parts.append(thinking_text)
                            elif delta_type == "signature_delta":
                                current_thinking_signature += get_field(delta, "signature", str, default="")

                        elif event_type == "content_block_stop":
                            if current_block_type == "tool_use":
                                json_str = "".join(current_tool_json_parts)
                                try:
                                    tool_args = narrow_value(loads(json_str), JsonObject) if json_str else JsonObject()
                                except (JSONDecodeError, TypeError):
                                    tool_args = JsonObject()
                                blocks.append(ToolUseBlock(
                                    id=current_tool_id,
                                    name=current_tool_name,
                                    input=tool_args,
                                ))
                                yield StreamEvent(type="tool_use_end")
                            elif current_block_type == "text":
                                text = "".join(current_text_parts)
                                if text:
                                    blocks.append(TextBlock(text=text))
                            elif current_block_type == "thinking":
                                blocks.append(ThinkingBlock(
                                    thinking="".join(current_thinking_parts),
                                    signature=current_thinking_signature,
                                ))
                            elif current_block_type == "redacted_thinking":
                                blocks.append(ThinkingBlock(
                                    data=current_redacted_data,
                                ))
                            current_block_type = ""

                        elif event_type == "message_delta":
                            delta = get_field(data, "delta", JsonObject, default={})
                            stop_reason = get_field(delta, "stop_reason", str, default=stop_reason)
                            # 合并 usage：取每个字段的最大值。
                            raw_usage = get_field(data, "usage", JsonObject, default={})
                            for k, v in raw_usage.items():
                                if k == "input_tokens" and isinstance(v, (int, float)):
                                    usage.input_tokens = max(usage.input_tokens, int(v))
                                elif k == "output_tokens" and isinstance(v, (int, float)):
                                    usage.output_tokens = max(usage.output_tokens, int(v))
                                elif k == "cache_read_input_tokens" and isinstance(v, (int, float)):
                                    usage.cache_read_input_tokens = max(usage.cache_read_input_tokens, int(v))
                                elif k == "cache_creation_input_tokens":
                                    usage.cache_creation_input_tokens = max(
                                        usage.cache_creation_input_tokens,
                                        _normalize_cache_creation(v),
                                    )

                        elif event_type == "message_stop":
                            message = Message(role="assistant", blocks=blocks)
                            response = Response(
                                message=message,
                                stop_reason=stop_reason,
                                usage=usage,
                            )
                            yield StreamEvent(type="response_done", response=response)

                    except Exception as e:
                        yield StreamEvent(
                            type="error",
                            error=f"Error processing SSE event '{event_type}': {e}",
                        )

                    event_type = ""
                    continue

                if not line:
                    event_type = ""
