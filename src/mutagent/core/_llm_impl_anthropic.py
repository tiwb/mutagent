"""mutagent.core._provider_impl_anthropic -- Anthropic Claude API provider."""

import json
import logging
import time
from typing import Any, AsyncGenerator, AsyncIterator, ClassVar

import httpx

from mutio.net.client import HttpClient
from ._llm_impl import get_default_context_window
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
)
from .llm import LLMApiClient

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

    def __init__(self, spec: dict):
        if not spec.get("auth_token"):
            raise ValueError("AnthropicProvider requires 'auth_token' in model spec.")
        model_id = spec.get("model_id", "")
        super().__init__(
            model_id=model_id,
            context_window=get_default_context_window(model_id),
            base_url=spec.get("base_url", "https://api.anthropic.com"),
            api_key=spec["auth_token"],
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
        payload: dict[str, Any] = {
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

def _block_to_claude(block: ContentBlock) -> dict[str, Any] | None:
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
        result: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
        }
        if block.is_error:
            result["is_error"] = True
        return result
    # 未知类型跳过
    return None


def _messages_to_claude(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert internal Message list to Claude API messages format.

    处理 blocks 模型：
    - assistant 消息中的 ToolUseBlock → tool_use content
    - user 消息中的 ToolResultBlock → tool_result content
    - 保证 user/assistant 严格交替（通过 merge）
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "assistant":
            assistant_content: list[dict[str, Any]] = []
            for block in msg.blocks:
                api_block = _block_to_claude(block)
                if api_block:
                    assistant_content.append(api_block)

            if assistant_content:
                result.append({"role": "assistant", "content": assistant_content})
        else:
            # user / system 消息
            content: list[dict[str, Any]] = []
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


def _prompts_to_claude(prompts: list[Message]) -> list[dict[str, Any]]:
    """将 prompt Messages 转换为 Claude API system 字段的 content block 数组。"""
    system_blocks: list[dict[str, Any]] = []
    for msg in prompts:
        for block in msg.blocks:
            if isinstance(block, TextBlock) and block.text:
                entry: dict[str, Any] = {"type": "text", "text": block.text}
                if msg.cacheable:
                    entry["cache_control"] = {"type": "ephemeral"}
                system_blocks.append(entry)
    return system_blocks


def _merge_consecutive_roles(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive messages with the same role into one."""
    if not messages:
        return messages
    merged: list[dict[str, Any]] = [messages[0]]
    for msg in messages[1:]:
        if msg["role"] == merged[-1]["role"]:
            prev = merged[-1]
            prev_content = _to_content_blocks(prev["content"])
            cur_content = _to_content_blocks(msg["content"])
            prev["content"] = prev_content + cur_content
        else:
            merged.append(msg)
    return merged


def _to_content_blocks(content: Any) -> list[dict[str, Any]]:
    """Normalize message content to a list of content blocks."""
    if isinstance(content, list):
        return content
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return []


def _tools_to_claude(tools: list[ToolSchema]) -> list[dict[str, Any]]:
    """Convert internal ToolSchema list to Claude API tools format."""
    result = []
    for tool in tools:
        entry: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema or {"type": "object", "properties": {}},
        }
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Claude API → 内部模型转换
# ---------------------------------------------------------------------------

def _response_from_claude(data: dict[str, Any]) -> Response:
    """Convert Claude API response to internal Response."""
    stop_reason = data.get("stop_reason", "")
    usage = data.get("usage", {})

    blocks: list[ContentBlock] = []
    for block_data in data.get("content", []):
        block_type = block_data.get("type", "")
        if block_type == "text":
            blocks.append(TextBlock(text=block_data.get("text", "")))
        elif block_type == "tool_use":
            blocks.append(ToolUseBlock(
                id=block_data.get("id", ""),
                name=block_data.get("name", ""),
                input=block_data.get("input", {}),
            ))
        elif block_type == "thinking":
            blocks.append(ThinkingBlock(
                thinking=block_data.get("thinking", ""),
                signature=block_data.get("signature", ""),
            ))
        elif block_type == "redacted_thinking":
            blocks.append(ThinkingBlock(
                data=block_data.get("data", ""),
            ))

    message = Message(role="assistant", blocks=blocks)
    return Response(message=message, stop_reason=stop_reason, usage=usage)


def _response_to_dict(response: Response) -> dict[str, Any]:
    """Convert a Response object to a plain dict for recording."""
    content: list[dict[str, Any]] = []
    for block in response.message.blocks:
        if isinstance(block, TextBlock) and block.text:
            content.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolUseBlock):
            content.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
        elif isinstance(block, ThinkingBlock):
            if block.data:
                content.append({"type": "redacted_thinking", "data": block.data})
            elif block.thinking:
                content.append({
                    "type": "thinking",
                    "thinking": block.thinking,
                    "signature": block.signature,
                })
    return {
        "content": content,
        "stop_reason": response.stop_reason,
    }


# ---------------------------------------------------------------------------
# HTTP 发送
# ---------------------------------------------------------------------------

async def _send_no_stream(
    base_url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> AsyncIterator[StreamEvent]:
    """Non-streaming path: make a regular HTTP request and wrap as StreamEvents."""
    async with HttpClient.create(timeout=httpx.Timeout(None, connect=10)) as client:
        resp = await client.post(
            f"{base_url}/v1/messages",
            headers=headers,
            json=payload,
        )
    data = resp.json()
    if resp.status_code != 200:
        error_msg = data.get("error", {}).get("message", json.dumps(data))
        logger.warning("API error (%d): %s", resp.status_code, error_msg)
        yield StreamEvent(
            type="error",
            error=f"Claude API error ({resp.status_code}): {error_msg}",
        )
        return

    response = _response_from_claude(data)

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
    payload: dict[str, Any],
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
                    data = json.loads(body)
                    error_msg = data.get("error", {}).get("message", json.dumps(data))
                except Exception:
                    error_msg = f"HTTP {resp.status_code}"
                logger.warning("API stream error (%d): %s", resp.status_code, error_msg)
                yield StreamEvent(
                    type="error",
                    error=f"Claude API error ({resp.status_code}): {error_msg}",
                )
                return

            # Accumulate blocks for final Response
            blocks: list[ContentBlock] = []
            stop_reason = ""
            usage: dict[str, Any] = {}

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
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    try:
                        if event_type == "message_start":
                            msg_data = data.get("message", {})
                            usage.update(msg_data.get("usage", {}))

                        elif event_type == "content_block_start":
                            block = data.get("content_block", {})
                            current_block_type = block.get("type", "")
                            if current_block_type == "tool_use":
                                current_tool_id = block.get("id", "")
                                current_tool_name = block.get("name", "")
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
                            delta = data.get("delta", {})
                            delta_type = delta.get("type", "")
                            if delta_type == "text_delta":
                                text = delta.get("text", "")
                                if text:
                                    current_text_parts.append(text)
                                    yield StreamEvent(
                                        type="text_delta", text=text
                                    )
                            elif delta_type == "input_json_delta":
                                json_chunk = delta.get("partial_json", "")
                                if json_chunk:
                                    current_tool_json_parts.append(json_chunk)
                                    yield StreamEvent(
                                        type="tool_use_delta",
                                        tool_json_delta=json_chunk,
                                    )
                            elif delta_type == "thinking_delta":
                                thinking_text = delta.get("thinking", "")
                                if thinking_text:
                                    current_thinking_parts.append(thinking_text)
                            elif delta_type == "signature_delta":
                                current_thinking_signature += delta.get("signature", "")

                        elif event_type == "content_block_stop":
                            if current_block_type == "tool_use":
                                json_str = "".join(current_tool_json_parts)
                                try:
                                    arguments = json.loads(json_str) if json_str else {}
                                except json.JSONDecodeError:
                                    arguments = {}
                                blocks.append(ToolUseBlock(
                                    id=current_tool_id,
                                    name=current_tool_name,
                                    input=arguments,
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
                            delta = data.get("delta", {})
                            stop_reason = delta.get("stop_reason", stop_reason)
                            # 合并 usage：取每个字段的最大值。
                            # message_start 携带 input_tokens 等初始值，
                            # message_delta 携带最终 output_tokens，
                            # 但某些代理会在 message_delta 中附带 input_tokens=0，
                            # 用 max 避免错误覆盖。
                            for k, v in data.get("usage", {}).items():
                                if isinstance(v, (int, float)):
                                    usage[k] = max(usage.get(k, 0), v)
                                elif isinstance(v, dict):
                                    # 嵌套 usage（如 cache_creation）
                                    existing = usage.get(k, {})
                                    if isinstance(existing, dict):
                                        for sk, sv in v.items():
                                            if isinstance(sv, (int, float)):
                                                existing[sk] = max(existing.get(sk, 0), sv)
                                        usage[k] = existing
                                    else:
                                        usage[k] = v
                                else:
                                    usage[k] = v

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
