"""mutagent.core._provider_impl_openai -- OpenAI Chat Completions API provider."""

import json
import logging
from typing import Any, AsyncGenerator, AsyncIterator, ClassVar

import httpx

from mutio.net.client import HttpClient
from ._llm_impl import get_default_context_window
from .messages import (
    ContentBlock,
    ImageBlock,
    Message,
    Response,
    StreamEvent,
    TextBlock,
    ToolResultBlock,
    ToolSchema,
    ToolUseBlock,
)
from .llm import LLMApiClient

logger = logging.getLogger(__name__)


class OpenAIApiClient(LLMApiClient):
    """OpenAI Chat Completions 格式 API provider。

    兼容所有使用 OpenAI Chat Completions 格式的 API（如 OpenAI、Groq 等）。

    Attributes:
        base_url: API 基础 URL（如 "https://api.openai.com/v1"）。
        api_key: API key。
    """

    api_type: ClassVar[str] = "OpenAI"

    base_url: str
    api_key: str

    def __init__(self, spec: dict):
        if not spec.get("auth_token"):
            raise ValueError("OpenAIProvider requires 'auth_token' in model spec.")
        model_id = spec.get("model_id", "")
        super().__init__(
            model_id=model_id,
            context_window=get_default_context_window(model_id),
            base_url=spec.get("base_url", "https://api.openai.com/v1"),
            api_key=spec["auth_token"],
        )

    async def send(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        prompts: list[Message] | None = None,
        stream: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Send messages to OpenAI-compatible API and yield streaming events."""
        openai_messages = _messages_to_openai(messages)
        if prompts:
            # 将 prompts 转换为 system 消息插入到最前面
            for msg in reversed(prompts):
                for block in msg.blocks:
                    if isinstance(block, TextBlock) and block.text:
                        openai_messages.insert(0, {"role": "system", "content": block.text})

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": openai_messages,
        }
        if tools:
            payload["tools"] = _tools_to_openai(tools)

        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

        if stream:
            async for event in _send_stream(self.base_url, payload, headers):
                yield event
        else:
            async for event in _send_no_stream(self.base_url, payload, headers):
                yield event


# ---------------------------------------------------------------------------
# Message → OpenAI API 转换
# ---------------------------------------------------------------------------

def _messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert internal Message list to OpenAI messages format.

    处理 blocks 模型：
    - assistant 消息中 ToolUseBlock → tool_calls 字段
    - user 消息中 ToolResultBlock → 生成 role:"tool" 结果消息
    - ThinkingBlock 忽略
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "assistant":
            # 构建 assistant 消息
            content_parts: list[str] = []
            tool_calls_list: list[dict[str, Any]] = []

            for block in msg.blocks:
                if isinstance(block, TextBlock) and block.text:
                    content_parts.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    tool_calls_list.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    })
                # ThinkingBlock, ImageBlock 等 → 忽略

            entry: dict[str, Any] = {"role": "assistant"}
            content = "\n".join(content_parts) if content_parts else None
            entry["content"] = content
            if tool_calls_list:
                entry["tool_calls"] = tool_calls_list
            result.append(entry)
        else:
            # user / system 消息
            content_parts = []
            image_parts: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []
            for block in msg.blocks:
                if isinstance(block, TextBlock) and block.text:
                    content_parts.append(block.text)
                elif isinstance(block, ImageBlock):
                    if block.url:
                        image_parts.append({
                            "type": "image_url",
                            "image_url": {"url": block.url},
                        })
                    elif block.data:
                        data_uri = f"data:{block.media_type};base64,{block.data}"
                        image_parts.append({
                            "type": "image_url",
                            "image_url": {"url": data_uri},
                        })
                elif isinstance(block, ToolResultBlock):
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": block.tool_use_id,
                        "content": block.content,
                    })

            if image_parts:
                # 多模态：content 是 array
                parts: list[dict[str, Any]] = []
                if content_parts:
                    parts.append({"type": "text", "text": "\n".join(content_parts)})
                parts.extend(image_parts)
                result.append({"role": msg.role, "content": parts})
            elif content_parts:
                result.append({"role": msg.role, "content": "\n".join(content_parts)})

            result.extend(tool_results)

    return _merge_consecutive_openai(result)


def _merge_consecutive_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive same-role messages for OpenAI format.

    Tool-role messages are never merged (each has a unique tool_call_id).
    """
    if not messages:
        return messages
    merged: list[dict[str, Any]] = [messages[0]]
    for msg in messages[1:]:
        prev = merged[-1]
        if msg["role"] == prev["role"] and msg["role"] not in ("tool",):
            prev_content = prev.get("content") or ""
            cur_content = msg.get("content") or ""
            if prev_content and cur_content:
                prev["content"] = prev_content + "\n\n" + cur_content
            elif cur_content:
                prev["content"] = cur_content
        else:
            merged.append(msg)
    return merged


def _tools_to_openai(tools: list[ToolSchema]) -> list[dict[str, Any]]:
    """Convert internal ToolSchema list to OpenAI tools format."""
    result = []
    for tool in tools:
        result.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema or {"type": "object", "properties": {}},
            },
        })
    return result


# ---------------------------------------------------------------------------
# OpenAI API → 内部模型转换
# ---------------------------------------------------------------------------

def _response_from_openai(data: dict[str, Any]) -> Response:
    """Convert OpenAI API response to internal Response."""
    choice = data.get("choices", [{}])[0]
    message_data = choice.get("message", {})
    finish_reason = choice.get("finish_reason") or ""

    stop_reason_map = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
        "content_filter": "content_filter",
    }
    stop_reason = stop_reason_map.get(finish_reason, finish_reason)

    blocks: list[ContentBlock] = []

    # Text content
    content = message_data.get("content", "") or ""
    if content:
        blocks.append(TextBlock(text=content))

    # Tool calls
    for tc_data in message_data.get("tool_calls", []):
        func = tc_data.get("function", {})
        try:
            arguments = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            arguments = {}
        blocks.append(ToolUseBlock(
            id=tc_data.get("id", ""),
            name=func.get("name", ""),
            input=arguments,
        ))

    # Usage
    usage_data = data.get("usage", {})
    usage: dict[str, int] = {}
    if "prompt_tokens" in usage_data:
        usage["input_tokens"] = usage_data["prompt_tokens"]
    if "completion_tokens" in usage_data:
        usage["output_tokens"] = usage_data["completion_tokens"]
    prompt_details = usage_data.get("prompt_tokens_details", {})
    if isinstance(prompt_details, dict) and "cached_tokens" in prompt_details:
        usage["cache_read_input_tokens"] = prompt_details["cached_tokens"]

    message = Message(role="assistant", blocks=blocks)
    return Response(message=message, stop_reason=stop_reason, usage=usage)


# ---------------------------------------------------------------------------
# HTTP 发送
# ---------------------------------------------------------------------------

async def _send_no_stream(
    base_url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> AsyncIterator[StreamEvent]:
    """Non-streaming path for OpenAI API."""
    async with HttpClient.create(timeout=httpx.Timeout(None, connect=10)) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
    data = resp.json()
    if resp.status_code != 200:
        error_msg = data.get("error", {}).get("message", json.dumps(data))
        logger.warning("OpenAI API error (%d): %s", resp.status_code, error_msg)
        yield StreamEvent(
            type="error",
            error=f"OpenAI API error ({resp.status_code}): {error_msg}",
        )
        return

    response = _response_from_openai(data)

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
    """Streaming path: parse OpenAI SSE and yield StreamEvents."""
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}

    async with HttpClient.create(timeout=httpx.Timeout(None, connect=10)) as client:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
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
                logger.warning("OpenAI API stream error (%d): %s", resp.status_code, error_msg)
                yield StreamEvent(
                    type="error",
                    error=f"OpenAI API error ({resp.status_code}): {error_msg}",
                )
                return

            text_parts: list[str] = []
            # Track tool call state by index
            tool_call_data: dict[int, dict[str, Any]] = {}
            stop_reason = ""
            usage: dict[str, int] = {}
            finish_reason = ""

            async for raw_line in resp.aiter_lines():
                line = raw_line

                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    # Finalize pending tool calls
                    tool_use_blocks: list[ToolUseBlock] = []
                    for idx in sorted(tool_call_data.keys()):
                        tc_info = tool_call_data[idx]
                        json_str = tc_info.get("args_json", "")
                        try:
                            arguments = json.loads(json_str) if json_str else {}
                        except json.JSONDecodeError:
                            arguments = {}
                        tool_use_blocks.append(ToolUseBlock(
                            id=tc_info.get("id", ""),
                            name=tc_info.get("name", ""),
                            input=arguments,
                        ))
                        yield StreamEvent(type="tool_use_end")

                    # Map finish_reason
                    stop_reason_map = {
                        "stop": "end_turn",
                        "tool_calls": "tool_use",
                        "length": "max_tokens",
                    }
                    stop_reason = stop_reason_map.get(finish_reason, finish_reason)

                    # Build blocks
                    blocks: list[ContentBlock] = []
                    text = "".join(text_parts)
                    if text:
                        blocks.append(TextBlock(text=text))
                    blocks.extend(tool_use_blocks)

                    message = Message(role="assistant", blocks=blocks)
                    response = Response(
                        message=message,
                        stop_reason=stop_reason,
                        usage=usage,
                    )
                    yield StreamEvent(type="response_done", response=response)
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Usage chunk
                if data.get("usage"):
                    usage_data = data["usage"]
                    if "prompt_tokens" in usage_data:
                        usage["input_tokens"] = usage_data["prompt_tokens"]
                    if "completion_tokens" in usage_data:
                        usage["output_tokens"] = usage_data["completion_tokens"]
                    prompt_details = usage_data.get("prompt_tokens_details", {})
                    if isinstance(prompt_details, dict) and "cached_tokens" in prompt_details:
                        usage["cache_read_input_tokens"] = prompt_details["cached_tokens"]

                choices = data.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                fr = choices[0].get("finish_reason")
                if fr:
                    finish_reason = fr

                # Text content
                content = delta.get("content")
                if content:
                    text_parts.append(content)
                    yield StreamEvent(type="text_delta", text=content)

                # Tool calls
                for tc_delta in delta.get("tool_calls", []):
                    idx = tc_delta.get("index", 0)
                    if idx not in tool_call_data:
                        func = tc_delta.get("function", {})
                        tool_call_data[idx] = {
                            "id": tc_delta.get("id", ""),
                            "name": func.get("name", ""),
                            "args_json": func.get("arguments", ""),
                        }
                        tc = ToolUseBlock(
                            id=tc_delta.get("id", ""),
                            name=func.get("name", ""),
                        )
                        yield StreamEvent(type="tool_use_start", tool_call=tc)
                    else:
                        func = tc_delta.get("function", {})
                        args_chunk = func.get("arguments", "")
                        if args_chunk:
                            tool_call_data[idx]["args_json"] += args_chunk
                            yield StreamEvent(
                                type="tool_use_delta",
                                tool_json_delta=args_chunk,
                            )
