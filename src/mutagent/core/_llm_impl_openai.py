"""mutagent.core._provider_impl_openai -- OpenAI Chat Completions API provider."""

import logging
from typing import AsyncGenerator, AsyncIterator, ClassVar

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
    Usage,
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

    def __init__(self, spec: JsonObject):
        auth_token = get_field(spec, "auth_token", str, default="")
        if not auth_token:
            raise ValueError("OpenAIProvider requires 'auth_token' in model spec.")
        model_id = get_field(spec, "model_id", str, default="")
        super().__init__(
            model_id=model_id,
            context_window=get_default_context_window(model_id),
            base_url=get_field(spec, "base_url", str, default="https://api.openai.com/v1"),
            api_key=auth_token,
        )

    async def send(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        prompts: list[Message] | None = None,
        stream: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Send messages to OpenAI-compatible API and yield streaming events."""
        openai_messages = messages_to_openai(messages)
        if prompts:
            # 将 prompts 转换为 system 消息插入到最前面
            for msg in reversed(prompts):
                for block in msg.blocks:
                    if isinstance(block, TextBlock) and block.text:
                        openai_messages.insert(0, {"role": "system", "content": block.text})

        payload: JsonObject = {
            "model": self.model_id,
            "messages": openai_messages,
        }
        if tools:
            payload["tools"] = tools_to_openai(tools)

        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

        if stream:
            async for event in send_stream(self.base_url, payload, headers):
                yield event
        else:
            async for event in send_no_stream(self.base_url, payload, headers):
                yield event


# ---------------------------------------------------------------------------
# Message → OpenAI API 转换
# ---------------------------------------------------------------------------

def messages_to_openai(messages: list[Message]) -> list[JsonObject]:
    """Convert internal Message list to OpenAI messages format.

    处理 blocks 模型：
    - assistant 消息中 ToolUseBlock → tool_calls 字段
    - user 消息中 ToolResultBlock → 生成 role:"tool" 结果消息
    - ThinkingBlock 忽略
    """
    result: list[JsonObject] = []
    for msg in messages:
        if msg.role == "assistant":
            # 构建 assistant 消息
            content_parts: list[str] = []
            tool_calls_list: list[JsonObject] = []

            for block in msg.blocks:
                if isinstance(block, TextBlock) and block.text:
                    content_parts.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    tool_calls_list.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": dumps(block.input),
                        },
                    })
                # ThinkingBlock, ImageBlock 等 → 忽略

            entry: JsonObject = {"role": "assistant"}
            content = "\n".join(content_parts) if content_parts else None
            entry["content"] = content
            if tool_calls_list:
                entry["tool_calls"] = tool_calls_list
            result.append(entry)
        else:
            # user / system 消息
            content_parts = []
            image_parts: list[JsonObject] = []
            tool_results: list[JsonObject] = []
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
                parts: list[JsonObject] = []
                if content_parts:
                    parts.append({"type": "text", "text": "\n".join(content_parts)})
                parts.extend(image_parts)
                result.append({"role": msg.role, "content": parts})
            elif content_parts:
                result.append({"role": msg.role, "content": "\n".join(content_parts)})

            result.extend(tool_results)

    return _merge_consecutive_openai(result)


def _merge_consecutive_openai(messages: list[JsonObject]) -> list[JsonObject]:
    """Merge consecutive same-role messages for OpenAI format.

    Tool-role messages are never merged (each has a unique tool_call_id).
    """
    if not messages:
        return messages
    merged: list[JsonObject] = [messages[0]]
    for msg in messages[1:]:
        prev = merged[-1]
        if msg["role"] == prev["role"] and msg["role"] not in ("tool",):
            prev_content = get_field(prev, "content", str, default="")
            cur_content = get_field(msg, "content", str, default="")
            if prev_content and cur_content:
                prev["content"] = prev_content + "\n\n" + cur_content
            elif cur_content:
                prev["content"] = cur_content
        else:
            merged.append(msg)
    return merged


def tools_to_openai(tools: list[ToolSchema]) -> list[JsonObject]:
    """Convert internal ToolSchema list to OpenAI tools format."""
    result: list[JsonObject] = []
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

def _response_from_openai(data: JsonObject) -> Response:
    """Convert OpenAI API response to internal Response."""
    choices = get_field(data, "choices", list[JsonObject], default=[])
    choice = choices[0] if choices else JsonObject()
    message_data = get_field(choice, "message", JsonObject, default={})
    finish_reason = get_field(choice, "finish_reason", str, default="")

    stop_reason_map = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
        "content_filter": "content_filter",
    }
    stop_reason = stop_reason_map.get(finish_reason, finish_reason)

    blocks: list[ContentBlock] = []

    # Text content
    content = get_field(message_data, "content", str, default="")
    if content:
        blocks.append(TextBlock(text=content))

    # Tool calls
    tool_calls = get_field(message_data, "tool_calls", list[JsonObject], default=[])
    for tc_data in tool_calls:
        func = get_field(tc_data, "function", JsonObject, default={})
        args_str = get_field(func, "arguments", str, default="{}")
        try:
            arguments = narrow_value(loads(args_str), JsonObject)
        except (JSONDecodeError, TypeError):
            arguments = JsonObject()
        blocks.append(ToolUseBlock(
            id=get_field(tc_data, "id", str, default=""),
            name=get_field(func, "name", str, default=""),
            input=arguments,
        ))

    # Usage
    raw_usage = get_field(data, "usage", JsonObject, default={})
    prompt_details = get_field(raw_usage, "prompt_tokens_details", JsonObject, default={})
    usage = Usage(
        input_tokens=get_field(raw_usage, "prompt_tokens", int, default=0),
        output_tokens=get_field(raw_usage, "completion_tokens", int, default=0),
        cache_read_input_tokens=get_field(prompt_details, "cached_tokens", int, default=0),
    )

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


async def send_no_stream(
    base_url: str,
    payload: JsonObject,
    headers: dict[str, str],
) -> AsyncIterator[StreamEvent]:
    """Non-streaming path for OpenAI API."""
    async with HttpClient.create(timeout=httpx.Timeout(None, connect=10)) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
    data_raw: JsonValue = resp.json()
    if resp.status_code != 200:
        error_msg = _extract_error_msg(data_raw, resp.status_code, "OpenAI API error")
        logger.warning("OpenAI API error (%d): %s", resp.status_code, error_msg)
        yield StreamEvent(
            type="error",
            error=error_msg,
        )
        return

    if not isinstance(data_raw, dict):
        yield StreamEvent(type="error", error="OpenAI API response is not a JSON object")
        return
    response = _response_from_openai(data_raw)

    for block in response.message.blocks:
        if isinstance(block, TextBlock) and block.text:
            yield StreamEvent(type="text_delta", text=block.text)
        elif isinstance(block, ToolUseBlock):
            yield StreamEvent(type="tool_use_start", tool_call=block)
            yield StreamEvent(type="tool_use_end")

    yield StreamEvent(type="response_done", response=response)


async def send_stream(
    base_url: str,
    payload: JsonObject,
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
                    error_data = loads(body)
                    error_msg = _extract_error_msg(error_data, resp.status_code, "OpenAI API error")
                except Exception:
                    error_msg = f"OpenAI API error ({resp.status_code}): HTTP {resp.status_code}"
                logger.warning("OpenAI API stream error (%d): %s", resp.status_code, error_msg)
                yield StreamEvent(
                    type="error",
                    error=error_msg,
                )
                return

            text_parts: list[str] = []
            # Track tool call state by index
            tool_call_data: dict[int, JsonObject] = {}
            stop_reason = ""
            usage = Usage()
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
                        json_str = get_field(tc_info, "args_json", str, default="")
                        try:
                            arguments = narrow_value(loads(json_str), JsonObject) if json_str else JsonObject()
                        except (JSONDecodeError, TypeError):
                            arguments = JsonObject()
                        tool_use_blocks.append(ToolUseBlock(
                            id=get_field(tc_info, "id", str, default=""),
                            name=get_field(tc_info, "name", str, default=""),
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
                    data_raw = loads(data_str)
                except JSONDecodeError:
                    continue

                # SSE 事件体始终是 JSON object
                if not isinstance(data_raw, dict):
                    continue
                data: JsonObject = data_raw

                # Usage chunk
                if data.get("usage"):
                    raw_usage = narrow_value(data["usage"], JsonObject)
                    usage.input_tokens = get_field(raw_usage, "prompt_tokens", int, default=usage.input_tokens)
                    usage.output_tokens = get_field(raw_usage, "completion_tokens", int, default=usage.output_tokens)
                    prompt_details = get_field(raw_usage, "prompt_tokens_details", JsonObject, default={})
                    usage.cache_read_input_tokens = get_field(prompt_details, "cached_tokens", int, default=usage.cache_read_input_tokens)

                choices = get_field(data, "choices", list[JsonValue], default=[])
                if not choices:
                    continue

                choice = narrow_value(choices[0], JsonObject)
                delta = get_field(choice, "delta", JsonObject, default={})
                fr = get_field(choice, "finish_reason", str, default="")
                if fr:
                    finish_reason = fr

                # Text content
                content = get_field(delta, "content", str, default="")
                if content:
                    text_parts.append(content)
                    yield StreamEvent(type="text_delta", text=content)

                # Tool calls
                tc_deltas = get_field(delta, "tool_calls", list[JsonObject], default=[])
                for tc_delta in tc_deltas:
                    idx = get_field(tc_delta, "index", int, default=0)
                    if idx not in tool_call_data:
                        func = get_field(tc_delta, "function", JsonObject, default={})
                        tool_call_data[idx] = {
                            "id": get_field(tc_delta, "id", str, default=""),
                            "name": get_field(func, "name", str, default=""),
                            "args_json": get_field(func, "arguments", str, default=""),
                        }
                        tc = ToolUseBlock(
                            id=narrow_value(tool_call_data[idx]["id"], str),
                            name=narrow_value(tool_call_data[idx]["name"], str),
                        )
                        yield StreamEvent(type="tool_use_start", tool_call=tc)
                    else:
                        func = get_field(tc_delta, "function", JsonObject, default={})
                        args_chunk = get_field(func, "arguments", str, default="")
                        if args_chunk:
                            tool_call_data[idx]["args_json"] = narrow_value(tool_call_data[idx]["args_json"], str) + args_chunk
                            yield StreamEvent(
                                type="tool_use_delta",
                                tool_json_delta=args_chunk,
                            )
