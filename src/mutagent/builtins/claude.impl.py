"""mutagent.builtins.claude -- Claude API implementation for LLMClient."""

import json
from typing import Any, AsyncIterator

import aiohttp

import mutagent
from mutagent.client import LLMClient
from mutagent.messages import (
    Message,
    Response,
    StreamEvent,
    ToolCall,
    ToolResult,
    ToolSchema,
)


def _messages_to_claude(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert internal Message list to Claude API messages format."""
    result = []
    for msg in messages:
        if msg.role == "user" and msg.tool_results:
            # Tool results are sent as user messages with tool_result content blocks
            content = []
            for tr in msg.tool_results:
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": tr.tool_call_id,
                    "content": tr.content,
                }
                if tr.is_error:
                    block["is_error"] = True
                content.append(block)
            result.append({"role": "user", "content": content})
        elif msg.role == "assistant" and msg.tool_calls:
            # Assistant messages with tool calls have mixed content blocks
            content: list[dict[str, Any]] = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            result.append({"role": "assistant", "content": content})
        else:
            # Simple text message
            result.append({"role": msg.role, "content": msg.content})
    return result


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


def _response_from_claude(data: dict[str, Any]) -> Response:
    """Convert Claude API response to internal Response."""
    stop_reason = data.get("stop_reason", "")
    usage = data.get("usage", {})

    # Parse content blocks
    content_blocks = data.get("content", [])
    text_parts = []
    tool_calls = []

    for block in content_blocks:
        if block["type"] == "text":
            text_parts.append(block["text"])
        elif block["type"] == "tool_use":
            tool_calls.append(ToolCall(
                id=block["id"],
                name=block["name"],
                arguments=block.get("input", {}),
            ))

    message = Message(
        role="assistant",
        content="\n".join(text_parts),
        tool_calls=tool_calls,
    )

    return Response(
        message=message,
        stop_reason=stop_reason,
        usage=usage,
    )


async def _send_message_no_stream(
    self: LLMClient,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> AsyncIterator[StreamEvent]:
    """Non-streaming path: make a regular HTTP request and wrap as StreamEvents."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{self.base_url}/v1/messages",
            headers=headers,
            json=payload,
        ) as resp:
            data = await resp.json()
            if resp.status != 200:
                error_msg = data.get("error", {}).get("message", json.dumps(data))
                yield StreamEvent(
                    type="error",
                    error=f"Claude API error ({resp.status}): {error_msg}",
                )
                return

            response = _response_from_claude(data)

            # Yield text as a single delta
            if response.message.content:
                yield StreamEvent(type="text_delta", text=response.message.content)

            # Yield each tool call
            for tc in response.message.tool_calls:
                yield StreamEvent(type="tool_use_start", tool_call=tc)
                yield StreamEvent(type="tool_use_end")

            yield StreamEvent(type="response_done", response=response)


async def _send_message_stream(
    self: LLMClient,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> AsyncIterator[StreamEvent]:
    """Streaming path: parse SSE events from Claude API and yield StreamEvents."""
    payload["stream"] = True

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{self.base_url}/v1/messages",
            headers=headers,
            json=payload,
        ) as resp:
            if resp.status != 200:
                try:
                    data = await resp.json()
                    error_msg = data.get("error", {}).get("message", json.dumps(data))
                except Exception:
                    error_msg = f"HTTP {resp.status}"
                yield StreamEvent(
                    type="error",
                    error=f"Claude API error ({resp.status}): {error_msg}",
                )
                return

            # State for assembling the final Response
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            stop_reason = ""
            usage: dict[str, int] = {}

            # Current content block being streamed
            current_block_type: str = ""
            current_tool_id: str = ""
            current_tool_name: str = ""
            current_tool_json_parts: list[str] = []

            # Parse SSE stream
            event_type = ""
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")

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
                                tc = ToolCall(
                                    id=current_tool_id,
                                    name=current_tool_name,
                                )
                                yield StreamEvent(
                                    type="tool_use_start", tool_call=tc
                                )

                        elif event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            delta_type = delta.get("type", "")
                            if delta_type == "text_delta":
                                text = delta.get("text", "")
                                if text:
                                    text_parts.append(text)
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

                        elif event_type == "content_block_stop":
                            if current_block_type == "tool_use":
                                # Parse accumulated JSON for tool arguments
                                json_str = "".join(current_tool_json_parts)
                                try:
                                    arguments = json.loads(json_str) if json_str else {}
                                except json.JSONDecodeError:
                                    arguments = {}
                                tool_calls.append(ToolCall(
                                    id=current_tool_id,
                                    name=current_tool_name,
                                    arguments=arguments,
                                ))
                                yield StreamEvent(type="tool_use_end")
                            current_block_type = ""

                        elif event_type == "message_delta":
                            delta = data.get("delta", {})
                            stop_reason = delta.get("stop_reason", stop_reason)
                            usage.update(data.get("usage", {}))

                        elif event_type == "message_stop":
                            # Assemble the final Response
                            message = Message(
                                role="assistant",
                                content="".join(text_parts),
                                tool_calls=tool_calls,
                            )
                            response = Response(
                                message=message,
                                stop_reason=stop_reason,
                                usage=usage,
                            )
                            yield StreamEvent(
                                type="response_done", response=response
                            )

                    except Exception as e:
                        yield StreamEvent(
                            type="error",
                            error=f"Error processing SSE event '{event_type}': {e}",
                        )

                    event_type = ""
                    continue

                # Empty line = end of SSE event (already handled above)
                if not line:
                    event_type = ""


@mutagent.impl(LLMClient.send_message)
async def send_message(
    self: LLMClient,
    messages: list[Message],
    tools: list[ToolSchema],
    system_prompt: str = "",
    stream: bool = True,
) -> AsyncIterator[StreamEvent]:
    """Send messages to Claude API and yield streaming events."""
    claude_messages = _messages_to_claude(messages)
    payload: dict[str, Any] = {
        "model": self.model,
        "messages": claude_messages,
        "max_tokens": 4096,
    }
    if system_prompt:
        payload["system"] = system_prompt
    if tools:
        payload["tools"] = _tools_to_claude(tools)

    headers = {
        "authorization": f"Bearer {self.api_key}",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    if stream:
        async for event in _send_message_stream(self, payload, headers):
            yield event
    else:
        async for event in _send_message_no_stream(self, payload, headers):
            yield event
