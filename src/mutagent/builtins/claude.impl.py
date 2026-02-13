"""mutagent.builtins.claude -- Claude API implementation for LLMClient."""

import json
from typing import Any

import aiohttp

import mutagent
from mutagent.client import LLMClient
from mutagent.messages import Message, Response, ToolCall, ToolResult, ToolSchema


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


@mutagent.impl(LLMClient.send_message)
async def send_message(
    self: LLMClient,
    messages: list[Message],
    tools: list[ToolSchema],
) -> Response:
    """Send messages to Claude API and return the response."""
    claude_messages = _messages_to_claude(messages)
    payload: dict[str, Any] = {
        "model": self.model,
        "messages": claude_messages,
        "max_tokens": 4096,
    }
    if tools:
        payload["tools"] = _tools_to_claude(tools)

    headers = {
        "x-api-key": self.api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{self.base_url}/v1/messages",
            headers=headers,
            json=payload,
        ) as resp:
            data = await resp.json()
            if resp.status != 200:
                error_msg = data.get("error", {}).get("message", json.dumps(data))
                raise RuntimeError(
                    f"Claude API error ({resp.status}): {error_msg}"
                )
            return _response_from_claude(data)
