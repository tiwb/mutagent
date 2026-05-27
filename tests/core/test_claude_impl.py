"""Tests for Claude API implementation (builtins/anthropic_provider.py)."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mutagent.core.messages import Message, Response, StreamEvent, TextBlock, ToolUseBlock, ToolSchema
from mutagent.core._llm_impl_anthropic import (
    AnthropicApiClient,
    _messages_to_claude,
    _tools_to_claude,
    _response_from_claude,
)


# ---------------------------------------------------------------------------
# Helper: extract text/tool_calls from new Message model
# ---------------------------------------------------------------------------

def _get_text(msg: Message) -> str:
    return "".join(b.text for b in msg.blocks if isinstance(b, TextBlock))


def _get_tool_calls(msg: Message) -> list[ToolUseBlock]:
    return [b for b in msg.blocks if isinstance(b, ToolUseBlock)]


class TestMessagesToClaude:

    def test_simple_user_message(self):
        msgs = [Message(role="user", blocks=[TextBlock(text="Hello")])]
        result = _messages_to_claude(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_simple_assistant_message(self):
        msgs = [Message(role="assistant", blocks=[TextBlock(text="Hi there")])]
        result = _messages_to_claude(msgs)
        assert result == [{"role": "assistant", "content": [{"type": "text", "text": "Hi there"}]}]

    def test_assistant_with_tool_calls(self):
        tc = ToolUseBlock(id="tc_1", name="Module-view_source", input={"target": "mutagent"})
        msgs = [Message(role="assistant", blocks=[TextBlock(text="Let me check."), tc])]
        result = _messages_to_claude(msgs)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        content = result[0]["content"]
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "Let me check."}
        assert content[1] == {
            "type": "tool_use",
            "id": "tc_1",
            "name": "Module-view_source",
            "input": {"target": "mutagent"},
        }

    def test_assistant_tool_calls_no_text(self):
        tc = ToolUseBlock(id="tc_1", name="run_code", input={"code": "1+1"})
        msgs = [Message(role="assistant", blocks=[tc])]
        result = _messages_to_claude(msgs)

        content = result[0]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "tool_use"

    def test_assistant_with_completed_tool_calls(self):
        """Completed ToolUseBlock generates tool_result in a user message."""
        tc = ToolUseBlock(id="tc_1", name="run_code", input={"code": "1+1"},
                          status="done", result="42")
        msgs = [Message(role="assistant", blocks=[tc])]
        result = _messages_to_claude(msgs)

        assert len(result) == 2
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "user"
        content = result[1]["content"]
        assert len(content) == 1
        assert content[0] == {
            "type": "tool_result",
            "tool_use_id": "tc_1",
            "content": "42",
        }

    def test_tool_result_with_error(self):
        tc = ToolUseBlock(id="tc_1", name="run_code", input={},
                          status="done", result="Error: not found", is_error=True)
        msgs = [Message(role="assistant", blocks=[tc])]
        result = _messages_to_claude(msgs)

        # assistant message + user tool_result message
        assert len(result) == 2
        block = result[1]["content"][0]
        assert block["is_error"] is True

    def test_multi_turn_conversation(self):
        msgs = [
            Message(role="user", blocks=[TextBlock(text="Hi")]),
            Message(role="assistant", blocks=[TextBlock(text="Hello!")]),
            Message(role="user", blocks=[TextBlock(text="Help me")]),
        ]
        result = _messages_to_claude(msgs)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"


class TestToolsToClaude:

    def test_single_tool(self):
        tools = [ToolSchema(
            name="Module-view_source",
            description="View source code",
            input_schema={
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Module path"},
                },
                "required": ["target"],
            },
        )]
        result = _tools_to_claude(tools)
        assert len(result) == 1
        assert result[0]["name"] == "Module-view_source"
        assert result[0]["description"] == "View source code"
        assert "properties" in result[0]["input_schema"]

    def test_empty_tools(self):
        result = _tools_to_claude([])
        assert result == []

    def test_tool_with_empty_schema(self):
        tools = [ToolSchema(name="noop", description="Does nothing")]
        result = _tools_to_claude(tools)
        assert result[0]["input_schema"] == {"type": "object", "properties": {}}


class TestResponseFromClaude:

    def test_text_response(self):
        data = {
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        resp = _response_from_claude(data)
        assert resp.message.role == "assistant"
        assert _get_text(resp.message) == "Hello!"
        assert resp.stop_reason == "end_turn"
        assert resp.usage == {"input_tokens": 10, "output_tokens": 5}

    def test_tool_use_response(self):
        data = {
            "content": [
                {"type": "text", "text": "I'll check that."},
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "Module-view_source",
                    "input": {"target": "mutagent.client"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 15},
        }
        resp = _response_from_claude(data)
        assert _get_text(resp.message) == "I'll check that."
        tool_calls = _get_tool_calls(resp.message)
        assert len(tool_calls) == 1
        tc = tool_calls[0]
        assert tc.id == "toolu_123"
        assert tc.name == "Module-view_source"
        assert tc.input == {"target": "mutagent.client"}
        assert resp.stop_reason == "tool_use"

    def test_multiple_tool_calls(self):
        data = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Module-view_source",
                    "input": {"target": "a"},
                },
                {
                    "type": "tool_use",
                    "id": "toolu_2",
                    "name": "run_code",
                    "input": {"code": "1+1"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {},
        }
        resp = _response_from_claude(data)
        tool_calls = _get_tool_calls(resp.message)
        assert len(tool_calls) == 2
        assert _get_text(resp.message) == ""

    def test_empty_content(self):
        data = {
            "content": [],
            "stop_reason": "end_turn",
            "usage": {},
        }
        resp = _response_from_claude(data)
        assert _get_text(resp.message) == ""
        assert _get_tool_calls(resp.message) == []


def _make_client():
    """创建测试用 AnthropicProvider。"""
    return AnthropicApiClient({
        "base_url": "https://api.anthropic.com",
        "auth_token": "test-key",
    })


async def _async_events(*events: StreamEvent):
    """Helper: yield StreamEvent objects as an async iterator."""
    for event in events:
        yield event


class TestSendMessageIntegration:

    async def test_send_message_success(self):
        """Test provider.send() returning text response events."""
        response = Response(
            message=Message(role="assistant", blocks=[TextBlock(text="Hello from Claude!")]),
            stop_reason="end_turn",
            usage={"input_tokens": 5, "output_tokens": 3},
        )

        mock_events = _async_events(
            StreamEvent(type="text_delta", text="Hello from Claude!"),
            StreamEvent(type="response_done", response=response),
        )

        provider = _make_client()
        with patch.object(provider, "send", return_value=mock_events):
            messages = [Message(role="user", blocks=[TextBlock(text="Hi")])]
            events = [e async for e in provider.send(
                messages, [], stream=False
            )]

        resp_event = [e for e in events if e.type == "response_done"][0]
        resp = resp_event.response
        assert _get_text(resp.message) == "Hello from Claude!"
        assert resp.stop_reason == "end_turn"

    async def test_send_message_with_tools(self):
        """Test provider.send() includes tools and returns tool_use response."""
        tc = ToolUseBlock(id="toolu_abc", name="Module-view_source", input={"target": "mutagent"})
        response = Response(
            message=Message(role="assistant", blocks=[tc]),
            stop_reason="tool_use",
            usage={"input_tokens": 10, "output_tokens": 8},
        )

        mock_events = _async_events(
            StreamEvent(type="tool_use_start", tool_call=tc),
            StreamEvent(type="tool_use_end"),
            StreamEvent(type="response_done", response=response),
        )

        tools = [ToolSchema(
            name="Module-view_source",
            description="View source code",
            input_schema={"type": "object", "properties": {"target": {"type": "string"}}},
        )]

        provider = _make_client()
        with patch.object(provider, "send", return_value=mock_events):
            messages = [Message(role="user", blocks=[TextBlock(text="Show me the code")])]
            events = [e async for e in provider.send(
                messages, tools, stream=False
            )]

        resp_event = [e for e in events if e.type == "response_done"][0]
        resp = resp_event.response
        assert resp.stop_reason == "tool_use"
        tool_calls = _get_tool_calls(resp.message)
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "Module-view_source"

    async def test_send_message_api_error(self):
        """Test provider.send() yields error event on API error."""
        mock_events = _async_events(
            StreamEvent(
                type="error",
                error="Claude API error (401): Invalid API key",
            ),
        )

        provider = _make_client()
        with patch.object(provider, "send", return_value=mock_events):
            events = [e async for e in provider.send(
                [Message(role="user", blocks=[TextBlock(text="Hi")])], [], stream=False
            )]

        assert len(events) == 1
        assert events[0].type == "error"
        assert "Invalid API key" in events[0].error


_has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))


@pytest.mark.skipif(not _has_api_key, reason="ANTHROPIC_API_KEY not set")
class TestClaudeRealAPI:
    """Integration tests using the real Claude API (skipped without API key)."""

    def _make_real_client(self):
        return AnthropicApiClient(
            model_id="claude-sonnet-4-20250514",
            base_url="https://api.anthropic.com",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )

    async def test_real_send_message(self):
        """Send a real message to Claude API and verify the response structure."""
        provider = self._make_real_client()
        messages = [Message(role="user", blocks=[TextBlock(text="Reply with exactly: PONG")])]
        events = [e async for e in provider.send(messages, [])]

        resp_event = [e for e in events if e.type == "response_done"][0]
        resp = resp_event.response
        assert isinstance(resp, Response)
        assert resp.message.role == "assistant"
        assert _get_text(resp.message)
        assert resp.stop_reason == "end_turn"
        assert resp.usage.get("input_tokens", 0) > 0
        assert resp.usage.get("output_tokens", 0) > 0

    async def test_real_send_message_with_tool_use(self):
        """Send a real message with tools and verify tool_use response."""
        provider = self._make_real_client()
        tools = [ToolSchema(
            name="get_weather",
            description="Get current weather for a city.",
            input_schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        )]
        messages = [Message(role="user", blocks=[TextBlock(text="What's the weather in Tokyo?")])]
        events = [e async for e in provider.send(messages, tools)]

        resp_event = [e for e in events if e.type == "response_done"][0]
        resp = resp_event.response
        assert isinstance(resp, Response)
        assert resp.message.role == "assistant"
        assert resp.stop_reason == "tool_use"
        tool_calls = _get_tool_calls(resp.message)
        assert len(tool_calls) >= 1
        tc = tool_calls[0]
        assert tc.name == "get_weather"
        assert "city" in tc.input
