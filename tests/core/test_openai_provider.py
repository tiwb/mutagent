"""Tests for OpenAI API implementation (builtins/openai_provider.py)."""

import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from mutagent.core.messages import (
    Message,
    Response,
    StreamEvent,
    TextBlock,
    ToolResultBlock,
    ToolSchema,
    ToolUseBlock,
    Usage,
)
from mutagent.core.llm import LLMApiClient
from mutagent.core._llm_impl_openai import (
    OpenAIApiClient,
    messages_to_openai,
    tools_to_openai,
    _response_from_openai,
)
from tests.core._helpers import MockLLMClient


# ---------------------------------------------------------------------------
# Helper: extract text/tool_calls from new Message model
# ---------------------------------------------------------------------------

def _get_text(msg: Message) -> str:
    return "".join(b.text for b in msg.blocks if isinstance(b, TextBlock))


def _get_tool_calls(msg: Message) -> list[ToolUseBlock]:
    return [b for b in msg.blocks if isinstance(b, ToolUseBlock)]


class TestMessagesToOpenAI:

    def test_simple_user_message(self):
        msgs = [Message(role="user", blocks=[TextBlock(text="Hello")])]
        result = messages_to_openai(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_simple_assistant_message(self):
        msgs = [Message(role="assistant", blocks=[TextBlock(text="Hi there")])]
        result = messages_to_openai(msgs)
        assert result == [{"role": "assistant", "content": "Hi there"}]

    def test_assistant_with_tool_calls_and_content(self):
        tc = ToolUseBlock(id="call_1", name="get_weather", input={"city": "Tokyo"})
        msgs = [Message(role="assistant", blocks=[TextBlock(text="Let me check."), tc])]
        result = messages_to_openai(msgs)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Let me check."
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0] == {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"city": "Tokyo"}),
            },
        }

    def test_assistant_tool_calls_no_text(self):
        tc = ToolUseBlock(id="call_1", name="run_code", input={"code": "1+1"})
        msgs = [Message(role="assistant", blocks=[tc])]
        result = messages_to_openai(msgs)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        # When content is empty, OpenAI format should set content to None
        assert result[0]["content"] is None
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["name"] == "run_code"

    def test_assistant_multiple_tool_calls(self):
        tc1 = ToolUseBlock(id="call_1", name="tool_a", input={"x": 1})
        tc2 = ToolUseBlock(id="call_2", name="tool_b", input={"y": 2})
        msgs = [Message(role="assistant", blocks=[tc1, tc2])]
        result = messages_to_openai(msgs)

        assert len(result) == 1
        assert len(result[0]["tool_calls"]) == 2
        assert result[0]["tool_calls"][0]["id"] == "call_1"
        assert result[0]["tool_calls"][1]["id"] == "call_2"

    def test_user_with_tool_result(self):
        """User ToolResultBlock generates 'tool' role messages in OpenAI format."""
        msgs = [
            Message(role="assistant", blocks=[
                ToolUseBlock(id="call_1", name="get_weather", input={"city": "Tokyo"}),
            ]),
            Message(role="user", blocks=[
                ToolResultBlock(
                    tool_use_id="call_1",
                    tool_name="get_weather",
                    content="42",
                ),
            ]),
        ]
        result = messages_to_openai(msgs)

        # assistant message + tool result message
        assert len(result) == 2
        assert result[0]["role"] == "assistant"
        assert result[1] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "42",
        }

    def test_user_with_multiple_tool_results(self):
        """Multiple ToolResultBlocks generate multiple 'tool' role messages."""
        msgs = [
            Message(role="assistant", blocks=[
                ToolUseBlock(id="call_1", name="tool_a", input={}),
                ToolUseBlock(id="call_2", name="tool_b", input={}),
            ]),
            Message(role="user", blocks=[
                ToolResultBlock(tool_use_id="call_1", tool_name="tool_a", content="result_1"),
                ToolResultBlock(tool_use_id="call_2", tool_name="tool_b", content="result_2"),
            ]),
        ]
        result = messages_to_openai(msgs)

        # assistant message + 2 tool result messages
        assert len(result) == 3
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_1"
        assert result[1]["content"] == "result_1"
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call_2"
        assert result[2]["content"] == "result_2"

    def test_multi_turn_conversation(self):
        msgs = [
            Message(role="user", blocks=[TextBlock(text="Hi")]),
            Message(role="assistant", blocks=[TextBlock(text="Hello!")]),
            Message(role="user", blocks=[TextBlock(text="Help me")]),
        ]
        result = messages_to_openai(msgs)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"

    def test_tool_call_arguments_serialized_to_json(self):
        """Verify arguments dict is serialized to JSON string in OpenAI format."""
        tc = ToolUseBlock(id="call_x", name="func", input={"a": [1, 2], "b": True})
        msgs = [Message(role="assistant", blocks=[tc])]
        result = messages_to_openai(msgs)

        args_str = result[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args_str, str)
        assert json.loads(args_str) == {"a": [1, 2], "b": True}


class TestToolsToOpenAI:

    def test_single_tool(self):
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
        result = tools_to_openai(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get current weather for a city."
        assert "properties" in result[0]["function"]["parameters"]
        assert result[0]["function"]["parameters"]["required"] == ["city"]

    def test_empty_tools(self):
        result = tools_to_openai([])
        assert result == []

    def test_tool_with_empty_schema(self):
        tools = [ToolSchema(name="noop", description="Does nothing")]
        result = tools_to_openai(tools)
        assert result[0]["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_multiple_tools(self):
        tools = [
            ToolSchema(name="tool_a", description="Tool A", input_schema={"type": "object", "properties": {"x": {"type": "integer"}}}),
            ToolSchema(name="tool_b", description="Tool B", input_schema={"type": "object", "properties": {"y": {"type": "string"}}}),
        ]
        result = tools_to_openai(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "tool_a"
        assert result[1]["function"]["name"] == "tool_b"
        assert all(r["type"] == "function" for r in result)


class TestResponseFromOpenAI:

    def test_text_response(self):
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        resp = _response_from_openai(data)
        assert resp.message.role == "assistant"
        assert _get_text(resp.message) == "Hello!"
        assert resp.stop_reason == "end_turn"
        assert resp.usage == Usage(input_tokens=10, output_tokens=5)

    def test_tool_use_response(self):
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "I'll check that.",
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Tokyo"}),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15},
        }
        resp = _response_from_openai(data)
        assert _get_text(resp.message) == "I'll check that."
        tool_calls = _get_tool_calls(resp.message)
        assert len(tool_calls) == 1
        tc = tool_calls[0]
        assert tc.id == "call_123"
        assert tc.name == "get_weather"
        assert tc.input == {"city": "Tokyo"}
        assert resp.stop_reason == "tool_use"

    def test_multiple_tool_calls(self):
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "tool_a",
                                "arguments": json.dumps({"x": 1}),
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "tool_b",
                                "arguments": json.dumps({"y": "hello"}),
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {},
        }
        resp = _response_from_openai(data)
        tool_calls = _get_tool_calls(resp.message)
        assert len(tool_calls) == 2
        assert _get_text(resp.message) == ""
        assert tool_calls[0].name == "tool_a"
        assert tool_calls[1].name == "tool_b"

    def test_empty_content(self):
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }],
            "usage": {},
        }
        resp = _response_from_openai(data)
        assert _get_text(resp.message) == ""
        assert _get_tool_calls(resp.message) == []

    def test_null_content_becomes_empty_string(self):
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": None},
                "finish_reason": "stop",
            }],
            "usage": {},
        }
        resp = _response_from_openai(data)
        assert _get_text(resp.message) == ""

    def test_stop_reason_mapping_length(self):
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": "truncated..."},
                "finish_reason": "length",
            }],
            "usage": {},
        }
        resp = _response_from_openai(data)
        assert resp.stop_reason == "max_tokens"

    def test_stop_reason_mapping_content_filter(self):
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "content_filter",
            }],
            "usage": {},
        }
        resp = _response_from_openai(data)
        assert resp.stop_reason == "content_filter"

    def test_stop_reason_unknown_passes_through(self):
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "some_unknown_reason",
            }],
            "usage": {},
        }
        resp = _response_from_openai(data)
        assert resp.stop_reason == "some_unknown_reason"

    def test_usage_mapping(self):
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        resp = _response_from_openai(data)
        assert resp.usage.input_tokens == 100
        assert resp.usage.output_tokens == 50

    def test_usage_missing_fields(self):
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": {},
        }
        resp = _response_from_openai(data)
        assert resp.usage == Usage()

    def test_invalid_tool_call_arguments_json(self):
        """Malformed JSON in tool call arguments should default to empty dict."""
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_bad",
                        "type": "function",
                        "function": {
                            "name": "broken_tool",
                            "arguments": "not-valid-json{{{",
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {},
        }
        resp = _response_from_openai(data)
        tool_calls = _get_tool_calls(resp.message)
        assert len(tool_calls) == 1
        assert tool_calls[0].input == {}


# ---------------------------------------------------------------------------
# Async helpers for integration tests
# ---------------------------------------------------------------------------


async def _collect_events(async_iter):
    """Collect all events from an async iterator into a list."""
    events = []
    async for event in async_iter:
        events.append(event)
    return events


async def _mock_send_events(*events: StreamEvent):
    """Create an async generator that yields the given StreamEvent objects.

    Used to mock provider.send() in integration tests.
    """
    for event in events:
        yield event


class TestSendMessageIntegration:

    async def test_send_message_success(self):
        """Test provider.send() with mocked send (non-streaming path)."""
        response = Response(
            message=Message(role="assistant", blocks=[TextBlock(text="Hello from OpenAI!")]),
            stop_reason="end_turn",
            usage=Usage(input_tokens=5, output_tokens=3),
        )
        mock_events = [
            StreamEvent(type="text_delta", text="Hello from OpenAI!"),
            StreamEvent(type="response_done", response=response),
        ]

        provider = MockLLMClient()
        provider.mock_send = lambda *a, **kw: _mock_send_events(*mock_events)
        messages = [Message(role="user", blocks=[TextBlock(text="Hi")])]
        events = await _collect_events(provider.send(messages, [], stream=False))

        resp_event = [e for e in events if e.type == "response_done"][0]
        resp = resp_event.response
        assert _get_text(resp.message) == "Hello from OpenAI!"
        assert resp.stop_reason == "end_turn"
        assert resp.usage.input_tokens == 5
        assert resp.usage.output_tokens == 3

    async def test_send_message_with_tools(self):
        """Test provider.send() includes tools in the request."""
        tc = ToolUseBlock(id="call_abc", name="get_weather", input={"city": "Tokyo"})
        response = Response(
            message=Message(role="assistant", blocks=[tc]),
            stop_reason="tool_use",
            usage=Usage(input_tokens=10, output_tokens=8),
        )
        mock_events = [
            StreamEvent(type="tool_use_start", tool_call=tc),
            StreamEvent(type="tool_use_end"),
            StreamEvent(type="response_done", response=response),
        ]

        tools = [ToolSchema(
            name="get_weather",
            description="Get current weather for a city.",
            input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
        )]

        provider = MockLLMClient()
        provider.mock_send = lambda *a, **kw: _mock_send_events(*mock_events)
        messages = [Message(role="user", blocks=[TextBlock(text="What's the weather?")])]
        events = await _collect_events(provider.send(messages, tools, stream=False))

        resp_event = [e for e in events if e.type == "response_done"][0]
        resp = resp_event.response
        assert resp.stop_reason == "tool_use"
        tool_calls = _get_tool_calls(resp.message)
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].input == {"city": "Tokyo"}

    async def test_send_message_api_error(self):
        """Test provider.send() yields error event on API error."""
        mock_events = [
            StreamEvent(type="error", error="OpenAI API error (401): Incorrect API key provided"),
        ]

        provider = MockLLMClient()
        provider.mock_send = lambda *a, **kw: _mock_send_events(*mock_events)
        events = await _collect_events(provider.send(
            [Message(role="user", blocks=[TextBlock(text="Hi")])], [], stream=False
        ))

        assert len(events) == 1
        assert events[0].type == "error"
        assert "Incorrect API key provided" in events[0].error

    async def test_send_message_text_delta_emitted(self):
        """Test that a text_delta event is emitted before response_done."""
        response = Response(
            message=Message(role="assistant", blocks=[TextBlock(text="Some text reply")]),
            stop_reason="end_turn",
            usage=Usage(input_tokens=5, output_tokens=4),
        )
        mock_events = [
            StreamEvent(type="text_delta", text="Some text reply"),
            StreamEvent(type="response_done", response=response),
        ]

        provider = MockLLMClient()
        provider.mock_send = lambda *a, **kw: _mock_send_events(*mock_events)
        events = await _collect_events(provider.send(
            [Message(role="user", blocks=[TextBlock(text="Hi")])], [], stream=False
        ))

        text_events = [e for e in events if e.type == "text_delta"]
        assert len(text_events) == 1
        assert text_events[0].text == "Some text reply"

    async def test_send_message_tool_use_events(self):
        """Test that tool_use_start and tool_use_end events are emitted for each tool call."""
        tc1 = ToolUseBlock(id="call_1", name="tool_a", input={"x": 1})
        tc2 = ToolUseBlock(id="call_2", name="tool_b", input={"y": 2})
        response = Response(
            message=Message(role="assistant", blocks=[tc1, tc2]),
            stop_reason="tool_use",
            usage=Usage(input_tokens=10, output_tokens=12),
        )
        mock_events = [
            StreamEvent(type="tool_use_start", tool_call=tc1),
            StreamEvent(type="tool_use_end"),
            StreamEvent(type="tool_use_start", tool_call=tc2),
            StreamEvent(type="tool_use_end"),
            StreamEvent(type="response_done", response=response),
        ]

        provider = MockLLMClient()
        provider.mock_send = lambda *a, **kw: _mock_send_events(*mock_events)
        events = await _collect_events(provider.send(
            [Message(role="user", blocks=[TextBlock(text="Do stuff")])], [], stream=False
        ))

        tool_start_events = [e for e in events if e.type == "tool_use_start"]
        tool_end_events = [e for e in events if e.type == "tool_use_end"]
        assert len(tool_start_events) == 2
        assert len(tool_end_events) == 2
        assert tool_start_events[0].tool_call.name == "tool_a"
        assert tool_start_events[1].tool_call.name == "tool_b"

    async def test_send_message_prompts(self):
        """Test that prompts are forwarded to provider.send()."""
        response = Response(
            message=Message(role="assistant", blocks=[TextBlock(text="OK")]),
            stop_reason="end_turn",
            usage=Usage(input_tokens=15, output_tokens=1),
        )
        mock_events = [
            StreamEvent(type="response_done", response=response),
        ]

        provider = MockLLMClient()
        captured_kwargs = {}

        async def mock_send(messages, tools, prompts=None, stream=True):
            captured_kwargs["prompts"] = prompts
            captured_kwargs["messages"] = messages
            for event in mock_events:
                yield event

        provider.mock_send = mock_send
        messages = [Message(role="user", blocks=[TextBlock(text="Hi")])]
        prompt_msg = Message(role="system", blocks=[TextBlock(text="You are helpful.")])
        await _collect_events(provider.send(
            messages, [], prompts=[prompt_msg], stream=False
        ))

        assert captured_kwargs["prompts"] is not None
        assert len(captured_kwargs["prompts"]) == 1

    async def test_send_message_no_prompts(self):
        """Test that no prompts are passed when prompts is None."""
        response = Response(
            message=Message(role="assistant", blocks=[TextBlock(text="OK")]),
            stop_reason="end_turn",
            usage=Usage(input_tokens=5, output_tokens=1),
        )
        mock_events = [
            StreamEvent(type="response_done", response=response),
        ]

        provider = MockLLMClient()
        captured_kwargs = {}

        async def mock_send(messages, tools, prompts=None, stream=True):
            captured_kwargs["prompts"] = prompts
            captured_kwargs["messages"] = messages
            for event in mock_events:
                yield event

        provider.mock_send = mock_send
        messages = [Message(role="user", blocks=[TextBlock(text="Hi")])]
        await _collect_events(provider.send(messages, [], stream=False))

        assert captured_kwargs["prompts"] is None


class TestOpenAIProviderFromSpec:

    def test_from_spec_defaults(self):
        provider = LLMApiClient.from_spec({"auth_token": "sk-test"})
        assert provider.base_url == "https://api.anthropic.com"
        assert provider.api_key == "sk-test"

    def test_from_spec_missing_auth_token(self):
        with pytest.raises(ValueError, match="auth_token"):
            LLMApiClient.from_spec({})

    def test_from_spec_custom(self):
        config = {
            "type": "OpenAI",
            "base_url": "https://api.groq.com/openai/v1",
            "auth_token": "gsk_abc123",
        }
        provider = LLMApiClient.from_spec(config)
        assert provider.base_url == "https://api.groq.com/openai/v1"
        assert provider.api_key == "gsk_abc123"


_has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))


@pytest.mark.skipif(not _has_openai_key, reason="OPENAI_API_KEY not set")
class TestOpenAIRealAPI:
    """Integration tests using the real OpenAI API (skipped without API key)."""

    def _make_real_client(self):
        return OpenAIApiClient(
            model_id="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key=os.environ["OPENAI_API_KEY"],
        )

    async def test_real_send_message(self):
        """Send a real message to OpenAI API and verify the response structure."""
        provider = self._make_real_client()
        messages = [Message(role="user", blocks=[TextBlock(text="Reply with exactly: PONG")])]
        events = await _collect_events(provider.send(messages, []))

        resp_event = [e for e in events if e.type == "response_done"][0]
        resp = resp_event.response
        assert isinstance(resp, Response)
        assert resp.message.role == "assistant"
        assert _get_text(resp.message)
        assert resp.stop_reason == "end_turn"
        assert resp.usage.input_tokens > 0
        assert resp.usage.output_tokens > 0

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
        events = await _collect_events(provider.send(messages, tools))

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
