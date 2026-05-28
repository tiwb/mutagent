from __future__ import annotations

import json

from mutagent.llmproxy.translation import (
    anthropic_request_to_openai,
    anthropic_response_to_openai,
    anthropic_sse_to_openai_chunks,
    openai_request_to_anthropic,
    openai_response_to_anthropic,
    openai_sse_to_anthropic_events,
)


def test_anthropic_request_to_openai_converts_system_tools_and_results():
    body = {
        "model": "claude-sonnet-4-20250514",
        "system": [{"type": "text", "text": "Be concise"}],
        "messages": [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Calling tool"},
                    {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "x"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "done"},
                ],
            },
        ],
        "tools": [
            {
                "name": "search",
                "description": "Search",
                "input_schema": {"type": "object"},
            }
        ],
        "tool_choice": {"type": "any"},
    }

    result = anthropic_request_to_openai(body)
    assert result["messages"][0] == {"role": "system", "content": "Be concise"}
    assert result["messages"][1] == {"role": "user", "content": "Hello"}
    assert result["messages"][2]["tool_calls"][0]["function"]["name"] == "search"
    assert result["messages"][3] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "done",
    }
    assert result["tool_choice"] == "required"


def test_openai_request_to_anthropic_converts_system_and_tool_messages():
    body = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Need tool",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "search", "arguments": json.dumps({"q": "x"})},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        ],
        "tool_choice": "required",
    }

    result = openai_request_to_anthropic(body)
    assert result["system"] == [{"type": "text", "text": "You are helpful"}]
    assert result["messages"][0] == {"role": "user", "content": "Hello"}
    assistant_blocks = result["messages"][1]["content"]
    assert assistant_blocks[0] == {"type": "text", "text": "Need tool"}
    assert assistant_blocks[1]["type"] == "tool_use"
    assert result["messages"][2]["role"] == "user"
    assert result["messages"][2]["content"][0]["type"] == "tool_result"
    assert result["tool_choice"] == {"type": "any"}


def test_openai_and_anthropic_response_translations():
    openai_response = {
        "id": "resp_1",
        "model": "gpt-4.1",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "search", "arguments": json.dumps({"q": "x"})},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }
    anthropic = openai_response_to_anthropic(openai_response, model="claude-sonnet-4")
    assert anthropic["model"] == "claude-sonnet-4"
    assert anthropic["stop_reason"] == "tool_use"
    assert anthropic["usage"] == {"input_tokens": 11, "output_tokens": 7}

    anthropic_response = {
        "id": "msg_1",
        "model": "claude-sonnet-4",
        "content": [
            {"type": "text", "text": "Hi"},
            {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "x"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }
    openai = anthropic_response_to_openai(anthropic_response, model="gpt-4.1")
    assert openai["model"] == "gpt-4.1"
    assert openai["choices"][0]["finish_reason"] == "tool_calls"
    assert openai["usage"] == {"prompt_tokens": 5, "completion_tokens": 3}


def test_sse_translation_openai_to_anthropic():
    lines = [
        'data: {"id":"chatcmpl_1","model":"gpt-4.1","choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":2}}',
        "data: [DONE]",
    ]

    events = list(openai_sse_to_anthropic_events(iter(lines), model="claude-sonnet-4"))
    assert events[0][0] == "message_start"
    assert any(event_type == "content_block_delta" for event_type, _ in events)
    assert events[-1][0] == "message_stop"


def test_sse_translation_anthropic_to_openai():
    lines = [
        'event: message_start',
        'data: {"message":{"id":"msg_1","model":"claude-sonnet-4","usage":{"input_tokens":9}}}',
        'event: content_block_start',
        'data: {"index":0,"content_block":{"type":"text","text":""}}',
        'event: content_block_delta',
        'data: {"index":0,"delta":{"type":"text_delta","text":"Hi"}}',
        'event: message_delta',
        'data: {"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}',
        'event: message_stop',
        'data: {"type":"message_stop"}',
    ]

    chunks = list(anthropic_sse_to_openai_chunks(iter(lines), model="gpt-4.1"))
    assert chunks[-1] == "data: [DONE]\n\n"
    payload = json.loads(chunks[0][6:].strip())
    assert payload["choices"][0]["delta"]["role"] == "assistant"
