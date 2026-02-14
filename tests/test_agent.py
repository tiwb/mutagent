"""Tests for Agent declaration and main loop implementation."""

from unittest.mock import AsyncMock

import pytest

import mutagent
from mutagent.agent import Agent
from mutagent.base import MutagentMeta
from mutagent.client import LLMClient
from mutagent.essential_tools import EssentialTools
from mutagent.main import load_builtins
from mutagent.messages import (
    Message,
    Response,
    StreamEvent,
    ToolCall,
    ToolResult,
    ToolSchema,
)
from mutagent.runtime.module_manager import ModuleManager
from mutagent.selector import ToolSelector
from forwardpy.core import _DECLARED_METHODS

# Load all impls (idempotent)
load_builtins()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _collect_events(async_iter):
    """Collect all StreamEvents from an async iterator into a list."""
    events = []
    async for event in async_iter:
        events.append(event)
    return events


async def _collect_text(async_iter):
    """Collect text from text_delta events."""
    text_parts = []
    async for event in async_iter:
        if event.type == "text_delta":
            text_parts.append(event.text)
    return "".join(text_parts)


def _make_stream_events_for_response(response: Response) -> list[StreamEvent]:
    """Build the list of StreamEvents that a non-streaming send_message would yield."""
    events = []
    if response.message.content:
        events.append(StreamEvent(type="text_delta", text=response.message.content))
    for tc in response.message.tool_calls:
        events.append(StreamEvent(type="tool_use_start", tool_call=tc))
        events.append(StreamEvent(type="tool_use_end"))
    events.append(StreamEvent(type="response_done", response=response))
    return events


async def _mock_send_message_gen(events_list):
    """Create a mock async generator that yields from a list of StreamEvent lists.

    Each call to the mock pops the first list and yields its events.
    """
    idx = 0

    async def _gen(*args, **kwargs):
        nonlocal idx
        events = events_list[idx]
        idx += 1
        for event in events:
            yield event

    return _gen


# ---------------------------------------------------------------------------
# Declaration tests
# ---------------------------------------------------------------------------

class TestAgentDeclaration:

    def test_inherits_from_mutagent_object(self):
        assert issubclass(Agent, mutagent.Object)

    def test_uses_mutagent_meta(self):
        assert isinstance(Agent, MutagentMeta)

    def test_declared_methods(self):
        declared = getattr(Agent, _DECLARED_METHODS, set())
        assert "run" in declared
        assert "step" in declared
        assert "handle_tool_calls" in declared


# ---------------------------------------------------------------------------
# Agent loop tests (adapted for streaming interface)
# ---------------------------------------------------------------------------

class TestAgentLoop:

    @pytest.fixture
    def mock_client(self):
        """Create a mock LLM client."""
        client = LLMClient(
            model="test-model",
            api_key="test-key",
            base_url="https://api.test.com",
        )
        return client

    @pytest.fixture
    def agent(self, mock_client):
        mgr = ModuleManager()
        tools = EssentialTools(module_manager=mgr)
        selector = ToolSelector(essential_tools=tools)
        agent = Agent(
            client=mock_client,
            tool_selector=selector,
            system_prompt="You are a helpful assistant.",
            messages=[],
        )
        yield agent
        mgr.cleanup()

    @pytest.mark.asyncio
    async def test_simple_response(self, agent):
        """Agent receives a simple text response (no tool calls)."""
        response = Response(
            message=Message(role="assistant", content="Hello! How can I help?"),
            stop_reason="end_turn",
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        events = _make_stream_events_for_response(response)
        call_count = 0

        async def mock_send(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            for e in events:
                yield e

        agent.client.send_message = mock_send

        text = await _collect_text(agent.run("Hi"))

        assert text == "Hello! How can I help?"
        assert len(agent.messages) == 2  # user + assistant
        assert agent.messages[0].role == "user"
        assert agent.messages[0].content == "Hi"
        assert agent.messages[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_tool_call_then_response(self, agent):
        """Agent handles a tool call then gets final response."""
        # First response: tool call
        tool_response = Response(
            message=Message(
                role="assistant",
                content="Let me run some code.",
                tool_calls=[ToolCall(id="tc_1", name="run_code", arguments={"code": "print(1+1)"})],
            ),
            stop_reason="tool_use",
        )
        # Second response: final text
        final_response = Response(
            message=Message(role="assistant", content="The result is 2."),
            stop_reason="end_turn",
        )

        events_1 = _make_stream_events_for_response(tool_response)
        events_2 = _make_stream_events_for_response(final_response)
        call_idx = 0

        async def mock_send(*args, **kwargs):
            nonlocal call_idx
            evts = [events_1, events_2][call_idx]
            call_idx += 1
            for e in evts:
                yield e

        agent.client.send_message = mock_send

        text = await _collect_text(agent.run("What is 1+1?"))

        assert text == "Let me run some code.The result is 2."
        assert len(agent.messages) == 4  # user, assistant(tool_call), user(tool_result), assistant(final)
        assert agent.messages[2].role == "user"
        assert len(agent.messages[2].tool_results) == 1
        assert "2" in agent.messages[2].tool_results[0].content

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self, agent):
        """Agent handles multiple tool calls in one response."""
        tool_response = Response(
            message=Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(id="tc_1", name="run_code", arguments={"code": "print('a')"}),
                    ToolCall(id="tc_2", name="run_code", arguments={"code": "print('b')"}),
                ],
            ),
            stop_reason="tool_use",
        )
        final_response = Response(
            message=Message(role="assistant", content="Done."),
            stop_reason="end_turn",
        )

        events_1 = _make_stream_events_for_response(tool_response)
        events_2 = _make_stream_events_for_response(final_response)
        call_idx = 0

        async def mock_send(*args, **kwargs):
            nonlocal call_idx
            evts = [events_1, events_2][call_idx]
            call_idx += 1
            for e in evts:
                yield e

        agent.client.send_message = mock_send

        text = await _collect_text(agent.run("Run two things"))

        assert text == "Done."
        assert len(agent.messages[2].tool_results) == 2

    @pytest.mark.asyncio
    async def test_step_yields_events(self, agent):
        """step() yields StreamEvents from client.send_message."""
        response = Response(
            message=Message(role="assistant", content="Response"),
            stop_reason="end_turn",
        )
        events = _make_stream_events_for_response(response)

        async def mock_send(*args, **kwargs):
            for e in events:
                yield e

        agent.client.send_message = mock_send
        agent.messages.append(Message(role="user", content="Test"))

        collected = await _collect_events(agent.step())

        assert len(collected) == 2  # text_delta + response_done
        assert collected[0].type == "text_delta"
        assert collected[0].text == "Response"
        assert collected[1].type == "response_done"
        assert collected[1].response is response

    @pytest.mark.asyncio
    async def test_handle_tool_calls_dispatches(self, agent):
        """handle_tool_calls dispatches each call through the selector."""
        calls = [
            ToolCall(id="tc_1", name="run_code", arguments={"code": "print(42)"}),
        ]
        results = await agent.handle_tool_calls(calls)

        assert len(results) == 1
        assert results[0].tool_call_id == "tc_1"
        assert "42" in results[0].content


# ---------------------------------------------------------------------------
# Streaming event sequence tests
# ---------------------------------------------------------------------------

class TestStreamingEventSequence:

    @pytest.fixture
    def agent(self):
        mgr = ModuleManager()
        tools = EssentialTools(module_manager=mgr)
        selector = ToolSelector(essential_tools=tools)
        client = LLMClient(
            model="test-model",
            api_key="test-key",
            base_url="https://api.test.com",
        )
        agent = Agent(
            client=client,
            tool_selector=selector,
            system_prompt="test",
            messages=[],
        )
        yield agent
        mgr.cleanup()

    @pytest.mark.asyncio
    async def test_event_order_simple(self, agent):
        """Simple response yields: text_delta, response_done."""
        response = Response(
            message=Message(role="assistant", content="Hello"),
            stop_reason="end_turn",
        )

        async def mock_send(*args, **kwargs):
            yield StreamEvent(type="text_delta", text="Hello")
            yield StreamEvent(type="response_done", response=response)

        agent.client.send_message = mock_send

        events = await _collect_events(agent.run("Hi"))
        types = [e.type for e in events]

        assert types == ["text_delta", "response_done"]

    @pytest.mark.asyncio
    async def test_event_order_with_tool_call(self, agent):
        """Tool call response yields: text, tool_use events, response_done,
        tool_exec_start, tool_exec_end, then second LLM call events."""
        tool_response = Response(
            message=Message(
                role="assistant",
                content="Thinking...",
                tool_calls=[ToolCall(id="tc_1", name="run_code", arguments={"code": "1+1"})],
            ),
            stop_reason="tool_use",
        )
        final_response = Response(
            message=Message(role="assistant", content="Done"),
            stop_reason="end_turn",
        )

        call_idx = 0

        async def mock_send(*args, **kwargs):
            nonlocal call_idx
            if call_idx == 0:
                call_idx += 1
                yield StreamEvent(type="text_delta", text="Thinking...")
                yield StreamEvent(
                    type="tool_use_start",
                    tool_call=ToolCall(id="tc_1", name="run_code"),
                )
                yield StreamEvent(type="tool_use_end")
                yield StreamEvent(type="response_done", response=tool_response)
            else:
                yield StreamEvent(type="text_delta", text="Done")
                yield StreamEvent(type="response_done", response=final_response)

        agent.client.send_message = mock_send

        events = await _collect_events(agent.run("Calc"))
        types = [e.type for e in events]

        assert types == [
            "text_delta",        # "Thinking..."
            "tool_use_start",    # LLM constructs tool call
            "tool_use_end",
            "response_done",     # first LLM call done
            "tool_exec_start",   # Agent executes tool
            "tool_exec_end",     # tool result
            "text_delta",        # "Done"
            "response_done",     # second LLM call done
        ]

        # Verify tool_exec events carry correct data
        exec_start = events[4]
        assert exec_start.tool_call.name == "run_code"
        exec_end = events[5]
        assert exec_end.tool_result is not None
        assert exec_end.tool_result.tool_call_id == "tc_1"

    @pytest.mark.asyncio
    async def test_error_event_stops_loop(self, agent):
        """An error event from LLM stops the agent loop."""
        async def mock_send(*args, **kwargs):
            yield StreamEvent(type="error", error="API failed")

        agent.client.send_message = mock_send

        events = await _collect_events(agent.run("Hi"))

        assert len(events) == 1
        assert events[0].type == "error"
        assert events[0].error == "API failed"
        # Only user message should be in history (no assistant message added)
        assert len(agent.messages) == 1
        assert agent.messages[0].role == "user"

    @pytest.mark.asyncio
    async def test_stream_false_produces_events(self, agent):
        """stream=False still yields events through the same interface."""
        response = Response(
            message=Message(role="assistant", content="Non-streamed"),
            stop_reason="end_turn",
        )

        async def mock_send(*args, **kwargs):
            yield StreamEvent(type="text_delta", text="Non-streamed")
            yield StreamEvent(type="response_done", response=response)

        agent.client.send_message = mock_send

        text = await _collect_text(agent.run("Test", stream=False))
        assert text == "Non-streamed"
