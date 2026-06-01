"""Tests for Agent declaration and public API (submit / subscribe / cancel)."""

import asyncio

import pytest

import mutobj
from mutagent.core.agent import Agent
from mutagent.core.context import AgentContext
from mutagent.core._agent_impl import MAX_TOOL_ROUNDS
from mutagent.core._llm_impl_anthropic import AnthropicApiClient
from mutagent.core.messages import (
    Message,
    Response,
    StreamEvent,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from mutagent.core.tools import ToolSet, Toolkit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_text(msg: Message) -> str:
    """从 Message 中提取文本内容。"""
    return "".join(b.text for b in msg.blocks if isinstance(b, TextBlock))


class _TestToolkit(Toolkit):
    """Minimal toolkit for testing tool dispatch."""

    _tool_prefix = "Test"

    def echo(self, text: str = "") -> str:
        """Echo the input."""
        return f"echo: {text}"

    def inspect(self, target: str = "") -> str:
        """Inspect a target."""
        return f"inspecting {target}: OK"


def _make_agent(llm=None, model="test-model"):
    """Create an Agent with a minimal ToolSet for testing."""
    if llm is None:
        llm = AnthropicApiClient({"base_url": "https://api.test.com", "auth_token": "test-key"})
    tool_set = ToolSet()
    tool_set.add(_TestToolkit())
    context = AgentContext()
    context.prompts.append(
        Message(role="system", blocks=[TextBlock(text="You are a test assistant.")], label="base")
    )
    agent = Agent(llm=llm, model=model, tools=tool_set, context=context)
    return agent


def _make_response(text: str, stop_reason: str = "end_turn") -> Response:
    """Create a simple text Response."""
    return Response(
        message=Message(role="assistant", blocks=[TextBlock(text=text)]),
        stop_reason=stop_reason,
    )


def _make_tool_response(tool_blocks: list[ToolUseBlock],
                        stop_reason: str = "tool_use",
                        text: str = "") -> Response:
    """Create a Response with tool calls."""
    blocks: list = []
    if text:
        blocks.append(TextBlock(text=text))
    blocks.extend(tool_blocks)
    return Response(
        message=Message(role="assistant", blocks=blocks),
        stop_reason=stop_reason,
    )


async def _await_turn(agent: Agent) -> None:
    """Wait for the current submit task to complete."""
    task = getattr(agent, "_current_task", None)
    if task is not None:
        try:
            await task
        except asyncio.CancelledError:
            pass


async def _collect_events(agent: Agent, expected_last: str = "turn_done") -> list[StreamEvent]:
    """Helper: submit text, await turn, return collected events."""
    events: list[StreamEvent] = []
    agent.subscribe(lambda e: events.append(e))
    await agent.submit("test input")
    await _await_turn(agent)
    return events


# ---------------------------------------------------------------------------
# Declaration tests
# ---------------------------------------------------------------------------

class TestAgentDeclaration:

    def test_inherits_from_mutagent_declaration(self):
        assert issubclass(Agent, mutobj.Declaration)

    def test_uses_declaration_meta(self):
        assert issubclass(Agent, mutobj.Declaration)

    def test_declared_methods(self):
        assert mutobj.get_declaration_func(Agent, "submit") is not None
        assert mutobj.get_declaration_func(Agent, "cancel") is not None
        assert mutobj.get_declaration_func(Agent, "subscribe") is not None
        assert mutobj.get_declaration_func(Agent, "is_busy") is not None
        assert mutobj.get_declaration_func(Agent, "handle_tool_calls") is not None

    def test_stub_submit_does_nothing(self):
        agent = _make_agent()
        # stub is awaited but does nothing
        asyncio.run(agent.submit("test"))

    def test_stub_subscribe_returns_cancel_fn(self):
        agent = _make_agent()
        cancel = agent.subscribe(lambda e: None)
        assert callable(cancel)
        cancel()


# ---------------------------------------------------------------------------
# Basic response tests
# ---------------------------------------------------------------------------

class TestBasicResponse:

    @pytest.fixture
    def agent(self):
        return _make_agent()

    async def test_simple_text_response(self, agent):
        """submit emits correct event sequence for a simple text response."""
        response = _make_response("Hello!")
        events: list[StreamEvent] = []
        agent.subscribe(lambda e: events.append(e))

        async def mock_send(messages, tools, prompts=None, stream=True):
            yield StreamEvent(type="text_delta", text="Hello!")
            yield StreamEvent(type="response_done", response=response)

        agent.llm.send = mock_send

        await agent.submit("Hi")
        await _await_turn(agent)

        types = [e.type for e in events]
        assert types == ["response_start", "text_delta", "response_done", "turn_done"]
        assert agent.is_busy() is False
        assert len(agent.context.messages) == 2
        assert agent.context.messages[0].role == "user"
        assert _get_text(agent.context.messages[0]) == "Hi"
        assert agent.context.messages[1].role == "assistant"
        assert _get_text(agent.context.messages[1]) == "Hello!"

    async def test_empty_text_delta_still_works(self, agent):
        """Empty text_delta events are forwarded but yield empty response."""
        response = _make_response("")
        events: list[StreamEvent] = []
        agent.subscribe(lambda e: events.append(e))

        async def mock_send(messages, tools, prompts=None, stream=True):
            yield StreamEvent(type="text_delta", text="")
            yield StreamEvent(type="response_done", response=response)

        agent.llm.send = mock_send

        await agent.submit("Hi")
        await _await_turn(agent)

        types = [e.type for e in events]
        assert "response_done" in types
        assert "turn_done" in types


# ---------------------------------------------------------------------------
# Tool call tests
# ---------------------------------------------------------------------------

class TestToolCalls:

    @pytest.fixture
    def agent(self):
        return _make_agent()

    async def test_single_tool_call_then_response(self, agent):
        """Agent executes tool call then gets final response."""
        tool_response = _make_tool_response([
            ToolUseBlock(id="tc_1", name="Test-echo", input={"text": "hello"}),
        ], text="Let me echo that.")
        final_response = _make_response("The echo returned: echo: hello")

        call_idx = 0
        async def mock_send(messages, tools, prompts=None, stream=True):
            nonlocal call_idx
            if call_idx == 0:
                call_idx += 1
                yield StreamEvent(type="text_delta", text="Let me echo that.")
                yield StreamEvent(
                    type="tool_use_start",
                    tool_call=ToolUseBlock(id="tc_1", name="Test-echo"),
                )
                yield StreamEvent(type="tool_use_end")
                yield StreamEvent(type="response_done", response=tool_response)
            else:
                yield StreamEvent(type="text_delta", text="The echo returned: echo: hello")
                yield StreamEvent(type="response_done", response=final_response)

        agent.llm.send = mock_send

        events = await _collect_events(agent)
        types = [e.type for e in events]

        assert "tool_exec_start" in types
        assert "tool_exec_end" in types
        assert types == [
            "response_start", "text_delta", "tool_use_start", "tool_use_end",
            "response_done", "tool_exec_start", "tool_exec_end",
            "response_start", "text_delta", "response_done", "turn_done",
        ]
        # user + assistant(tool) + user(tool_result) + assistant(final)
        assert len(agent.context.messages) == 4

    async def test_multiple_tool_calls_in_one_response(self, agent):
        """Agent handles multiple tool calls in a single response."""
        tool_response = _make_tool_response([
            ToolUseBlock(id="tc_1", name="Test-echo", input={"text": "a"}),
            ToolUseBlock(id="tc_2", name="Test-inspect", input={"target": "x"}),
        ])
        final_response = _make_response("Done.")

        call_idx = 0
        async def mock_send(messages, tools, prompts=None, stream=True):
            nonlocal call_idx
            if call_idx == 0:
                call_idx += 1
                yield StreamEvent(
                    type="tool_use_start",
                    tool_call=ToolUseBlock(id="tc_1", name="Test-echo"),
                )
                yield StreamEvent(type="tool_use_end")
                yield StreamEvent(
                    type="tool_use_start",
                    tool_call=ToolUseBlock(id="tc_2", name="Test-inspect"),
                )
                yield StreamEvent(type="tool_use_end")
                yield StreamEvent(type="response_done", response=tool_response)
            else:
                yield StreamEvent(type="text_delta", text="Done.")
                yield StreamEvent(type="response_done", response=final_response)

        agent.llm.send = mock_send

        events = await _collect_events(agent)
        exec_starts = [e for e in events if e.type == "tool_exec_start"]
        exec_ends = [e for e in events if e.type == "tool_exec_end"]
        assert len(exec_starts) == 2
        assert len(exec_ends) == 2
        tc_blocks = [b for b in agent.context.messages[1].blocks if isinstance(b, ToolUseBlock)]
        assert len(tc_blocks) == 2
        result_blocks = [
            b for b in agent.context.messages[2].blocks if isinstance(b, ToolResultBlock)
        ]
        assert len(result_blocks) == 2

    async def test_tool_call_without_text(self, agent):
        """Tool call response with no text block."""
        tool_response = _make_tool_response([
            ToolUseBlock(id="tc_1", name="Test-echo", input={"text": "silent"}),
        ])
        final_response = _make_response("Processed.")

        call_idx = 0
        async def mock_send(messages, tools, prompts=None, stream=True):
            nonlocal call_idx
            if call_idx == 0:
                call_idx += 1
                yield StreamEvent(
                    type="tool_use_start",
                    tool_call=ToolUseBlock(id="tc_1", name="Test-echo"),
                )
                yield StreamEvent(type="tool_use_end")
                yield StreamEvent(type="response_done", response=tool_response)
            else:
                yield StreamEvent(type="text_delta", text="Processed.")
                yield StreamEvent(type="response_done", response=final_response)

        agent.llm.send = mock_send

        events = await _collect_events(agent)
        types = [e.type for e in events]
        assert "tool_exec_start" in types
        assert types[-1] == "turn_done"


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestErrorHandling:

    @pytest.fixture
    def agent(self):
        return _make_agent()

    async def test_provider_error_produces_turn_done(self, agent):
        """Provider error event does not crash the agent, turn_done still fires."""
        async def mock_send(messages, tools, prompts=None, stream=True):
            yield StreamEvent(type="error", error="API rate limited")

        agent.llm.send = mock_send

        events = await _collect_events(agent)
        types = [e.type for e in events]

        assert types == ["response_start", "error", "turn_done"]
        assert events[1].error == "API rate limited"
        # Context not corrupted — only the user message was added
        assert len(agent.context.messages) == 1
        assert agent.context.messages[0].role == "user"

    async def test_no_response_done_produces_error(self, agent):
        """If provider doesn't yield response_done, agent yields error + turn_done."""
        async def mock_send(messages, tools, prompts=None, stream=True):
            # Only text_delta, no response_done
            yield StreamEvent(type="text_delta", text="unfinished...")

        agent.llm.send = mock_send

        events = await _collect_events(agent)
        types = [e.type for e in events]

        assert "error" in types
        assert types[-1] == "turn_done"
        assert agent.is_busy() is False

    async def test_submit_while_busy_raises(self, agent):
        """Submitting while busy raises RuntimeError."""
        # Make send hang
        async def mock_send(messages, tools, prompts=None, stream=True):
            await asyncio.sleep(0.5)
            yield StreamEvent(type="text_delta", text="late")
            yield StreamEvent(type="response_done", response=_make_response("late"))

        agent.llm.send = mock_send

        await agent.submit("first")
        with pytest.raises(RuntimeError, match="busy"):
            await agent.submit("second")

        # Clean up
        task = getattr(agent, "_current_task")
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


# ---------------------------------------------------------------------------
# Cancel tests
# ---------------------------------------------------------------------------

class TestCancel:

    @pytest.fixture
    def agent(self):
        return _make_agent()

    async def test_cancel_mid_turn(self, agent):
        """cancel() cancels the running task, turn_done still fires."""
        async def mock_send(messages, tools, prompts=None, stream=True):
            yield StreamEvent(type="text_delta", text="partial output...")
            await asyncio.sleep(10)  # Simulate long running

        agent.llm.send = mock_send

        events: list[StreamEvent] = []
        agent.subscribe(lambda e: events.append(e))

        await agent.submit("Stop me")
        await asyncio.sleep(0.05)

        assert agent.is_busy() is True
        assert agent.cancel() is True

        await _await_turn(agent)
        types = [e.type for e in events]

        assert types[:2] == ["response_start", "text_delta"]
        assert types[-1] == "turn_done"
        assert agent.is_busy() is False

    async def test_cancel_when_idle_returns_false(self, agent):
        """cancel() when no task returns False."""
        assert agent.cancel() is False


# ---------------------------------------------------------------------------
# Subscribe tests
# ---------------------------------------------------------------------------

class TestSubscribe:

    @pytest.fixture
    def agent(self):
        return _make_agent()

    async def test_unsubscribe_stops_events(self, agent):
        """After cancel_fn(), callback no longer receives events."""
        response = _make_response("Hello")

        async def mock_send(messages, tools, prompts=None, stream=True):
            yield StreamEvent(type="text_delta", text="A")
            yield StreamEvent(type="response_done", response=response)

        agent.llm.send = mock_send

        events_a: list[StreamEvent] = []
        events_b: list[StreamEvent] = []
        cancel_a = agent.subscribe(lambda e: events_a.append(e))
        cancel_b = agent.subscribe(lambda e: events_b.append(e))

        await agent.submit("Hi")
        await _await_turn(agent)
        assert len(events_a) > 0
        assert len(events_b) > 0

        # Cancel A, submit again
        cancel_a()
        events_a.clear()
        events_b.clear()

        await agent.submit("Hi again")
        await _await_turn(agent)

        assert len(events_a) == 0  # unsubscribed
        assert len(events_b) > 0   # still subscribed

        # Cleanup
        cancel_b()


# ---------------------------------------------------------------------------
# handle_tool_calls direct test
# ---------------------------------------------------------------------------

class TestHandleToolCalls:

    @pytest.fixture
    def agent(self):
        return _make_agent()

    async def test_dispatches_tool_blocks(self, agent):
        """handle_tool_calls dispatches each ToolUseBlock through the tool set."""
        blocks = [
            ToolUseBlock(id="tc_1", name="Test-echo", input={"text": "direct"}),
        ]
        results = await agent.handle_tool_calls(blocks)

        assert results[0].tool_use_id == "tc_1"
        assert results[0].content == "echo: direct"

    async def test_dispatches_multiple_blocks(self, agent):
        """Multiple tool blocks all get dispatched."""
        blocks = [
            ToolUseBlock(id="tc_1", name="Test-echo", input={"text": "a"}),
            ToolUseBlock(id="tc_2", name="Test-inspect", input={"target": "mod"}),
        ]
        results = await agent.handle_tool_calls(blocks)

        assert results[0].content == "echo: a"
        assert "OK" in results[1].content

    async def test_unknown_tool_sets_error_flag(self, agent):
        """Unknown tool name results in done status with is_error flag."""
        blocks = [
            ToolUseBlock(id="tc_1", name="Test-nonexistent", input={}),
        ]
        results = await agent.handle_tool_calls(blocks)

        assert results[0].is_error is True
        assert "Unknown" in results[0].content


# ---------------------------------------------------------------------------
# Max tool rounds test
# ---------------------------------------------------------------------------

class TestMaxToolRounds:

    @pytest.fixture
    def agent(self):
        return _make_agent()

    async def test_max_tool_rounds_injects_summary(self, agent):
        """After MAX_TOOL_ROUNDS tool calls, agent injects summary request.

        Need MAX_TOOL_ROUNDS+1 tool_use responses: the first 25 drive
        tool_round up to 25, then the 26th triggers the >= check.
        """
        call_idx = 0
        async def mock_send(messages, tools, prompts=None, stream=True):
            nonlocal call_idx
            idx = call_idx
            call_idx += 1
            if idx <= MAX_TOOL_ROUNDS:  # 0..25 → 26 tool_use responses
                resp = _make_tool_response([
                    ToolUseBlock(id=f"tc_{idx}", name="Test-echo", input={"text": f"round{idx}"}),
                ], text=f"Round {idx}")
                yield StreamEvent(type="text_delta", text=f"Round {idx}")
                yield StreamEvent(
                    type="tool_use_start",
                    tool_call=ToolUseBlock(id=f"tc_{idx}", name="Test-echo"),
                )
                yield StreamEvent(type="tool_use_end")
                yield StreamEvent(type="response_done", response=resp)
            else:
                # Summary response after limit
                resp = _make_response("Summary of all work done.")
                yield StreamEvent(type="text_delta", text="Summary of all work done.")
                yield StreamEvent(type="response_done", response=resp)

        agent.llm.send = mock_send

        events = await _collect_events(agent)
        tool_exec_starts = [e for e in events if e.type == "tool_exec_start"]
        assert len(tool_exec_starts) == MAX_TOOL_ROUNDS

        # Check summary request was injected
        limit_msgs = [
            m for m in agent.context.messages
            if m.role == "user" and "Tool call limit reached" in _get_text(m)
        ]
        assert len(limit_msgs) == 1


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:

    @pytest.fixture
    def agent(self):
        return _make_agent()

    async def test_end_turn_with_tool_calls_executes_tools(self, agent):
        """stop_reason=end_turn but tool_calls exist → tools execute, then loop continues."""
        tool_response = _make_tool_response(
            [ToolUseBlock(id="tc_1", name="Test-echo", input={"text": "late"})],
            stop_reason="end_turn",
            text="Here you go.",
        )
        final_response = _make_response("Now really done.")

        call_idx = 0
        async def mock_send(messages, tools, prompts=None, stream=True):
            nonlocal call_idx
            idx = call_idx
            call_idx += 1
            if idx == 0:
                yield StreamEvent(type="text_delta", text="Here you go.")
                yield StreamEvent(
                    type="tool_use_start",
                    tool_call=ToolUseBlock(id="tc_1", name="Test-echo"),
                )
                yield StreamEvent(type="tool_use_end")
                yield StreamEvent(type="response_done", response=tool_response)
            else:
                yield StreamEvent(type="text_delta", text="Now really done.")
                yield StreamEvent(type="response_done", response=final_response)

        agent.llm.send = mock_send

        events = await _collect_events(agent)
        types = [e.type for e in events]
        assert "tool_exec_start" in types
        assert types[-1] == "turn_done"
        # user + assistant(tool) + user(tool_result) + assistant(final)
        assert len(agent.context.messages) == 4

    async def test_end_turn_without_tool_calls_ends_normally(self, agent):
        """stop_reason=end_turn, no tool_calls → normal end."""
        response = _make_response("All done.")
        async def mock_send(messages, tools, prompts=None, stream=True):
            yield StreamEvent(type="text_delta", text="All done.")
            yield StreamEvent(type="response_done", response=response)

        agent.llm.send = mock_send

        events = await _collect_events(agent)
        types = [e.type for e in events]
        assert types == ["response_start", "text_delta", "response_done", "turn_done"]
        assert len(agent.context.messages) == 2

    async def test_turn_done_does_not_append_turn_block(self, agent):
        """turn_done 由 runtime 事件表达，不修改 assistant transcript。"""
        response = _make_response("Done.")
        async def mock_send(messages, tools, prompts=None, stream=True):
            yield StreamEvent(type="text_delta", text="Done.")
            yield StreamEvent(type="response_done", response=response)

        agent.llm.send = mock_send

        await _collect_events(agent)
        last_msg = agent.context.messages[-1]
        assert last_msg.role == "assistant"
        assert all(not isinstance(b, ToolResultBlock) for b in last_msg.blocks)
        assert len(last_msg.blocks) == 1
