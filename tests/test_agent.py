"""Tests for Agent declaration and main loop implementation."""

import asyncio
from unittest.mock import MagicMock

import pytest

import mutagent
from mutagent.builtins.main_impl import DictConfig
from mutagent.agent import Agent
from mutagent.client import LLMClient
from mutagent.context import AgentContext
from mutagent.builtins.anthropic_provider import AnthropicProvider
from mutagent.toolkits.module_toolkit import ModuleToolkit
from mutagent.toolkits.log_toolkit import LogToolkit
from mutagent.messages import (
    Message,
    Response,
    StreamEvent,
    TextBlock,
    ToolSchema,
    ToolUseBlock,
    TurnStartBlock,
)
from mutagent.runtime.module_manager import ModuleManager
from mutagent.tools import ToolSet
from mutobj.core import DeclarationMeta, _DECLARED_METHODS

import mutagent.builtins  # noqa: F401  -- register all @impl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_text(msg: Message) -> str:
    """从 Message 中提取文本内容。"""
    return "".join(b.text for b in msg.blocks if isinstance(b, TextBlock))


def _get_tool_calls(msg: Message) -> list[ToolUseBlock]:
    """从 Message 中提取 ToolUseBlock。"""
    return [b for b in msg.blocks if isinstance(b, ToolUseBlock)]


async def _single_input(text: str):
    """Create an async iterator yielding a single user Message with TurnStartBlock."""
    from uuid import uuid4
    yield Message(
        role="user",
        blocks=[TurnStartBlock(turn_id=uuid4().hex[:12]), TextBlock(text=text)],
    )


async def _multi_input(*texts: str):
    """Create an async iterator yielding multiple user Messages with TurnStartBlock."""
    from uuid import uuid4
    for text in texts:
        yield Message(
            role="user",
            blocks=[TurnStartBlock(turn_id=uuid4().hex[:12]), TextBlock(text=text)],
        )


async def _collect_events(aiter):
    """Collect all StreamEvents from an async iterator into a list."""
    return [event async for event in aiter]


async def _collect_text(aiter):
    """Collect text from text_delta events."""
    text_parts = []
    async for event in aiter:
        if event.type == "text_delta":
            text_parts.append(event.text)
    return "".join(text_parts)


def _make_stream_events_for_response(response: Response) -> list[StreamEvent]:
    """Build the list of StreamEvents that a non-streaming send_message would yield."""
    events = []
    for block in response.message.blocks:
        if isinstance(block, TextBlock) and block.text:
            events.append(StreamEvent(type="text_delta", text=block.text))
        elif isinstance(block, ToolUseBlock):
            events.append(StreamEvent(type="tool_use_start", tool_call=block))
            events.append(StreamEvent(type="tool_use_end"))
    events.append(StreamEvent(type="response_done", response=response))
    return events


def _make_agent(mock_client=None):
    """Create an Agent with ToolSet for testing."""
    if mock_client is None:
        provider = AnthropicProvider(base_url="https://api.test.com", api_key="test-key")
        mock_client = LLMClient(provider=provider, model="test-model")
    mgr = ModuleManager()
    module_tools = ModuleToolkit(module_manager=mgr)
    tool_set = ToolSet()
    tool_set.add(module_tools)
    context = AgentContext()
    context.prompts.append(
        Message(role="system", blocks=[TextBlock(text="You are a helpful assistant.")], label="base")
    )
    agent = Agent(
        llm=mock_client,
        tools=tool_set,
        context=context,
    )
    tool_set.agent = agent
    return agent, mgr


def _mock_send(events_list):
    """Create a mock async send_message that yields events from a list or list of lists."""
    if events_list and isinstance(events_list[0], list):
        call_idx = 0

        async def mock_send(*args, **kwargs):
            nonlocal call_idx
            evts = events_list[call_idx] if call_idx < len(events_list) else events_list[-1]
            call_idx += 1
            for e in evts:
                yield e

        return mock_send
    else:
        async def mock_send(*args, **kwargs):
            for e in events_list:
                yield e

        return mock_send


# ---------------------------------------------------------------------------
# Declaration tests
# ---------------------------------------------------------------------------

class TestAgentDeclaration:

    def test_inherits_from_mutagent_declaration(self):
        assert issubclass(Agent, mutagent.Declaration)

    def test_uses_declaration_meta(self):
        assert isinstance(Agent, DeclarationMeta)

    def test_declared_methods(self):
        declared = getattr(Agent, _DECLARED_METHODS, set())
        assert "run" in declared
        assert "step" in declared
        assert "handle_tool_calls" in declared


# ---------------------------------------------------------------------------
# Agent loop tests
# ---------------------------------------------------------------------------

class TestAgentLoop:

    @pytest.fixture
    def mock_client(self):
        provider = AnthropicProvider(base_url="https://api.test.com", api_key="test-key")
        return LLMClient(provider=provider, model="test-model")

    @pytest.fixture
    def agent(self, mock_client):
        agent, mgr = _make_agent(mock_client)
        yield agent
        mgr.cleanup()

    async def test_simple_response(self, agent):
        """Agent receives a simple text response (no tool calls)."""
        response = Response(
            message=Message(role="assistant", blocks=[TextBlock(text="Hello! How can I help?")]),
            stop_reason="end_turn",
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        events = _make_stream_events_for_response(response)
        agent.llm.send_message = _mock_send(events)

        text = await _collect_text(agent.run(_single_input("Hi")))

        assert text == "Hello! How can I help?"
        assert len(agent.context.messages) == 2  # user + assistant
        assert agent.context.messages[0].role == "user"
        assert _get_text(agent.context.messages[0]) == "Hi"
        assert agent.context.messages[1].role == "assistant"

    async def test_tool_call_then_response(self, agent):
        """Agent handles a tool call then gets final response."""
        tool_response = Response(
            message=Message(
                role="assistant",
                blocks=[
                    TextBlock(text="Let me inspect the module."),
                    ToolUseBlock(id="tc_1", name="Module-inspect", input={"module_path": "mutagent"}),
                ],
            ),
            stop_reason="tool_use",
        )
        final_response = Response(
            message=Message(role="assistant", blocks=[TextBlock(text="The result is ready.")]),
            stop_reason="end_turn",
        )

        events_1 = _make_stream_events_for_response(tool_response)
        events_2 = _make_stream_events_for_response(final_response)
        agent.llm.send_message = _mock_send([events_1, events_2])

        text = await _collect_text(agent.run(_single_input("What is 1+1?")))

        assert text == "Let me inspect the module.The result is ready."
        # user, assistant(tool_call with result), assistant(final)
        assert len(agent.context.messages) == 3
        # Tool result is on the ToolUseBlock itself
        tc_blocks = _get_tool_calls(agent.context.messages[1])
        assert len(tc_blocks) == 1
        assert tc_blocks[0].status == "done"
        assert "mutagent" in tc_blocks[0].result

    async def test_multiple_tool_calls(self, agent):
        """Agent handles multiple tool calls in one response."""
        tool_response = Response(
            message=Message(
                role="assistant",
                blocks=[
                    ToolUseBlock(id="tc_1", name="Module-inspect", input={"module_path": "mutagent"}),
                    ToolUseBlock(id="tc_2", name="Module-inspect", input={"module_path": "mutagent.agent"}),
                ],
            ),
            stop_reason="tool_use",
        )
        final_response = Response(
            message=Message(role="assistant", blocks=[TextBlock(text="Done.")]),
            stop_reason="end_turn",
        )

        events_1 = _make_stream_events_for_response(tool_response)
        events_2 = _make_stream_events_for_response(final_response)
        agent.llm.send_message = _mock_send([events_1, events_2])

        text = await _collect_text(agent.run(_single_input("Run two things")))

        assert text == "Done."
        tc_blocks = _get_tool_calls(agent.context.messages[1])
        assert len(tc_blocks) == 2
        assert all(b.status == "done" for b in tc_blocks)

    async def test_step_yields_events(self, agent):
        """step() yields StreamEvents from llm.send_message."""
        response = Response(
            message=Message(role="assistant", blocks=[TextBlock(text="Response")]),
            stop_reason="end_turn",
        )
        events = _make_stream_events_for_response(response)
        agent.llm.send_message = _mock_send(events)
        agent.context.messages.append(Message(role="user", blocks=[TextBlock(text="Test")]))

        collected = await _collect_events(agent.step())

        assert len(collected) == 2  # text_delta + response_done
        assert collected[0].type == "text_delta"
        assert collected[0].text == "Response"
        assert collected[1].type == "response_done"
        assert collected[1].response is response

    async def test_handle_tool_calls_dispatches(self, agent):
        """handle_tool_calls dispatches each call through the tool set."""
        blocks = [
            ToolUseBlock(id="tc_1", name="Module-define", input={"module_path": "test_dispatch.mod", "source": "x = 42\n"}),
        ]
        await agent.handle_tool_calls(blocks)

        assert blocks[0].status == "done"
        assert "OK" in blocks[0].result


# ---------------------------------------------------------------------------
# Streaming event sequence tests
# ---------------------------------------------------------------------------

class TestStreamingEventSequence:

    @pytest.fixture
    def agent(self):
        agent, mgr = _make_agent()
        yield agent
        mgr.cleanup()

    async def test_event_order_simple(self, agent):
        """Simple response yields: response_start, text_delta, response_done, turn_done."""
        response = Response(
            message=Message(role="assistant", blocks=[TextBlock(text="Hello")]),
            stop_reason="end_turn",
        )

        async def mock_send(*args, **kwargs):
            yield StreamEvent(type="text_delta", text="Hello")
            yield StreamEvent(type="response_done", response=response)

        agent.llm.send_message = mock_send

        events = await _collect_events(agent.run(_single_input("Hi")))
        types = [e.type for e in events]

        assert types == ["response_start", "text_delta", "response_done", "turn_done"]

    async def test_event_order_with_tool_call(self, agent):
        """Tool call response yields correct event sequence including turn_done."""
        tool_response = Response(
            message=Message(
                role="assistant",
                blocks=[
                    TextBlock(text="Thinking..."),
                    ToolUseBlock(id="tc_1", name="Module-define", input={"module_path": "test_evt.mod", "source": "x=1\n"}),
                ],
            ),
            stop_reason="tool_use",
        )
        final_response = Response(
            message=Message(role="assistant", blocks=[TextBlock(text="Done")]),
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
                    tool_call=ToolUseBlock(id="tc_1", name="Module-define"),
                )
                yield StreamEvent(type="tool_use_end")
                yield StreamEvent(type="response_done", response=tool_response)
            else:
                yield StreamEvent(type="text_delta", text="Done")
                yield StreamEvent(type="response_done", response=final_response)

        agent.llm.send_message = mock_send

        events = await _collect_events(agent.run(_single_input("Calc")))
        types = [e.type for e in events]

        assert types == [
            "response_start",    # 1st LLM call starts
            "text_delta",        # "Thinking..."
            "tool_use_start",    # LLM constructs tool call
            "tool_use_end",
            "response_done",     # first LLM call done
            "tool_exec_start",   # Agent executes tool
            "tool_exec_end",     # tool result (via tool_call field)
            "response_start",    # 2nd LLM call starts
            "text_delta",        # "Done"
            "response_done",     # second LLM call done
            "turn_done",         # turn complete
        ]

        # Verify tool_exec events carry correct data
        exec_start = events[5]
        assert exec_start.tool_call.name == "Module-define"
        exec_end = events[6]
        assert exec_end.tool_call is not None
        assert exec_end.tool_call.id == "tc_1"
        assert exec_end.tool_call.status == "done"

    async def test_error_event_stops_turn(self, agent):
        """An error event from LLM stops the current turn but yields turn_done."""
        async def mock_send(*args, **kwargs):
            yield StreamEvent(type="error", error="API failed")

        agent.llm.send_message = mock_send

        events = await _collect_events(agent.run(_single_input("Hi")))
        types = [e.type for e in events]

        assert types == ["response_start", "error", "turn_done"]
        assert events[1].error == "API failed"
        assert len(agent.context.messages) == 1
        assert agent.context.messages[0].role == "user"

    async def test_stream_false_produces_events(self, agent):
        """stream=False still yields events through the same interface."""
        response = Response(
            message=Message(role="assistant", blocks=[TextBlock(text="Non-streamed")]),
            stop_reason="end_turn",
        )

        async def mock_send(*args, **kwargs):
            yield StreamEvent(type="text_delta", text="Non-streamed")
            yield StreamEvent(type="response_done", response=response)

        agent.llm.send_message = mock_send

        text = await _collect_text(agent.run(_single_input("Test"), stream=False))
        assert text == "Non-streamed"


# ---------------------------------------------------------------------------
# Runtime API tests
# ---------------------------------------------------------------------------

class TestAgentRuntimeAPI:

    @pytest.fixture
    def agent(self):
        agent, mgr = _make_agent()
        agent.config = DictConfig(
            _data={
                "providers": {
                    "test": {
                        "provider": "AnthropicProvider",
                        "models": {
                            "alpha": {"model_id": "model-alpha"},
                            "beta": {"model_id": "model-beta"},
                        },
                        "auth_token": "test-key",
                        "base_url": "https://api.test.com",
                    }
                },
                "default_model": "alpha",
            },
            _listeners=[],
        )
        yield agent
        mgr.cleanup()

    async def test_submit_emits_events(self, agent):
        response = Response(
            message=Message(role="assistant", blocks=[TextBlock(text="Hello from submit")]),
            stop_reason="end_turn",
        )

        async def mock_send(*args, **kwargs):
            yield StreamEvent(type="text_delta", text="Hello from submit")
            yield StreamEvent(type="response_done", response=response)

        agent.llm.send_message = mock_send
        received: list[str] = []
        agent.subscribe(lambda event: received.append(event.type))

        await agent.submit("Hi")
        task = getattr(agent, "_current_task")
        assert task is not None
        await task

        assert received == ["response_start", "text_delta", "response_done", "turn_done"]
        assert agent.is_busy() is False
        assert _get_text(agent.context.messages[-1]) == "Hello from submit"

    async def test_cancel_cancels_running_turn(self, agent):
        async def mock_send(*args, **kwargs):
            yield StreamEvent(type="text_delta", text="partial")
            await asyncio.sleep(1)

        agent.llm.send_message = mock_send
        received: list[str] = []
        agent.subscribe(lambda event: received.append(event.type))

        await agent.submit("Stop soon")
        await asyncio.sleep(0.05)
        assert agent.cancel() is True
        task = getattr(agent, "_current_task")
        assert task is not None
        with pytest.raises(asyncio.CancelledError):
            await task

        assert received[:2] == ["response_start", "text_delta"]
        assert received[-1] == "turn_done"
        assert agent.is_busy() is False

    def test_select_model_switches_client(self, agent):
        assert {model["name"] for model in agent.list_models()} == {"alpha", "beta"}
        agent.select_model("beta")
        assert agent.llm.model == "model-beta"


# ---------------------------------------------------------------------------
# Multi-turn and error recovery tests
# ---------------------------------------------------------------------------

class TestMultiTurnAndErrorRecovery:

    @pytest.fixture
    def agent(self):
        agent, mgr = _make_agent()
        yield agent
        mgr.cleanup()

    async def test_multi_turn_single_run(self, agent):
        """Multiple Messages are processed through a single agent.run() call."""
        response_1 = Response(
            message=Message(role="assistant", blocks=[TextBlock(text="Hi there")]),
            stop_reason="end_turn",
        )
        response_2 = Response(
            message=Message(role="assistant", blocks=[TextBlock(text="I'm fine")]),
            stop_reason="end_turn",
        )

        call_idx = 0

        async def mock_send(*args, **kwargs):
            nonlocal call_idx
            if call_idx == 0:
                call_idx += 1
                yield StreamEvent(type="text_delta", text="Hi there")
                yield StreamEvent(type="response_done", response=response_1)
            else:
                yield StreamEvent(type="text_delta", text="I'm fine")
                yield StreamEvent(type="response_done", response=response_2)

        agent.llm.send_message = mock_send

        events = await _collect_events(
            agent.run(_multi_input("Hello", "How are you?"))
        )
        types = [e.type for e in events]

        assert types == [
            "response_start", "text_delta", "response_done", "turn_done",
            "response_start", "text_delta", "response_done", "turn_done",
        ]

        assert len(agent.context.messages) == 4
        assert _get_text(agent.context.messages[0]) == "Hello"
        assert _get_text(agent.context.messages[1]) == "Hi there"
        assert _get_text(agent.context.messages[2]) == "How are you?"
        assert _get_text(agent.context.messages[3]) == "I'm fine"

    async def test_error_then_continue(self, agent):
        """After an error in one turn, agent continues processing the next input."""
        response_ok = Response(
            message=Message(role="assistant", blocks=[TextBlock(text="OK now")]),
            stop_reason="end_turn",
        )

        call_idx = 0

        async def mock_send(*args, **kwargs):
            nonlocal call_idx
            if call_idx == 0:
                call_idx += 1
                yield StreamEvent(type="error", error="API timeout")
            else:
                yield StreamEvent(type="text_delta", text="OK now")
                yield StreamEvent(type="response_done", response=response_ok)

        agent.llm.send_message = mock_send

        events = await _collect_events(
            agent.run(_multi_input("First", "Second"))
        )
        types = [e.type for e in events]

        assert types == [
            "response_start", "error", "turn_done",
            "response_start", "text_delta", "response_done", "turn_done",
        ]

        assert len(agent.context.messages) == 3
        assert _get_text(agent.context.messages[0]) == "First"
        assert _get_text(agent.context.messages[1]) == "Second"
        assert _get_text(agent.context.messages[2]) == "OK now"


# ---------------------------------------------------------------------------
# stop_reason vs tool_calls mismatch tests
# ---------------------------------------------------------------------------

class TestStopReasonToolCallsMismatch:

    @pytest.fixture
    def agent(self):
        agent, mgr = _make_agent()
        yield agent
        mgr.cleanup()

    async def test_end_turn_with_tool_calls_executes_tools(self, agent):
        """When stop_reason=end_turn but tool_calls exist, tools should still be executed."""
        tool_response = Response(
            message=Message(
                role="assistant",
                blocks=[
                    TextBlock(text="Let me check:"),
                    ToolUseBlock(id="tc_1", name="Module-inspect", input={}),
                ],
            ),
            stop_reason="end_turn",  # mismatch
        )
        final_response = Response(
            message=Message(role="assistant", blocks=[TextBlock(text="Here are the results.")]),
            stop_reason="end_turn",
        )

        events_1 = _make_stream_events_for_response(tool_response)
        events_2 = _make_stream_events_for_response(final_response)
        agent.llm.send_message = _mock_send([events_1, events_2])

        events = await _collect_events(agent.run(_single_input("Check modules")))
        types = [e.type for e in events]

        assert "tool_exec_start" in types
        assert "tool_exec_end" in types
        # user, assistant(tool), assistant(final)
        assert len(agent.context.messages) == 3

    async def test_end_turn_without_tool_calls_ends_turn(self, agent):
        """When stop_reason=end_turn and no tool_calls, turn ends normally."""
        response = Response(
            message=Message(role="assistant", blocks=[TextBlock(text="All done.")]),
            stop_reason="end_turn",
        )
        agent.llm.send_message = _mock_send(_make_stream_events_for_response(response))

        events = await _collect_events(agent.run(_single_input("Hi")))
        types = [e.type for e in events]

        assert types == ["response_start", "text_delta", "response_done", "turn_done"]
        assert len(agent.context.messages) == 2

    async def test_tool_use_with_tool_calls_still_works(self, agent):
        """When stop_reason=tool_use and tool_calls exist, behavior unchanged."""
        tool_response = Response(
            message=Message(
                role="assistant",
                blocks=[
                    TextBlock(text="Inspecting..."),
                    ToolUseBlock(id="tc_1", name="Module-inspect", input={"module_path": "mutagent"}),
                ],
            ),
            stop_reason="tool_use",
        )
        final_response = Response(
            message=Message(role="assistant", blocks=[TextBlock(text="Done.")]),
            stop_reason="end_turn",
        )

        events_1 = _make_stream_events_for_response(tool_response)
        events_2 = _make_stream_events_for_response(final_response)
        agent.llm.send_message = _mock_send([events_1, events_2])

        events = await _collect_events(agent.run(_single_input("Inspect")))
        types = [e.type for e in events]

        assert "tool_exec_start" in types
        assert "tool_exec_end" in types
        assert len(agent.context.messages) == 3


# ---------------------------------------------------------------------------
# max_tool_rounds tests
# ---------------------------------------------------------------------------

class TestMaxToolRounds:

    def _make_tool_loop_agent(self, total_tool_responses):
        """Create an agent that will produce consecutive tool-calling responses."""
        agent, mgr = _make_agent()

        call_idx = 0

        async def mock_send(*args, **kwargs):
            nonlocal call_idx
            idx = call_idx
            call_idx += 1
            if idx < total_tool_responses:
                resp = Response(
                    message=Message(
                        role="assistant",
                        blocks=[
                            TextBlock(text=f"Round {idx}"),
                            ToolUseBlock(id=f"tc_{idx}", name="Module-inspect", input={}),
                        ],
                    ),
                    stop_reason="tool_use",
                )
                for e in _make_stream_events_for_response(resp):
                    yield e
            else:
                resp = Response(
                    message=Message(
                        role="assistant",
                        blocks=[TextBlock(text="Summary of progress.")],
                    ),
                    stop_reason="end_turn",
                )
                for e in _make_stream_events_for_response(resp):
                    yield e

        agent.llm.send_message = mock_send
        return agent, mgr

    async def test_max_tool_rounds_stops_loop(self):
        """Agent stops after MAX_TOOL_ROUNDS (25) tool rounds."""
        from mutagent.builtins.agent_impl import MAX_TOOL_ROUNDS
        agent, mgr = self._make_tool_loop_agent(total_tool_responses=30)
        try:
            events = await _collect_events(agent.run(_single_input("Do work")))
            types = [e.type for e in events]

            tool_exec_starts = [e for e in events if e.type == "tool_exec_start"]
            assert len(tool_exec_starts) == MAX_TOOL_ROUNDS

            limit_msgs = [
                m for m in agent.context.messages
                if m.role == "user" and "Tool call limit reached" in _get_text(m)
            ]
            assert len(limit_msgs) == 1
            assert types[-1] == "turn_done"
        finally:
            mgr.cleanup()

    async def test_below_max_tool_rounds_normal(self):
        """Agent completes normally when tool rounds are below limit."""
        agent, mgr = self._make_tool_loop_agent(total_tool_responses=3)
        try:
            events = await _collect_events(agent.run(_single_input("Small task")))
            types = [e.type for e in events]

            tool_exec_starts = [e for e in events if e.type == "tool_exec_start"]
            assert len(tool_exec_starts) == 3

            limit_msgs = [
                m for m in agent.context.messages
                if m.role == "user" and "Tool call limit reached" in _get_text(m)
            ]
            assert len(limit_msgs) == 0
            assert types[-1] == "turn_done"
        finally:
            mgr.cleanup()
