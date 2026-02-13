"""Tests for Agent declaration and main loop implementation."""

from unittest.mock import AsyncMock

import pytest

import mutagent
from mutagent.agent import Agent
from mutagent.base import MutagentMeta
from mutagent.client import LLMClient
from mutagent.essential_tools import EssentialTools
from mutagent.main import load_builtins
from mutagent.messages import Message, Response, ToolCall, ToolResult, ToolSchema
from mutagent.runtime.module_manager import ModuleManager
from mutagent.selector import ToolSelector
from forwardpy.core import _DECLARED_METHODS

# Load all impls (idempotent)
load_builtins()


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
        agent.client.send_message = AsyncMock(return_value=response)

        result = await agent.run("Hi")

        assert result == "Hello! How can I help?"
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
        agent.client.send_message = AsyncMock(side_effect=[tool_response, final_response])

        result = await agent.run("What is 1+1?")

        assert result == "The result is 2."
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
        agent.client.send_message = AsyncMock(side_effect=[tool_response, final_response])

        result = await agent.run("Run two things")

        assert result == "Done."
        assert len(agent.messages[2].tool_results) == 2

    @pytest.mark.asyncio
    async def test_step_calls_send_message(self, agent):
        """step() calls client.send_message with messages and tools."""
        response = Response(
            message=Message(role="assistant", content="Response"),
            stop_reason="end_turn",
        )
        agent.client.send_message = AsyncMock(return_value=response)
        agent.messages.append(Message(role="user", content="Test"))

        result = await agent.step()

        assert result is response
        agent.client.send_message.assert_called_once()
        call_args = agent.client.send_message.call_args
        # First arg: messages, second arg: tools
        assert len(call_args[0][0]) == 1  # 1 message
        assert len(call_args[0][1]) == 5  # 5 tool schemas

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
