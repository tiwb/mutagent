"""End-to-end integration tests for mutagent Agent."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from mutagent.agent import Agent
from mutagent.client import LLMClient
from mutagent.essential_tools import EssentialTools
from mutagent.main import create_agent
from mutagent.messages import Message, Response, ToolCall, ToolResult, ToolSchema
from mutagent.runtime.module_manager import ModuleManager
from mutagent.selector import ToolSelector


class TestCreateAgent:

    def test_create_agent_returns_agent(self):
        agent = create_agent(api_key="test-key")
        assert isinstance(agent, Agent)
        assert agent.client.api_key == "test-key"
        assert agent.client.model == "claude-sonnet-4-20250514"
        assert agent.system_prompt
        assert agent.messages == []

    def test_create_agent_custom_params(self):
        agent = create_agent(
            api_key="key",
            model="custom-model",
            system_prompt="Custom prompt",
        )
        assert agent.client.model == "custom-model"
        assert agent.system_prompt == "Custom prompt"


class TestEndToEnd:
    """Simulate full Agent workflow with mock LLM responses."""

    @pytest.fixture
    def agent(self):
        agent = create_agent(api_key="test-key")
        yield agent

    @pytest.mark.asyncio
    async def test_inspect_then_patch_then_run(self, agent, tmp_path):
        """Simulate: Agent inspects module -> patches code -> runs code -> saves."""
        # Step 1: LLM asks to inspect a module
        inspect_response = Response(
            message=Message(
                role="assistant",
                content="Let me inspect the module structure.",
                tool_calls=[ToolCall(
                    id="tc_1",
                    name="inspect_module",
                    arguments={"module_path": "mutagent", "depth": 1},
                )],
            ),
            stop_reason="tool_use",
        )

        # Step 2: LLM patches a new module
        patch_response = Response(
            message=Message(
                role="assistant",
                content="I'll create a helper module.",
                tool_calls=[ToolCall(
                    id="tc_2",
                    name="patch_module",
                    arguments={
                        "module_path": "test_e2e.helper",
                        "source": "def add(a, b):\n    return a + b\n",
                    },
                )],
            ),
            stop_reason="tool_use",
        )

        # Step 3: LLM runs the code to verify
        run_response = Response(
            message=Message(
                role="assistant",
                content="Let me verify the module works.",
                tool_calls=[ToolCall(
                    id="tc_3",
                    name="run_code",
                    arguments={
                        "code": "from test_e2e.helper import add\nprint(add(2, 3))",
                    },
                )],
            ),
            stop_reason="tool_use",
        )

        # Step 4: LLM saves the module
        save_response = Response(
            message=Message(
                role="assistant",
                content="Saving the module.",
                tool_calls=[ToolCall(
                    id="tc_4",
                    name="save_module",
                    arguments={
                        "module_path": "test_e2e.helper",
                        "file_path": str(tmp_path),
                    },
                )],
            ),
            stop_reason="tool_use",
        )

        # Step 5: Final response
        final_response = Response(
            message=Message(
                role="assistant",
                content="Done! I created a helper module with an add function.",
            ),
            stop_reason="end_turn",
        )

        agent.client.send_message = AsyncMock(side_effect=[
            inspect_response,
            patch_response,
            run_response,
            save_response,
            final_response,
        ])

        result = await agent.run("Create a helper module with an add function")

        # Verify final result
        assert "Done" in result

        # Verify the tool interactions happened
        assert len(agent.messages) == 10  # user + 4*(assistant+tool_result) + final_assistant

        # Verify inspect result was in messages
        inspect_result = agent.messages[2].tool_results[0]
        assert "mutagent" in inspect_result.content

        # Verify patch result
        patch_result = agent.messages[4].tool_results[0]
        assert "OK" in patch_result.content

        # Verify run result
        run_result = agent.messages[6].tool_results[0]
        assert "5" in run_result.content

        # Verify save result
        save_result = agent.messages[8].tool_results[0]
        assert "OK" in save_result.content

        # Verify file was actually saved
        saved_file = tmp_path / "test_e2e" / "helper.py"
        assert saved_file.exists()
        assert "def add" in saved_file.read_text()

    @pytest.mark.asyncio
    async def test_view_source_of_patched_module(self, agent):
        """Agent patches a module then views its source."""
        # Patch
        patch_resp = Response(
            message=Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(
                    id="tc_1",
                    name="patch_module",
                    arguments={
                        "module_path": "test_e2e.src",
                        "source": "class Greeter:\n    def greet(self):\n        return 'hi'\n",
                    },
                )],
            ),
            stop_reason="tool_use",
        )

        # View source
        view_resp = Response(
            message=Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(
                    id="tc_2",
                    name="view_source",
                    arguments={"target": "test_e2e.src.Greeter"},
                )],
            ),
            stop_reason="tool_use",
        )

        final_resp = Response(
            message=Message(role="assistant", content="Here's the Greeter class."),
            stop_reason="end_turn",
        )

        agent.client.send_message = AsyncMock(side_effect=[patch_resp, view_resp, final_resp])

        result = await agent.run("Show me the Greeter class")

        # Verify view_source returned the source
        view_result = agent.messages[4].tool_results[0]
        assert "class Greeter" in view_result.content
        assert "return 'hi'" in view_result.content

    @pytest.mark.asyncio
    async def test_simple_chat_no_tools(self, agent):
        """Agent can respond without using any tools."""
        response = Response(
            message=Message(role="assistant", content="Hello! I'm mutagent."),
            stop_reason="end_turn",
        )
        agent.client.send_message = AsyncMock(return_value=response)

        result = await agent.run("Hello")
        assert result == "Hello! I'm mutagent."
        assert len(agent.messages) == 2
