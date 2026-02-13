"""End-to-end integration tests for mutagent Agent."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import forwardpy

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


class TestSelfEvolution:
    """Verify Agent can create new tool modules, patch ToolSelector, and use them."""

    @pytest.fixture
    def agent(self):
        agent = create_agent(api_key="test-key")
        yield agent
        # Cleanup: unregister override impls, remove virtual modules,
        # then re-load the original selector impl to restore original impls.
        mgr = agent.tool_selector.essential_tools.module_manager
        forwardpy.unregister_module_impls("user_tools.selector_ext")
        mgr.cleanup()
        # Re-execute the original selector impl to restore ToolSelector impls
        self._reload_selector_impl()

    @staticmethod
    def _reload_selector_impl():
        """Re-execute the builtins/selector.impl.py to restore original @impls."""
        from pathlib import Path
        selector_impl = Path(__file__).resolve().parent.parent / "src" / "mutagent" / "builtins" / "selector.impl.py"
        source = selector_impl.read_text(encoding="utf-8")
        mod = sys.modules.get("mutagent.builtins.selector")
        if mod is not None:
            code = compile(source, str(selector_impl), "exec")
            exec(code, mod.__dict__)

    @pytest.mark.asyncio
    async def test_create_tool_and_use_it(self, agent):
        """Self-evolution: Agent creates a new tool class, patches ToolSelector, then uses it."""
        # Step 1: Agent creates a new tool class declaration
        create_decl_resp = Response(
            message=Message(
                role="assistant",
                content="I'll create a math tools module.",
                tool_calls=[ToolCall(
                    id="tc_1",
                    name="patch_module",
                    arguments={
                        "module_path": "user_tools.math_tools",
                        "source": (
                            "import mutagent\n"
                            "\n"
                            "class MathTools(mutagent.Object):\n"
                            "    def factorial(self, n: int) -> str:\n"
                            "        '''Compute factorial of n.'''\n"
                            "        ...\n"
                        ),
                    },
                )],
            ),
            stop_reason="tool_use",
        )

        # Step 2: Agent provides the implementation
        create_impl_resp = Response(
            message=Message(
                role="assistant",
                content="Now I'll implement the factorial method.",
                tool_calls=[ToolCall(
                    id="tc_2",
                    name="patch_module",
                    arguments={
                        "module_path": "user_tools.math_tools_impl",
                        "source": (
                            "import mutagent\n"
                            "from user_tools.math_tools import MathTools\n"
                            "\n"
                            "@mutagent.impl(MathTools.factorial)\n"
                            "def factorial(self, n: int) -> str:\n"
                            "    result = 1\n"
                            "    for i in range(2, n + 1):\n"
                            "        result *= i\n"
                            "    return str(result)\n"
                        ),
                    },
                )],
            ),
            stop_reason="tool_use",
        )

        # Step 3: Agent verifies the new tool works via run_code
        verify_resp = Response(
            message=Message(
                role="assistant",
                content="Let me verify it works.",
                tool_calls=[ToolCall(
                    id="tc_3",
                    name="run_code",
                    arguments={
                        "code": (
                            "from user_tools.math_tools import MathTools\n"
                            "mt = MathTools()\n"
                            "print(mt.factorial(5))\n"
                        ),
                    },
                )],
            ),
            stop_reason="tool_use",
        )

        # Step 4: Agent creates a SEPARATE override module for ToolSelector
        # (does NOT patch mutagent.builtins.selector, preserving original helpers)
        patch_selector_resp = Response(
            message=Message(
                role="assistant",
                content="Now I'll extend the ToolSelector to include the new tool.",
                tool_calls=[ToolCall(
                    id="tc_4",
                    name="patch_module",
                    arguments={
                        "module_path": "user_tools.selector_ext",
                        "source": (
                            "import asyncio\n"
                            "import mutagent\n"
                            "from mutagent.selector import ToolSelector\n"
                            "from mutagent.messages import ToolResult\n"
                            "from mutagent.builtins.selector import make_schema_from_method, _TOOL_METHODS\n"
                            "\n"
                            "@mutagent.impl(ToolSelector.get_tools, override=True)\n"
                            "async def get_tools(self, context):\n"
                            "    schemas = []\n"
                            "    for name in _TOOL_METHODS:\n"
                            "        schemas.append(make_schema_from_method(self.essential_tools, name))\n"
                            "    from user_tools.math_tools import MathTools\n"
                            "    mt = MathTools()\n"
                            "    schemas.append(make_schema_from_method(mt, 'factorial'))\n"
                            "    return schemas\n"
                            "\n"
                            "@mutagent.impl(ToolSelector.dispatch, override=True)\n"
                            "async def dispatch(self, tool_call):\n"
                            "    method = getattr(self.essential_tools, tool_call.name, None)\n"
                            "    if method is None:\n"
                            "        from user_tools.math_tools import MathTools\n"
                            "        mt = MathTools()\n"
                            "        method = getattr(mt, tool_call.name, None)\n"
                            "    if method is None:\n"
                            "        return ToolResult(tool_call_id=tool_call.id, content='Unknown tool', is_error=True)\n"
                            "    try:\n"
                            "        result = method(**tool_call.arguments)\n"
                            "        if asyncio.iscoroutine(result):\n"
                            "            result = await result\n"
                            "        return ToolResult(tool_call_id=tool_call.id, content=str(result))\n"
                            "    except Exception as e:\n"
                            "        return ToolResult(tool_call_id=tool_call.id, content=str(e), is_error=True)\n"
                        ),
                    },
                )],
            ),
            stop_reason="tool_use",
        )

        # Step 5: Agent uses the new tool directly (dispatched by patched ToolSelector)
        use_new_tool_resp = Response(
            message=Message(
                role="assistant",
                content="Let me compute factorial(6).",
                tool_calls=[ToolCall(
                    id="tc_5",
                    name="factorial",
                    arguments={"n": 6},
                )],
            ),
            stop_reason="tool_use",
        )

        # Step 6: Final response
        final_resp = Response(
            message=Message(
                role="assistant",
                content="factorial(6) = 720. The self-evolution is complete!",
            ),
            stop_reason="end_turn",
        )

        agent.client.send_message = AsyncMock(side_effect=[
            create_decl_resp,
            create_impl_resp,
            verify_resp,
            patch_selector_resp,
            use_new_tool_resp,
            final_resp,
        ])

        result = await agent.run("Create a factorial tool and use it")

        # Verify the full workflow completed
        assert "720" in result

        # Verify run_code result: factorial(5) = 120
        run_result = agent.messages[6].tool_results[0]
        assert "120" in run_result.content

        # Verify the new tool was dispatched and returned correct result
        factorial_result = agent.messages[10].tool_results[0]
        assert "720" in factorial_result.content
        assert factorial_result.is_error is False
