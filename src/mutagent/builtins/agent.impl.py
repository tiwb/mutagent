"""mutagent.builtins.agent -- Agent main loop implementation."""

import mutagent
from mutagent.agent import Agent
from mutagent.messages import Message, Response, ToolCall, ToolResult


@mutagent.impl(Agent.run)
async def run(self: Agent, user_input: str) -> str:
    """Run the agent loop until end_turn."""
    # Add user message
    self.messages.append(Message(role="user", content=user_input))

    while True:
        response = await self.step()

        # Add assistant message to history
        self.messages.append(response.message)

        if response.stop_reason == "tool_use" and response.message.tool_calls:
            # Handle tool calls
            results = await self.handle_tool_calls(response.message.tool_calls)
            # Add tool results as a user message
            self.messages.append(Message(role="user", tool_results=results))
        else:
            # end_turn or no tool calls
            return response.message.content


@mutagent.impl(Agent.step)
async def step(self: Agent) -> Response:
    """Execute a single LLM call."""
    # Get available tools
    tools = await self.tool_selector.get_tools({})

    # Build messages with system prompt
    messages = list(self.messages)

    # Send to LLM (system prompt passed as first user message if needed)
    # For Claude API, system prompt is handled via the system parameter
    # For MVP, we prepend it as context
    if self.system_prompt and messages and messages[0].role == "user":
        # Inject system context into the conversation
        pass  # Claude handles system prompt separately via the API

    return await self.client.send_message(messages, tools)


@mutagent.impl(Agent.handle_tool_calls)
async def handle_tool_calls(
    self: Agent, tool_calls: list[ToolCall]
) -> list[ToolResult]:
    """Dispatch tool calls through the selector."""
    results = []
    for call in tool_calls:
        result = await self.tool_selector.dispatch(call)
        results.append(result)
    return results
