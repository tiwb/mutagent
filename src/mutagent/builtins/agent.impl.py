"""mutagent.builtins.agent -- Agent main loop implementation."""

from typing import AsyncIterator

import mutagent
from mutagent.agent import Agent
from mutagent.messages import Message, StreamEvent, ToolCall, ToolResult


@mutagent.impl(Agent.run)
async def run(
    self: Agent, user_input: str, stream: bool = True
) -> AsyncIterator[StreamEvent]:
    """Run the agent loop until end_turn, yielding streaming events."""
    # Add user message
    self.messages.append(Message(role="user", content=user_input))

    while True:
        response = None
        async for event in self.step(stream=stream):
            yield event
            if event.type == "response_done":
                response = event.response
            elif event.type == "error":
                return  # Stop on error

        if response is None:
            yield StreamEvent(
                type="error", error="No response_done event received from LLM"
            )
            return

        # Add assistant message to history
        self.messages.append(response.message)

        if response.stop_reason == "tool_use" and response.message.tool_calls:
            # Handle tool calls, yielding execution events
            results = []
            for call in response.message.tool_calls:
                yield StreamEvent(type="tool_exec_start", tool_call=call)
                result = await self.tool_selector.dispatch(call)
                yield StreamEvent(
                    type="tool_exec_end", tool_call=call, tool_result=result
                )
                results.append(result)
            # Add tool results as a user message
            self.messages.append(Message(role="user", tool_results=results))
        else:
            # end_turn or no tool calls — done
            return


@mutagent.impl(Agent.step)
async def step(
    self: Agent, stream: bool = True
) -> AsyncIterator[StreamEvent]:
    """Execute a single LLM call, yielding streaming events."""
    tools = await self.tool_selector.get_tools({})
    async for event in self.client.send_message(
        self.messages, tools, system_prompt=self.system_prompt, stream=stream,
    ):
        yield event


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
