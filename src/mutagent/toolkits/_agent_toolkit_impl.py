"""mutagent.toolkits._agent_toolkit_impl -- AgentToolkit delegate implementation."""

import asyncio
import logging

import mutobj
from .agent_toolkit import AgentToolkit
from mutagent.core.messages import StreamEvent

logger = logging.getLogger(__name__)


@mutobj.impl(AgentToolkit.delegate)
async def agent_toolkit_delegate(self: AgentToolkit, agent_name: str, task: str) -> str:
    """Delegate a task to a named Sub-Agent (async)."""
    agent = self.agents.get(agent_name)
    if agent is None:
        available = list(self.agents.keys())
        return f"Unknown agent: {agent_name}. Available: {available}"

    logger.info("Delegating to sub-agent '%s': %.100s", agent_name, task)

    # Clear message history (each call is independent)
    agent.context.messages.clear()

    text_parts: list[str] = []
    done = asyncio.Event()

    def on_event(event: StreamEvent) -> None:
        if event.type == "text_delta" and event.text:
            text_parts.append(event.text)
        elif event.type == "turn_done":
            done.set()
        elif event.type == "error":
            text_parts.append(f"\n[Error: {event.error}]")
            done.set()

    disposable = agent.subscribe(on_event)
    try:
        await agent.submit(task)
        await done.wait()
        result = "".join(text_parts)
        logger.info("Sub-agent '%s' completed (%d chars)", agent_name, len(result))
        return result
    finally:
        disposable()
